from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Stock


@dataclass(frozen=True)
class _LiquidityConfig:
    min_avg_turnover: float


def get_universe(session: Session, asof_date: date, config) -> pd.DataFrame:
    """取得可用股票 universe（目前維持既有邏輯：上市且 security_type=stock）。"""
    _ = asof_date
    _ = config
    stmt = (
        select(Stock.stock_id)
        .where(Stock.security_type == "stock")
        .where(Stock.is_listed == True)
        .order_by(Stock.stock_id)
    )
    rows = session.execute(stmt).fetchall()
    return pd.DataFrame({"stock_id": [str(row[0]) for row in rows]})


def apply_liquidity_filter(price_df: pd.DataFrame, config) -> pd.DataFrame:
    """以近 20 日平均成交值（close * volume）做流動性過濾。"""
    if price_df.empty:
        return pd.DataFrame(columns=["stock_id", "avg_turnover"])

    df = price_df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["stock_id", "trading_date", "close", "volume"])
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "avg_turnover"])

    recent = (
        df.sort_values(["stock_id", "trading_date"])
        .groupby("stock_id", as_index=False, group_keys=False)
        .tail(20)
        .copy()
    )
    recent["turnover"] = recent["close"] * recent["volume"]
    avg_turnover = (
        recent.groupby("stock_id")["turnover"]
        .mean()
        .rename("avg_turnover")
        .reset_index()
    )
    min_amt_20 = float(getattr(config, "min_amt_20", 0.0) or 0.0)
    if min_amt_20 > 0:
        threshold = min_amt_20
    else:
        # 向後相容：舊版使用「億元」門檻
        threshold = float(getattr(config, "min_avg_turnover", 0.0)) * 1e8
    if threshold > 0:
        avg_turnover = avg_turnover[avg_turnover["avg_turnover"] >= threshold]
    return avg_turnover.reset_index(drop=True)


def pick_topn(scores_df: pd.DataFrame, topn: int) -> pd.DataFrame:
    if scores_df.empty:
        return scores_df.copy()
    return scores_df.sort_values("score", ascending=False).head(topn).copy()


def compute_atr(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """計算每支股票的 ATR（Average True Range）。

    若 price_df 含有 high/low 欄位則使用完整 True Range；
    否則退回以 |close - prev_close| 近似（close-to-close ATR）。

    Returns:
        DataFrame with columns: stock_id, trading_date, atr
    """
    if price_df.empty:
        return pd.DataFrame(columns=["stock_id", "trading_date", "atr"])

    df = price_df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    has_hl = "high" in df.columns and "low" in df.columns
    if has_hl:
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")

    df = df.sort_values(["stock_id", "trading_date"]).copy()
    df["prev_close"] = df.groupby("stock_id")["close"].shift(1)

    if has_hl:
        hl = (df["high"] - df["low"]).abs()
        hc = (df["high"] - df["prev_close"]).abs()
        lc = (df["low"] - df["prev_close"]).abs()
        df["tr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    else:
        df["tr"] = (df["close"] - df["prev_close"]).abs()

    # Wilder 指數移動平均（EWM span = period）
    df["atr"] = df.groupby("stock_id")["tr"].transform(
        lambda x: x.ewm(span=period, adjust=False).mean()
    )

    return (
        df[["stock_id", "trading_date", "atr"]]
        .dropna(subset=["atr"])
        .reset_index(drop=True)
    )


def compute_position_weights(
    scores_df: pd.DataFrame,
    method: str = "equal",
    price_df: Optional[pd.DataFrame] = None,
    atr_period: int = 14,
) -> pd.DataFrame:
    """計算每檔股票的倉位權重。

    Args:
        scores_df: DataFrame with 'stock_id' and 'score'，依 score 降序排列
        method:
            - "equal": 等權重（預設）
            - "score_tiered": 依 score 分層（前 25% 雙倍、後 25% 半倍）
            - "vol_inverse": 波動率反比（需提供 price_df）
        price_df: 近期價格資料（vol_inverse 模式使用）
        atr_period: ATR 計算週期

    Returns:
        DataFrame with columns: stock_id, weight（總和為 1.0）
    """
    if scores_df.empty:
        return pd.DataFrame(columns=["stock_id", "weight"])

    df = scores_df[["stock_id"]].copy().reset_index(drop=True)
    n = len(df)

    if method == "score_tiered":
        top_cut = max(1, n // 4)
        bot_cut = min(n, max(top_cut + 1, (n * 3) // 4))
        raw = np.ones(n, dtype=float)
        raw[:top_cut] = 2.0
        raw[bot_cut:] = 0.5
        weights = raw / raw.sum()

    elif method == "vol_inverse" and price_df is not None and not price_df.empty:
        atr_df = compute_atr(price_df, period=atr_period)
        latest_atr = (
            atr_df.sort_values("trading_date")
            .groupby("stock_id")["atr"]
            .last()
            .reset_index()
        )
        df = df.merge(latest_atr, on="stock_id", how="left")
        median_atr = df["atr"].median()
        df["atr"] = df["atr"].fillna(median_atr if pd.notna(median_atr) else 1.0)
        df["atr"] = df["atr"].clip(lower=1e-6)
        raw = 1.0 / df["atr"].values
        weights = raw / raw.sum()

    else:
        weights = np.ones(n) / n

    result = scores_df[["stock_id"]].copy().reset_index(drop=True)
    result["weight"] = weights
    return result


def apply_stoploss(
    trades_or_positions_df: pd.DataFrame,
    price_df: pd.DataFrame,
    stoploss_pct: float,
    per_stock_stoploss: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """套用停損規則，回傳每筆部位的最終出場價與是否觸發停損。

    Args:
        trades_or_positions_df: 需含 stock_id / entry_date / planned_exit_date / entry_price
        price_df: 持有期間內的價格資料
        stoploss_pct: 全局停損百分比（如 -0.07 = -7%）
        per_stock_stoploss: 個股動態停損 {stock_id: stoploss_pct}，優先於全局值
    """
    if trades_or_positions_df.empty:
        return pd.DataFrame(columns=["stock_id", "entry_price", "exit_price", "exit_date", "stoploss_triggered"])

    results = []
    for _, row in trades_or_positions_df.iterrows():
        stock_id = str(row["stock_id"])
        entry_date = row["entry_date"]
        planned_exit_date = row["planned_exit_date"]
        entry_price = float(row["entry_price"])
        if entry_price <= 0:
            continue

        effective_stoploss = (
            per_stock_stoploss.get(stock_id, stoploss_pct)
            if per_stock_stoploss
            else stoploss_pct
        )

        stock_prices = price_df[
            (price_df["stock_id"].astype(str) == stock_id)
            & (price_df["trading_date"] >= entry_date)
            & (price_df["trading_date"] <= planned_exit_date)
        ].sort_values("trading_date")
        if stock_prices.empty:
            continue

        exit_price = entry_price
        exit_date = entry_date
        stoploss_triggered = False
        for _, px_row in stock_prices.iterrows():
            trading_date = px_row["trading_date"]
            close = float(px_row["close"])
            if trading_date == entry_date:
                exit_price = close
                exit_date = trading_date
                continue
            current_ret = close / entry_price - 1
            if effective_stoploss < 0 and current_ret <= effective_stoploss:
                exit_price = close
                exit_date = trading_date
                stoploss_triggered = True
                break
            exit_price = close
            exit_date = trading_date

        results.append(
            {
                "stock_id": stock_id,
                "entry_date": entry_date,
                "planned_exit_date": planned_exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_date": exit_date,
                "stoploss_triggered": stoploss_triggered,
            }
        )
    return pd.DataFrame(results)


def apply_trailing_stop(
    trades_or_positions_df: pd.DataFrame,
    price_df: pd.DataFrame,
    trailing_stop_pct: float,
    stoploss_pct: float = -0.07,
    per_stock_stoploss: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """移動停利 + 固定停損：從持有期最高收盤回落觸發出場，或跌破固定停損，以先觸發者為準。

    Args:
        trades_or_positions_df: 需含 stock_id / entry_date / planned_exit_date / entry_price
        price_df: 持有期間內的價格資料
        trailing_stop_pct: 從最高點回落觸發比例（如 -0.12 = -12%）
        stoploss_pct: 以進場價為基準的固定停損（如 -0.07 = -7%）
        per_stock_stoploss: 個股動態固定停損 dict，優先於全局 stoploss_pct
    """
    if trades_or_positions_df.empty:
        return pd.DataFrame(columns=["stock_id", "entry_price", "exit_price", "exit_date", "stoploss_triggered"])

    results = []
    for _, row in trades_or_positions_df.iterrows():
        stock_id = str(row["stock_id"])
        entry_date = row["entry_date"]
        planned_exit_date = row["planned_exit_date"]
        entry_price = float(row["entry_price"])
        if entry_price <= 0:
            continue

        effective_stoploss = (
            per_stock_stoploss.get(stock_id, stoploss_pct)
            if per_stock_stoploss
            else stoploss_pct
        )

        stock_prices = price_df[
            (price_df["stock_id"].astype(str) == stock_id)
            & (price_df["trading_date"] >= entry_date)
            & (price_df["trading_date"] <= planned_exit_date)
        ].sort_values("trading_date")
        if stock_prices.empty:
            continue

        exit_price = entry_price
        exit_date = entry_date
        stoploss_triggered = False
        peak_close = entry_price

        for _, px_row in stock_prices.iterrows():
            trading_date = px_row["trading_date"]
            close = float(px_row["close"])

            if trading_date == entry_date:
                exit_price = close
                exit_date = trading_date
                peak_close = max(peak_close, close)
                continue

            if close > peak_close:
                peak_close = close

            # 固定停損（以進場價為基準）
            hard_ret = close / entry_price - 1
            if effective_stoploss < 0 and hard_ret <= effective_stoploss:
                exit_price = close
                exit_date = trading_date
                stoploss_triggered = True
                break

            # 移動停利（從最高點回落）
            drawdown_from_peak = close / peak_close - 1
            if trailing_stop_pct < 0 and drawdown_from_peak <= trailing_stop_pct:
                exit_price = close
                exit_date = trading_date
                stoploss_triggered = True
                break

            exit_price = close
            exit_date = trading_date

        results.append(
            {
                "stock_id": stock_id,
                "entry_date": entry_date,
                "planned_exit_date": planned_exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_date": exit_date,
                "stoploss_triggered": stoploss_triggered,
            }
        )
    return pd.DataFrame(results)

