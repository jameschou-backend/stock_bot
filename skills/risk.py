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
    atr_stoploss_multiplier: Optional[float] = None,
    atr_at_entry: Optional[pd.Series] = None,
    stagnant_days: int = 10,
    stagnant_threshold: float = 0.03,
) -> pd.DataFrame:
    """移動停利 + 固定停損：從持有期最高收盤回落觸發出場，或跌破固定停損，以先觸發者為準。

    Args:
        trades_or_positions_df: 需含 stock_id / entry_date / planned_exit_date / entry_price
        price_df: 持有期間內的價格資料
        trailing_stop_pct: 從最高點回落觸發比例（如 -0.12 = -12%），若有 ATR 設定，此為保底的最小容忍度
        stoploss_pct: 以進場價為基準的固定停損（如 -0.07 = -7%）
        per_stock_stoploss: 個股動態固定停損 dict，優先於全局 stoploss_pct
        atr_stoploss_multiplier: ATR 倍數（如 2.5），若設定則作為動態移動停利基準
        atr_at_entry: 每檔股票進場時的 ATR 值（Series, index: stock_id）
        stagnant_days: 超過幾天不創新高啟動淘汰 (預設 10 天)
        stagnant_threshold: 在 stagnant_days 期間即使沒破底但漲幅低於此比例就視為死魚股淘汰 (預設 0.03)
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

        # ── 計算動態移動停利 (Adaptive Trailing Stop) ──
        effective_trailing_stop = trailing_stop_pct
        if atr_stoploss_multiplier is not None and atr_at_entry is not None and stock_id in atr_at_entry.index:
            atr_val = float(atr_at_entry[stock_id])
            dynamic_trailing = -(atr_stoploss_multiplier * atr_val / entry_price)
            # 若有指定全域 trailing_stop_pct，把它當作最嚴格的門檻（比如不讓停利無限制寬鬆）
            # 或者當作較寬的門檻，讓 ATR 主導。這裡採「取較小值（即容忍度較寬的）」
            # 例如: dynamic = -0.05, trailing_stop_pct = -0.15 -> -0.15 (保護牛皮股不會太快被洗)
            # 例如: dynamic = -0.15, trailing_stop_pct = -0.08 -> -0.15 (保護飆股不被預設的 -8% 洗掉)
            if trailing_stop_pct is not None and trailing_stop_pct < 0:
                effective_trailing_stop = min(trailing_stop_pct, dynamic_trailing)
            else:
                effective_trailing_stop = dynamic_trailing
            
            # 最大容忍度 30%
            effective_trailing_stop = max(effective_trailing_stop, -0.30)
        elif trailing_stop_pct is None:
             effective_trailing_stop = -0.10 # default fallback if nothing is meant to be trailing


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
        
        # 紀錄創新高天數，用於汰弱留強
        days_held = 0
        days_since_new_high = 0

        for _, px_row in stock_prices.iterrows():
            trading_date = px_row["trading_date"]
            close = float(px_row["close"])

            if trading_date == entry_date:
                exit_price = close
                exit_date = trading_date
                peak_close = max(peak_close, close)
                continue

            days_held += 1

            if close > peak_close:
                peak_close = close
                days_since_new_high = 0
            else:
                days_since_new_high += 1

            # 固定停損（以進場價為基準）
            hard_ret = close / entry_price - 1
            if effective_stoploss < 0 and hard_ret <= effective_stoploss:
                exit_price = close
                exit_date = trading_date
                stoploss_triggered = True
                break

            # 階段性停利保護 (Stage-based Take Profit)
            # 若最高點獲利超過特定門檻，緊縮 trailing_stop
            current_drawdown_stop = effective_trailing_stop
            peak_ret = peak_close / entry_price - 1
            if peak_ret >= 0.20:
                # 獲利超過 20% 時，拉緊停利為 5% 或是至少保本 +10%
                current_drawdown_stop = max(current_drawdown_stop, -0.05)
                # 另外一種保底條件：不能跌破進場的1.1倍
                if close < entry_price * 1.10:
                    exit_price = close
                    exit_date = trading_date
                    stoploss_triggered = True
                    break
            elif peak_ret >= 0.10:
                # 獲利超過 10% 時，停利至少改為保本 (或小賺 2%)
                if close < entry_price * 1.02:
                    exit_price = close
                    exit_date = trading_date
                    stoploss_triggered = True
                    break

            # 移動停利（從最高點回落）
            drawdown_from_peak = close / peak_close - 1
            if current_drawdown_stop < 0 and drawdown_from_peak <= current_drawdown_stop:
                exit_price = close
                exit_date = trading_date
                stoploss_triggered = True
                break

            # 汰弱留強機制 (Time-Stop Rotation)
            if days_since_new_high >= stagnant_days and hard_ret < stagnant_threshold:
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

