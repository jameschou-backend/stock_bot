from __future__ import annotations

from dataclasses import dataclass
from datetime import date

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


def apply_stoploss(
    trades_or_positions_df: pd.DataFrame,
    price_df: pd.DataFrame,
    stoploss_pct: float,
) -> pd.DataFrame:
    """套用停損規則，回傳每筆部位的最終出場價與是否觸發停損。"""
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
            if stoploss_pct < 0 and current_ret <= stoploss_pct:
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

