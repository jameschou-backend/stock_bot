"""Recent coverage by market; a global MAX(date) cannot establish completeness."""
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text


def recent_market_counts(session, end: date, days: int = 21) -> pd.DataFrame:
    return pd.read_sql(text("""
        SELECT p.trading_date, s.market, COUNT(*) AS rows_count
        FROM raw_prices p JOIN stocks s ON s.stock_id=p.stock_id
        WHERE p.trading_date BETWEEN :start AND :end AND p.close>0
          AND s.market IN ('TWSE','TPEX') AND s.security_type='stock'
        GROUP BY p.trading_date, s.market ORDER BY p.trading_date, s.market
    """), session.get_bind(), params={"start": end - timedelta(days=days), "end": end})


def incomplete_dates(counts: pd.DataFrame, *, ratio: float = .9) -> list[date]:
    if counts.empty:
        return []
    table = counts.pivot(index="trading_date", columns="market", values="rows_count")
    table = table.reindex(columns=["TWSE", "TPEX"]).fillna(0).sort_index()
    # Each market has an independent reference; zero on a one-market day stays visible.
    reference = table.max()
    bad = (table < reference * ratio).any(axis=1) | (table == 0).any(axis=1)
    return [pd.Timestamp(day).date() for day in table.index[bad]]
