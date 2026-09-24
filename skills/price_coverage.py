"""Recent coverage by market; a global MAX(date) cannot establish completeness."""
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text


def recent_market_counts(session, end: date, days: int = 21) -> pd.DataFrame:
    counts = pd.read_sql(text("""
        SELECT p.trading_date, s.market, COUNT(*) AS rows_count
        FROM raw_prices p JOIN stocks s ON s.stock_id=p.stock_id
        WHERE p.trading_date BETWEEN :start AND :end AND p.close>0
          AND s.market IN ('TWSE','TPEX') AND s.security_type='stock'
        GROUP BY p.trading_date, s.market ORDER BY p.trading_date, s.market
    """), session.connection(), params={"start": end - timedelta(days=days), "end": end})
    counts['trading_date'] = pd.to_datetime(counts.trading_date).dt.date
    return counts


def incomplete_dates(counts: pd.DataFrame, *, ratio: float = .9,
                     required_dates: list[date] | None = None) -> list[date]:
    if counts.empty:
        return sorted(set(required_dates or []))
    table = counts.pivot(index="trading_date", columns="market", values="rows_count")
    table = table.reindex(columns=["TWSE", "TPEX"]).fillna(0).sort_index()
    # Each market has an independent reference; zero on a one-market day stays visible.
    reference = table.max()
    if required_dates is not None:
        table = table.reindex(sorted(set(table.index) | set(required_dates)), fill_value=0)
    bad = (table < reference * ratio).any(axis=1) | (table == 0).any(axis=1)
    return [pd.Timestamp(day).date() for day in table.index[bad]]


def require_market_coverage(session, target_date: date | None = None) -> dict:
    """Fail before expensive work or publication on partial-market inputs.

    The rolling count is a missing-market guard, not certification of every
    security's historical membership, corporate actions, or individual quote.
    Explicit calendar dates catch days missing from *both* markets.
    """
    from sqlalchemy import func
    from app.models import RawPrice
    from app.market_calendar import get_latest_trading_day, get_trading_days_from_db

    if target_date is None:
        latest = session.query(func.max(RawPrice.trading_date)).scalar()
        expected = get_latest_trading_day(session)
        dates = [d for d in (latest, expected) if d is not None]
        if not dates:
            raise ValueError('上市／上櫃行情檢查失敗：沒有價格或交易日')
        target_date = max(dates)
    counts = recent_market_counts(session, target_date)
    first = target_date - timedelta(days=7)
    required = get_trading_days_from_db(session, first, target_date)
    required = sorted(set(required) | {target_date})
    gaps = [d for d in incomplete_dates(counts, required_dates=required) if d >= first]
    if gaps:
        raise ValueError('上市／上櫃行情不完整：' + '、'.join(map(str, gaps)) +
                         '；請先補齊兩市場行情，再建置特徵、訓練與選股')
    return {'market_coverage_target': str(target_date),
            'market_coverage_dates': [str(d) for d in required],
            'market_coverage_ratio': .9}
