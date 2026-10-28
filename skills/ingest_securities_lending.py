"""FinMind borrowing transactions: resumable, bounded, all-market daily ingestion.

Official schema: https://finmind.github.io/tutor/TaiwanMarket/Chip/#taiwanstocksecuritieslending
transaction_type is a trading method (競價/議借/定價), not lend/return direction.
These flows cannot reconstruct outstanding balances or actual return dates.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.finmind import FinMindError, fetch_dataset
from app.ingest_dates import resolve_ingest_end_date
from app.job_utils import finish_job, start_job, update_job
from app.market_calendar import get_latest_trading_day
from app.models import SecuritiesLendingDaily, SecuritiesLendingDay, TradingCalendar

DATASET = 'TaiwanStockSecuritiesLending'
COLUMN_MAP = {'date': 'trading_date', 'stock_id': 'stock_id', 'volume': 'transaction_volume',
              'fee_rate': 'fee_rate', 'transaction_type': 'transaction_type'}
METHODS = {'競價', '議借', '定價'}
MAX_REQUESTS_PER_RUN = 90


def _aggregate_lending(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise FinMindError(f'{DATASET}: missing columns {sorted(missing)}')
    frame = df.rename(columns=COLUMN_MAP).copy()
    frame['stock_id'] = frame.stock_id.astype(str)
    frame = frame[frame.stock_id.str.fullmatch(r'\d{4}')].copy()
    frame['trading_date'] = pd.to_datetime(frame.trading_date, errors='coerce').dt.date
    for c in ('transaction_volume', 'fee_rate'):
        frame[c] = pd.to_numeric(frame[c], errors='coerce')
    invalid = (frame.trading_date.isna() | ~frame.transaction_type.isin(METHODS)
               | ~np.isfinite(frame.transaction_volume) | (frame.transaction_volume <= 0)
               | (frame.transaction_volume % 1 != 0)
               | ~np.isfinite(frame.fee_rate) | (frame.fee_rate < 0))
    if invalid.any():
        raise FinMindError(f'{DATASET}: invalid date, volume, fee or transaction method')
    records = []
    for (sid, day), g in frame.groupby(['stock_id', 'trading_date']):
        volume = int(g.transaction_volume.sum())
        records.append(dict(stock_id=sid, trading_date=day,
                            transaction_volume=volume, transaction_count=len(g),
                            fee_rate=float((g.transaction_volume * g.fee_rate).sum() / volume),
                            methods_json=g.transaction_type.value_counts().astype(int).to_dict()))
    return pd.DataFrame(records)


def run(config, db_session, *, start_date=None, end_date=None, max_requests=MAX_REQUESTS_PER_RUN, **kwargs):
    today = datetime.now(ZoneInfo(config.tz)).date()
    published = get_latest_trading_day(db_session)
    end_date = end_date or min(resolve_ingest_end_date(today), published or today - timedelta(days=1))
    start_date = start_date or end_date - timedelta(days=89)
    if start_date > end_date or (published is not None and end_date > published):
        raise ValueError('Lending range must be ordered and end on a published trading day')
    if not 1 <= max_requests <= MAX_REQUESTS_PER_RUN:
        raise ValueError(f'max_requests must be 1..{MAX_REQUESTS_PER_RUN}')
    days = db_session.execute(select(TradingCalendar.trading_date).where(
        TradingCalendar.is_open.is_(True), TradingCalendar.trading_date.between(start_date, end_date)
    ).order_by(TradingCalendar.trading_date)).scalars().all()
    if not days:
        raise FinMindError('No open trading days in lending range; populate the calendar first')
    completed = set(db_session.execute(select(SecuritiesLendingDay.trading_date).where(
        SecuritiesLendingDay.trading_date.between(start_date, end_date))).scalars())
    pending = [day for day in days if day not in completed]
    if len(pending) > max_requests:
        raise ValueError(f'Lending needs {len(pending)} requests; split the range (budget {max_requests})')
    job = start_job(db_session, 'ingest_securities_lending', commit=True)
    logs = dict(start_date=str(start_date), end_date=str(end_date), dataset=DATASET,
                planned_requests=len(pending), requests=0, rows=0, completed_days=len(completed),
                semantics='transactions_not_balances', volume_unit='finmind_reported',
                historical_publication_verified=False)
    try:
        for day in pending:
            df = fetch_dataset(dataset=DATASET, start_date=day, end_date=day, data_id=None,
                               token=config.finmind_token,
                               requests_per_hour=config.finmind_requests_per_hour,
                               max_retries=config.finmind_retry_max,
                               backoff_seconds=config.finmind_retry_backoff, timeout=120)
            logs['requests'] += 1
            if df.empty:
                raise FinMindError(f'{DATASET}: empty published market day {day}; retry later')
            if 'date' not in df or set(pd.to_datetime(df['date'], errors='coerce').dt.date) != {day}:
                raise FinMindError(f'{DATASET}: unexpected response date for {day}')
            rows = _aggregate_lending(df)
            if rows.empty:
                raise FinMindError(f'{DATASET}: no valid four-digit securities for {day}')
            # Preserve duplicate trades and non-stock source evidence exactly. Only
            # normalized four-digit codes become SecuritiesLendingDaily stock rows.
            payload = json.loads(df.to_json(orient='records', force_ascii=False))
            canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
            db_session.add_all(SecuritiesLendingDaily(**row) for row in rows.to_dict('records'))
            db_session.add(SecuritiesLendingDay(trading_date=day, observed_at=datetime.utcnow(),
                source_rows=len(df), stock_rows=len(rows), payload_json=payload,
                payload_sha256=hashlib.sha256(canonical.encode()).hexdigest()))
            logs['rows'] += len(rows)
            logs['last_completed_day'] = str(day)
            update_job(db_session, job, logs=logs)
            db_session.commit()  # checkpoint and rows succeed/fail together
        finish_job(db_session, job, 'success', logs=logs)
        db_session.commit()
        return logs
    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job, 'failed', error_text=str(exc), logs=logs)
        db_session.commit()
        raise


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from app.config import load_config
    from app.db import get_session
    from app.file_lock import file_lock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=date.fromisoformat)
    parser.add_argument('--end', type=date.fromisoformat)
    parser.add_argument('--max-requests', type=int, default=MAX_REQUESTS_PER_RUN)
    args = parser.parse_args()
    with file_lock(Path('.cache/research-or-update.lock'), timeout=0):
        with get_session() as session:
            print(json.dumps(run(load_config(), session, start_date=args.start, end_date=args.end,
                                 max_requests=args.max_requests), ensure_ascii=False, indent=2))
