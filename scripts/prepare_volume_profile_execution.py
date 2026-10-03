#!/usr/bin/env python3
"""Bounded missing execution-cache preparation for the 2024 POC account study.

Only existing exact FinMind query shapes are allowed. This never requests an
official exchange URL, overwrites evidence, retries, or changes account rules.
"""
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from skills.frozen_dividend_copy import ensure_dividend_copy

BASE = ROOT/'.cache/volume-profile-account-20261003/execution-preparation-v1'
DEST = ROOT/'.cache/market-input-repair-20261002/execution-v1'
DATASETS = ('TaiwanStockDividend', 'TaiwanStockPriceLimit')


def _prepare(stock_id, dataset, *, resume_permission_failure=False):
    if not (isinstance(stock_id, str) and len(stock_id) == 4 and stock_id.isdigit()) or dataset not in DATASETS:
        raise ValueError('Only four-digit stock execution dividend/limit sources allowed')
    path = DEST/(stock_id+'-'+dataset+'.parquet')
    metadata = path.with_suffix('.json')
    query = dict(stock_id=stock_id, dataset=dataset, start='2018-01-01', end='2026-09-09')
    attempt = BASE/'attempts'/(stock_id+'-'+dataset+'.json')
    if path.exists() or metadata.exists():
        if not path.exists() or not metadata.exists():
            raise ValueError('Incomplete existing execution source; never overwrite')
        meta = read(metadata)
        if sha(path) != meta['sha256'] or any(meta.get(k) != v for k,v in query.items()):
            raise ValueError('Existing execution source conflicts with requested query')
        return dict(status='already_present', query=query, raw_path=str(path.relative_to(ROOT)), sha256=sha(path))
    attempt.parent.mkdir(parents=True, exist_ok=True)
    if attempt.exists():
        if not resume_permission_failure or read(attempt).get('error_type') != 'PermissionError':
            raise ValueError('Previous execution data attempt exists; automatic retry forbidden')
        attempt = attempt.with_name(attempt.stem+'.permission-resume.json')
        if attempt.exists():
            raise ValueError('Permission continuation already attempted')
    if len(list(attempt.parent.glob('*.json'))) >= 100:
        raise ValueError('Persistent execution preparation budget 100 exhausted')
    record = dict(query=query, status='started', started_at=datetime.now(timezone.utc).isoformat(),
                  maximum_requests=100, retries=0, shared_requests_per_hour=5400,
                  helper_sha256=sha(Path(__file__)))
    # Exclusive reservation survives interruption and prohibits retries.
    with attempt.open('x') as stream:
        json.dump(record, stream)
    from app.config import load_config
    from app.finmind import fetch_dataset
    config = load_config()
    try:
        frame = fetch_dataset(dataset, date(2018,1,1), date(2026,9,9),
            data_id=stock_id, token=config.finmind_token,
            requests_per_hour=min(5400, config.finmind_requests_per_hour),
            max_retries=0, timeout=40)
        if not frame.empty:
            import pandas as pd
            if set(frame.stock_id) != {stock_id} or not pd.to_datetime(frame.date).between('2018-01-01','2026-09-09').all():
                raise ValueError('Provider returned a different stock/date range')
        DEST.mkdir(parents=True, exist_ok=True)
        if path.exists() or metadata.exists():
            raise ValueError('Execution cache appeared during request; do not overwrite')
        frame.to_parquet(path, index=False)
        write(metadata, dict(query, sha256=sha(path)))
        if dataset == 'TaiwanStockDividend':
            ensure_dividend_copy(frame, DEST/'dividends'/(stock_id+'.parquet'), prepare=True)
        record.update(status='received', raw_path=str(path.relative_to(ROOT)), raw_sha256=sha(path),
            metadata_path=str(metadata.relative_to(ROOT)), metadata_sha256=sha(metadata), rows=len(frame),
            cache_hit=bool(frame.attrs.get('cache_hit', False)), retrieved_at=frame.attrs.get('retrieved_at'))
    except Exception as exc:
        # Do not persist arbitrary provider error text containing credentials.
        record.update(status='failed', error_type=type(exc).__name__)
        write(attempt, record)
        raise RuntimeError('Execution preparation failed; see redacted attempt receipt') from None
    write(attempt, record)
    return record


def prepare(stock_id, dataset, *, resume_permission_failure=False):
    from app.file_lock import file_lock
    with file_lock(BASE/'.run.lock',timeout=0):
        return _prepare(stock_id,dataset,resume_permission_failure=resume_permission_failure)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock-id', required=True)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--resume-permission-failure', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.stock_id, args.dataset,
                            resume_permission_failure=args.resume_permission_failure), ensure_ascii=False))
