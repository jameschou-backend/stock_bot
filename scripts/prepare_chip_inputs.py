#!/usr/bin/env python3
"""Bounded, resumable chip evidence; never writes sealed parent inputs or DB."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import requests
from sqlalchemy import bindparam, text
from app.config import load_config
from app.db import get_engine
from app.file_lock import file_lock
from app.finmind import (fetch_dataset, FinMindError, FinMindQuotaError,
                         _http_session, _build_headers, _retry_after)
from app.finmind_cache import cache_path, read_cache, write_cache
from app.rate_limiter import get_rate_limiter
from scripts.research_exit_scenarios import read, write, sha

OUTPUT = ROOT / '.cache/chip-inputs'
SPEC = ROOT / 'docs/prereg_chip_20260911.md'
BROKER_URL = 'https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report'


def fetch_broker(sid, day, token):
    """Independent endpoint, same process-shared quota and credential-isolated cache."""
    params = dict(data_id=sid, date=day)
    path = cache_path(dict(endpoint=BROKER_URL, **params), token)
    with file_lock(path.with_suffix('.lock'), timeout=60):
        saved = read_cache(path, 300)
        if saved is not None:
            result = pd.DataFrame(saved['data'])
            result.attrs.update(cache_hit=True, retrieved_at=saved['retrieved_at'])
            return result
        limiter = get_rate_limiter(5400)
        if not limiter.acquire(timeout=0):
            raise FinMindQuotaError(limiter.get_stats().retry_after_seconds)
        try:
            response = _http_session().get(BROKER_URL, params=params,
                headers=_build_headers(token), timeout=60)
        except requests.RequestException as exc:
            raise FinMindError('Broker network error: '+type(exc).__name__) from None
        if response.status_code in (402, 429):
            limiter.defer(_retry_after(response.headers.get('Retry-After')))
            raise FinMindQuotaError(limiter.get_stats().retry_after_seconds)
        if response.status_code != 200:
            raise FinMindError('Broker HTTP '+str(response.status_code))
        try:
            payload = response.json()
        except ValueError:
            raise FinMindError('Broker invalid JSON') from None
        if not isinstance(payload, dict):
            raise FinMindError('Broker invalid payload')
        if str(payload.get('status')) in ('402', '429'):
            limiter.defer(_retry_after(response.headers.get('Retry-After')))
            raise FinMindQuotaError(limiter.get_stats().retry_after_seconds)
        if payload.get('status') not in (200, '200', None):
            raise FinMindError('Broker invalid status')
        rows = payload.get('data')
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise FinMindError('Broker missing data')
        now = time.time()
        write_cache(path, rows, now)
        result = pd.DataFrame(rows)
        result.attrs.update(cache_hit=False, retrieved_at=now)
        return result


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with file_lock(OUTPUT / 'prepare.lock', timeout=1):
        _prepare()


def refresh_primary():
    """A failed DB/raw spot check requires complete candidate-only source refresh."""
    token = load_config().finmind_token
    pool = sorted({e['members'][0] for e in read(ROOT/'.cache/million-replay-signals/signals.json')['entries']})
    jobs = [(ds, sid) for ds in ('TaiwanStockInstitutionalInvestorsBuySell', 'TaiwanStockMarginPurchaseShortSale') for sid in pool]
    def task(item):
        ds, sid = item
        path = OUTPUT/'raw'/ds/(sid+'.parquet')
        meta = path.with_suffix('.json')
        if meta.exists():
            if read(meta)['sha256'] != sha(path):
                raise ValueError('Primary source changed')
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = fetch_dataset(ds, date(2021,1,1), date(2026,9,9), token=token,
            data_id=sid, requests_per_hour=5400, max_retries=0)
        if not frame.empty and (not frame.stock_id.eq(sid).all() or not pd.to_datetime(frame.date).between('2021-01-01', '2026-09-09').all()):
            raise ValueError('Primary provider identity mismatch')
        frame.to_parquet(path, index=False)
        write(meta, dict(sha256=sha(path), rows=len(frame), retrieved_at=frame.attrs.get('retrieved_at'),
            cache_hit=frame.attrs.get('cache_hit', False)))
        return int(not frame.attrs.get('cache_hit', False))
    count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for offset in range(0, len(jobs), 4):
            count += sum(executor.map(task, jobs[offset:offset+4]))
            if offset % 40 == 0:
                print('primary refresh', min(offset+4, len(jobs)), '/', len(jobs), flush=True)
    inst = pd.concat([pd.read_parquet(OUTPUT/'raw/TaiwanStockInstitutionalInvestorsBuySell'/(sid+'.parquet')) for sid in pool], ignore_index=True)
    if inst.duplicated(['date','stock_id','name']).any():
        raise ValueError('Duplicate institutional identity')
    inst['net'] = pd.to_numeric(inst.buy)-pd.to_numeric(inst.sell)
    wide = inst.pivot(index=['date','stock_id'], columns='name', values='net')
    fresh = pd.DataFrame(dict(trust_net=wide.Investment_Trust,
        foreign_net=wide.Foreign_Investor+wide.Foreign_Dealer_Self)).reset_index()
    fresh['date'] = pd.to_datetime(fresh.date)
    old = pd.read_parquet(OUTPUT/'raw_institutional.parquet')
    both = fresh.merge(old, on=['date','stock_id'], suffixes=('_api','_db'))
    comparison = {actor: int((both[actor+'_net_api'] != both[actor+'_net_db']).sum()) for actor in ('trust','foreign')}
    fresh.to_parquet(OUTPUT/'institutional_verified.parquet', index=False)
    margin = pd.concat([pd.read_parquet(OUTPUT/'raw/TaiwanStockMarginPurchaseShortSale'/(sid+'.parquet')) for sid in pool], ignore_index=True)
    margin = margin[['date','stock_id','MarginPurchaseTodayBalance']].rename(columns={'MarginPurchaseTodayBalance':'margin_purchase_balance'})
    margin['date'] = pd.to_datetime(margin.date)
    old_margin = pd.read_parquet(OUTPUT/'raw_margin_short.parquet')
    joined = margin.merge(old_margin,on=['date','stock_id'],suffixes=('_api','_db'))
    margin.to_parquet(OUTPUT/'margin_verified.parquet',index=False)
    write(OUTPUT/'primary_audit.json',dict(institution_rows=len(fresh), compared_rows=len(both),
        net_different_rows=comparison, margin_rows=len(margin), margin_compared_rows=len(joined),
        margin_different_rows=int((joined.margin_purchase_balance_api != joined.margin_purchase_balance_db).sum()),
        this_run_new_requests=count, reason='2026-04-07 2330 trust net DB 306851 vs API 420851',
        publication_revisions_remain_possible=True))
    meta = read(OUTPUT/'manifest.json')
    meta['files_sha256'] = {str(p.relative_to(ROOT)):sha(p) for p in OUTPUT.rglob('*')
        if p.suffix in ('.parquet','.json') and p.name!='manifest.json' and 'execution-feeds' not in p.parts and 'dividends' not in p.parts}
    meta['primary_refresh_new_requests'] = count
    write(OUTPUT/'manifest.json',meta)
    print('primary refresh complete',comparison,flush=True)


def _prepare():
    began = time.monotonic()
    entries = read(ROOT / '.cache/million-replay-signals/signals.json')['entries']
    pool = sorted({e['members'][0] for e in entries})
    if len(entries) != 458 or len(pool) != 280:
        raise ValueError('Unexpected frozen candidate population')
    stamp = OUTPUT / 'prereg.json'
    identity = dict(spec_sha256=sha(SPEC), signals_sha256=sha(ROOT / '.cache/million-replay-signals/signals.json'))
    if stamp.exists() and read(stamp) != identity:
        raise ValueError('Preregistered inputs changed')
    write(stamp, identity)
    engine = get_engine()
    for table in ('raw_institutional', 'raw_margin_short'):
        target = OUTPUT / (table+'.parquet')
        if target.exists():
            continue
        query = text(f'SELECT * FROM {table} WHERE stock_id IN :pool AND trading_date BETWEEN :start AND :end').bindparams(bindparam('pool', expanding=True))
        with engine.connect() as connection:
            frame = pd.read_sql(query, connection, params=dict(pool=pool, start='2021-01-01', end='2026-09-09'))
        frame['date'] = pd.to_datetime(frame.pop('trading_date'))
        if frame.duplicated(['stock_id', 'date']).any():
            raise ValueError('Duplicate DB rows')
        frame.to_parquet(target, index=False)
        print(table, len(frame), flush=True)
    token = load_config().finmind_token
    tasks = []
    for sid in pool:
        own = [e for e in entries if e['members'][0] == sid]
        start = str((pd.Timestamp(min(e['signal_date'] for e in own))-pd.Timedelta(days=100)).date())
        end = max(e['signal_date'] for e in own)
        for dataset in ('TaiwanStockHoldingSharesPer', 'TaiwanDailyShortSaleBalances'):
            tasks.append((dataset, sid, start, end))
    for entry in entries:
        tasks.append(('broker', entry['members'][0], entry['signal_date'], entry['signal_date']))
    if len(tasks) > 1300:
        raise ValueError('Preparation request budget exceeded')

    def task(item):
        dataset, sid, start, end = item
        directory = OUTPUT / 'raw' / dataset
        directory.mkdir(parents=True, exist_ok=True)
        name = sid+'_'+start+'_'+end
        path, meta = directory/(name+'.parquet'), directory/(name+'.json')
        if meta.exists():
            saved = read(meta)
            if saved.get('sha256') != sha(path):
                raise ValueError('Saved raw evidence changed')
            return 0
        frame = (fetch_broker(sid, start, token) if dataset == 'broker' else
            fetch_dataset(dataset, date.fromisoformat(start), date.fromisoformat(end),
                token=token, data_id=sid, requests_per_hour=5400, max_retries=0))
        if not frame.empty:
            if not {'date', 'stock_id'}.issubset(frame):
                raise ValueError('Provider identity columns missing')
            if not frame.stock_id.eq(sid).all() or not pd.to_datetime(frame.date).between(start, end).all():
                raise ValueError('Provider returned wrong stock/date')
        frame.to_parquet(path, index=False)
        write(meta, dict(dataset=dataset, stock_id=sid, start=start, end=end,
            rows=len(frame), retrieved_at=frame.attrs.get('retrieved_at'),
            cache_hit=frame.attrs.get('cache_hit', False), sha256=sha(path)))
        return int(not frame.attrs.get('cache_hit', False))

    # Submit bounded waves: an error prevents any subsequent wave from starting.
    count = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for offset in range(0, len(tasks), 4):
            count += sum(executor.map(task, tasks[offset:offset+4]))
            if offset % 40 == 0:
                print(f'chip evidence {min(offset+4,len(tasks))}/{len(tasks)}, new requests {count}', flush=True)
    files = {str(p.relative_to(ROOT)): sha(p) for p in OUTPUT.rglob('*')
             if p.suffix in ('.parquet', '.json') and p.name != 'manifest.json'}
    write(OUTPUT / 'manifest.json', dict(**identity, files_sha256=files,
        preparation_code_sha256=sha(Path(__file__)), candidate_count=458, stock_count=280,
        this_run_new_requests=count, elapsed_seconds=time.monotonic()-began,
        historical_first_publication_verified=False))
    print('chip evidence complete', count, round(time.monotonic()-began, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-primary', action='store_true')
    args = parser.parse_args()
    if args.refresh_primary:
        with file_lock(OUTPUT/'prepare.lock',timeout=1):
            refresh_primary()
    else:
        prepare()
