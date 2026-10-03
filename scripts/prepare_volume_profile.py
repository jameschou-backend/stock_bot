#!/usr/bin/env python3
"""Bounded, restartable raw tick acquisition for a frozen, outcome-blind cohort.

No DB writes, no daily-volume approximation, no automatic retries. A saved
empty/error/interrupted attempt remains evidence rather than being refetched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from scripts.research_early_signal_losses import digest

SEED = 'volume-profile-pilot-20261003-v1'
RANK_SHA = 'e9c3e1826d012bf5c960f53d9ab1fb7a172d39c7a92e3c24d9fc054abd36752a'
PREREG = ROOT/'docs/prereg_volume_profile_20261003.md'
BUNDLE = ROOT/'.cache/all-signals-2019-20261002/inputs'
RANKS = ROOT/'.cache/signal-rank-20261003/rank-v2/signal-ranks.parquet'
KNOWN_GAPS = {'2018-12-22', '2019-02-20', '2019-02-21', '2019-02-22', '2019-05-16'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def validate_cohort(cohort, ranks, calendar):
    if cohort['seed'] != SEED or cohort['rank_sha256'] != RANK_SHA:
        raise ValueError('Unregistered cohort identity')
    selected = []
    for year in range(2019, 2027):
        rows = ranks.loc[ranks.signal_date.str.startswith(str(year))].to_dict('records')
        rows.sort(key=lambda r: (hashlib.sha256((SEED+'|'+r['signal_id']).encode()).hexdigest(), r['signal_id']))
        selected += rows[:20]
    expected = {}
    for row in selected:
        i = calendar.index(row['signal_date'])
        if i < 20: raise ValueError('Insufficient canonical calendar warmup')
        expected[row['signal_id']] = {k: row[k] for k in ('signal_id', 'stock_id', 'signal_date')}
        expected[row['signal_id']]['prior_dates'] = calendar[i-20:i]
    actual = {r['signal_id']: r for r in cohort['rows']}
    if len(actual) != 160 or len(cohort['rows']) != 160 or actual != expected:
        raise ValueError('Cohort differs from outcome-independent 20/year selection')
    return sorted({(r['stock_id'], day) for r in cohort['rows'] for day in r['prior_dates']})


def local(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Evidence escapes repository')
    return path


def verify_saved(item, query):
    if item['query'] != query: raise ValueError('Checkpoint query changed')
    if item.get('raw_path'):
        p = local(item['raw_path'])
        if digest(p) != item['raw_sha256']: raise ValueError('Checkpoint raw bytes changed')
    return item


def reuse(item, query):
    if item.get('content_conflict'):
        return dict(query=query, status='cached_versions_conflict', sources=item['sources'])
    sources = item['sources']
    if not sources: raise ValueError('Empty reuse source list')
    for source in sources:
        p, mp = local(source['path']), local(source['metadata_path'])
        meta = json.loads(mp.read_text())
        if digest(p) != source['raw_sha256'] or meta.get('raw_sha256') != source['raw_sha256'] or meta.get('query') != query:
            raise ValueError('Reusable source hash/query changed')
    source = sorted(sources, key=lambda s: s['path'])[0]
    return dict(query=query, status='cached', raw_path=str(local(source['path']).relative_to(ROOT)),
                raw_sha256=source['raw_sha256'], retrieved_at=source.get('retrieved_at'),
                rows=source.get('rows'), metadata_path=source['metadata_path'],
                metadata_sha256=digest(local(source['metadata_path'])))


def run(inventory, output, online=False):
    inventory, output = local(inventory), local(output)
    output.mkdir(parents=True, exist_ok=True)
    cohort_file, reuse_file = inventory/'cohort.json', inventory/'reuse-index.json'
    if digest(RANKS) != RANK_SHA: raise ValueError('Rank source changed')
    ranks = pd.read_parquet(RANKS, columns=['signal_id', 'stock_id', 'signal_date'])
    days = pd.to_datetime(pd.read_parquet(BUNDLE/'close-official.parquet', columns=['date']).date).dt.strftime('%Y-%m-%d').tolist()
    cohort = json.loads(cohort_file.read_text())
    coordinates = validate_cohort(cohort, ranks, days)
    if len(coordinates) > 3200: raise ValueError('Research request budget exceeded')
    reuse_index = json.loads(reuse_file.read_text())
    identity = dict(cohort_sha256=digest(cohort_file), reuse_index_sha256=digest(reuse_file),
                    prereg_sha256=digest(PREREG), rank_sha256=RANK_SHA, maximum_requests=3200,
                    workers=4, max_retries=0, coordinates=[list(k) for k in coordinates])
    plan = output/'plan.json'
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError('Existing acquisition plan changed; do not overwrite')
    write(plan, identity)
    from app.config import load_config
    from app.finmind import fetch_dataset, FinMindQuotaError, FinMindError
    config = load_config() if online else None
    stopped = threading.Event()
    def one(coordinate):
        sid, day = coordinate
        if not re.fullmatch(r'\d{4}', sid) or date.fromisoformat(day).isoformat() != day:
            raise ValueError('Invalid acquisition identity')
        key = sid+'-'+day
        p = output/'receipts'/(key+'.json')
        query = dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)
        if p.exists(): return verify_saved(json.loads(p.read_text()), query)
        if key in reuse_index:
            result = reuse(reuse_index[key], query); write(p, result); return result
        if day in KNOWN_GAPS or day < '2018-12-07':
            result = dict(query=query, status='provider_documented_gap',
                source_url='https://finmind.github.io/tutor/TaiwanMarket/Technical/')
            write(p, result); return result
        if not online or stopped.is_set(): return dict(query=query, status='not_requested')
        result = dict(query=query, status='started', started_at=datetime.now(timezone.utc).isoformat())
        write(p, result)  # Reserve before I/O. A crash never creates a free retry.
        try:
            frame = fetch_dataset('TaiwanStockPriceTick', date.fromisoformat(day), data_id=sid,
                token=config.finmind_token, requests_per_hour=min(6000, config.finmind_requests_per_hour),
                max_retries=0, timeout=40)
        except FinMindQuotaError as exc:
            stopped.set()
            result.update(status='quota_paused', retry_after_seconds=exc.retry_after_seconds)
        except FinMindError:
            result.update(status='provider_error', error_type='FinMindError')
        else:
            raw = output/'raw'/(key+'.parquet'); raw.parent.mkdir(exist_ok=True)
            temp = raw.with_suffix('.tmp.parquet'); frame.to_parquet(temp, index=False); temp.replace(raw)
            result.update(status='received' if len(frame) else 'empty', raw_path=str(raw.relative_to(ROOT)),
                raw_sha256=digest(raw), rows=len(frame), retrieved_at=frame.attrs.get('retrieved_at'),
                cache_hit=frame.attrs.get('cache_hit', False))
        write(p, result)
        return result
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index in range(0, len(coordinates), 4):
            records.extend(pool.map(one, coordinates[index:index+4]))
            if index % 100 == 0 or index+4 >= len(coordinates):
                from collections import Counter
                print(json.dumps(dict(processed=len(records), required=len(coordinates),
                    statuses=dict(Counter(r['status'] for r in records)))), flush=True)
    from collections import Counter
    counts = dict(Counter(r['status'] for r in records))
    report = dict(schema='volume_profile_acquisition_v1', plan_sha256=digest(plan),
        script_sha256=digest(__file__), required_stock_days=len(coordinates), counts=counts,
        adapter_attempts=sum('started_at' in r for r in records),
        complete_raw_responses=all(r['status'] in ('cached','received') for r in records),
        raw_rows=sum(r.get('rows') or 0 for r in records), database_mutations=False,
        ordinary_aggregate_verified=False, live_qualified=False,
        receipt_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((output/'receipts').glob('*.json'))})
    write(output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='receipt_sha256'}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=ROOT/'.cache/volume-profile-20261003/inventory')
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/volume-profile-20261003/tapes-v1')
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    with file_lock(args.output/'.run.lock', timeout=0):
        run(args.inventory, args.output, args.fetch)
