#!/usr/bin/env python3
"""Freeze course research inputs without modifying existing data or strategies."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sqlalchemy import select
from skills.strategy_scanner.data import load_bundle

OUT = ROOT / '.cache/course-bundle-20261028'
SPEC = ROOT / 'docs/prereg_course_bundle_20261028.json'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    tmp.replace(path)


def acquire(sid, cfg):
    from app.finmind import fetch_dataset
    folder = OUT / 'turnover-stock'
    dest = folder / (sid + '.parquet')
    receipt = dest.with_suffix('.json')
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior.get('empty'):
            return sid, 'empty'
        if sha(dest) != prior['sha256']:
            raise ValueError('Changed turnover source: ' + sid)
        return sid, 'reused'
    frame = fetch_dataset('TaiwanStockPrice', date(2021, 9, 1), date(2026, 10, 22),
        token=cfg.finmind_token, data_id=sid,
        requests_per_hour=min(5400, cfg.finmind_requests_per_hour), max_retries=0,
        timeout=45, cache_ttl=365 * 86400)
    if frame.empty:
        save_json(receipt, {'empty': True, 'stock_id': sid})
        return sid, 'empty'
    required = ['date', 'stock_id', 'Trading_money', 'Trading_Volume', 'close']
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing actual turnover source columns: ' + sid)
    if set(frame.stock_id) != {sid} or frame.duplicated(['date', 'stock_id']).any():
        raise ValueError('Conflicting source identity: ' + sid)
    attrs = dict(frame.attrs)
    frame = frame[required].copy()
    if not pd.to_datetime(frame.date).between('2021-09-01', '2026-10-22').all():
        raise ValueError('Unexpected source date: ' + sid)
    frame.to_parquet(dest, index=False)
    save_json(receipt, {'stock_id': sid, 'dataset': 'TaiwanStockPrice', 'rows': len(frame),
        'sha256': sha(dest), 'retrieved_at': attrs.get('retrieved_at'),
        'cache_hit': attrs.get('cache_hit'), 'actual_turnover': True})
    return sid, 'fetched'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Use 1-8 workers with the shared FinMind quota')
    OUT.mkdir(exist_ok=True)
    (OUT / 'turnover-stock').mkdir(exist_ok=True)
    spec = json.loads(SPEC.read_text())
    manifest = OUT / 'inputs.json'
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if value['spec_sha256'] != sha(SPEC):
            raise ValueError('Preregistered specification changed after input freeze')
        for name, expected in value['files_sha256'].items():
            if sha(OUT / name) != expected:
                raise ValueError('Frozen research input changed: ' + name)
        print('Frozen inputs verified; no new fetch.', flush=True)
        return
    bundle = load_bundle(ROOT / spec['source_bundle'], spec['start'], spec['end'])
    bars = bundle['bars']
    bars.to_parquet(OUT / 'bars.parquet', index=False)
    save_json(OUT / 'source.json', bundle['provenance'])
    save_json(OUT / 'names.json', bundle['names'])
    from app.config import load_config
    from app.db import get_session
    from app.models import RawInstitutional
    cfg = load_config()
    dest = OUT / 'institutional.parquet'
    if not dest.exists():
        with get_session() as session:
            query = select(RawInstitutional.stock_id, RawInstitutional.trading_date,
                           RawInstitutional.trust_net, RawInstitutional.foreign_net).where(
                RawInstitutional.trading_date.between(date(2021, 9, 1), date(2026, 10, 22)),
                RawInstitutional.stock_id.in_(bundle['universe']))
            frame = pd.read_sql(query, session.get_bind()).rename(columns={'trading_date': 'date'})
        frame.to_parquet(dest, index=False)
    ids = sorted(s for s in bundle['universe'] if s != '0050')
    if any(len(s)!=4 or not s.isascii() or not s.isdigit() for s in ids):
        raise ValueError('Course universe must contain four-digit Taiwan stock IDs')
    print(f'Acquiring actual turnover for {len(ids)} stocks; preserved cohort; 4-digit IDs only.', flush=True)
    errors, counts = [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire, sid, cfg): sid for sid in ids}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                sid, status = future.result()
                counts[status] = counts.get(status, 0) + 1
            except Exception as exc:
                errors.append({'stock_id': futures[future], 'error': str(exc)})
                from app.finmind import FinMindQuotaError
                if isinstance(exc, FinMindQuotaError):
                    for item in futures:
                        item.cancel()
                    save_json(OUT / 'acquire-errors.json', errors)
                    raise
            if n % 50 == 0 or n == len(ids):
                print(n, '/', len(ids), counts, 'errors', len(errors), flush=True)
    save_json(OUT / 'acquire-errors.json', errors)
    if errors:
        raise ValueError(f'{len(errors)} source fetches failed; completed snapshots retained')
    frames = []
    receipts = {}
    for sid in ids:
        rec = json.loads((OUT / 'turnover-stock' / (sid + '.json')).read_text())
        if rec.get('empty'):
            continue
        path = OUT / 'turnover-stock' / (sid + '.parquet')
        if sha(path) != rec['sha256']:
            raise ValueError('Turnover snapshot changed')
        frames.append(pd.read_parquet(path))
        receipts[str(path.relative_to(OUT))] = rec['sha256']
    pd.concat(frames, ignore_index=True).to_parquet(OUT / 'turnover.parquet', index=False)
    files = ['bars.parquet', 'source.json', 'names.json', 'institutional.parquet', 'turnover.parquet']
    save_json(manifest, {'experiment': spec['experiment'], 'start': spec['start'], 'end': spec['end'],
        'spec_sha256': sha(SPEC), 'files_sha256': {name: sha(OUT / name) for name in files},
        'turnover_source_sha256': receipts, 'fetch_counts': counts,
        'db_mutations': False, 'historical_institutional_first_publication_verified': False})
    print('Research inputs sealed.', flush=True)


if __name__ == '__main__':
    main()
