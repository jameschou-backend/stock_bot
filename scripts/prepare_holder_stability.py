#!/usr/bin/env python3
"""Bounded append-only holder snapshot preparation through the shared client."""
from __future__ import annotations
import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from app.file_lock import file_lock
from scripts.extend_scanner_daily_inputs import digest

SOURCE_DIRS = ('.cache/theme-chip-inputs-20260927', '.cache/theme-chip-calendar-20260927')
SPEC = ROOT / 'docs/prereg_holder_stability_600_20261102.md'


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def run(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / '.cache'):
        raise ValueError('Preparation belongs in a new project cache directory')
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / 'prepare.lock', timeout=0):
        if (output / 'manifest.json').exists():
            manifest = json.loads((output / 'manifest.json').read_text())
            for p, h in manifest['source_sha256'].items():
                if digest(ROOT / p) != h:
                    raise ValueError('Completed source changed: ' + p)
            print('Verified completed preparation; no requests', flush=True)
            return manifest
        refs = {str(SPEC.relative_to(ROOT)): digest(SPEC)}
        files = {}
        for name in SOURCE_DIRS:
            folder = ROOT / name
            manifest = json.loads((folder / 'manifest.json').read_text())
            refs[name + '/manifest.json'] = digest(folder / 'manifest.json')
            for day, meta in manifest['files'].items():
                path = folder / (day + '.parquet')
                if digest(path) != meta['sha256']:
                    raise ValueError('Original holder snapshot changed: ' + day)
                if not meta['rows']:
                    continue
                if day in files:
                    raise ValueError('Conflicting historical week: ' + day)
                files[day] = dict(path=str(path.relative_to(ROOT)), sha256=meta['sha256'])
                refs[str(path.relative_to(ROOT))] = meta['sha256']
                receipt = folder / (day + '.json')
                if json.loads(receipt.read_text()) != {k: v for k, v in meta.items() if k != 'reused'}:
                    # Older manifests can contain additional run-level metadata.
                    recorded = json.loads(receipt.read_text())
                    if any(recorded.get(k) != meta.get(k) for k in ('date', 'rows', 'sha256', 'retrieved_at', 'cache_hit')):
                        raise ValueError('Historical receipt changed')
                refs[str(receipt.relative_to(ROOT))] = digest(receipt)
        plan = dict(schema='holder_stability_preparation_v1', start='2026-09-05', end='2026-10-30',
                    dataset='TaiwanStockHoldingSharesPer', maximum_requests=12, retries=0,
                    prereg_sha256=digest(SPEC), historical_weeks=len(files))
        if (output / 'plan.json').exists():
            if json.loads((output / 'plan.json').read_text()) != plan:
                raise ValueError('Acquisition plan changed')
        else:
            save(output / 'plan.json', plan)
        config = load_config()
        receipts = []

        def acquire(key, start, end=None, sid=None):
            query = dict(dataset=plan['dataset'], start_date=start, end_date=end, data_id=sid)
            path, receipt = output / (key + '.parquet'), output / (key + '.json')
            attempt = output / (key + '.attempt.json')
            if receipt.exists():
                meta = json.loads(receipt.read_text())
                if meta['query'] != query or digest(path) != meta['sha256']:
                    raise ValueError('Saved request differs')
                frame = pd.read_parquet(path)
            else:
                if attempt.exists():
                    raise ValueError('Prior incomplete attempt requires explicit review: ' + key)
                if len(list(output.glob('*.attempt.json'))) >= plan['maximum_requests']:
                    raise ValueError('Preregistered request budget exhausted')
                save(attempt, query)
                frame = fetch_dataset(plan['dataset'], date.fromisoformat(start),
                    date.fromisoformat(end) if end else None, data_id=sid, token=config.finmind_token,
                    requests_per_hour=min(config.finmind_requests_per_hour, 5400), max_retries=0, timeout=45)
                frame.to_parquet(path, index=False)
                meta = dict(query=query, sha256=digest(path), rows=len(frame),
                            cache_hit=bool(frame.attrs.get('cache_hit')),
                            retrieved_at=frame.attrs.get('retrieved_at'))
                save(receipt, meta)
            for p in (path, receipt, attempt):
                refs[str(p.relative_to(ROOT))] = digest(p)
            receipts.append(meta)
            print(key, len(frame), 'rows', flush=True)
            return frame, path

        dates = set()
        for start, end in [('2026-09-05', '2026-09-30'), ('2026-10-01', '2026-10-30')]:
            frame, _ = acquire('calendar-' + start, start, end, '2330')
            if frame.empty or not {'stock_id', 'date', 'HoldingSharesLevel'}.issubset(frame):
                raise ValueError('Calendar probe unavailable')
            observed = pd.to_datetime(frame.date)
            if set(frame.stock_id.astype(str)) != {'2330'} or not observed.between(start, end).all():
                raise ValueError('Calendar probe identity/date mismatch')
            dates.update(observed.dt.strftime('%Y-%m-%d'))
        if not dates or max(dates) != plan['end']:
            raise ValueError('Latest weekly observation has not been published')
        for day in sorted(dates - set(files)):
            frame, path = acquire(day, day)
            if frame.empty or not {'date', 'stock_id', 'HoldingSharesLevel', 'people', 'unit', 'percent'}.issubset(frame):
                raise ValueError('Required market snapshot unavailable: ' + day)
            if set(pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')) != {day}:
                raise ValueError('Market snapshot date mismatch')
            files[day] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
        refs[str((output / 'plan.json').relative_to(ROOT))] = digest(output / 'plan.json')
        refs[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
        result = dict(**plan, files=dict(sorted(files.items())), source_sha256=refs,
                      network_requests=sum(not r['cache_hit'] for r in receipts),
                      request_receipts=len(receipts), original_snapshots_changed=False)
        save(output / 'manifest.json', result)
        (output / 'manifest.sha256').write_text(digest(output / 'manifest.json') + '\n')
        print(json.dumps(dict(weeks=len(files), latest=max(files), requests=result['network_requests'])), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
