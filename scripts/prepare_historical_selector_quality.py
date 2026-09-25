#!/usr/bin/env python3
"""Bounded, checkpointed second-price snapshot for the 49 omitted stocks."""
from datetime import date, datetime, timezone
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from app.config import load_config
from scripts.prepare_historical_cohort_supplement import verify as verify_raw, DIRECTORY
from scripts.research_exit_scenarios import read, write, sha

OUTPUT = ROOT / '.cache/historical-selector-quality-20260925'


def validate(frame, sid):
    if frame.empty:
        return  # An empty response remains missing, not fabricated prices.
    if not {'stock_id', 'date', 'close'}.issubset(frame):
        raise ValueError('Adjusted quote columns missing: ' + sid)
    dates = pd.to_datetime(frame.date, errors='raise')
    if (set(frame.stock_id.astype(str)) != {sid} or dates.duplicated().any()
            or not dates.between('2021-01-01', '2026-09-09').all()
            or not dates.eq(dates.dt.normalize()).all()):
        raise ValueError('Adjusted quote identity/date differs: ' + sid)
    pd.to_numeric(frame.close, errors='raise')


def verify(output=OUTPUT):
    info = read(output / 'manifest.json')
    if sha(output / 'manifest.json') != (output / 'manifest.sha256').read_text().strip():
        raise ValueError('Adjusted snapshot manifest changed')
    if read(output / 'plan.json') != info['plan']:
        raise ValueError('Adjusted snapshot plan changed')
    for name, digest in info['files_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError('Adjusted snapshot changed: ' + name)
    if set(info['plan']['stock_ids']) != {p[:-8] for p in info['files_sha256'] if p.endswith('.parquet')}:
        raise ValueError('Incomplete adjusted stock coverage')
    return info


def prepare(output=OUTPUT):
    raw = verify_raw(DIRECTORY)
    plan = dict(dataset='TaiwanStockPriceAdj', stock_ids=raw['plan']['stock_ids'],
                start='2021-01-01', end='2026-09-09', maximum_calls=60, max_retries=0,
                effective_hourly_limit=5400, raw_manifest_sha256=sha(DIRECTORY / 'manifest.json'))
    if (output / 'manifest.json').exists():
        result = verify(output)
        if result['plan'] != plan:
            raise ValueError('Prepared adjusted source plan differs')
        return result
    if (output / 'plan.json').exists() and read(output / 'plan.json') != plan:
        raise ValueError('Use a new output when the adjusted source plan changes')
    write(output / 'plan.json', plan)
    ledger = read(output / 'attempts.json') if (output / 'attempts.json').exists() else []
    config = load_config()
    if not config.finmind_token:
        raise ValueError('FINMIND_TOKEN is required; use the existing local configuration')
    for sid in plan['stock_ids']:
        path = output / (sid + '.parquet')
        if path.exists():
            validate(pd.read_parquet(path), sid)
            continue
        if len(ledger) >= plan['maximum_calls']:
            raise ValueError('Adjusted source lifetime request budget exhausted')
        ledger.append(dict(stock_id=sid, requested_at=datetime.now(timezone.utc).isoformat()))
        write(output / 'attempts.json', ledger)
        frame = fetch_dataset(plan['dataset'], date(2021, 1, 1), date(2026, 9, 9),
            data_id=sid, token=config.finmind_token, max_retries=0, timeout=30, requests_per_hour=5400)
        validate(frame, sid)
        temporary = path.with_suffix('.tmp')
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        print(sid, len(frame), flush=True)
    info = dict(plan=plan, calls_reserved=len(ledger), source_is_revised_snapshot=True,
        live_qualified=False, files_sha256={p.name: sha(p) for p in output.iterdir()
            if p.name in ('plan.json', 'attempts.json') or p.suffix == '.parquet'})
    write(output / 'manifest.json', info)
    (output / 'manifest.sha256').write_text(sha(output / 'manifest.json') + '\n')
    return verify(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with file_lock(OUTPUT / '.prepare.lock', timeout=0):
        result = verify() if args.verify else prepare()
    print('stocks', len(result['plan']['stock_ids']), 'calls_reserved', result['calls_reserved'])
