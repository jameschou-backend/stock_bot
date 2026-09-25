#!/usr/bin/env python3
"""Prepare only feeds needed by the rebuilt candidate union, with a fixed quota."""
from datetime import date, datetime, timezone
from pathlib import Path
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from app.config import load_config
from app.finmind import fetch_dataset
from scripts.research_exit_scenarios import read, write, sha
from skills.replay_market_feeds import ReplayMarketFeeds

PARENT = ROOT / '.cache/backtest-corporate-completion-20260925/probe-v2'
OUTPUT = ROOT / '.cache/historical-selector-execution-20260925'


def verify(output=OUTPUT):
    result = read(output / 'manifest.json')
    if sha(output / 'manifest.json') != (output / 'manifest.sha256').read_text().strip():
        raise ValueError('Execution preparation manifest changed')
    for name, digest in result['files_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError('Prepared execution source changed: ' + name)
    return result


def prepare(signals, output=OUTPUT):
    signals = Path(signals).resolve()
    paths = [signals / arm / 'signals.json' for arm in ('original', 'identity', 'omitted', 'combined')]
    ids = sorted({'0050'} | {e['members'][0] for p in paths for e in read(p)['entries']})
    plan = dict(stock_ids=ids, signal_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                parent_manifest_sha256=sha(PARENT / 'manifest.json'), maximum_calls=100,
                start='2021-01-01', end='2026-12-31', official_requests=0, max_retries=0)
    if (output / 'manifest.json').exists():
        result = verify(output)
        if result['plan'] != plan:
            raise ValueError('Execution candidate plan changed; use a new output')
        return result
    if (output / 'plan.json').exists() and read(output / 'plan.json') != plan:
        raise ValueError('Execution source plan changed; use a new output')
    write(output / 'plan.json', plan)
    inputs = output / 'inputs'
    if not inputs.exists():
        for name, digest in read(PARENT / 'manifest.json')['files_sha256'].items():
            if sha(PARENT / name) != digest:
                raise ValueError('Parent evidence changed: ' + name)
        shutil.copytree(PARENT / 'inputs', inputs)
    path = output / 'attempts.json'
    attempts = read(path) if path.exists() else []
    config = load_config()
    def bounded_fetch(dataset, start, end, **kwargs):
        if dataset not in ('TaiwanStockDividend', 'TaiwanStockPriceLimit'):
            raise ValueError('Unexpected dataset in bounded execution preparation')
        if len(attempts) >= plan['maximum_calls']:
            raise ValueError('Execution preparation lifetime request ceiling reached')
        attempts.append(dict(dataset=dataset, stock_id=kwargs['data_id'],
                             requested_at=datetime.now(timezone.utc).isoformat()))
        write(path, attempts)
        kwargs.update(token=config.finmind_token, max_retries=0, requests_per_hour=5400, timeout=30)
        return fetch_dataset(dataset, start, end, **kwargs)
    def no_official(*args, **kwargs):
        raise RuntimeError('This preparation forbids official HTTP requests')
    feeds = ReplayMarketFeeds(inputs / 'execution-feeds', offline=False, token=config.finmind_token,
                              finmind_fetch=bounded_fetch, http_get=no_official)
    for sid in ids:
        dividend = inputs / 'dividends' / (sid + '.parquet')
        if not dividend.exists():
            frame = bounded_fetch('TaiwanStockDividend', date(2021, 1, 1), date(2026, 12, 31), data_id=sid)
            if not frame.empty and ('stock_id' not in frame or set(frame.stock_id.astype(str)) != {sid}):
                raise ValueError('Dividend stock identity differs: ' + sid)
            temp = dividend.with_suffix('.tmp')
            frame.to_parquet(temp, index=False)
            temp.replace(dividend)
            print('dividend', sid, len(frame), flush=True)
        existed = 'limits:' + sid in read(inputs / 'execution-feeds/index.json')['entries']
        rows = feeds.get_limits(sid)
        if not existed:
            print('limits', sid, len(rows), flush=True)
    result = dict(plan=plan, calls_reserved=len(attempts), database_writes=0,
        files_sha256={str(p.relative_to(output)): sha(p) for p in output.rglob('*')
            if p.is_file() and p.suffix != '.lock' and p.name not in ('manifest.json', 'manifest.sha256')})
    write(output / 'manifest.json', result)
    (output / 'manifest.sha256').write_text(sha(output / 'manifest.json') + '\n')
    return verify(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signals', type=Path, required=True)
    args = parser.parse_args()
    with file_lock(OUTPUT / '.prepare.lock', timeout=0):
        result = prepare(args.signals)
    print('calls_reserved', result['calls_reserved'])
