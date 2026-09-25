#!/usr/bin/env python3
"""Extend a sealed execution snapshot, carrying forward its lifetime request count."""
from datetime import date, datetime, timezone
from pathlib import Path
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts.research_exit_scenarios import read, write, sha
from skills.replay_market_feeds import ReplayMarketFeeds


def prepare(source, signals, output):
    source, signals, output = (Path(p).resolve() for p in (source, signals, output))
    source_manifest = read(source / 'manifest.json')
    if sha(source / 'manifest.json') != (source / 'manifest.sha256').read_text().strip():
        raise ValueError('Parent execution manifest changed')
    for name, digest in source_manifest['files_sha256'].items():
        if sha(source / name) != digest:
            raise ValueError('Parent execution source changed: ' + name)
    paths = [signals / arm / 'signals.json' for arm in ('original', 'identity', 'omitted', 'combined')]
    ids = sorted({'0050'} | {e['members'][0] for path in paths for e in read(path)['entries']})
    plan = dict(parent_path=str(source.relative_to(ROOT)), parent_sha256=sha(source / 'manifest.json'),
        inherited_calls=source_manifest['calls_reserved'], stock_ids=ids, maximum_calls=100,
        signal_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths}, code_sha256=sha(__file__))
    if (output / 'plan.json').exists() and read(output / 'plan.json') != plan:
        raise ValueError('Use a new directory for a changed extension plan')
    if (output / 'manifest.json').exists():
        result = read(output / 'manifest.json')
        for name, digest in result['files_sha256'].items():
            if sha(output / name) != digest:
                raise ValueError('Execution extension source changed')
        return result
    write(output / 'plan.json', plan)
    if not (output / 'inputs').exists():
        shutil.copytree(source / 'inputs', output / 'inputs')
    shutil.copyfile(__file__, output / 'preparation-source.py')
    calls_file = output / 'attempts.json'
    calls = read(calls_file) if calls_file.exists() else []
    token = load_config().finmind_token
    def bounded(dataset, start, end, **kwargs):
        if dataset not in ('TaiwanStockDividend', 'TaiwanStockPriceLimit') or kwargs['data_id'] not in ids:
            raise ValueError('Unplanned execution source request')
        if plan['inherited_calls'] + len(calls) >= plan['maximum_calls']:
            raise ValueError('Shared execution preparation lifetime budget exhausted')
        calls.append(dict(dataset=dataset, stock_id=kwargs['data_id'], at=datetime.now(timezone.utc).isoformat()))
        write(calls_file, calls)
        kwargs.update(token=token, timeout=30, max_retries=0, requests_per_hour=5400)
        return fetch_dataset(dataset, start, end, **kwargs)
    def denied(*args, **kwargs):
        raise RuntimeError('Official origin requests are prohibited in this preparation')
    feeds = ReplayMarketFeeds(output / 'inputs/execution-feeds', offline=False, token=token,
                             finmind_fetch=bounded, http_get=denied)
    for sid in ids:
        dividend = output / 'inputs/dividends' / (sid + '.parquet')
        if not dividend.exists():
            frame = bounded('TaiwanStockDividend', date(2021, 1, 1), date(2026, 12, 31), data_id=sid)
            if not frame.empty and ('stock_id' not in frame or set(frame.stock_id.astype(str)) != {sid}):
                raise ValueError('Dividend identity differs')
            frame.to_parquet(dividend, index=False)
            print('dividend', sid, len(frame), flush=True)
        feeds.get_limits(sid)
    result = dict(plan=plan, calls_reserved=plan['inherited_calls'] + len(calls), new_calls=len(calls),
        files_sha256={str(p.relative_to(output)): sha(p) for p in output.rglob('*')
            if p.is_file() and p.suffix != '.lock' and p.name not in ('manifest.json', 'manifest.sha256')})
    write(output / 'manifest.json', result)
    (output / 'manifest.sha256').write_text(sha(output / 'manifest.json') + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'signals', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    with file_lock(args.output / '.prepare.lock', timeout=0):
        result = prepare(args.source, args.signals, args.output)
    print('new_calls', result.get('new_calls'), 'calls_reserved', result['calls_reserved'])
