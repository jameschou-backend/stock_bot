#!/usr/bin/env python3
"""Replay a sealed monthly snapshot daily; verify cutoff and future-price invariance.

Read-only research: this never creates orders, fills, or forward ledger records.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.audit_signal_causality import NAMES, digest, project, sha, transform
from scripts.prepare_million_signals import build_signals


def run(source, historical, output):
    if output.exists():
        raise ValueError('Output already exists; choose a new path to preserve evidence')
    started = time.perf_counter()
    manifest_path = source / 'manifest.json'
    historical_manifest_path = historical / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    historical_manifest = json.loads(historical_manifest_path.read_text())
    expected = {
        manifest_path: sha(manifest_path),
        historical_manifest_path: manifest['historical_manifest_sha256'],
        historical / 'companies.parquet': historical_manifest['files_sha256']['companies.parquet'],
        Path(__file__).resolve(): sha(__file__),
        ROOT / 'scripts/audit_signal_causality.py': sha(ROOT / 'scripts/audit_signal_causality.py'),
    }
    expected.update({source / name: manifest['sha256'][name] for name in (*NAMES, 'signals.json')})
    expected.update({ROOT / name: value for name, value in manifest['code_sha256'].items()})
    for path, value in expected.items():
        if not path.is_file() or sha(path) != value:
            raise ValueError('Sealed input/code changed: ' + str(path))
    sealed = json.loads((source / 'signals.json').read_text())
    start, end = sealed['start'], sealed['signal_end']
    frames = [pd.read_parquet(source / name).set_index('date') for name in NAMES]
    for frame in frames:
        frame.index = pd.to_datetime(frame.index)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError('Price dates must be sorted and unique')
        if not frame.index.equals(frames[0].index):
            raise ValueError('Price frames must share the same calendar')
    companies = pd.read_parquet(historical / 'companies.parquet')
    names = companies.set_index('stock_id')['name'].to_dict()
    full = build_signals(*frames, companies, start=start, signal_end=end)
    matches = all(full[key] == sealed[key] for key in ('entries', 'rejections'))
    if not matches:
        raise ValueError('Full signal rebuild differs from the sealed snapshot')
    dates = frames[0].index[(frames[0].index >= start) & (frames[0].index <= end)]
    if len(dates) == 0 or str(dates[-1].date()) != end:
        raise ValueError('No complete replay range in the snapshot')
    daily, cases = [], []
    for date in dates:
        cutoff = str(date.date())
        target = project(full, cutoff)
        truncated = None
        for mode in ('truncate', 'mutate'):
            altered = transform(frames, cutoff, mode)
            rebuilt = build_signals(*altered, companies, start=start, signal_end=cutoff)
            actual = project(rebuilt, cutoff)
            differences = [key for key in target if actual[key] != target[key]]
            cases.append(dict(cutoff=cutoff, mode=mode, passed=not differences,
                              differing_sections=differences, expected_sha256=digest(target),
                              actual_sha256=digest(actual)))
            if mode == 'truncate':
                truncated = rebuilt
            print(cutoff, mode, 'PASS' if not differences else 'FAIL', flush=True)
        signals = []
        for entry in truncated['entries']:
            if entry['signal_date'] != cutoff:
                continue
            sid = entry['members'][0]
            liquidity = entry['liquidity_at_signal']
            signals.append(dict(entry, stock_id=sid, name=names[sid],
                reference_close=float(frames[2].loc[date, sid]),
                passes_turnover_screen=bool(liquidity['complete_20_sessions']
                    and liquidity['mean_turnover20_twd'] >= 50_000_000),
                execution_status='unknown_no_historical_intraday_fill_evidence'))
        trend = next(x for x in truncated['trend'] if x['date'] == cutoff)
        daily.append(dict(date=cutoff, trend=trend, signals=signals,
                          account_orders_evaluated=False))
    unchanged = all(sha(path) == value for path, value in expected.items())
    result = dict(start=start, end=end, snapshot_prepared_at=manifest['prepared_at'],
        interpretation='Close-of-day candidates for next session; not fills or account-level orders',
        account_policy=dict(slots=3, idle_cash='cash', benchmark='0050'),
        limitations=['Historical reconstruction from a later snapshot, not contemporaneously recorded signals',
                     'Future-price invariance does not certify historical publication versions or universe',
                     'No historical fills; cash, occupied slots and position exits cannot be asserted'],
        source_sha256={str(path.resolve()): value for path, value in expected.items()},
        full_rebuild_matches_sealed=matches, sources_unchanged_after_run=unchanged,
        passed=unchanged and all(x['passed'] for x in cases),
        finmind_requests=0, live_qualified=False, daily=daily, cases=cases,
        elapsed_seconds=round(time.perf_counter() - started, 3))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--historical', type=Path, default=ROOT / '.cache/million-replay-signals')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.historical, args.output)
    print(json.dumps({key: result[key] for key in ('passed', 'start', 'end', 'elapsed_seconds')}, indent=2))
    sys.exit(0 if result['passed'] else 1)
