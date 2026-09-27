#!/usr/bin/env python3
"""Frozen market/sector context around all six named 6446 first-bar events."""
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO
import argparse
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.backtest_tool_ui import verified_bytes
from app.file_lock import file_lock
from scripts.audit_current_causality_20260925 import matrices, BASE, ORIGINAL, verify_hashes
from scripts.research_exit_scenarios import read, write, sha
from scripts.research_surge_anatomy import clean
from skills.market_context import asof
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

INPUT = ROOT / '.cache/pharma-context-inputs-20260927'
SPEC = ROOT / 'docs/prereg_pharma_context_20260927.md'
PARENT = ROOT / 'artifacts/forward_simulation/pharmaessentia_20260927.json'
PUBLICATION = ROOT / 'artifacts/forward_simulation/pharma_context_20260927.json'


def prepare():
    INPUT.mkdir(parents=True, exist_ok=True)
    with file_lock(INPUT / 'prepare.lock', timeout=0):
        output = INPUT / 'taiex-tr.parquet'; meta = INPUT / 'taiex-tr.json'
        if meta.exists():
            if sha(output) != read(meta)['sha256']: raise ValueError('Sealed TAIEX TR copy changed')
            return
        if output.exists(): raise ValueError('Inspect incomplete prior input copy')
        source = ROOT / 'artifacts/benchmark/taiex_tr.parquet'
        digest = sha(source); shutil.copyfile(source, output)
        if digest != sha(output) or digest != sha(source): raise ValueError('TAIEX TR changed while copying')
        write(meta, dict(sha256=digest, original_path=str(source.relative_to(ROOT)), copied_at=datetime.now(timezone.utc).isoformat(),
            first_publication_verified=False, new_network_requests=0))


def checks(frames, companies, tr):
    close, _, raw, volume = frames
    results = []
    for date in ('2023-12-29', '2024-12-31', '2026-03-31'):
        day = pd.Timestamp(date); baseline = asof(close, raw, volume, companies, tr, day)
        for mode in ('truncate', 'mutate'):
            changed = []
            for frame in (close, raw, volume):
                f = frame.loc[:day].copy() if mode == 'truncate' else frame.copy()
                if mode == 'mutate': f.loc[f.index > day] *= 3.7
                changed.append(f)
            index = tr.loc[:day].copy() if mode == 'truncate' else tr.copy()
            if mode == 'mutate': index.loc[index.index > day] *= 5
            actual = asof(*changed, companies, index, day)
            if clean(baseline[0]) != clean(actual[0]): raise ValueError('Future changed past context')
            for a, b in zip(baseline[1:], actual[1:]): pd.testing.assert_frame_equal(a, b)
            results.append(dict(cutoff=date, mode=mode, passed=True))
    return results


def run(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / '.cache'): raise ValueError('Use a new cache directory')
    tick = time.perf_counter()
    with file_lock(INPUT / 'research.lock', timeout=0), offline_only():
        if sha(PARENT) != PARENT.with_suffix('.sha256').read_text().strip(): raise ValueError('Parent publication changed')
        pub = read(PARENT)
        r = read(ROOT / pub['report']['path'])
        if sha(ROOT / pub['report']['path']) != pub['report']['sha256']: raise ValueError('Parent report changed')
        expected = {ROOT / p: h for p, h in r['source_sha256'].items()}
        for path in (SPEC, Path(__file__), ROOT / 'skills/market_context.py', ROOT / 'tests/test_market_context.py',
                     INPUT / 'taiex-tr.json', PARENT, ROOT / pub['report']['path']): expected[path] = sha(path)
        expected[INPUT / 'taiex-tr.parquet'] = read(INPUT / 'taiex-tr.json')['sha256']
        ref = r['artifacts']['first_events']; expected[ROOT / ref['path']] = ref['sha256']
        events = pd.read_csv(BytesIO(verified_bytes(ref, ROOT, '.csv')), dtype={'stock_id': str})
        verify_hashes(expected)
        if not events.stock_id.eq('6446').all() or len(events) != 6: raise ValueError('Frozen six-case scope changed')
        frames = matrices(BASE); close, _, raw, volume = frames
        companies = pd.read_parquet(ORIGINAL / 'companies.parquet')
        tr = pd.read_parquet(INPUT / 'taiex-tr.parquet'); tr['date'] = pd.to_datetime(tr.date)
        tr = tr.set_index('date').tr_index.sort_index()
        causal = checks(frames, companies, tr)
        contexts, sectors, leaders = [], [], []
        for event in events.to_dict('records'):
            pos = close.index.get_loc(pd.Timestamp(event['signal_date']))
            for offset in (-1, 0):
                day = close.index[pos + offset]
                meta = dict(event_id=event['event_id'], signal_date=event['signal_date'], offset=offset, feature_date=str(day.date()))
                context, sector, leader = asof(close, raw, volume, companies, tr, day)
                contexts.append(dict(meta, **context))
                sectors.append(sector.assign(**meta)); leaders.append(leader.assign(**meta))
        output.mkdir(parents=True); artifacts = {}
        for name, table in [('context', pd.DataFrame(contexts)), ('sectors', pd.concat(sectors, ignore_index=True)),
                            ('leaders', pd.concat(leaders, ignore_index=True))]:
            path = output / (name + '.csv'); table.to_csv(path, index=False, encoding='utf-8-sig')
            artifacts[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(table))
        verify_hashes(expected)
        report = clean(dict(schema='pharma_context_v1', completed=True, live_qualified=False, adopted=False,
            unseen_validation=False, portfolio_returns_computed=False, strategy_net_return=None,
            first_publication_verified=False, historical_membership_verified=False,
            events=6, snapshots=12, source_sha256={str(p.relative_to(ROOT)): h for p, h in expected.items()},
            taiex_tr_first=str(tr.index.min().date()), taiex_tr_last=str(tr.index.max().date()),
            cohort_rows=len(companies), artifacts=artifacts, causality_checks=causal,
            finmind_requests=0, database_writes=0, elapsed_seconds=round(time.perf_counter() - tick, 3),
            limitations=['Six already-seen user-named events; no causal or out-of-sample inference.',
                'Current company/industry snapshot omits historical classifications, exits and some board-transfer history.',
                'Observed close times volume approximates trading activity, not actual money inflows.',
                'Biotechnology industry includes diverse businesses; not a precise new-drug peer or official index.',
                'Peers exclude 6446; leaders use only trailing known data and are not revenue leaders.',
                'TAIEX TR is a stale local total-return-index cache; no forward fills or substitute benchmark.']))
        write(output / 'report.json', report)
        write(output / 'manifest.json', dict(report_sha256=sha(output / 'report.json'), files=artifacts, source_sha256=report['source_sha256']))
        append_trial_registry(dict(timestamp=datetime.now(timezone.utc).isoformat(), source='pharma_context', status='completed',
            output=str(output.relative_to(ROOT)), preregistration_sha256=sha(SPEC), events=6, portfolio_returns_computed=False))
        print(dict(events=6, snapshots=12, elapsed_seconds=report['elapsed_seconds']))


def publish(folders):
    folders = [Path(p).resolve() for p in folders]
    if folders[0] == folders[1] or any(not p.is_relative_to(ROOT / '.cache') for p in folders): raise ValueError('Two distinct cache runs required')
    comparable = []
    for folder in folders:
        r = read(folder / 'report.json'); m = read(folder / 'manifest.json')
        if (r['schema'] != 'pharma_context_v1' or r['completed'] is not True or r['strategy_net_return'] is not None
            or any(r.get(k) is not False for k in ('live_qualified', 'adopted', 'unseen_validation', 'portfolio_returns_computed'))):
            raise ValueError('Unsupported qualification flags')
        if m != dict(report_sha256=sha(folder / 'report.json'), files=r['artifacts'], source_sha256=r['source_sha256']):
            raise ValueError('Manifest mismatch')
        verify_hashes({ROOT / p: h for p, h in r['source_sha256'].items()})
        for ref in r['artifacts'].values():
            p = ROOT / ref['path']
            if p.parent != folder or sha(p) != ref['sha256']: raise ValueError('Changed output')
        r.pop('elapsed_seconds')
        r['artifacts'] = {k: {f: v for f, v in a.items() if f != 'path'} for k, a in r['artifacts'].items()}
        comparable.append(r)
    if comparable[0] != comparable[1]: raise ValueError('Independent context studies differ')
    write(PUBLICATION, dict(schema='pharma_context_publication_v1', live_qualified=False, adopted=False,
        report=dict(path=str((folders[0] / 'report.json').relative_to(ROOT)), sha256=sha(folders[0] / 'report.json')),
        reproducibility=dict(passed=True, runs=[dict(path=str((p / 'manifest.json').relative_to(ROOT)), sha256=sha(p / 'manifest.json')) for p in folders],
            csv_sha256={k: v['sha256'] for k, v in comparable[0]['artifacts'].items()})))
    PUBLICATION.with_suffix('.sha256').write_text(sha(PUBLICATION) + '\n')
    print('Published two identical market-context studies')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true'); g.add_argument('--output', type=Path); g.add_argument('--publish', nargs=2, type=Path)
    a = p.parse_args()
    if a.prepare: prepare()
    elif a.publish: publish(a.publish)
    else: run(a.output)
