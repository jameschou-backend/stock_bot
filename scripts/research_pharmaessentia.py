#!/usr/bin/env python3
"""Add the user-named 6446 case without changing sealed ten-stock evidence."""
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from app.backtest_tool_ui import verified_bytes
from app.file_lock import file_lock
from app.first_bar_ui import load as load_first
from app.launch_flows_ui import load as load_parent, PUBLICATION as PARENT
from scripts.audit_current_causality_20260925 import BASE, ORIGINAL, matrices, verify_hashes
from scripts.research_exit_scenarios import read, write, sha
from scripts.research_launch_flows import causal_checks
from scripts.research_surge_anatomy import clean
from skills.launch_flows import normalize, price_features, flow_features, snapshots
from skills.stock_launch import HORIZONS, features, outcomes, snapshot, episode_positions
from skills.stock_launch import causal_checks as launch_causal_checks
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

SID = '6446'
SPEC = ROOT / 'docs/prereg_pharmaessentia_20260927.md'
INPUT = ROOT / '.cache/pharmaessentia-inputs-20260927'
PUBLICATION = ROOT / 'artifacts/forward_simulation/pharmaessentia_20260927.json'


def prepare():
    from app.config import load_config
    from app.finmind import fetch_dataset
    INPUT.mkdir(parents=True, exist_ok=True)
    with file_lock(INPUT / 'prepare.lock', timeout=0):
        path = INPUT / '6446.parquet'
        meta = path.with_suffix('.json')
        if meta.exists():
            if sha(path) != read(meta)['sha256']:
                raise ValueError('Sealed 6446 source changed')
            print('Using sealed 6446 source; no new requests')
            return
        if path.exists():
            raise ValueError('Unsealed input exists; inspect it before collecting again')
        config = load_config()
        frame = fetch_dataset('TaiwanStockInstitutionalInvestorsBuySell', date(2021, 1, 1), date(2026, 9, 9),
            token=config.finmind_token, data_id=SID, requests_per_hour=config.finmind_requests_per_hour,
            max_retries=0, timeout=45)
        if (frame.empty or not frame.stock_id.eq(SID).all()
                or not pd.to_datetime(frame.date).between('2021-01-01', '2026-09-09').all()):
            raise ValueError('6446 source identity or interval mismatch')
        normalize(frame)
        frame.to_parquet(path, index=False)
        write(meta, dict(sha256=sha(path), rows=len(frame), dataset='TaiwanStockInstitutionalInvestorsBuySell',
            retrieved_at=frame.attrs.get('retrieved_at'), cache_hit=frame.attrs.get('cache_hit', False),
            historical_first_publication_verified=False))
        print('6446 source sealed:', len(frame), 'rows')


def describe(events, timeline):
    """One T-1 flow row and one T price row per event; never choose by return."""
    keys = ['event_id']
    prior_columns = [f'{actor}_{field}' for actor in ('foreign', 'trust', 'dealer')
        for field in ('net5', 'buy_streak', 'streak3')]
    current_columns = ['above20', 'above60', 'ma_stack', 'distance_ma20', 'distance_ma60', 'volume_ratio']
    current_columns += [f'{actor}_net' for actor in ('foreign', 'trust', 'dealer')]
    prior = timeline[timeline.offset.eq(-1)][keys + prior_columns].rename(
        columns={k: 'prior_' + k for k in prior_columns})
    current = timeline[timeline.offset.eq(0)][keys + current_columns]
    # Event tables may already contain the identical volume-ratio feature.
    return events.drop(columns=current_columns, errors='ignore').merge(prior, on=keys, how='left',
        validate='many_to_one').merge(current, on=keys, how='left', validate='many_to_one')


def price_coverage(frames):
    coverage = {}
    for name, frame in zip(('adjusted', 'quality', 'raw', 'volume'), frames):
        series = frame[SID].where(lambda s: np.isfinite(s) & s.gt(0)).dropna()
        coverage[name] = dict(first=str(series.index.min().date()) if len(series) else None,
            last=str(series.index.max().date()) if len(series) else None, valid_days=len(series),
            by_year=series.groupby(series.index.year).size().to_dict())
    return coverage


def run(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / '.cache'):
        raise ValueError('Use a new immutable cache directory')
    tick = time.perf_counter()
    with file_lock(INPUT / 'research.lock', timeout=0), offline_only():
        parent = load_parent()
        first_report, first_pub = load_first()
        expected = {ROOT / p: h for p, h in parent['source_sha256'].items()}
        expected.update({ROOT / p: h for p, h in first_report['source_sha256'].items()})
        input_path = INPUT / '6446.parquet'
        input_meta = read(input_path.with_suffix('.json'))
        expected[input_path] = input_meta['sha256']
        for p in (input_path.with_suffix('.json'), SPEC, Path(__file__), ROOT / PARENT,
                ROOT / 'tests/test_pharmaessentia.py'):
            expected[p] = sha(p)
        folder = Path(first_pub['report']['path']).parent
        def source(name):
            path = folder / (name + '.csv')
            h = first_pub['reproducibility']['files_sha256'][name + '.csv']
            expected[ROOT / path] = h
            return pd.read_csv(BytesIO(verified_bytes(dict(path=str(path), sha256=h), ROOT, '.csv')),
                dtype={'stock_id': str})
        first = source('events').query('stock_id == @SID').copy()
        result = source('outcomes').query('stock_id == @SID and lag == 8').copy()
        verify_hashes(expected)
        frames = [f[[SID, '0050']] for f in matrices(BASE)]
        close, quality, raw, volume = frames
        companies = pd.read_parquet(ORIGINAL / 'companies.parquet')
        companies = companies[companies.stock_id.eq(SID)]
        normalized = normalize(pd.read_parquet(input_path))
        price = price_features(close)
        flow = flow_features(normalized, close.index, volume)
        checks = causal_checks(close, volume, normalized, price, flow)
        checks += launch_causal_checks(close, raw, volume, companies,
            ('2023-12-29', '2024-12-31', '2026-03-31'))
        price['volume_ratio'] = volume / volume.shift(1).rolling(20, min_periods=20).mean()
        computed = features(close, raw, volume, companies)
        retro_rows = []
        for horizon in HORIZONS:
            label = outcomes(close, quality, horizon)
            panel = pd.concat([snapshot(computed, label, close.index, int(p), horizon, only=[SID], eligible_only=False)
                for p in np.flatnonzero(close.index >= pd.Timestamp('2022-01-03'))], ignore_index=True)
            for sid, pos in episode_positions(panel, close.index, horizon):
                row = panel[panel.signal_date.eq(str(close.index[pos].date()))].iloc[0].to_dict()
                row.update(event_id=f'retro-{horizon}-{row["signal_date"]}-{sid}', kind='retrospective')
                retro_rows.append(row)
        retro = pd.DataFrame(retro_rows, columns=None if retro_rows else ['event_id', 'stock_id', 'signal_date', 'horizon'])
        first['kind'] = 'first_bar'
        first['named_case'] = True  # Added after the original study, never rewrite its controls.
        first_timeline = snapshots(first, price, flow, close.index)
        retro_timeline = snapshots(retro, price, flow, close.index)
        # Fixed stock has events in this interval; absence must be visible, not silently publish blank joins.
        if first.empty or retro.empty:
            raise ValueError('No cases in the fixed interval; report absence explicitly')
        tables = dict(first_events=first, first_timeline=first_timeline, first_results=result,
            first_summary=describe(result, first_timeline), retrospective_events=retro,
            retrospective_timeline=retro_timeline, retrospective_summary=describe(retro, retro_timeline))
        output.mkdir(parents=True)
        files = {}
        for name, table in tables.items():
            path = output / (name + '.csv')
            table.to_csv(path, index=False, encoding='utf-8-sig')
            files[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(table))
        verify_hashes(expected)
        report = clean(dict(schema='pharmaessentia_case_v1', completed=True, stock_id=SID,
            start='2022-01-03', end='2026-09-09', live_qualified=False, adopted=False,
            historical_first_publication_verified=False, portfolio_returns_computed=False, strategy_net_return=None,
            price_coverage=price_coverage(frames),
            first_events=len(first), retrospective_counts=retro.groupby('horizon').size().to_dict(),
            causality_checks=checks, artifacts=files, source_sha256={str(p.relative_to(ROOT)): h for p, h in expected.items()},
            collection_requests=int(not input_meta['cache_hit']), research_network_calls=0, database_writes=0,
            unsupported_institutional_days=int((~normalized.schema_supported).sum()),
            elapsed_seconds=round(time.perf_counter() - tick, 3),
            limitations=['6446 is an additional user-selected historical case, not unseen validation.',
                'Previous ten-stock study controls included 6446; they are not independent eleven-stock controls.',
                'Sealed price coverage may start after the requested study interval; see price_coverage.',
                'Retrospective anchors use future returns and cannot be traded as signals.',
                'First-bar outcomes are gross prices without costs, fills or capital allocation.',
                'Historical revisions, first publication, current membership and correlated events remain limitations.']))
        write(output / 'report.json', report)
        write(output / 'manifest.json', dict(report_sha256=sha(output / 'report.json'), files=files,
            source_sha256=report['source_sha256']))
        append_trial_registry(dict(timestamp=datetime.now(timezone.utc).isoformat(), source='pharmaessentia_case',
            status='completed', output=str(output.relative_to(ROOT)), preregistration_sha256=sha(SPEC),
            first_events=len(first), retrospective_events=len(retro), portfolio_returns_computed=False))
        print({k: report[k] for k in ('first_events', 'retrospective_counts', 'elapsed_seconds')})
        return report


def publish(folders):
    folders = [Path(p).resolve() for p in folders]
    if len(folders) != 2 or folders[0] == folders[1] or any(not p.is_relative_to(ROOT / '.cache') for p in folders):
        raise ValueError('Two distinct cache runs required')
    reports = []
    for folder in folders:
        r = read(folder / 'report.json'); m = read(folder / 'manifest.json')
        if m != dict(report_sha256=sha(folder / 'report.json'), files=r['artifacts'], source_sha256=r['source_sha256']):
            raise ValueError('Manifest mismatch')
        verify_hashes({ROOT / p: h for p, h in r['source_sha256'].items()})
        for ref in r['artifacts'].values():
            path = ROOT / ref['path']
            if path.parent != folder or sha(path) != ref['sha256']:
                raise ValueError('Output changed')
        comparable = dict(r); comparable.pop('elapsed_seconds')
        comparable['artifacts'] = {k: {f: v for f, v in ref.items() if f != 'path'} for k, ref in r['artifacts'].items()}
        reports.append(comparable)
    if reports[0] != reports[1]:
        raise ValueError('Independent case studies differ')
    write(PUBLICATION, dict(schema='pharmaessentia_publication_v1', live_qualified=False, adopted=False,
        report=dict(path=str((folders[0] / 'report.json').relative_to(ROOT)), sha256=sha(folders[0] / 'report.json')),
        reproducibility=dict(passed=True, runs=[dict(path=str((p / 'manifest.json').relative_to(ROOT)),
            sha256=sha(p / 'manifest.json')) for p in folders], csv_sha256={k: v['sha256'] for k, v in reports[0]['artifacts'].items()})))
    PUBLICATION.with_suffix('.sha256').write_text(sha(PUBLICATION) + '\n')
    print('Published two identical 6446 case studies')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--output', type=Path)
    group.add_argument('--publish', nargs=2, type=Path)
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.publish: publish(args.publish)
    else: run(args.output)
