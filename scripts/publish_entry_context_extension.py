#!/usr/bin/env python3
"""Extend the two entry-context screens without revising sealed historical rows.

No outcome labels, account state, data acquisition, or live orders are used.
The published rows are first occurrences of the parent price/volume signal.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.publish_entry_context_terminal import digest, market_days, write_sealed
from scripts.research_sector_participation_20261003 import participation_features
from skills.rally_context_features import attach_context, build_signal_features, conjunction
from skills.strategy_scanner.data import load_bundle

PREVIOUS_PUBLICATION = 'artifacts/forward_simulation/entry_context_terminal_20261007.json'
PREVIOUS_SHA256 = '872d189be103546cc99d8976329377680a77a8a1026103d43a1c565a452cc0ba'
CALCULATION_FILES = ('scripts/publish_entry_context_extension.py',
    'scripts/publish_entry_context_terminal.py', 'skills/rally_context_features.py',
    'skills/strategy_scanner/engine.py', 'skills/strategy_scanner/research_rules.py',
    'skills/strategy_scanner/data.py', 'scripts/research_sector_participation_20261003.py')


def local_path(root, value):
    """Keep published references relative to the project, including symlinks."""
    path = Path(value)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError('Publication paths must stay relative to the project: ' + str(value))
    return resolved


def read_sealed(root, descriptor):
    path = local_path(root, descriptor['path'])
    expected = descriptor['sha256']
    if (digest(path) != expected
            or path.with_suffix('.sha256').read_text().strip() != expected):
        raise ValueError('Previous evidence checksum failed: ' + descriptor['path'])
    return json.loads(path.read_text())


def verify_continuity(previous, previous_report, *, start, end, rows, day_context):
    """An extension may append sessions, never silently revise published history."""
    if (previous.get('schema') != 'entry_context_terminal_v1'
            or previous_report.get('schema') != 'entry60_current_signal_check_v1'
            or previous.get('start') != start or previous_report.get('start') != start
            or previous_report.get('end') != previous.get('end')
            or previous.get('live_qualified') is not False
            or previous_report.get('live_qualified') is not False
            or previous_report.get('parent') != 'legacy_course_breakout'
            or previous_report.get('account_backtest') is not False):
        raise ValueError('Previous publication has incompatible scope')
    prior_end = previous['end']
    if end <= prior_end:
        raise ValueError('Publication extension must add a later session')
    if not day_context or day_context[-1]['date'] != end:
        raise ValueError('Publication end must be an observed market session')
    overlapping_days = [row for row in day_context if row['date'] <= prior_end]
    if ([row['date'] for row in overlapping_days] != previous['dates']
            or json.dumps(overlapping_days, sort_keys=True, allow_nan=False)
            != json.dumps(previous['day_context'], sort_keys=True, allow_nan=False)):
        raise ValueError('Previously published daily context changed')
    overlapping_rows = [row for row in rows if row['signal_date'] <= prior_end]
    if (json.dumps(overlapping_rows, sort_keys=True, allow_nan=False)
            != json.dumps(previous_report['all_first_signals'], sort_keys=True, allow_nan=False)):
        raise ValueError('Previously published parent events or conditions changed')
    return dict(previous_end=prior_end, unchanged_dates=len(overlapping_days),
                unchanged_parent_events=len(overlapping_rows), exact_comparison=True)


def publish(root=ROOT, *, bundle, manifest_sha256, output, publication, end,
            start='2026-09-01', previous_publication=PREVIOUS_PUBLICATION,
            previous_sha256=PREVIOUS_SHA256):
    root = Path(root).resolve()
    for value in (start, end):
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError('Publication dates must use YYYY-MM-DD')
    if start > end:
        raise ValueError('Publication start is after end')
    bundle_path, output_path = local_path(root, bundle), local_path(root, output)
    publication_path = local_path(root, publication)
    previous_descriptor = dict(path=previous_publication, sha256=previous_sha256)
    previous = read_sealed(root, previous_descriptor)
    previous_report_descriptor = previous['artifacts']['report']
    previous_report = read_sealed(root, previous_report_descriptor)
    if (publication_path == local_path(root, previous_publication)
            or output_path / 'report.json' == local_path(root, previous_report_descriptor['path'])):
        raise ValueError('Extension needs new output paths; previous evidence is frozen')
    manifest_path = bundle_path / 'manifest.json'
    if digest(manifest_path) != manifest_sha256:
        raise ValueError('Unexpected source manifest')
    manifest = json.loads(manifest_path.read_text())
    if manifest['end'] != end:
        raise ValueError('Source cutoff differs from publication')
    sources = {filename: digest(root / filename) for filename in CALCULATION_FILES}
    sources.update({str(manifest_path.relative_to(root)): manifest_sha256,
                    previous_descriptor['path']: previous_descriptor['sha256'],
                    previous_report_descriptor['path']: previous_report_descriptor['sha256']})
    data = load_bundle(bundle_path, start, end)
    events, features, days, ids = build_signal_features(data['bars'], data['calendar'],
        start=start, end=end, original_signals=data['original_signals'], poc=[],
        provenance=data['provenance'])
    events = events.loc[events.cohort.eq('legacy_course_breakout')].copy()
    day_context = market_days(features, days, ids, events, start, end)
    del features, data['bars']
    frames = {}
    for filename in ('close-official', 'raw-close', 'raw-volume', 'eligibility'):
        path = bundle_path / (filename + '.parquet')
        expected = manifest['files_sha256'][path.name]
        if digest(path) != expected:
            raise ValueError('Source matrix checksum failed: ' + path.name)
        sources[str(path.relative_to(root))] = expected
        frames[filename] = pd.read_parquet(path, filters=[
            ('date', '>=', pd.Timestamp('2025-01-01')), ('date', '<=', pd.Timestamp(end))]).set_index('date')
    observations = [dict(signal_id=row.event_id, stock_id=row.stock_id, signal_date=row.signal_date)
                    for row in events.itertuples(index=False)]
    peers, groups = participation_features(*(frames[k] for k in (
        'close-official', 'raw-close', 'raw-volume', 'eligibility')), observations)
    joined = attach_context(events, [], peers, frames['close-official'].index)
    joined['market_narrow'] = ~joined['market_breadth']
    joined['contraction_and_narrow'] = conjunction(joined.contraction, joined.market_narrow)
    joined['peer_and_narrow'] = conjunction(joined.peer_breadth, joined.market_narrow)
    cols = ['stock_id', 'signal_date', 'contraction_ratio', 'contraction', 'market_breadth_value',
            'market_breadth_coverage', 'market_narrow', 'peer_breadth_value', 'peer_issue',
            'peer_breadth', 'contraction_and_narrow', 'peer_and_narrow']
    rows = json.loads(joined[cols].to_json(orient='records', double_precision=15))
    companies = pd.read_parquet(bundle_path / 'companies.parquet')
    names = dict(zip(companies.stock_id, companies.name))
    peer_map = {(x['stock_id'], x['signal_date']): x for x in peers}
    for row in rows:
        row['name'] = names.get(row['stock_id'], row['stock_id'])
        p = peer_map[(row['stock_id'], row['signal_date'])]
        row['group_cutoff_date'] = p['group_cutoff_date']
        row['peer_ids'] = p['peer_ids']
    continuity = verify_continuity(previous, previous_report, start=start, end=end,
                                   rows=rows, day_context=day_context)
    for path, sha in sources.items():
        if digest(root / path) != sha:
            raise ValueError('Source changed during calculation: ' + path)
    report = dict(schema='entry60_current_signal_check_v1', start=start, end=end,
        parent='legacy_course_breakout', selection='Known first occurrence of parent; filters at completed signal close',
        all_first_signals=rows, source_sha256=sources, source_provenance=data['provenance'],
        monthly_peers=groups, live_qualified=False, account_backtest=False,
        network_requests=0, finmind_requests=0, reconstructs_historical_features=True,
        known_at_signal_realtime_receipt=False, continuity=continuity)
    report_sha = write_sealed(output_path / 'report.json', report)
    publication_data = dict(schema='entry_context_terminal_v1', start=start, end=end,
        dates=[x['date'] for x in day_context], day_context=day_context,
        artifacts=dict(report=dict(path=str((output_path / 'report.json').relative_to(root)), sha256=report_sha),
                       bundle=dict(path=str(manifest_path.relative_to(root)), sha256=manifest_sha256)),
        previous_publication=previous_descriptor, continuity=continuity,
        source_sha256=sources, live_qualified=False, account_independent=True,
        historical_period_already_researched=True, signal_timing='T_completed_close_then_next_session',
        signal_policy='Known first parent occurrence; no holdings or capital restriction; no 60-session research cooldown',
        note='兩組只看進場條件；同儕依前月以前的價格相關性建立，不是官方產業分類。每日訊號與歷史60日冷卻勝率樣本分開。')
    sha = write_sealed(publication_path, publication_data)
    return dict(publication=str(publication_path.relative_to(root)), sha256=sha,
                first_parent_events=len(rows), dates=len(day_context), source_end=end,
                finmind_requests=0, continuity=continuity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, help='Project-relative sealed bundle directory')
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True, help='New project-relative report directory')
    parser.add_argument('--publication', required=True, help='New project-relative publication JSON')
    parser.add_argument('--start', default='2026-09-01')
    parser.add_argument('--end', required=True)
    parser.add_argument('--previous-publication', default=PREVIOUS_PUBLICATION)
    parser.add_argument('--previous-sha256', default=PREVIOUS_SHA256)
    print(json.dumps(publish(**vars(parser.parse_args())), ensure_ascii=False))
