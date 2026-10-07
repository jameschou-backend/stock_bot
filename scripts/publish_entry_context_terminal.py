#!/usr/bin/env python3
"""Publish the two fixed entry-context screens from sealed local prices only.

No outcome labels, account state, data acquisition, or live orders are used.
The published rows are first occurrences of the parent price/volume signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.research_sector_participation_20261003 import participation_features
from skills.rally_context_features import attach_context, build_signal_features, conjunction
from skills.strategy_scanner.data import load_bundle

BUNDLE = '.cache/scanner-20261007/inputs-v2'
MANIFEST_SHA = 'a0590a896dc0fe23ffaf7a45460ff25ec488857011eed2d6dc1a23686817ae81'
OUTPUT = '.cache/entry-context-terminal-20261007-v2'
PUBLICATION = 'artifacts/forward_simulation/entry_context_terminal_20261007.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_sealed(path, value):
    """Deterministic and idempotent; never overwrite different frozen evidence."""
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Output already contains different evidence: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(raw)
        temporary.replace(path)
    sha = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_suffix('.sha256')
    if sidecar.exists() and sidecar.read_text().strip() != sha:
        raise ValueError('Output sidecar differs: ' + str(path))
    if not sidecar.exists():
        sidecar.write_text(sha + '\n')
    return sha


def market_days(features, days, ids, events, start, end):
    """Same valid-price breadth definition as build_signal_features, every day."""
    stock_ids = [sid for sid in ids if not sid.startswith('0')]
    close = features['c'][stock_ids]
    ma60 = close.rolling(60, min_periods=60).mean()
    eligible = features['eligible'][stock_ids].eq(True).fillna(False)
    valid = close.notna() & ma60.notna()
    denominator = valid.sum(axis=1)
    coverage = denominator / eligible.sum(axis=1).replace(0, np.nan)
    breadth = (close.gt(ma60) & valid).sum(axis=1) / denominator.replace(0, np.nan)
    breadth = breadth.where(denominator.ge(500) & coverage.ge(.8))
    counts = events.groupby('signal_date').size()
    rows = []
    for date in days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]:
        text = str(date.date())
        value = float(breadth.loc[date]) if np.isfinite(breadth.loc[date]) else None
        cover = float(coverage.loc[date]) if np.isfinite(coverage.loc[date]) else None
        rows.append(dict(date=text, market_breadth_value=value,
            market_breadth_coverage=cover, market_narrow=None if value is None else value < .5,
            valid60_stocks=int(denominator.loc[date]), eligible_stocks=int(eligible.loc[date].sum()),
            parent_candidates=int(counts.get(text, 0))))
    return rows


def publish(root=ROOT):
    root = Path(root)
    bundle, output = root / BUNDLE, root / OUTPUT
    manifest_path = bundle / 'manifest.json'
    if digest(manifest_path) != MANIFEST_SHA:
        raise ValueError('Unexpected source manifest')
    manifest = json.loads(manifest_path.read_text())
    start, end = '2026-09-01', '2026-10-06'
    if manifest['end'] != end:
        raise ValueError('Source cutoff differs from publication')
    data = load_bundle(bundle, start, end)
    events, features, days, ids = build_signal_features(data['bars'], data['calendar'],
        start=start, end=end, original_signals=data['original_signals'], poc=[],
        provenance=data['provenance'])
    events = events.loc[events.cohort.eq('legacy_course_breakout')].copy()
    day_context = market_days(features, days, ids, events, start, end)
    del features, data['bars']
    frames, sources = {}, {BUNDLE + '/manifest.json': MANIFEST_SHA}
    for filename in ('close-official', 'raw-close', 'raw-volume', 'eligibility'):
        path = bundle / (filename + '.parquet')
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
    companies = pd.read_parquet(bundle / 'companies.parquet')
    names = dict(zip(companies.stock_id, companies.name))
    peer_map = {(x['stock_id'], x['signal_date']): x for x in peers}
    for row in rows:
        row['name'] = names.get(row['stock_id'], row['stock_id'])
        p = peer_map[(row['stock_id'], row['signal_date'])]
        row['group_cutoff_date'] = p['group_cutoff_date']
        row['peer_ids'] = p['peer_ids']
    for filename in ('scripts/publish_entry_context_terminal.py', 'skills/rally_context_features.py',
                     'skills/strategy_scanner/engine.py', 'skills/strategy_scanner/research_rules.py',
                     'scripts/research_sector_participation_20261003.py'):
        sources[filename] = digest(root / filename)
    # Compare the prior user-visible result without depending on it for calculation.
    previous = root / '.cache/entry60-current-20261007/report.json'
    if previous.exists():
        old = json.loads(previous.read_text())['all_first_signals']
        if len(old) != len(rows):
            raise ValueError('Parent-event count differs from prior published answer')
        for a, b in zip(old, rows):
            for key in cols:
                av, bv = a[key], b[key]
                equal = np.isclose(av, bv, rtol=0, atol=1e-9) if type(av) is float else av == bv
                if not equal:
                    raise ValueError('Prior result mismatch: ' + str((a['signal_date'], a['stock_id'], key)))
    for path, sha in sources.items():
        if digest(root / path) != sha:
            raise ValueError('Source changed during calculation: ' + path)
    report = dict(schema='entry60_current_signal_check_v1', start=start, end=end,
        parent='legacy_course_breakout', selection='Known first occurrence of parent; filters at completed signal close',
        all_first_signals=rows, source_sha256=sources, source_provenance=data['provenance'],
        monthly_peers=groups, live_qualified=False, account_backtest=False,
        network_requests=0, finmind_requests=0, reconstructs_historical_features=True,
        known_at_signal_realtime_receipt=False)
    report_sha = write_sealed(output / 'report.json', report)
    publication = dict(schema='entry_context_terminal_v1', start=start, end=end,
        dates=[x['date'] for x in day_context], day_context=day_context,
        artifacts=dict(report=dict(path=OUTPUT + '/report.json', sha256=report_sha),
                       bundle=dict(path=BUNDLE + '/manifest.json', sha256=MANIFEST_SHA)),
        source_sha256=sources, live_qualified=False, account_independent=True,
        historical_period_already_researched=True, signal_timing='T_completed_close_then_next_session',
        signal_policy='Known first parent occurrence; no holdings or capital restriction; no 60-session research cooldown',
        note='兩組只看進場條件；同儕依前月以前的價格相關性建立，不是官方產業分類。每日訊號與歷史60日冷卻勝率樣本分開。')
    sha = write_sealed(root / PUBLICATION, publication)
    return dict(publication=PUBLICATION, sha256=sha, first_parent_events=len(rows),
                dates=len(day_context), source_end=end, finmind_requests=0)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(publish(), ensure_ascii=False))
