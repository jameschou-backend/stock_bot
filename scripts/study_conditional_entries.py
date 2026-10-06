#!/usr/bin/env python3
"""Measure added conditions within one frozen original-red candidate cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scan_market_strategies import digest
from skills.conditional_entries import compute_conditions
from skills.conditional_outcomes import analyze_conditions
from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.engine import _compile_rules, _prepare
from skills.strategy_scanner.outcomes import COSTS, _stats, measure_events
from skills.trial_registry import append_trial_registry


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def study(bars, calendar, *, original_signals, provenance, start, end):
    """Build causal candidates/conditions once, then attach future outcomes."""
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError('Study start exceeds end')
    if not provenance.get('original_candidates_complete'):
        raise ValueError('Conditional study requires complete hash-bound original candidate ledger')
    if pd.Timestamp(end) > pd.Timestamp(provenance['source_end']):
        raise ValueError('Study exceeds frozen source end')
    f, days, ids = _prepare(bars, calendar, pd.Timestamp(end))
    if pd.Timestamp(start) not in days or pd.Timestamp(end) not in days:
        raise ValueError('Study dates must be observed market sessions')
    z, masks, _, _ = _compile_rules(f, days, ids, original_signals=original_signals,
                                   provenance=provenance)
    matched, known, _, _ = masks['original_red']
    available = known & f['eligible'].eq(True).fillna(False)
    first = matched & available & available.shift(1, fill_value=False) & ~matched.shift(1, fill_value=False)
    first &= f['volume'].gt(0) & z['amount20'].ge(50_000_000)
    first.loc[(days < pd.Timestamp(start)) | (days > pd.Timestamp(end)), :] = False
    first.loc[:, [sid for sid in ids if sid.startswith('0')]] = False
    rows, columns = np.where(first.to_numpy(bool))
    del z, masks
    computed = compute_conditions(f)
    frames = []
    for identifier, condition in computed['conditions'].items():
        frames.append(pd.DataFrame(dict(signal_date=days[rows].strftime('%Y-%m-%d'),
            stock_id=np.asarray(ids)[columns], filter_id=identifier,
            known=condition['known'].to_numpy(bool)[rows, columns],
            matched=condition['matched'].to_numpy(bool)[rows, columns])))
    conditions = pd.concat(frames, ignore_index=True)
    # Outcome evaluation is deliberately last: no future field enters candidates.
    events = measure_events(f, days, ids, first, start=start, end=end)
    report = analyze_conditions(events, conditions, start, end,
                                filter_ids=list(computed['conditions']))
    report['condition_definitions'] = {key:{name:value for name,value in item.items()
        if name not in ('known', 'matched')} for key,item in computed['conditions'].items()}
    report['feature_definitions'] = computed['definitions']
    baseline = []
    for horizon in (5, 20, 60):
        window = events[events.horizon.eq(horizon)]
        for year in ['all'] + [str(y) for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year+1)]:
            group = window if year == 'all' else window[window.signal_date.str.startswith(year)]
            baseline.append(dict(horizon=horizon, year=year, **_stats(group)))
    report['baseline_summary'] = baseline
    report['candidate_count'] = len(rows)
    report['candidate_stocks'] = len(set(np.asarray(ids)[columns]))
    return report, events, conditions


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Choose a new empty output directory; results are immutable')
    started = time.perf_counter()
    sources = [Path(__file__).resolve(), ROOT/'skills/conditional_entries.py',
               ROOT/'skills/conditional_outcomes.py', ROOT/'skills/smc_research.py',
               ROOT/'scripts/scan_market_strategies.py', ROOT/'skills/trial_registry.py']
    sources += sorted((ROOT/'skills/strategy_scanner').glob('*.py'))
    code_hashes = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    prereg = ROOT/'docs/prereg_conditional_entries_20261006.md'
    prereg_sha = digest(prereg)
    data = load_bundle(args.bundle, start=args.start, end=args.end)
    report, events, conditions = study(data['bars'], data['calendar'],
        original_signals=data['original_signals'], provenance=data['provenance'],
        start=args.start, end=args.end)
    if any(digest(ROOT/path) != sha for path,sha in code_hashes.items()) or digest(prereg) != prereg_sha:
        raise RuntimeError('Source or preregistration changed during computation; refusing publication')
    events.to_parquet(output/'events.parquet', index=False)
    conditions.to_parquet(output/'conditions.parquet', index=False)
    report.update(schema='conditional_original_red_study_v1', start=args.start, end=args.end,
        created_at=datetime.now(timezone.utc).isoformat(), costs=COSTS,
        source_provenance=dict(data['provenance'], source_code_sha256=code_hashes,
            prereg_sha256=prereg_sha, external_data_requests=0),
        candidate_rule='original_red_known_first_plus_20mean_estimated_turnover_50m',
        entry_price='original_signal_T_plus_1_adjusted_open_proxy',
        exit_price='original_signal_T_plus_h_adjusted_close_proxy',
        study_type='descriptive_conditional_event_study_not_portfolio_backtest',
        registered_configurations=45, historical_period_already_researched=True,
        account_independent=True, cumulative_return=None, max_drawdown=None,
        multiple_testing_adjusted=False, live_qualified=False,
        elapsed_compute_seconds=round(time.perf_counter()-started,3),
        details={p.name:dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                             sha256=digest(p)) for p in sorted(output.iterdir())})
    write_json(output/'summary.json', report)
    base = dict(timestamp=datetime.now(timezone.utc).isoformat(), source='conditional_original_red',
        command=' '.join(sys.argv), study_type=report['study_type'], sharpe=None,
        start=args.start, end=args.end, report_sha256=digest(output/'summary.json'))
    trial_counts = []
    for row in report['baseline_summary']:
        if row['year'] == 'all':
            trial_counts.append(append_trial_registry(dict(base,
                params=dict(strategy_id='original_red_baseline', horizon=row['horizon'], costs=COSTS), outcome=row)))
    for key in ('summary','common_pool_summary'):
        for row in report[key]:
            if row['year'] == 'all':
                trial_counts.append(append_trial_registry(dict(base,
                    params=dict(filter_id=row['filter_id'], scope=row['scope'], horizon=row['horizon'], costs=COSTS), outcome=row)))
    if len(trial_counts) != 45:
        raise RuntimeError('Unexpected fixed configuration count')
    write_json(output/'receipt.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        files_sha256={p.name:digest(p) for p in sorted(output.iterdir())},
        trial_registry_records=len(trial_counts), trial_registry_last_count=trial_counts[-1]))
    print(json.dumps(dict(output=str(output), candidates=report['candidate_count'],
        condition_rows=len(conditions), outcome_rows=len(events), trials=len(trial_counts),
        elapsed_seconds=round(time.perf_counter()-started,3), external_data_requests=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'.cache/scanner-20261006/inputs-v1')
    parser.add_argument('--start', default='2024-01-02')
    parser.add_argument('--end', default='2026-10-05')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
