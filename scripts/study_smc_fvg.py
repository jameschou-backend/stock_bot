#!/usr/bin/env python3
"""Run the preregistered offline SMC/FVG study; never fetch, tune or trade."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scan_market_strategies import digest
from skills.smc_research import STRATEGY_IDS, compute_setups
from skills.smc_outcomes import compare_fvg
from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.engine import _prepare
from skills.strategy_scanner.outcomes import COSTS, _stats, measure_events
from skills.trial_registry import append_trial_registry

NAMES = {
    'research_breakout20': '20 日放量突破',
    'smc_bull_break': '確認波段高點突破（含初始）',
    'smc_bos': 'BOS 多頭延續',
    'smc_choch': 'CHoCH 多頭轉折',
    'smc_sweep': '假跌破波段低點收回',
    'fvg_form': 'FVG 形成後直接買',
    'fvg_retest': 'FVG 回測守穩後買',
    'smc_orderblock_retest': 'Order Block 回測守穩後買',
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def study(f, days, ids, *, start, end):
    """Keep signal construction separate from all forward outcome evaluation."""
    if pd.Timestamp(start) not in days or pd.Timestamp(end) != days[-1]:
        raise ValueError('Prepare inputs through exactly the observed study end')
    setups = compute_setups(f)
    event_rows, summary, context = [], [], []
    bench = f['c']['0050'].where(f['valid']['0050'] & f['eligible']['0050'].eq(True))
    ma = bench.rolling(60, min_periods=60).mean()
    regime = pd.Series('unknown', index=days)
    regime.loc[ma.notna() & bench.notna()] = 'at_or_below_ma60'
    regime.loc[bench.gt(ma)] = 'above_ma60'
    for strategy in STRATEGY_IDS:
        mask = pd.DataFrame(False, index=days, columns=ids)
        for event in setups['events']:
            if event['strategy_id'] == strategy:
                mask.at[pd.Timestamp(event['signal_date']), event['stock_id']] = True
        rows = measure_events(f, days, ids, mask, start=start, end=end)
        rows.insert(0, 'strategy_id', strategy)
        rows['market_context'] = pd.to_datetime(rows.signal_date).map(regime)
        event_rows.append(rows)
        for horizon in (5, 20, 60):
            window = rows[rows.horizon.eq(horizon)]
            for year in ['all'] + [str(y) for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year+1)]:
                group = window if year == 'all' else window[window.signal_date.str.startswith(year)]
                summary.append(dict(strategy_id=strategy, name=NAMES[strategy], horizon=horizon,
                                    year=year, **_stats(group)))
            for market in ('above_ma60', 'at_or_below_ma60', 'unknown'):
                context.append(dict(strategy_id=strategy, name=NAMES[strategy], horizon=horizon,
                                    market_context=market, **_stats(window[window.market_context.eq(market)])))
    paired_summary, paired_rows = compare_fvg(f, days, ids, setups['setups'], start, end)
    return dict(summary=summary, market_context=context, paired_summary=paired_summary,
                definitions=setups['definitions'], setup_counts_including_warmup=setups['counts']), pd.concat(event_rows, ignore_index=True), paired_rows, setups


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Choose a new empty output directory; study outputs are immutable')
    started = time.perf_counter()
    data = load_bundle(args.bundle, start=args.start, end=args.end)
    f, days, ids = _prepare(data['bars'], data['calendar'], pd.Timestamp(args.end))
    source_paths = [Path(__file__).resolve(), ROOT/'skills/smc_research.py', ROOT/'skills/smc_outcomes.py',
                    ROOT/'scripts/scan_market_strategies.py', ROOT/'skills/trial_registry.py']
    source_paths += sorted((ROOT/'skills/strategy_scanner').glob('*.py'))
    provenance = dict(data['provenance'], external_data_requests=0,
        source_code_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        prereg_sha256=digest(ROOT/'docs/prereg_smc_fvg_20261006.md'))
    report, events, paired, setups = study(f, days, ids, start=args.start, end=args.end)
    if any(digest(ROOT/path) != sha for path, sha in provenance['source_code_sha256'].items()):
        raise RuntimeError('Study source changed during computation; refuse misleading provenance')
    if digest(ROOT/'docs/prereg_smc_fvg_20261006.md') != provenance['prereg_sha256']:
        raise RuntimeError('Preregistration changed during computation')
    events.to_parquet(output/'events.parquet', index=False)
    paired.to_parquet(output/'fvg_pairs.parquet', index=False)
    write_json(output/'setups.json', setups)
    report.update(schema='smc_fvg_event_study_v1', start=args.start, end=args.end,
        created_at=datetime.now(timezone.utc).isoformat(), source_provenance=provenance,
        strategies=list(STRATEGY_IDS), strategy_names=NAMES, horizons=[5, 20, 60], costs=COSTS,
        hypothesis_count=24, paired_arm_window_count=6, registered_configurations=30,
        study_type='descriptive_overlapping_event_study_not_portfolio_backtest',
        entry_price='signal_T_plus_1_adjusted_open_proxy',
        exit_price='signal_T_plus_h_adjusted_close_proxy',
        paired_exit_price='formation_T_plus_h_adjusted_close_same_for_direct_wait_and_0050',
        market_context_timing='signal_close_0050_vs_trailing_60_session_mean',
        historical_period_already_researched=True, multiple_testing_adjusted=False,
        execution_capacity_verified=False, live_qualified=False, account_independent=True,
        cumulative_return=None, max_drawdown=None,
        limitations=[
            'Fixed daily bullish operational variants, not all SMC/ICT or proof of institutional activity.',
            'All signals overlap across stocks, dates and rules; events are not independent capital slots.',
            'Historical membership and raw source limitations inherited from sealed bundle; no fresh source certification.',
            'Fees/slippage included; queue, odd-lot fills, capacity and actual corporate-action cash not verified.',
            'Formation-date bootstrap leaves repeated-stock and overlapping-window dependence; no multiplicity correction.',
            'Same-window annual and market-context splits are diagnostics, not unseen tests or tuned trading rules.',
        ],
        elapsed_compute_seconds=round(time.perf_counter()-started, 3),
        details={p.name:dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                             sha256=digest(p)) for p in sorted(output.iterdir())})
    write_json(output/'summary.json', report)
    trial_counts = []
    base = dict(timestamp=datetime.now(timezone.utc).isoformat(), source='smc_fvg_event_study',
        command=' '.join(sys.argv), study_type=report['study_type'], sharpe=None,
        start=args.start, end=args.end, report_sha256=digest(output/'summary.json'))
    for row in report['summary']:
        if row['year'] == 'all':
            trial_counts.append(append_trial_registry(dict(base, params=dict(strategy_id=row['strategy_id'],
                horizon=row['horizon'], costs=COSTS), outcome=row)))
    for row in report['paired_summary']:
        if row['year'] == 'all':
            for arm in ('direct', 'wait', '0050'):
                trial_counts.append(append_trial_registry(dict(base, params=dict(strategy_id='paired_fvg_'+arm,
                    horizon=row['horizon'], costs=COSTS, common_formation_cohort=True), outcome=row)))
    write_json(output/'receipt.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        files_sha256={p.name:digest(p) for p in sorted(output.iterdir())},
        trial_registry_records=len(trial_counts), trial_registry_last_count=trial_counts[-1],
        source_provenance=provenance))
    print(json.dumps(dict(output=str(output), event_horizon_rows=len(events), paired_rows=len(paired),
        elapsed_seconds=round(time.perf_counter()-started, 3), registered_trials=len(trial_counts),
        external_data_requests=0, live_qualified=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'.cache/scanner-20261006/inputs-v1')
    parser.add_argument('--start', default='2024-01-02')
    parser.add_argument('--end', default='2026-10-05')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
