#!/usr/bin/env python3
"""Offline 600-lot stability study on frozen prices and weekly snapshots."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.extend_scanner_daily_inputs import digest
from scripts.prepare_holder_stability import save, SPEC
from skills.holder_stability import (ARMS, LAGS, EXIT_RULES, COSTS, aggregate_600,
    holder_changes, with_holders, observations, event_outcomes, summarize, matched_pairs, paired_statistics)
from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.engine import _prepare
from skills.verified_backtest_tool import offline_only
from skills.trial_registry import append_trial_registry

START, END = '2022-01-03', '2026-10-30'
CODE = ('scripts/research_holder_stability.py', 'skills/holder_stability.py',
        'scripts/prepare_holder_stability.py', 'skills/theme_chips.py', 'skills/holding_validation.py',
        'skills/strategy_scanner/data.py', 'skills/strategy_scanner/engine.py',
        'tests/test_holder_stability.py', 'docs/prereg_holder_stability_600_20261102.md')


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def verify(refs):
    for name, expected in refs.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)


def causal_checks(f, days, companies, obs, weekly, aligned):
    checks = []
    for cutoff in ('2023-12-29', '2024-12-31', '2026-03-31'):
        stamp = pd.Timestamp(cutoff)
        # Compare identical weekly anchors; a partial-week terminal row is only
        # a test input, never added retroactively to historical observations.
        expected = obs[obs.signal_date.le(cutoff)].reset_index(drop=True)
        for mode in ('truncate', 'mutate'):
            subdays = days[days <= stamp] if mode == 'truncate' else days
            altered = {}
            for key in ('c', 'valid', 'eligible', 'a', 'close'):
                altered[key] = f[key].loc[subdays].copy()
                if mode == 'mutate':
                    altered[key].loc[altered[key].index > stamp] = False if key in ('valid', 'eligible') else 999999.
            actual = observations(altered, subdays, companies, start=START, end=cutoff if mode == 'truncate' else END)
            actual = actual[actual.event_id.isin(expected.event_id)].reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, actual)
            checks.append(dict(kind='price_' + mode, cutoff=cutoff, passed=True, rows=len(actual)))
        for lag in LAGS:
            query = obs[obs.signal_date.le(cutoff)]
            expected_holder = aligned[aligned.holder_lag.eq(lag) & aligned.signal_date.le(cutoff)].reset_index(drop=True)
            unavailable = pd.to_datetime(weekly.date) + pd.Timedelta(days=lag) > stamp
            for mode in ('truncate', 'mutate'):
                changed = weekly.loc[~unavailable].copy() if mode == 'truncate' else weekly.copy()
                if mode == 'mutate':
                    changed.loc[unavailable, ['large_pct', 'large_units', 'total_units']] = 999999.
                    changed.loc[unavailable, 'valid'] = False
                actual = with_holders(query, holder_changes(changed), lag).reset_index(drop=True)
                pd.testing.assert_frame_equal(expected_holder, actual)
                checks.append(dict(kind='holder_' + mode, lag=lag, cutoff=cutoff, passed=True, rows=len(actual)))
    return checks


def run(args):
    began = time.perf_counter()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / '.cache'):
        raise ValueError('Choose a new immutable project-cache output directory')
    refs = {name: digest(ROOT / name) for name in CODE}
    holding = args.holding.resolve()
    manifest = json.loads((holding / 'manifest.json').read_text())
    if digest(holding / 'manifest.json') != (holding / 'manifest.sha256').read_text().strip():
        raise ValueError('Holder manifest changed')
    refs[str((holding / 'manifest.json').relative_to(ROOT))] = digest(holding / 'manifest.json')
    refs.update(manifest['source_sha256'])
    verify(refs)
    definitions = [dict(arm=arm, lag=lag, exit_rule=rule, cost=cost)
                   for arm in ARMS for lag in LAGS for rule in EXIT_RULES for cost in ('main', 'stress')]
    for trial in definitions:
        append_trial_registry(dict(source='holder_stability_600', status='started', **trial,
            output=str(output.relative_to(ROOT)), timestamp=datetime.now(timezone.utc).isoformat(),
            prereg_sha256=digest(SPEC), reproducibility=args.reproducibility))
    with offline_only():
        print('Loading frozen price bundle', flush=True)
        data = load_bundle(args.bundle, START, END)
        refs.update({str((args.bundle.resolve() / name).relative_to(ROOT)): sha
                     for name, sha in data['provenance']['source_hashes'].items()})
        f, days, ids = _prepare(data.pop('bars'), data['calendar'], pd.Timestamp(END))
        print('Building causal price observations', flush=True)
        obs = observations(f, days, data['companies'], start=START, end=END)
        cohort = [sid for sid in ids if not sid.startswith('0')]
        weeks = []
        for index, (day, descriptor) in enumerate(sorted(manifest['files'].items())):
            if not '2021-11-01' <= day <= END:
                continue
            path = ROOT / descriptor['path']
            if digest(path) != descriptor['sha256']:
                raise ValueError('Holder snapshot changed')
            weeks.append(aggregate_600(pd.read_parquet(path), day, cohort))
            if (index+1) % 50 == 0:
                print('Validated holder weeks', index+1, flush=True)
        weekly = pd.concat(weeks, ignore_index=True)
        changed = holder_changes(weekly)
        aligned = pd.concat([with_holders(obs, changed, lag) for lag in LAGS], ignore_index=True)
        print('Running cutoff/future-mutation checks', flush=True)
        checks = causal_checks(f, days, data['companies'], obs, weekly, aligned)
        selected = aligned[aligned.drawdown].drop_duplicates('event_id')
        print('Replaying selected event paths', len(selected), flush=True)
        outcomes = event_outcomes(f, days, selected)
        statistics = summarize(aligned, outcomes)
        pairs = matched_pairs(aligned)
        paired = paired_statistics(pairs, outcomes)
        latest = aligned[aligned.signal_date.eq(END)].copy()
        show = latest[latest.price_setup | latest.named_case].copy()
        show['scan_status'] = np.select([~show.price_known, ~show.price_setup, ~show.chip_known,
                                         show.stable_confirmed, show.stable, show.decreasing],
                                        ['price_unknown', 'not_deep_drawdown', 'holder_unknown',
                                         'stable_confirmed', 'stable_watch', 'holder_decreasing'], default='mixed')
        output.mkdir(parents=True)
        for name, frame in [('weekly', changed), ('observations', aligned), ('outcomes', outcomes),
                            ('pairs', pairs), ('latest', show)]:
            frame.to_parquet(output / (name + '.parquet'), index=False)
        save(output / 'statistics.json', clean(statistics))
        save(output / 'paired.json', clean(paired))
        save(output / 'latest.json', clean(show.sort_values(['holder_lag', 'amount20', 'stock_id'], ascending=[True, False, True]).to_dict('records')))
        period_counts = {}
        for lag in LAGS:
            subset = aligned[aligned.holder_lag.eq(lag) & ~aligned.named_case]
            price = subset[subset.price_setup]
            period_counts[str(lag)] = dict(price_setups=len(price), holder_known=int(price.chip_known.sum()),
                                          holder_unknown=int((~price.chip_known).sum()),
                                          holder_states=price.holder_state.value_counts().to_dict(),
                                          arms={arm: int(subset[arm].sum()) for arm in ARMS})
        verify(refs)
        report = clean(dict(schema='holder_stability_600_v1', completed=True, start=START, end=END,
            universe_count=len(cohort), observation_dates=obs.signal_date.nunique(), snapshot_weeks=len(weekly.date.unique()),
            counts=period_counts, latest_counts={str(lag): latest[latest.holder_lag.eq(lag) & latest.price_setup].holder_state.value_counts().to_dict() for lag in LAGS},
            latest=show.to_dict('records'), statistics=statistics, paired_statistics=paired,
            holder_quality_reasons=weekly.reason.value_counts().to_dict(),
            holder_change_reasons=changed.change_reason.value_counts().to_dict(),
            causal_checks=checks, costs=COSTS, trial_count=len(definitions), source_sha256=refs,
            price_provenance=data['provenance'], named_cases=['6442', '6223'],
            study_type='overlapping_event_backtest_not_finite_capital_account',
            capital_returns=None, account_max_drawdown=None, execution_capacity_verified=False,
            historical_first_publication_verified=False, live_qualified=False, adopted=False,
            unseen_validation=False, research_network_requests=0, db_writes=0,
            preparation_network_requests=manifest['network_requests'], original_strategy_changed=False,
            elapsed_seconds=time.perf_counter()-began,
            limitations=['Frozen current cohort and industry metadata, not a certified historical market universe.',
                'Observed weekly balances plus 8/15-day availability assumptions; historical first releases were not verified.',
                'Daily adjusted open/close are research return proxies; queues, whole-share sizing, minimum fees and capacity are not simulated.',
                'Overlapping weekly events are not independent account trades; no compound account return or account MDD is reported.',
                'The named examples are excluded from principal statistics. Previously studied history is not unseen validation.',
                'Matched controls use date/industry/confirmation/drawdown/liquidity, not verified historical market capitalization.',
                'Descriptive date-block intervals do not correct repeated research, reused controls or overlapping horizons.']))
        save(output / 'report.json', report)
        save(output / 'manifest.json', dict(schema='holder_stability_600_manifest_v1', source_sha256=refs,
             files_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
        for trial in definitions:
            append_trial_registry(dict(source='holder_stability_600', status='completed', **trial,
                output=str(output.relative_to(ROOT)), timestamp=datetime.now(timezone.utc).isoformat(),
                reproducibility=args.reproducibility, live_qualified=False))
    print(json.dumps(dict(elapsed_seconds=report['elapsed_seconds'], counts=report['counts'], latest=report['latest_counts']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--holding', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reproducibility', action='store_true')
    run(parser.parse_args())
