#!/usr/bin/env python3
"""Fixed offline census of rally windows and causal scanner precursors.

Future prices label outcomes only. This is not a selection model, execution
simulation or portfolio backtest. No remote data or model fitting occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

from scripts.scan_market_strategies import digest, load_poc
from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.engine import _compile_rules, _prepare
from skills.strategy_scanner.outcomes import COSTS, _net
from skills.trial_registry import append_trial_registry

CONFIGURATIONS = ((20, .30), (60, .50))
PROTOTYPES = [
    dict(id='research_true_gap_red', name='研究原型：真向上缺口＋紅K', family='gap',
         status='research_prototype', live_qualified=False,
         definition='adjusted low T > adjusted high T-1 and raw close T > open T',
         source_urls=['https://www.tradingview.com/support/solutions/43000675999-gaps/']),
    dict(id='research_gap_retest5', name='研究原型：缺口後五日內回測守穩', family='gap',
         status='research_prototype', live_qualified=False,
         definition='prior1..5 session true up-gap; T low touches its adjusted low, T close above it and red; no intervening close below pre-gap high',
         source_urls=['https://www.tradingview.com/support/solutions/43000675999-gaps/']),
]


def gap_prototypes(f):
    """New causal hypotheses derived from the documented up-gap definition.

    The provider defines the gap; the red/retest trading conditions are our
    fixed research additions, not a strategy certified by that provider.
    """
    valid = f['valid'] & f['eligible'].eq(True).fillna(False)
    gap_known = valid & valid.shift(1, fill_value=False)
    gap = f['l'].gt(f['h'].shift(1)) & gap_known
    red = f['close'].gt(f['open'])
    retest = pd.DataFrame(False, index=gap.index, columns=gap.columns)
    # Longest candidate needs T-6..T, preserving missing sessions explicitly.
    retest_known = valid.rolling(7, min_periods=7).sum().eq(7)
    for age in range(1, 6):
        upper, lower = f['l'].shift(age), f['h'].shift(age+1)
        held = f['c'].rolling(age, min_periods=age).min().ge(lower)
        retest |= gap.shift(age, fill_value=False) & f['l'].le(upper) & f['c'].gt(upper) & held & red
    return {'research_true_gap_red': (gap & red, gap_known),
            'research_gap_retest5': (retest & retest_known, retest_known)}


def forward_labels(f, horizon, threshold):
    """T+1 adjusted open to T+h close; require every holding session."""
    if type(horizon) is not int or horizon < 1 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Require a positive integer horizon and positive threshold')
    close = f['c'].to_numpy(float)
    open_adj = (f['open'] * f['c'] / f['close']).to_numpy(float)
    good = (f['valid'] & f['eligible'].eq(True).fillna(False)).to_numpy(bool)
    good &= np.isfinite(open_adj) & (open_adj > 0) & np.isfinite(close) & (close > 0)
    good &= np.isfinite(f['volume'].to_numpy(float)) & (f['volume'].to_numpy(float) > 0)
    n, width = close.shape
    mature = np.broadcast_to((np.arange(n) + horizon < n)[:, None], close.shape).copy()
    complete = np.zeros_like(good)
    gross = np.full(close.shape, np.nan)
    net = np.full(close.shape, np.nan)
    if horizon < n:
        missing = np.vstack([np.zeros((1, width), dtype=int), (~good).cumsum(axis=0)])
        rows = np.arange(n-horizon)
        complete[rows] = missing[rows+horizon+1] - missing[rows+1] == 0
        result = close[rows+horizon] / open_adj[rows+1] - 1
        gross[rows] = np.where(complete[rows], result, np.nan)
        result_net = _net(open_adj[rows+1], close[rows+horizon], COSTS['stock_sell_tax'])
        net[rows] = np.where(complete[rows], result_net, np.nan)
    return dict(mature=mature, complete=complete, gross=gross, net=net,
                rally=complete & (gross >= threshold))


def known_first(match, available):
    """A missing previous observation cannot establish a first occurrence."""
    return match & available & available.shift(1, fill_value=False) & ~match.shift(1, fill_value=False)


def profile_coverage(days, ids, profiles):
    """Only observed known POC profiles belong in its conditional comparison."""
    result = pd.DataFrame(False, index=days, columns=ids)
    for row in profiles or []:
        day, sid = pd.Timestamp(row['signal_date']), row['stock_id']
        if day in days and sid in ids and row.get('status') in ('up', 'down') and row.get('available') is True:
            result.at[day, sid] = True
    return result.to_numpy(bool)


def episode_coordinates(rally, horizon):
    """Ex-post case deduplication: earliest anchor, no overlapping entry paths."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError('Episode horizon must be a positive integer')
    coords = []
    for col in range(rally.shape[1]):
        previous = -horizon
        for row in np.flatnonzero(rally[:, col]):
            if row >= previous+horizon:
                coords.append((int(row), col))
                previous = row
    return sorted(coords)


def _ratio(a, b):
    return float(a / b) if b else None


def summarize(first, available, base, labels):
    """Compare each rule against its own known-data denominator, not just winners."""
    candidates = first & available & base
    evaluated = candidates & labels['complete']
    comparison = available & base & labels['complete']
    tp = int((evaluated & labels['rally']).sum())
    count = int(evaluated.sum())
    known_rallies = int((comparison & labels['rally']).sum())
    precision = _ratio(tp, count)
    baseline = _ratio(known_rallies, int(comparison.sum()))
    return dict(first_events=int(candidates.sum()), evaluated=count, tp=tp,
        false_positives=count-tp,
        immature=int((candidates & ~labels['mature']).sum()),
        missing_future_path=int((candidates & labels['mature'] & ~labels['complete']).sum()),
        precision=precision, baseline_rate=baseline,
        lift=precision/baseline if precision is not None and baseline else None,
        recall_of_known_rally_windows=_ratio(tp, known_rallies),
        known_rally_windows=known_rallies,
        known_eligible_windows=int(comparison.sum()),
        all_rally_windows=int((base & labels['rally']).sum()),
        signal_unknown_windows=int((base & ~available).sum()),
        profit_win_rate=float((labels['net'][evaluated] > 0).mean()) if count else None,
        mean_net_return=float(labels['net'][evaluated].mean()) if count else None)


def study(bars, calendar, *, start, end, names=None, original_signals=(), poc=None, provenance=None):
    started = time.perf_counter()
    start_day, end_day = pd.Timestamp(start), pd.Timestamp(end)
    if start_day > end_day:
        raise ValueError('Study start exceeds end')
    if provenance and provenance.get('source_end') and end_day > pd.Timestamp(provenance['source_end']):
        raise ValueError('Study exceeds sealed source coverage')
    f, days, ids = _prepare(bars, calendar, end_day)
    if start_day not in days or end_day not in days:
        raise ValueError('Study endpoints must be observed market sessions')
    z, masks, catalog, evidence = _compile_rules(f, days, ids,
        original_signals=original_signals, poc=poc, provenance=provenance)
    entries = [x for x in catalog if x['status'] == 'active' and x['kind'] == 'entry']
    # Shared liquidity restriction is observable at T and independent of future outcome.
    base = (f['valid'] & f['eligible'].eq(True).fillna(False)
            & z['amount20'].ge(50_000_000) & f['volume'].gt(0)).to_numpy(bool)
    base[(days < start_day) | (days > end_day)] = False
    base[:, [i for i, sid in enumerate(ids) if sid.startswith('0')]] = False
    del z, evidence
    years = ['all'] + [str(y) for y in range(start_day.year, end_day.year+1)]
    firsts, available_rules, classifiable_rules = {}, {}, {}
    poc_up = masks['poc_up_red'][0].to_numpy(bool).copy()
    for item in entries:
        match, known, _, _ = masks[item['id']]
        available = known & f['eligible'].eq(True).fillna(False)
        firsts[item['id']] = known_first(match, available).to_numpy(bool)
        available_rules[item['id']] = available.to_numpy(bool)
        classifiable_rules[item['id']] = (available & available.shift(1, fill_value=False)).to_numpy(bool)
    del masks
    prototypes = gap_prototypes(f)
    # Noncandidate "known false" POC evaluations are valid scanner semantics,
    # but not an observed POC comparison population. Keep its baseline conditional.
    if 'poc_up_red' in available_rules:
        available_rules['poc_up_red'] &= profile_coverage(days, ids, poc)
    recent_by_family, known_by_family = {}, {}
    for family in sorted({x['family'] for x in entries if x['id'] != 'poc_up_red'}):
        members = [x['id'] for x in entries if x['family'] == family and x['id'] != 'poc_up_red']
        union = np.logical_or.reduce([firsts[sid] & base for sid in members])
        known = np.logical_and.reduce([classifiable_rules[sid] for sid in members])
        recent_by_family[family] = pd.DataFrame(union).rolling(11, min_periods=11).max().eq(1).to_numpy(bool)
        known_by_family[family] = pd.DataFrame(known).rolling(11, min_periods=11).min().eq(1).to_numpy(bool)
    recent_by_family['any_non_poc'] = np.logical_or.reduce(list(recent_by_family.values()))
    known_by_family['any_non_poc'] = np.logical_and.reduce(list(known_by_family.values()))
    rows, case_rows, universes, episode_summaries, family_summary, paired_poc_red = [], [], [], [], [], []
    for horizon, threshold in CONFIGURATIONS:
        labels = forward_labels(f, horizon, threshold)
        for year in years:
            subset = base.copy()
            if year != 'all':
                subset[days.year != int(year)] = False
            mature_base = subset & labels['complete']
            universes.append(dict(horizon=horizon, threshold=threshold, year=year,
                eligible_windows=int(subset.sum()), complete_windows=int(mature_base.sum()),
                immature=int((subset & ~labels['mature']).sum()),
                missing_future_path=int((subset & labels['mature'] & ~labels['complete']).sum()),
                rally_windows=int((subset & labels['rally']).sum()),
                rally_stocks=int((subset & labels['rally']).any(axis=0).sum()),
                baseline_rate=_ratio(int((subset & labels['rally']).sum()), int(mature_base.sum()))))
            for item in entries:
                sid = item['id']
                rows.append(dict(strategy_id=sid, name=item['name'], family=item['family'],
                    horizon=horizon, threshold=threshold, year=year,
                    limited_coverage=sid == 'poc_up_red',
                    comparison_scope='known_profile_original_candidates_only' if sid == 'poc_up_red' else 'rule_known_eligible_stockdays',
                    **summarize(firsts[sid], available_rules[sid], subset, labels)))
            for item in PROTOTYPES:
                match, available = prototypes[item['id']]
                rows.append(dict(strategy_id=item['id'], name=item['name'], family=item['family'],
                    horizon=horizon, threshold=threshold, year=year,
                    limited_coverage=False, research_prototype=True,
                    comparison_scope='rule_known_eligible_stockdays',
                    **summarize(known_first(match, available).to_numpy(bool),
                                available.to_numpy(bool), subset, labels)))
            for family, recent in recent_by_family.items():
                sample = mature_base & known_by_family[family]
                positive = sample & labels['rally']
                negative = sample & ~labels['rally']
                signal = sample & recent
                hits = int((signal & labels['rally']).sum())
                baseline = _ratio(int(positive.sum()), int(sample.sum()))
                precision = _ratio(hits, int(signal.sum()))
                family_summary.append(dict(family=family, year=year, horizon=horizon,
                    known_windows=int(sample.sum()), rally_windows=int(positive.sum()),
                    precursor_windows=int(signal.sum()), precursor_rally_windows=hits,
                    prevalence_in_rallies=_ratio(int((positive & recent).sum()), int(positive.sum())),
                    prevalence_in_nonrallies=_ratio(int((negative & recent).sum()), int(negative.sum())),
                    precision=precision, baseline_rate=baseline,
                    lift=precision/baseline if baseline and precision is not None else None,
                    definition='any known first event of this family within previous10 sessions or anchor; all family rules known over11 sessions; POC excluded'))
            # Same red-first events, known POC coverage and holding dates in all arms.
            red_events = firsts['original_red'] & available_rules['poc_up_red'] & mature_base
            for arm, condition in [('all_known_profiles', np.ones(base.shape, dtype=bool)),
                                   ('poc_up', poc_up), ('poc_not_up', ~poc_up)]:
                population = red_events & condition
                total = int(population.sum())
                hits = int((population & labels['rally']).sum())
                paired_poc_red.append(dict(year=year, horizon=horizon, threshold=threshold,
                    arm=arm, events=total, rallies=hits, precision=_ratio(hits, total),
                    profit_win_rate=float((labels['net'][population] > 0).mean()) if total else None,
                    mean_net_return=float(labels['net'][population].mean()) if total else None,
                    selection='same_original_red_first_events_with_known_up_or_down_POC'))
        coordinates = episode_coordinates(base & labels['rally'], horizon)
        family_counts, strategy_counts, no_signal = Counter(), Counter(), 0
        for i, col in coordinates:
            signals = []
            for item in entries:
                sid = item['id']
                # Restrict both labels and displayed precursors to the stated study period.
                window = np.arange(max(days.get_loc(start_day), i-10), i+1)
                matches = window[firsts[sid][window, col] & base[window, col]]
                if len(matches):
                    k = int(matches[-1])
                    signals.append(dict(strategy_id=sid, name=item['name'], family=item['family'],
                                        signal_date=str(days[k].date()), lead_sessions=i-k))
            if not signals:
                no_signal += 1
            strategy_counts.update(s['strategy_id'] for s in signals)
            family_counts.update(set(s['family'] for s in signals))
            case_rows.append(dict(stock_id=ids[col], name=(names or {}).get(ids[col], ids[col]),
                anchor_date=str(days[i].date()), entry_date=str(days[i+1].date()),
                end_date=str(days[i+horizon].date()), horizon=horizon, threshold=threshold,
                gross_return=float(labels['gross'][i, col]), prior_signals=signals,
                classification='ex_post_earliest_nonoverlapping_rally_window_not_known_launch_date'))
        episode_summaries.append(dict(horizon=horizon, threshold=threshold,
            episode_count=len(coordinates), stocks=len({col for _, col in coordinates}),
            without_first_signal_in_prior10=no_signal,
            with_first_signal_in_prior10=len(coordinates)-no_signal,
            family_counts=dict(family_counts), strategy_counts=dict(strategy_counts)))
    return dict(schema='rally_precursor_study_v1', start=start, end=end,
        generated_at=datetime.now(timezone.utc).isoformat(), elapsed_compute_seconds=round(time.perf_counter()-started, 3),
        definitions=dict(targets=[dict(horizon=h, threshold=t) for h, t in CONFIGURATIONS],
            target='T+1 adjusted open to T+h adjusted close gross return >= threshold',
            signal='T close, known first day only; both current and previous rule availability required',
            universe='frozen individual stocks; eligible, valid at T; trailing20 mean raw-close-times-reported-share-volume estimated turnover proxy >= NTD50m; positive volume',
            maturity='all T+1..T+h stock sessions complete; missing path and immature excluded separately',
            baseline='same-date eligible stock windows; rule lift denominator further requires that rule known at T',
            precision='share of complete first-signal events reaching fixed endpoint rally threshold, not profit win rate',
            false_positive='did not reach rally threshold; may still be profitable',
            recall='same-anchor first-signal hits / known eligible rally windows; not recall of unique companies',
            episode='earliest qualifying window per stock, next anchor at least h sessions later; ex-post description only',
            precursor='known first signals on episode anchor or preceding10 observed market sessions within study period',
            precursor_lead='sessions before descriptive anchor, not prediction of future turning point',
            costs=COSTS, entry_price='T+1 adjusted open proxy', independent_windows=False),
        strategy_definitions=entries, research_prototype_definitions=PROTOTYPES,
        active_entry_rules=len(entries), research_prototypes=len(PROTOTYPES),
        universe_counts=universes, summary=rows,
        episodes=episode_summaries, family_summary=family_summary, paired_poc_red=paired_poc_red, examples=case_rows,
        source_provenance=provenance or {}, external_data_requests=0,
        account_backtest=False, live_qualified=False, historical_period_already_researched=True,
        multiple_testing_adjusted=False, parameters_selected_by_outcome=False,
        limitations=['股票日及不同策略會重疊，不能視為獨立樣本或把報酬相乘。',
            '歷史名冊與全部行情尚非完整獨立認證；缺資料窗口不計入成敗。',
            'POC只覆蓋部分2026原始候選，與其他策略母體不同；未將未知當不成立。',
            '此處飆升是固定期末報酬門檻，未達門檻不代表虧損；不是持有期間最高價。',
            '最早飆升窗口起點是事後標記，不是當時可知底部；案例只供回顧，不回灌買訊。',
            '無實際成交、資金名額、最小手續費、零股容量或漲跌停排隊驗證。'])


def run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; research runs are immutable')
    output.mkdir(parents=True, exist_ok=True)
    data = load_bundle(args.bundle, start=args.start, end=args.end)
    poc, poc_info = load_poc(args.poc_report, bundle=args.bundle,
        manifest_hash=data['provenance']['source_hashes']['manifest.json'])
    source = dict(data['provenance'], poc=poc_info,
        source_code_sha256={str(p.relative_to(ROOT)): digest(p) for p in
            sorted((ROOT/'skills/strategy_scanner').glob('*.py')) + [Path(__file__).resolve()]})
    report = study(data['bars'], data['calendar'], start=args.start, end=args.end,
        names=data['names'], original_signals=data['original_signals'], poc=poc, provenance=source)
    if args.directions:
        report['research_directions'] = json.loads(Path(args.directions).read_text())
        report['source_provenance']['research_directions_sha256'] = digest(args.directions)
    all_cases = report['examples']
    cases_path = output/'all-episodes.json'
    cases_path.write_text(json.dumps(all_cases, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    # UI examples are display-only: top12 fixed-window returns in each year/horizon.
    # The complete census and every outcome remain in the separate hashed artifact.
    display = []
    for horizon, _ in CONFIGURATIONS:
        for year in range(pd.Timestamp(args.start).year, pd.Timestamp(args.end).year+1):
            selection = [x for x in all_cases if x['horizon'] == horizon and x['anchor_date'].startswith(str(year))]
            display.extend(sorted(selection, key=lambda x: (-x['gross_return'], x['anchor_date'], x['stock_id']))[:12])
    report['examples'] = display
    report['all_episodes_artifact'] = dict(path=str(cases_path.relative_to(ROOT)), sha256=digest(cases_path), count=len(all_cases))
    report['examples_policy'] = 'display_only_top12_fixed_window_return_per_year_and_horizon; not used in aggregate calculations'
    report['trial_records'] = [dict(strategy_id=row['strategy_id'], horizon=row['horizon'],
        threshold=row['threshold'], prototype=row.get('research_prototype', False),
        signal_policy='known_first_day_only_T_close', cost_model=COSTS,
        outcome=row) for row in report['summary'] if row['year'] == 'all']
    report['hypothesis_count'] = len(report['trial_records'])
    report['descriptive_diagnostic_groups'] = dict(
        precursor_family_horizon=2*len({r['family'] for r in report['family_summary']}),
        paired_POC_red_first_arms=6, deduplicated_episode_horizons=2,
        independent_tests=False)
    target = output/'report.json'
    target.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    target.with_suffix('.sha256').write_text(digest(target)+'\n')
    registry_counts = []
    for record in report['trial_records']:
        registry_counts.append(append_trial_registry(dict(timestamp=report['generated_at'],
            source='rally_precursor_classification', command=' '.join(sys.argv), sharpe=None,
            study_type='exploratory_overlapping_rally_classification_not_account_backtest',
            start=args.start, end=args.end, params={k:v for k,v in record.items() if k != 'outcome'},
            outcome=record['outcome'], report_sha256=digest(target))))
    receipt = dict(schema='rally_precursor_receipt_v1',
        files_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()},
        trial_registry_records=len(registry_counts),
        trial_registry_last_count=registry_counts[-1] if registry_counts else None)
    (output/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), summary_rows=len(report['summary']),
        cases=len(all_cases), displayed_cases=len(display), elapsed_compute_seconds=report['elapsed_compute_seconds'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--poc-report', type=Path, required=True)
    parser.add_argument('--start', default='2024-01-02')
    parser.add_argument('--end', default='2026-10-05')
    parser.add_argument('--directions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
