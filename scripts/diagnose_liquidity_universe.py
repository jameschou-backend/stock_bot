#!/usr/bin/env python3
"""Audit winner retention on complete expanded-universe liquidity accounts."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from scripts.export_candidate_quality import verify_runs
from skills.account_cohort_attribution import cohort_outcomes
from skills.liquidity_universe import ARMS
from skills.liquidity_candidates import FILTERS


def main():
    parser = argparse.ArgumentParser()
    for arg in ('left', 'right', 'signals', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    a = parser.parse_args()
    a.output = a.output.resolve()
    a.output.relative_to(ROOT)
    if a.output.exists():
        raise ValueError('Preserve existing diagnostic')
    cases = verify_runs(a.left, a.right, ARMS)
    prepared = read(a.signals)
    reports = [read(p/'report.json') for p in (a.left, a.right)]
    signal_path = str(a.signals.resolve().relative_to(ROOT))
    if any(r['source_sha256'][signal_path] != sha(a.signals) for r in reports):
        raise ValueError('Candidate evidence differs from account inputs')
    outcomes = {arm: cohort_outcomes(cases[arm]) for arm in ARMS if arm != 'benchmark'}
    original = outcomes['liquid_universe']
    decisions = {r['event_id']: r for r in prepared['decisions']}
    focus = sorted(original.values(), key=lambda r: r['pnl'], reverse=True)[:10]
    comparison, concentration, stocks = {}, {}, {}
    for arm, current in outcomes.items():
        stock_pnl = defaultdict(float)
        for row in current.values():
            stock_pnl[row['stock_id']] += row['pnl']
        stocks[arm] = sorted((dict(stock_id=sid, pnl=pnl) for sid, pnl in stock_pnl.items()),
                            key=lambda r: r['pnl'], reverse=True)
        positive = sum(max(0, r['pnl']) for r in stocks[arm])
        net = cases[arm]['summary']['profit']
        concentration[arm] = dict(top=stocks[arm][:10], positive_stock_profit=positive,
            top1_fraction_positive=stocks[arm][0]['pnl']/positive if positive else None,
            top1_fraction_net=stocks[arm][0]['pnl']/net if net > 0 else None,
            top3_fraction_positive=sum(r['pnl'] for r in stocks[arm][:3])/positive if positive else None,
            net_profit=net)
        if arm == 'liquid_universe':
            continue
        removed = [original[eid] for eid in sorted(original.keys()-current.keys())]
        added = [current[eid] for eid in sorted(current.keys()-original.keys())]
        orders = defaultdict(list)
        for row in cases[arm]['account']['orders']:
            if row['side'] == 'buy':
                orders[row['event_id']].append(row)
        focus_outcomes = []
        for row in focus:
            eid = row['event_id']
            selected = decisions[eid]['passes'][arm]
            filled = eid in current
            if selected and not orders[eid]:
                raise ValueError('Selected focus signal has no account order')
            if filled != any(o['filled_qty'] > 0 for o in orders[eid]):
                raise ValueError('Funded focus cohort disagrees with orders')
            focus_outcomes.append(dict(original=row, feature_values=decisions[eid]['values'],
                passed_filter=selected, actual_cohort=current.get(eid),
                buy_order_failures=dict(Counter(o['failure'] for o in orders[eid] if o.get('failure')))))
        comparison[arm] = dict(same_cohorts=len(original.keys() & current.keys()),
            missed_original=removed, added=added,
            directly_filtered_funded=[original[eid] for eid in original if not decisions[eid]['passes'][arm]],
            missed_closed_winners=sum(r['closed'] and r['pnl'] > 0 for r in removed),
            missed_closed_doublers=sum(r['closed'] and r['return_on_cost'] >= 1 for r in removed),
            focus=focus_outcomes,
            exact_original_account=cases[arm]['account'] == cases['liquid_universe']['account'])
    # The earlier independent-signal population overlaps, but is not an account.
    prior = ROOT/'artifacts/forward_simulation/liquidity_screen_20261001'
    prior_report = read(prior/'report.json')
    if sha(prior/'signals.parquet') != prior_report['exports_sha256']['signals.parquet']:
        raise ValueError('Independent liquidity evidence changed')
    independent = pd.read_parquet(prior/'signals.parquet').set_index('event_id')
    matched = 0
    for eid, row in decisions.items():
        if eid not in independent.index:
            continue
        for arm in FILTERS:
            if row['passes'][arm] != bool(independent.at[eid, arm]):
                raise ValueError('Overlapping liquidity decision differs')
        matched += 1
    if not matched:
        raise ValueError('No independent candidate overlap')
    # Keep a compact complete evidence table instead of duplicating 72 MB of entries.
    a.output.mkdir(parents=True)
    table = [dict(event_id=r['event_id'], stock_id=r['stock_id'], signal_date=r['signal_date'],
                  entry_date=r['entry_date'], **r['values'], **r['passes'],
                  unknown_features=','.join(r['unknown_features'])) for r in prepared['decisions']]
    pd.DataFrame(table).to_parquet(a.output/'candidates.parquet', index=False)
    pd.DataFrame(table).to_csv(a.output/'candidates.csv', index=False, encoding='utf-8-sig')
    for arm, rows in outcomes.items():
        pd.DataFrame(rows.values()).to_csv(a.output/(arm+'-cohorts.csv'), index=False, encoding='utf-8-sig')
    refs = {str(p.resolve().relative_to(ROOT)): sha(p) for p in (a.signals,
        a.left/'report.json', a.right/'report.json', Path(__file__),
        ROOT/'skills/account_cohort_attribution.py', prior/'report.json', prior/'signals.parquet')}
    write(a.output/'report.json', dict(candidate_counts={k: len(v) for k, v in prepared['entries'].items()},
        unknown_count=sum(bool(r['unknown_features']) for r in prepared['decisions']),
        prefix_checks=prepared['prefix_checks'], independent_overlap_verified=matched,
        comparison=comparison, concentration=concentration, stock_profit=stocks,
        source_sha256=refs, exports_sha256={p.name: sha(p) for p in a.output.iterdir()},
        live_qualified=False, unseen_validation=False, attribution='account paths, not causal effects'))
    print({arm: {k: v for k, v in d.items() if k not in
        ('missed_original', 'added', 'directly_filtered_funded', 'focus')} for arm, d in comparison.items()})


if __name__ == '__main__':
    main()
