#!/usr/bin/env python3
"""Reconcile portfolio changes and identify which original winners were missed."""
from collections import defaultdict
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from scripts.export_candidate_quality import verify_runs
from scripts.export_midpoint_2025_report import close
from skills.liquidity_candidates import ARMS, FILTERS


def cohort_outcomes(case):
    account, summary = case['account'], case['summary']
    # These fixed accounts have no remaining receivables. Do not silently value
    # future cash/stock rights at zero if this diagnostic is reused elsewhere.
    if summary['final_receivables']:
        raise ValueError('Outstanding rights require a separate valuation audit')
    events = {r['event_id']: r for r in account['cohorts']}
    action_events = {r['action_id']: r['event_id'] for r in account['corporate_actions']}
    cash, cost, marks = defaultdict(float), defaultdict(float), defaultdict(float)
    for row in account['cash_ledger']:
        if row['kind'] == 'initial_deposit':
            continue
        eid = row.get('event_id') or action_events.get(row.get('action_id'))
        if eid is None and row['kind'] == 'fractional_share_payment':
            matches = {r['event_id'] for r in account['corporate_actions']
                       if r['kind'] == 'share_delivery' and r['stock_id'] == row['stock_id']
                       and r['date'] == row['date'] and r.get('fraction', 0) > 0}
            if len(matches) != 1:
                raise ValueError('Ambiguous fractional cash allocation')
            eid = matches.pop()
        if eid not in events:
            raise ValueError('Cash movement has no unique cohort')
        cash[eid] += row['cash_change']
        if row['kind'] == 'buy':
            cost[eid] -= row['cash_change']
    for row in summary['final_holdings']:
        marks[row['event_id']] += row['market_value']
    close(sum(cash.values())+summary['initial_cash'], summary['cash'], 'Cohort cash allocation')
    close(sum(cash.values())+sum(marks.values()), summary['profit'], 'Cohort profit allocation')
    result = {}
    for eid, event in events.items():
        if cost[eid] <= 0:
            raise ValueError('Funded cohort requires positive cost')
        result[eid] = dict(event_id=eid, stock_id=event['stock_id'], name=event['name'],
            entry_date=event['entry_date'], exit_date=event.get('exit_date'),
            closed=eid not in marks, cash_out=cost[eid], pnl=cash[eid]+marks.get(eid, 0),
            return_on_cost=(cash[eid]+marks.get(eid, 0))/cost[eid])
    return result


def main():
    parser = argparse.ArgumentParser()
    for arg in ('left', 'right', 'signals', 'output'):
        parser.add_argument('--'+arg, type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing diagnostic')
    cases = verify_runs(a.left, a.right, ARMS)
    prepared = read(a.signals)
    reports = [read(p/'report.json') for p in (a.left, a.right)]
    signal_path = str(a.signals.resolve().relative_to(ROOT))
    if any(r['source_sha256'][signal_path] != sha(a.signals) for r in reports):
        raise ValueError('Candidate evidence differs from account inputs')
    outcomes = {arm: cohort_outcomes(cases[arm]) for arm in ('original', *FILTERS)}
    original = outcomes['original']
    comparison = {}
    for arm in FILTERS:
        current = outcomes[arm]
        removed = [original[eid] for eid in sorted(original.keys()-current.keys())]
        added = [current[eid] for eid in sorted(current.keys()-original.keys())]
        comparison[arm] = dict(same_cohorts=len(original.keys() & current.keys()),
            missed_original=removed, added=added,
            missed_closed_winners=sum(r['closed'] and r['pnl'] > 0 for r in removed),
            missed_closed_doublers=sum(r['closed'] and r['return_on_cost'] >= 1 for r in removed),
            exact_original_account=cases[arm]['account'] == cases['original']['account'])
    excluded = [dict(row, original_outcome=original.get(row['event_id']))
                for row in prepared['decisions'] if not all(row['passes'].values())]
    refs = {str(p.resolve().relative_to(ROOT)): sha(p) for p in
            (a.signals, a.left/'report.json', a.right/'report.json', Path(__file__))}
    write(a.output, dict(candidate_counts={k: len(v) for k, v in prepared['entries'].items()},
        excluded=excluded, comparison=comparison, cohorts=outcomes, source_sha256=refs,
        live_qualified=False, unseen_validation=False, attribution='account paths, not causal effects'))
    print({arm: {k: v for k, v in d.items() if k not in ('missed_original', 'added')}
           for arm, d in comparison.items()})


if __name__ == '__main__':
    main()
