"""Allocate audited cash, holdings and cash receivables to funded signal cohorts.

This describes a realized account path, not the counterfactual return obtained
by deleting a trade. Unpriced share rights are deliberately unsupported.
"""
from collections import defaultdict
import math


def same(actual, expected, label):
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(
            actual, expected, abs_tol=.011, rel_tol=0):
        raise ValueError(label)


def cohort_outcomes(case):
    account, summary = case['account'], case['summary']
    events = {row['event_id']: row for row in account['cohorts']}
    if len(events) != len(account['cohorts']):
        raise ValueError('Duplicate funded cohort')
    action_events = defaultdict(set)
    for row in account['corporate_actions']:
        action_events[(row['action_id'], row['stock_id'])].add(row['event_id'])
    cash, cost, marks, rights = [defaultdict(float) for _ in range(4)]
    for row in account['cash_ledger']:
        if row['kind'] == 'initial_deposit':
            continue
        eid = row.get('event_id')
        if not eid:
            matches = action_events[(row.get('action_id'), row['stock_id'])]
            if not matches and row['kind'] == 'fractional_share_payment':
                matches = {r['event_id'] for r in account['corporate_actions']
                           if r['kind'] == 'share_delivery' and r['stock_id'] == row['stock_id']
                           and r['date'] == row['date'] and r.get('fraction', 0) > 0}
            if len(matches) != 1:
                raise ValueError('Cash movement has no unique cohort')
            eid = next(iter(matches))
        if eid not in events or events[eid]['stock_id'] != row['stock_id']:
            raise ValueError('Cash movement cohort/stock differs')
        cash[eid] += row['cash_change']
        if row['kind'] == 'buy':
            cost[eid] -= row['cash_change']
    for row in summary['final_holdings']:
        eid = row['event_id']
        if eid not in events or events[eid]['stock_id'] != row['stock_id']:
            raise ValueError('Holding cohort/stock differs')
        marks[eid] += row['market_value']
    if summary['final_receivables'] != account['receivables']:
        raise ValueError('Final receivable journal differs')
    for row in summary['final_receivables']:
        if row['kind'] != 'cash':
            raise ValueError('Non-cash rights require an explicit valuation audit')
        eid = row['event_id']
        if eid not in events or events[eid]['stock_id'] != row['stock_id']:
            raise ValueError('Receivable cohort/stock differs')
        if not math.isfinite(row['amount']) or row['amount'] < 0:
            raise ValueError('Invalid cash receivable')
        rights[eid] += row['amount']
    same(sum(rights.values()), summary['receivable'], 'Receivables do not reconcile')
    same(sum(marks.values()), summary['market_value'], 'Holdings do not reconcile')
    same(sum(cash.values())+summary['initial_cash'], summary['cash'], 'Cash does not reconcile')
    same(sum(cash.values())+sum(marks.values())+sum(rights.values()), summary['profit'],
         'Cohort profit does not reconcile')
    result = {}
    for eid, event in events.items():
        if cost[eid] <= 0:
            raise ValueError('Funded cohort requires positive cost')
        pnl = cash[eid]+marks.get(eid, 0)+rights.get(eid, 0)
        result[eid] = dict(event_id=eid, stock_id=event['stock_id'], name=event['name'],
            signal_date=event.get('signal_date'), entry_date=event['entry_date'],
            exit_date=event.get('exit_date'), closed=eid not in marks,
            settled=eid not in marks and eid not in rights, cash_out=cost[eid],
            cash_net=cash[eid], market_value=marks.get(eid, 0),
            receivable=rights.get(eid, 0), pnl=pnl, return_on_cost=pnl/cost[eid])
    return result
