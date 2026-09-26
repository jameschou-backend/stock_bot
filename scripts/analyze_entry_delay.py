#!/usr/bin/env python3
"""Reconcile sealed entry-delay paths; no new strategy or counterfactual fills."""
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.residual_slots_ui import load, REPORT
from app.backtest_tool_ui import verified_bytes
from scripts.research_exit_scenarios import read, write, sha
from skills.backtest_case_cache import file_identities

COMPONENTS = ('entry_gross', 'exit_gross', 'execution_cost', 'distribution_cash',
              'terminal_holdings', 'terminal_receivables')


def number(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError('Nonfinite account amount')
    return result


def close(left, right, message):
    if abs(number(left)-number(right)) > Decimal('.03'):
        raise ValueError(message)


def event_book(case):
    if case.get('completed') is not True:
        raise ValueError('Incomplete account cannot explain completed returns')
    account = case['account']
    rows = {}
    for cohort in account['cohorts']:
        event = cohort['event_id']
        if event in rows:
            raise ValueError('Duplicate cohort event')
        rows[event] = dict(event_id=event, stock_id=cohort['stock_id'], name=cohort['name'],
            entry_date=cohort['entry_date'], cohort_exit_date=cohort['exit_date'],
            initial_quantity=0, buys=[], sells=[],
            components={k:Decimal(0) for k in COMPONENTS}, ledger_cash=Decimal(0))

    def row_for(event, sid):
        if event not in rows or rows[event]['stock_id'] != sid:
            raise ValueError('Cash or asset has no matching cohort')
        return rows[event]

    for trade in account['trades']:
        row = row_for(trade['event_id'], trade['stock_id'])
        if trade['side'] not in ('buy', 'sell'):
            raise ValueError('Unknown trade side')
        buy = trade['side'] == 'buy'
        gross, cost = number(trade['gross']), number(trade['total_cost'])
        close((-gross if buy else gross)-cost, trade['cash_change'], 'Trade cash does not reconcile')
        row['components']['entry_gross' if buy else 'exit_gross'] += -gross if buy else gross
        row['components']['execution_cost'] -= cost
        row['buys' if buy else 'sells'].append({k:trade[k] for k in
            ('date', 'qty', 'reference_price', 'reason', 'cash_change')})
        if buy:
            row['initial_quantity'] += trade['qty']

    # Use dated corporate-action evidence to assign payments after disposal;
    # never attribute them to whichever position happens to be held that day.
    for item in account['cash_ledger']:
        if item['kind'] == 'initial_deposit':
            continue
        if item['kind'] not in ('buy', 'sell', 'dividend_payment', 'fractional_share_payment'):
            raise ValueError('Unsupported cash movement in strategy profit')
        event = item.get('event_id')
        if not event:
            if item['kind'] == 'dividend_payment':
                events = {a['event_id'] for a in account['corporate_actions']
                    if a['kind'] == 'payment' and a.get('action_id') == item.get('action_id')
                    and a['date'] == item['date'] and a['stock_id'] == item['stock_id']}
            elif item['kind'] == 'fractional_share_payment':
                events = {a['event_id'] for a in account['corporate_actions']
                    if a['kind'] == 'share_delivery' and a.get('fraction', 0)
                    and a['date'] == item['date'] and a['stock_id'] == item['stock_id']}
            else:
                raise ValueError('Unsupported cash movement without an event')
            if len(events) != 1:
                raise ValueError('Corporate cash has missing or ambiguous event evidence')
            event = events.pop()
        row = row_for(event, item['stock_id'])
        row['ledger_cash'] += number(item['cash_change'])
        if item['kind'] not in ('buy', 'sell'):
            row['components']['distribution_cash'] += number(item['cash_change'])
    for row in rows.values():
        close(sum(row['components'].values()), row['ledger_cash'], 'Event trade/corporate cash does not reconcile')
        if row['initial_quantity'] <= 0:
            raise ValueError('Cohort has no initial purchased shares')

    final = account['daily'][-1]
    for holding in account['holdings']:
        if holding['date'] == final['date']:
            row_for(holding['event_id'], holding['stock_id'])['components']['terminal_holdings'] += number(holding['market_value'])
    rights = defaultdict(list)
    for right in account['receivables']:
        row_for(right['event_id'], right['stock_id'])
        rights[right['stock_id']].append(right)
    for sid, items in rights.items():
        # Sealed per-stock valuations cover cash plus shares, including stock
        # rights with zero physical inventory. Mixed owners need per-right marks.
        events = {r['event_id'] for r in items}
        if len(events) == 1:
            value = number(case['stock_pnl'][sid]['receivable_value'])
            rows[next(iter(events))]['components']['terminal_receivables'] += value
        elif all(r['kind'] == 'cash' for r in items):
            for right in items:
                rows[right['event_id']]['components']['terminal_receivables'] += number(right['amount'])
        else:
            raise ValueError('Mixed event stock-right valuations require additional evidence')
    for sid, stock in case['stock_pnl'].items():
        selected = [r for r in rows.values() if r['stock_id'] == sid]
        for actual, target in ((sum(r['ledger_cash'] for r in selected), stock['cash_flow']),
                (sum(r['components']['terminal_holdings'] for r in selected), stock['holding_value']),
                (sum(r['components']['terminal_receivables'] for r in selected), stock['receivable_value'])):
            close(actual, target, 'Event allocation differs from sealed stock totals')
    profit = sum(sum(r['components'].values()) for r in rows.values())
    close(profit, number(final['nav'])-number(account['settings']['initial_cash']), 'Event profit differs from account NAV')
    return {event:dict(row, ledger_cash=float(row['ledger_cash']),
        components={k:float(v) for k,v in row['components'].items()},
        profit=float(sum(row['components'].values()))) for event,row in sorted(rows.items())}


def compare(base, delayed):
    a, b = base['account'], delayed['account']
    if a['settings']['initial_cash'] != b['settings']['initial_cash'] or [r['date'] for r in a['daily']] != [r['date'] for r in b['daily']]:
        raise ValueError('Both account capital and dates must match')
    left, right = event_book(base), event_book(delayed)
    events = []
    totals = dict(removed_events=Decimal(0), added_events=Decimal(0), common_quantity=Decimal(0),
                  **{'common_'+k:Decimal(0) for k in COMPONENTS})
    for event in sorted(set(left)|set(right)):
        first, second = left.get(event), right.get(event)
        pnl_a = number(first['profit']) if first else Decimal(0)
        pnl_b = number(second['profit']) if second else Decimal(0)
        row = dict(event_id=event, stock_id=(first or second)['stock_id'], name=(first or second)['name'],
            base=first, delayed=second, profit_change=float(pnl_b-pnl_a))
        if first and second:
            if first['stock_id'] != second['stock_id']:
                raise ValueError('Common event points to different stocks')
            qa, qb = number(first['initial_quantity']), number(second['initial_quantity'])
            quantity = (qb-qa)*(pnl_a/qa+pnl_b/qb)/2
            pieces = {k:(number(second['components'][k])/qb-number(first['components'][k])/qa)*(qa+qb)/2 for k in COMPONENTS}
            close(quantity+sum(pieces.values()), pnl_b-pnl_a, 'Common-event decomposition does not reconcile')
            totals['common_quantity'] += quantity
            for k,v in pieces.items():
                totals['common_'+k] += v
            row.update(category='common', quantity_effect=float(quantity), per_initial_share_effects={k:float(v) for k,v in pieces.items()})
        else:
            category = 'removed_events' if first else 'added_events'
            totals[category] += pnl_b-pnl_a
            other = b if first else a
            row.update(category=category, other_account_orders=[{k:r.get(k) for k in
                ('date', 'side', 'reason', 'failure', 'requested_qty', 'filled_qty')}
                for r in other['orders'] if r.get('event_id') == event])
        events.append(row)
    gap = number(b['daily'][-1]['nav'])-number(a['daily'][-1]['nav'])
    close(sum(totals.values()), gap, 'Decomposition differs from final account gap')
    annual = []
    for year in sorted({r['date'][:4] for r in a['daily']}):
        al = [r for r in a['daily'] if r['date'].startswith(year)]
        bl = [r for r in b['daily'] if r['date'].startswith(year)]
        opening = number(bl[0]['opening_nav'])-number(al[0]['opening_nav'])
        ending = number(bl[-1]['nav'])-number(al[-1]['nav'])
        annual.append(dict(year=year, opening_asset_gap=float(opening), ending_asset_gap=float(ending),
                           change_in_asset_gap=float(ending-opening)))
    close(sum(number(r['change_in_asset_gap']) for r in annual), gap, 'Annual gap does not reconcile')
    return dict(base_return=base['summary']['total_return'], delayed_return=delayed['summary']['total_return'],
        final_asset_gap=float(gap), attribution_twd={k:float(v) for k,v in totals.items()}, annual_asset_gap=annual,
        event_counts=dict(base=len(left), delayed=len(right), common=len(set(left)&set(right)),
            removed=len(set(left)-set(right)), added=len(set(right)-set(left))),
        events=sorted(events,key=lambda r:r['profit_change']),
        attribution_is_counterfactual=False, live_qualified=False)


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Retain previous diagnosis; choose a new output directory')
    publication = load()
    refs = {str(REPORT.relative_to(ROOT)):sha(REPORT),
        str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        'docs/prereg_entry_delay_diagnosis_20260927.md':sha(ROOT/'docs/prereg_entry_delay_diagnosis_20260927.md')}
    cases = {}
    for mask in range(8):
        descriptor = publication['cases']['release_'+str(mask)]['result']
        cases[mask] = json.loads(verified_bytes(descriptor, ROOT, '.json'))
        refs[descriptor['path']] = descriptor['sha256']
    pairs = {str(mask)+'_to_'+str(mask+2):compare(cases[mask],cases[mask+2]) for mask in (0,1,4,5)}
    if file_identities([ROOT/p for p in refs], ROOT) != refs:
        raise ValueError('Evidence changed during diagnosis')
    result = dict(schema='entry_delay_diagnosis_v1', source_sha256=refs, pairs=pairs,
        period=dict(start=publication['start'],end=publication['end']), initial_cash=1_000_000,
        new_strategy_trials=0, network_calls=0, database_writes=0, live_qualified=False,
        attribution_is_counterfactual=False,
        limitations=['Arithmetic on two completed paths does not establish a separately executable causal intervention',
            'Per-initial-share exit changes combine timing, stops, partial exits and corporate actions',
            'Selection and quantity changes include feedback from prior profits, available cash and occupied slots',
            'Historical daily fills and prior source limitations remain; no broker screenshots are used'])
    write(output/'analysis.json',result)
    (output/'analysis.sha256').write_text(sha(output/'analysis.json')+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k:dict(v,events='omitted') for k,v in result['pairs'].items()},ensure_ascii=False,indent=2))
