#!/usr/bin/env python3
"""Compare complete account drawdowns and overlapping winning-stock holdings."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse
import math
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_three_black_exit import verify_runs
from scripts.research_exit_scenarios import read, write, sha
from skills.account_cohort_attribution import cohort_outcomes
from skills.three_black_exit import ARMS


def outcomes_with_pending_shares(case, closing_prices):
    """Attribute non-tradable share rights without reclassifying the real ledger.

    The legacy attribution helper accepts cash rights only. Give it a temporary
    valuation view, independently reconcile the amount, then explicitly retain
    the unpaid shares and open-cohort status in the analysis result.
    """
    pending = [r for r in case['account']['receivables'] if r['kind'] != 'cash']
    if not pending:
        return cohort_outcomes(case)
    if case['summary']['final_receivables'] != case['account']['receivables']:
        raise ValueError('Final rights differ from the account journal')
    view, amounts = deepcopy(case), {}
    for row in view['account']['receivables']:
        if row['kind'] == 'cash':
            continue
        price = closing_prices.get(row['stock_id'])
        if (row['kind'] != 'shares' or row.get('tradable') is not False
                or row.get('delivery_status') != 'pending_unannounced'
                or row.get('pay_date') is not None or row.get('fractional_cash_per_share') != 0
                or type(row['qty']) is not int or row['qty'] < 0
                or price is None or not math.isfinite(price) or price <= 0):
            raise ValueError('Unsupported pending-share valuation')
        value = row['qty']*price
        amounts[row['event_id']] = amounts.get(row['event_id'], 0.)+value
        row.update(kind='cash', amount=value)
    view['summary']['final_receivables'] = deepcopy(view['account']['receivables'])
    result = cohort_outcomes(view)  # Reconciles all cash, holdings and rights to NAV.
    for eid, value in amounts.items():
        result[eid].update(closed=False, settled=False, pending_share_value=value,
                           pending_share_valuation_basis='end_date_ordinary_share_close_proxy')
    return result


def drawdown_window(daily):
    frame = pd.DataFrame(daily).set_index('date')
    peak = max(float(frame.iloc[0].opening_nav), float(frame.iloc[0].nav))
    worst, peak_day, worst_peak, trough = 0., frame.index[0], frame.index[0], frame.index[0]
    for day, row in frame.iterrows():
        if row.nav > peak:
            peak, peak_day = row.nav, day
        if row.nav/peak-1 < worst:
            worst, worst_peak, trough = row.nav/peak-1, peak_day, day
    peak_nav = max(float(frame.iloc[0].opening_nav), float(frame.at[worst_peak, 'nav']))
    recovered = frame.loc[trough:]
    recovered = recovered.loc[recovered.nav.ge(peak_nav)]
    recovery = recovered.index[0] if len(recovered) else None
    return dict(max_drawdown=worst, peak_date=worst_peak, trough_date=trough,
                peak_nav=peak_nav, trough_nav=float(frame.at[trough, 'nav']),
                recovery_date=recovery,
                peak_to_trough_sessions=int(frame.index.get_loc(trough)-frame.index.get_loc(worst_peak)),
                peak_to_recovery_sessions=int(frame.index.get_loc(recovery)-frame.index.get_loc(worst_peak))
                if recovery else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT)
    if args.output.exists():
        raise ValueError('Preserve prior analysis')
    cases = verify_runs(args.left, args.right, ARMS)
    quotes_path = ROOT/'.cache/partial-risk-2019-20260929/inputs-final/quotes-unmasked.parquet'
    quotes = pd.read_parquet(quotes_path, columns=['stock_id', 'date', 'close'])
    closing = quotes.loc[pd.to_datetime(quotes.date).eq(cases['control']['summary']['end'])]
    if closing.stock_id.duplicated().any():
        raise ValueError('Duplicate end-date valuation quote')
    prices = closing.set_index('stock_id').close.to_dict()
    outcomes = {arm: outcomes_with_pending_shares(case, prices)
                for arm, case in cases.items() if arm != 'benchmark'}
    diagnosis_path = ROOT/'artifacts/forward_simulation/drawdown_control_20261001_diagnosis.json'
    diagnosis = read(diagnosis_path)
    start, end = diagnosis['peak_date'], diagnosis['trough_date']
    reference = sorted(outcomes['control'].values(), key=lambda row: row['pnl'], reverse=True)[:5]
    comparisons, winners, problem_holdings = [], {}, {}
    for arm, case in cases.items():
        a, s = case['account'], case['summary']
        frame = pd.DataFrame(a['daily']).set_index('date')
        window = drawdown_window(a['daily'])
        if abs(window['max_drawdown']-s['max_drawdown']) > 1e-10:
            raise ValueError('Drawdown reconstruction differs')
        closed = [row for row in outcomes.get(arm, {}).values() if row['closed']]
        positions = {day: i for i, day in enumerate(frame.index)}
        sales = [row for row in a['trades'] if row['side']=='sell' and row.get('signal_date') in positions]
        lags = [positions[row['date']]-positions[row['signal_date']] for row in sales]
        comparisons.append(dict(arm=arm, **window,
            original_drawdown_window_return=float(frame.at[end, 'nav']/frame.at[start, 'nav']-1),
            average_stock_exposure=float((frame.market_value/frame.nav).mean()),
            average_cash_fraction=float((frame.cash/frame.nav).mean()),
            closed_cohorts=len(closed), winning_closed_cohorts=sum(row['pnl']>0 for row in closed),
            losing_closed_cohorts=sum(row['pnl']<0 for row in closed),
            cooling_blocked_days=sum(row['blocked'] for row in a.get('cooling_log', [])),
            cooling_blocked_candidates=sum(len(row['blocked_events']) for row in a.get('cooling_log', [])),
            median_executed_sell_delay_sessions=float(pd.Series(lags).median()) if lags else None,
            longest_executed_sell_delay_sessions=max(lags) if lags else None,
            delayed_sale_fills=sum(lag > 1 for lag in lags),
            sale_reasons=dict(Counter(row['reason'] for row in a['trades'] if row['side']=='sell'))))
        if arm == 'benchmark':
            continue
        winners[arm] = []
        for old in reference:
            # A different entry date can still catch the same stock's wave.
            # Report entire cohort P&L explicitly, not a fictitious common-window return.
            matches = [row for row in outcomes[arm].values() if row['stock_id']==old['stock_id']
                       and row['entry_date'] <= (old['exit_date'] or s['end'])
                       and (row['exit_date'] or s['end']) >= old['entry_date']]
            winners[arm].append(dict(reference_event=old, overlapping_cohorts=matches,
                                      full_cohort_profit=sum(row['pnl'] for row in matches)))
        problem_holdings[arm] = []
        for row in outcomes[arm].values():
            if row['stock_id'] not in ('2630', '6625') or not row['entry_date'] <= start < (row['exit_date'] or s['end']):
                continue
            sales = [{k:t[k] for k in ('date', 'qty', 'reference_price', 'reason')}
                     for t in a['trades'] if t['event_id']==row['event_id'] and t['side']=='sell']
            problem_holdings[arm].append(dict(**row, sales=sales))
    sources = [Path(__file__), ROOT/'scripts/export_three_black_exit.py',
               ROOT/'skills/account_cohort_attribution.py', diagnosis_path, quotes_path]
    sources += [p/'report.json' for p in (args.left, args.right)]
    sources += [p/(arm+'.json') for p in (args.left, args.right) for arm in ARMS]
    result = dict(comparison=comparisons, original_drawdown_window=dict(start=start, end=end),
                  top_winning_waves=winners, original_peak_holdings=problem_holdings,
                  cohort_outcomes=outcomes,
                  wave_comparison_basis='overlapping holding intervals; full cohort cash/rights P&L',
                  source_sha256={str(p.resolve().relative_to(ROOT)): sha(p) for p in sources},
                  live_qualified=False, unseen_validation=False, actual_fill_verified=False)
    write(args.output, result)
    print(pd.DataFrame(comparisons).drop(columns=['sale_reasons']).to_string(index=False))


if __name__ == '__main__':
    main()
