#!/usr/bin/env python3
"""Describe frozen candle/volume accounts; future prices are outcomes only."""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal as D, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING
import argparse
import hashlib
import json
import math
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.account_cohort_attribution import cohort_outcomes


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(actual, expected, label, tolerance=.02):
    if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0):
        raise ValueError(label)


def arithmetic(case):
    """Rebuild cash, cost and marked NAV without the replay's audit routines."""
    a, s = case['account'], case['summary']
    cash = D(0)
    daily_cash = {}
    for row in a['cash_ledger']:
        cash = (cash + D(str(row['cash_change']))).quantize(D('.01'), rounding=ROUND_HALF_UP)
        check(cash, row['cash_after'], 'Cash ledger')
        if cash < 0:
            raise ValueError('Negative cash')
        daily_cash[row['date']] = cash
    costs = D(0)
    for t in a['trades']:
        if type(t['qty']) is not int or t['qty'] <= 0:
            raise ValueError('Invalid filled shares')
        raw = D(str(t['reference_price'])) * t['qty']
        gross = raw.quantize(D('.01'), rounding=ROUND_HALF_UP)
        fee = max(D(20), (raw * D('.001425')).quantize(D(1), rounding=ROUND_HALF_UP))
        slip = (raw * D('.0045')).quantize(D(1), rounding=ROUND_CEILING)
        tax = ((raw * D('.001' if t['stock_id'] == '0050' else '.003')).quantize(D(1), rounding=ROUND_FLOOR)
               if t['side'] == 'sell' else D(0))
        for field, expected in dict(gross=gross, commission=fee, slippage=slip, tax=tax,
                                    total_cost=fee+slip+tax,
                                    cash_change=(gross if t['side'] == 'sell' else -gross)-fee-slip-tax).items():
            check(t[field], expected, 'Independent trade ' + field)
        costs += fee+slip+tax
    marks = defaultdict(float)
    for row in a['holdings']:
        check(row['price'] * row['qty'], row['market_value'], 'Holding mark')
        marks[row['date']] += row['market_value']
    previous = peak = 1_000_000.
    cash = D(1_000_000)
    worst = 0.
    for row in a['daily']:
        cash = daily_cash.get(row['date'], cash)
        check(cash, row['cash'], 'Daily cash')
        check(marks[row['date']], row['market_value'], 'Daily market value')
        check(float(cash)+marks[row['date']]+row['receivable'], row['nav'], 'Daily NAV')
        check(previous, row['opening_nav'], 'Opening NAV')
        peak = max(peak, row['nav'])
        worst = min(worst, row['nav']/peak-1)
        check(row['drawdown'], row['nav']/peak-1, 'Daily drawdown', 1e-10)
        previous = row['nav']
    check(s['final_nav'], previous, 'Final NAV')
    check(s['total_return'], previous/1_000_000-1, 'Total return', 1e-10)
    check(s['max_drawdown'], worst, 'Worst drawdown', 1e-10)
    check(s['costs']['total_cost'], costs, 'Total cost')
    return dict(daily_cash_and_equity=len(a['daily']), filled_trade_costs=len(a['trades']), passed=True)


def analyze(reports, output, *, replace_incomplete=False):
    output = output.resolve()
    output.relative_to(ROOT / '.cache/red-volume-exit-20261003')
    if output.exists():
        raise ValueError('Choose a fresh analysis directory')
    refs = {}

    def bind(path):
        path = path.resolve()
        name = str(path.relative_to(ROOT))
        digest = sha(path)
        if name in refs and refs[name] != digest:
            raise ValueError('Changed source ' + name)
        refs[name] = digest
        return read(path)

    cases = {}
    attempts = defaultdict(list)
    parent_sources = []
    for path in reports:
        report = bind(path)
        parent_sources.append(report['source_sha256'])
        for name, item in report['cases'].items():
            p = ROOT / item['path']
            if sha(p) != item['sha256']:
                raise ValueError('Case hash mismatch')
            value = bind(p)
            if name in cases:
                if not replace_incomplete or cases[name]['completed']:
                    raise ValueError('Cannot replace duplicate/completed arm')
                if cases[name].get('family_rules') != value.get('family_rules'):
                    raise ValueError('Completion changed the registered strategy rules')
            attempts[name].append(dict(path=str(p.relative_to(ROOT)), sha256=item['sha256'],
                                       completed=value['completed'], reason=value.get('reason')))
            cases[name] = value
    inputs = ROOT / '.cache/market-input-repair-20261002/inputs-v2'
    for name in ('manifest.json', 'close-official.parquet', 'quotes-unmasked.parquet'):
        path = inputs / name
        if any(sources.get(str(path.relative_to(ROOT))) != sha(path) for sources in parent_sources):
            raise ValueError('Analysis inputs differ from the bound parent account: ' + name)
    manifest = bind(inputs / 'manifest.json')
    for name in ('close-official.parquet', 'quotes-unmasked.parquet'):
        path = inputs / name
        if sha(path) != manifest['files_sha256'][name]:
            raise ValueError('Input source changed')
        refs[str(path.relative_to(ROOT))] = sha(path)
    prices = pd.read_parquet(inputs / 'close-official.parquet').set_index('date')
    prices.index = pd.to_datetime(prices.index)
    raw = pd.read_parquet(inputs / 'quotes-unmasked.parquet', columns=['stock_id', 'date', 'open', 'close'])
    raw['date'] = pd.to_datetime(raw['date'])
    if raw.duplicated(['stock_id', 'date']).any():
        raise ValueError('Duplicate raw candles')
    candles = raw.set_index(['stock_id', 'date'])
    calendar = prices.index
    summaries, cohorts, future = {}, {}, {}
    for name, case in cases.items():
        if not case['completed']:
            summaries[name] = dict(completed=False, reason=case['reason'], full_period_return=None)
            continue
        a, s = case['account'], case['summary']
        result = dict(completed=True, **{k: s[k] for k in
            ('total_return', 'final_nav', 'max_drawdown', 'annual', 'costs', 'buy_count', 'sell_count', 'stock_cohorts')})
        result['independent_arithmetic'] = arithmetic(case)
        if name == 'benchmark':
            summaries[name] = result
            continue
        rows = cohort_outcomes(case)
        for eid, row in rows.items():
            bar = candles.loc[(row['stock_id'], pd.Timestamp(row['signal_date']))]
            if not all(math.isfinite(value) and value > 0 for value in (bar['open'], bar['close'])):
                raise ValueError('Unknown funded signal candle')
            row['signal_candle'] = 'red' if bar['close'] > bar['open'] else 'black' if bar['close'] < bar['open'] else 'doji'
            sells = [t for t in a['trades'] if t['event_id'] == eid and t['side'] == 'sell']
            row['exit_reasons'] = sorted({t['reason'] for t in sells})
            row['first_sell_date'] = min((t['date'] for t in sells), default=None)
        cohorts[name] = rows
        closed = [r for r in rows.values() if r['closed']]
        settled = [r for r in rows.values() if r['settled']]
        result.update(closed_events=len(closed), closed_wins=sum(r['pnl'] > 0 for r in closed),
                      closed_losses=sum(r['pnl'] < 0 for r in closed),
                      closed_breakeven=sum(r['pnl'] == 0 for r in closed),
                      closed_unsettled=sum(not r['settled'] for r in closed),
                      settled_events=len(settled), settled_wins=sum(r['pnl'] > 0 for r in settled),
                      closed_win_rate_definition='No remaining shares; P&L includes confirmed cash receivables even if not paid.',
                      closed_win_rate=sum(r['pnl'] > 0 for r in closed)/len(closed) if closed else None,
                      funded_candles=dict(Counter(r['signal_candle'] for r in rows.values())),
                      exit_reason_events=dict(Counter(reason for r in rows.values() for reason in r['exit_reasons'])),
                      volume_decision_statuses=dict(Counter(r['status'] for r in case.get('volume_exit_log', []))),
                      entry_gate_statuses=dict(Counter(r['status'] for r in case.get('entry_gate_decisions', []))))
        if case['family_rules']['red_gate'] and any(r['signal_candle'] != 'red' for r in rows.values()):
            raise ValueError('Red-gated account bought a non-red signal')
        diagnostics = []
        for log in case.get('volume_exit_log', []):
            if not log['trigger']:
                continue
            eid, sid = log['event_id'], log['stock_id']
            row = rows[eid]
            sells = [t for t in a['trades'] if t['event_id'] == eid and t['reason'].startswith('volume_dry')]
            first = min((t['date'] for t in sells), default=None)
            record = dict(event_id=eid, stock_id=sid, name=row['name'], decision_date=log['decision_date'],
                          first_sell_date=first, complete_exit_date=row['exit_date'], cohort_pnl=row['pnl'],
                          baseline_volume=log['baseline_volume'], observations=log['observations'])
            left, right = log['observations']
            signal_price = prices.at[pd.Timestamp(log['baseline_date']), sid]
            known_prices = all(pd.notna(v) and math.isfinite(v) and v > 0 for v in
                               (left['adjusted_close'], right['adjusted_close'], signal_price))
            record['price_confirmation_at_trigger'] = dict(available=known_prices,
                up_from_previous=bool(right['adjusted_close'] > left['adjusted_close']) if known_prices else None,
                below_both_previous_and_signal=bool(right['adjusted_close'] < left['adjusted_close']
                    and right['adjusted_close'] < signal_price) if known_prices else None)
            if first:
                base_index = calendar.get_loc(pd.Timestamp(first))
                base = prices.at[pd.Timestamp(first), sid]
                for horizon in (20, 60):
                    selected = calendar[base_index+1:base_index+horizon+1]
                    selected = selected[selected <= pd.Timestamp('2026-09-09')]
                    values = prices.loc[selected, sid]
                    valid = len(values) > 0 and pd.notna(base) and base > 0 and values.notna().all() and (values > 0).all()
                    record[str(horizon)+'_sessions'] = dict(complete_window=len(values) == horizon,
                        observed_sessions=len(values), valid=bool(valid),
                        maximum_close_return=float(values.max()/base-1) if valid else None,
                        minimum_close_return=float(values.min()/base-1) if valid else None,
                        ending_close_return=float(values.iloc[-1]/base-1) if valid else None)
                record['later_same_stock_entries'] = [r['entry_date'] for other, r in rows.items()
                    if other != eid and r['stock_id'] == sid and r['entry_date'] > first]
            diagnostics.append(record)
        future[name] = diagnostics
        result['new_volume_trigger_events'] = len(diagnostics)
        result['volume_triggers_with_price_up'] = sum(
            r['price_confirmation_at_trigger']['up_from_previous'] is True for r in diagnostics)
        result['volume_triggers_below_both_prices'] = sum(
            r['price_confirmation_at_trigger']['below_both_previous_and_signal'] is True for r in diagnostics)
        summaries[name] = result
    base = cohorts.get('poc_base', {})
    baseline_nonred = sorted([r for r in base.values() if r['signal_candle'] != 'red'], key=lambda r: r['pnl'])
    output.mkdir(parents=True)
    snapshot = output / 'analyzer_source.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    refs[str(snapshot.relative_to(ROOT))] = sha(snapshot)
    refs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    refs['tests/test_candle_volume_analysis.py'] = sha(ROOT/'tests/test_candle_volume_analysis.py')
    refs['skills/account_cohort_attribution.py'] = sha(ROOT/'skills/account_cohort_attribution.py')
    for name, value in [('cohorts.json', cohorts), ('post-exit-prices.json', future)]:
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')
    result = dict(summaries=summaries, baseline_nonred_funded_events=baseline_nonred,
        attempt_history=dict(attempts),
        source_sha256=refs, live_qualified=False, network_requests=0,
        limitations=['Repeatedly researched history, not unseen validation.',
            'Future prices are descriptive adjusted closing prices relative to first sale day close, not executable missed profits or counterfactual NAV.',
            'Shortened end-of-study price windows are flagged, not treated as complete horizons.',
            'Three-slot path changes also change future capital and selection; cohort differences are not isolated causal effects.',
            'Uses parent HL2 and legacy total-session capacity assumptions; actual fills remain unverified.'],
        output_sha256={p.name: sha(p) for p in output.iterdir()})
    for path, digest in refs.items():
        if sha(ROOT/path) != digest:
            raise ValueError('Analysis source changed')
    (output/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replace-incomplete', action='store_true',
                        help='Retain failed attempts and use later same-rule completion; never replace a completed arm')
    args = parser.parse_args()
    analyze(args.report, args.output, replace_incomplete=args.replace_incomplete)
