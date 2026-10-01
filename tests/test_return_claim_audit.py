from copy import deepcopy
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from skills.return_claim_audit import audit_account, trade_costs


def fixture_account():
    eid, sid = 'test-entry', '2330'
    days = ['2024-01-02', '2024-01-03']
    trades = []
    for i, (side, price, gross, fee, slip, tax, change, cash, remaining) in enumerate([
        ('buy', 100, 100000, 143, 450, 0, -100593, 99407, 1000),
        ('sell', 110, 110000, 157, 495, 330, 109018, 208425, 0),
    ]):
        trades.append(dict(date=days[i], sequence=i+1, event_id=eid, stock_id=sid,
            channel='board', side=side, qty=1000, source_high=price, source_low=price,
            source_volume=1000000, reference_price=price, participation_limit=.01,
            capacity_qty=10000, signal_date='2024-01-01' if i == 0 else days[0],
            gross=gross, commission=fee, slippage=slip, tax=tax, total_cost=fee+slip+tax,
            cash_change=change, cash_after=cash, remaining_shares=remaining))
    ledger = [dict(date=days[0], kind='initial_deposit', cash_change=200000, cash_after=200000)]
    ledger += [dict(date=t['date'], kind=t['side'], cash_change=t['cash_change'],
                    cash_after=t['cash_after'], stock_id=sid, event_id=eid, channel='board') for t in trades]
    daily = []
    previous = 200000
    for i, (cash, mv, cost) in enumerate([(99407, 101000, 593), (208425, 0, 982)]):
        nav = cash+mv
        daily.append(dict(date=days[i], cash=cash, market_value=mv, receivable=0, nav=nav,
            opening_nav=previous, cost=cost, daily_return=nav/previous-1,
            total_return=nav/200000-1, drawdown=0, holdings=int(mv > 0), stale_holdings=0))
        previous = nav
    holding = dict(date=days[0], event_id=eid, stock_id=sid, qty=1000, price=101,
                   mark_date=days[0], stale=False, market_value=101000)
    account = dict(settings=dict(price_formula='(high+low)/2', benchmark=False, initial_cash=200000),
        daily=daily, trades=trades, corporate_actions=[], holdings=[holding], cash_ledger=ledger,
        cohorts=[dict(event_id=eid, stock_id=sid, name='測試', entry_date=days[0])], receivables=[])
    summary = dict(final_nav=208425, total_return=.042125, max_drawdown=0,
        costs=dict(commission=300, slippage=945, tax=330, total_cost=1575),
        annual=[dict(year='2024', start_nav=200000, end_nav=208425, total_return=.042125)])
    marks = {(days[0], sid):(101, days[0]), (days[1], sid):(110, days[1])}
    execution = {(days[0], sid, 'board'):(100, 100, 1000000, 1000000),
                 (days[1], sid, 'board'):(110, 110, 1000000, 1000000)}
    return dict(account=account, summary=summary), marks, execution


def check(case, marks, execution, policies=None):
    return audit_account(case, marks, execution, fractional_rounding=policies or {})


def test_independent_cash_and_compounding():
    result = check(*fixture_account())
    assert result['final_nav'] == 208425
    assert result['total_return'] == .042125
    assert result['costs']['total_cost'] == 1575
    assert result['top_five_net_profit'][0]['net_pnl'] == 8425
    assert result['actual_fill_verified'] is False


@pytest.mark.parametrize('fault', ['extra_deposit', 'duplicate_fill', 'ignored_fee', 'wrong_mark',
                                  'future_mark', 'future_signal', 'missing_shares', 'oversold',
                                  'inflated_nav', 'invented_receivable', 'low_capacity'])
def test_rejects_monetary_and_timing_faults(fault):
    case, marks, quotes = fixture_account()
    a = case['account']
    if fault == 'extra_deposit':
        a['cash_ledger'].append(dict(date='2024-01-03', kind='initial_deposit', cash_change=100000, cash_after=308425))
    elif fault == 'duplicate_fill':
        a['trades'].append(deepcopy(a['trades'][-1]))
    elif fault == 'ignored_fee':
        a['trades'][0]['commission'] = 0
    elif fault == 'wrong_mark':
        marks[('2024-01-02','2330')] = (202, '2024-01-02')
    elif fault == 'future_mark':
        marks[('2024-01-02','2330')] = (101, '2024-01-03')
    elif fault == 'future_signal':
        a['trades'][0]['signal_date'] = '2024-01-02'
    elif fault == 'missing_shares':
        a['holdings'] = []
    elif fault == 'oversold':
        a['trades'][-1]['remaining_shares'] = -1
    elif fault == 'inflated_nav':
        a['daily'][-1]['nav'] += 100000
    elif fault == 'invented_receivable':
        a['daily'][0]['receivable'] = 1000
    else:
        quotes[('2024-01-02','2330','board')] = (100,100,1000,1000000)
    with pytest.raises(ValueError):
        check(case, marks, quotes)


def test_split_does_not_create_profit():
    case, marks, quotes = fixture_account()
    a = case['account']
    a['corporate_actions'] = [dict(date='2024-01-03', stock_id='2330', event_id='test-entry',
        action_id='split-1', kind='split', entitled_qty=1000, multiplier=2, qty_after=2000)]
    a['trades'][1].update(qty=2000, source_high=55, source_low=55, reference_price=55)
    quotes[('2024-01-03','2330','board')] = (55,55,1000000,1000000)
    assert check(case, marks, quotes)['final_nav'] == 208425
    a['corporate_actions'][0]['multiplier'] = 20
    with pytest.raises(ValueError, match='Split shares differ'):
        check(case, marks, quotes)


def test_stock_delivery_retains_unliquidated_value_once():
    case, marks, quotes = fixture_account()
    a = case['account']
    common = dict(date='2024-01-03', stock_id='2330', event_id='test-entry',
                  action_id='div-stock', pay_date='2024-01-03')
    a['corporate_actions'] = [
        dict(common, kind='stock_dividend', entitled_qty=1000, shares_per_share=.1,
             whole_new_shares=100, fractional_right=0, fractional_cash_per_share=10),
        dict(common, kind='share_delivery', qty=100, fraction=0),
    ]
    a['cash_ledger'].insert(2, dict(date='2024-01-03', kind='fractional_share_payment',
        stock_id='2330', cash_change=0, cash_after=99407))
    a['trades'][1]['remaining_shares'] = 100
    a['holdings'].append(dict(date='2024-01-03', event_id='test-entry', stock_id='2330',
        qty=100, price=110, market_value=11000, mark_date='2024-01-03', stale=False))
    a['daily'][1].update(market_value=11000, nav=219425, holdings=1,
                         daily_return=219425/200407-1, total_return=.097125)
    case['summary'].update(final_nav=219425, total_return=.097125)
    case['summary']['annual'][0].update(end_nav=219425, total_return=.097125)
    assert check(case, marks, quotes, {'div-stock':'half_up_cents'})['final_nav'] == 219425
    a['corporate_actions'].append(deepcopy(a['corporate_actions'][-1]))
    with pytest.raises(ValueError, match='Payment without'):
        check(case, marks, quotes, {'div-stock':'half_up_cents'})


def test_unknown_corporate_event_does_not_silently_pass():
    case, marks, quotes = fixture_account()
    case['account']['corporate_actions'] = [dict(date='2024-01-03', stock_id='2330',
        event_id='test-entry', action_id='unknown-1', kind='capital_reduction')]
    with pytest.raises(ValueError, match='Unsupported corporate'):
        check(case, marks, quotes)


def test_cash_dividend_is_counted_once_at_payment():
    case, marks, quotes = fixture_account()
    a = case['account']
    common = dict(date='2024-01-03', stock_id='2330', event_id='test-entry',
                  action_id='cash-1', pay_date='2024-01-03')
    a['corporate_actions'] = [
        dict(common, kind='cash_dividend', entitled_qty=1000, cash_per_share=.1, entitlement_value=100),
        dict(common, kind='payment', amount=100),
    ]
    a['cash_ledger'].insert(2, dict(date='2024-01-03', kind='dividend_payment', stock_id='2330',
        action_id='cash-1', cash_change=100, cash_after=99507))
    a['cash_ledger'][-1]['cash_after'] = 208525
    a['trades'][1]['cash_after'] = 208525
    a['daily'][1].update(cash=208525, nav=208525, daily_return=208525/200407-1, total_return=.042625)
    case['summary'].update(final_nav=208525, total_return=.042625)
    case['summary']['annual'][0].update(end_nav=208525, total_return=.042625)
    assert check(case, marks, quotes)['final_nav'] == 208525
    a['corporate_actions'][0]['pay_date'] = '2024-01-04'
    with pytest.raises(ValueError, match='due date'):
        check(case, marks, quotes)


def test_exact_hl2_exposes_one_dollar_float_boundary_errors():
    exact = trade_costs(57.3, 55.4, 20000, 'sell')
    old = trade_costs(57.3, 55.4, 20000, 'sell', legacy_float=True)
    assert exact['tax'] == Decimal(3381)
    assert old['tax'] == Decimal(3380)
    exact = trade_costs(53.1, 50.7, 20000, 'sell')
    old = trade_costs(53.1, 50.7, 20000, 'sell', legacy_float=True)
    assert exact['slippage'] == Decimal(4671)
    assert old['slippage'] == Decimal(4672)


def test_entry_audit_ignores_prices_after_signal():
    from scripts.audit_three_black_return import prefix_signals
    days = pd.bdate_range('2023-01-02', periods=150)
    price = pd.DataFrame({'2330':100*1.001**np.arange(150),
                          '0050':100*1.0001**np.arange(150)}, index=days)
    volume = pd.DataFrame(1_000_000., index=days, columns=price.columns)
    volume.loc[days[140], '2330'] = 2_000_000
    frames = {name:price.copy() for name in ('raw-close','close-official','close-quality')}
    frames.update({'raw-volume':volume, 'eligibility':price.gt(0)})
    relative = (price.iloc[140]/price.iloc[120]-1)
    c = dict(event_id='entry', stock_id='2330', signal_date=str(days[140].date()),
             entry_date=str(days[141].date()), group_cutoff_date=str(days[130].date()),
             priority=relative['2330']-relative['0050'])
    account = dict(cohorts=[c], black_log=[], trades=[])
    companies = pd.DataFrame([dict(stock_id='2330',listed_date='2000-01-01')])
    quotes = pd.DataFrame(columns=['date', 'stock_id'])
    before = prefix_signals(account, frames, companies, quotes)
    for name in ('raw-close','close-official','close-quality','raw-volume'):
        frames[name].loc[days[141]:] = 0
    assert prefix_signals(account, frames, companies, quotes) == before
    frames['raw-volume'].loc[days[140], '2330'] = 1
    with pytest.raises(ValueError, match='Technical buy predicate'):
        prefix_signals(account, frames, companies, quotes)
