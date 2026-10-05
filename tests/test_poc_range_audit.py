"""Daily intraday-odd scenarios must not masquerade as verified timed fills."""
from copy import deepcopy

import pandas as pd
import pytest

from skills import poc_range_audit as m
from skills.poc_range_execution import match_range_odd, match_range_board
from test_poc_executable_replay import audit_case  # noqa: F401
from test_poc_gap_audit import gap_case  # noqa: F401


def window(plan):
    plan.update(order_time=m.OPEN, expires_at=m.END, odd_order_time=m.ODD_OPEN, odd_expires_at=m.ODD_END)


def settings(account, count=0):
    account['settings'].update(execution=m.MODEL, odd_participation=.01, board_participation=.01,
        buy_fraction=.5, sell_fraction=.5, board_tick_verified=False, live_qualified=False,
        intraday_tick_verified=False, odd_tick_verified=False, actual_fill_verified=False,
        within_window_execution_verified=False, price_level_volume_verified=False,
        data_gap_policy=m.GAP_POLICY, posthoc_data_exclusion=True, excluded_event_count=count)


@pytest.fixture
def range_case(audit_case):
    account, ticks, odds, routes, quotes, days, corp, feeds = audit_case
    plan = account['tick_plans'][0]; window(plan); settings(account)
    account['data_gap_exclusions'] = []
    source = dict(after_hours=False, odd_high=102., odd_low=98., odd_shares=10_000,
        source_date=plan['date'], market='TWSE',
        volume_scope='intraday_odd_session', volume_unit='shares', price_unit='TWD_per_share',
        evidence_status='official_intraday_daily_table', execution_evidence='intraday_odd_daily_hl2_proxy',
        intraday_tick_verified=False, actual_fill_verified=False)
    odds.get_odd = lambda *args: deepcopy(source)
    ticks.audit_day=lambda *args: {}
    data, digest=ticks.get('', '', '')
    quotes['open']=100.; quotes['high']=101.; quotes['low']=100.
    base=account['orders'][0]
    base.update(order_time=m.OPEN,expires_at=m.END,tape_evidence={},
        **match_range_board(data,'buy',plan['limit_price'],plan['board_qty'],1_000_000))
    account['trades'][0]=dict(base,qty=base['filled_qty'],**m.costs(base['reference_price'],base['filled_qty'],'buy','2330'))
    old = account['orders'][1]
    row = {k: old[k] for k in ('date', 'stock_id', 'event_id', 'side', 'signal_date', 'channel',
        'limit_price', 'requested_qty', 'prior_avg_volume20', 'prior_avg_amount20')}
    row.update(order_time=m.ODD_OPEN, expires_at=m.ODD_END,
               **match_range_odd(source, 'buy', plan['odd_limit'], plan['odd_qty']))
    account['orders'][1] = row
    account['trades'][1] = dict(row, qty=row['filled_qty'], **m.costs(row['reference_price'], row['filled_qty'], 'buy', '2330'))
    account['daily'][0]['cash'] = m.money(account['settings']['initial_cash']+sum(t['cash_change'] for t in account['trades']))
    return audit_case


def test_daily_proxy_is_reconstructed_without_asserting_intraday_time_or_tick_evidence(range_case):
    result = m.audit_range_execution(*range_case)
    assert result['intraday_daily_proxies_rebuilt'] == 1
    assert result['chronological_board_allocations_rebuilt'] == 0
    assert result['ordinary_daily_proxies_rebuilt'] == 1
    assert result['original_planned_children'] == 2
    assert result['fills_reconciled'] == 2
    assert result['intraday_odd_tick_verified'] is False
    assert result['intraday_sequence_verified'] is False
    assert result['proxy_not_tick'] is True
    assert result['actual_fill_verified'] is False and result['live_qualified'] is False
    assert result['all_original_planned_children_source_data_present'] is True
    assert result['all_original_planned_children_execution_evidence_complete'] is False
    assert result['all_purchases_funded_before_sales'] is True


@pytest.mark.parametrize('change', ['clock', 'tick_claim', 'window_claim', 'fill_claim', 'price',
    'capacity', 'volume', 'five_percent', 'source_afterhours', 'source_scope', 'source_price',
    'duplicate_order', 'same_day', 'reused_budget', 'missing_child', 'missing_trade', 'settings',
    'source_date', 'source_market', 'source_legal_range', 'level_volume_claim', 'proxy_value',
    'trade_identity', 'trade_clock', 'trade_claim', 'settings_claim'])
def test_proxy_audit_rejects_fabricated_precision_or_changed_cash_and_denominators(range_case, change):
    account, _, odds, *_ = range_case; row = account['orders'][1]
    if change == 'clock': row['last_fill_time'] = '10:00:00'
    elif change == 'tick_claim': row['odd_tick_verified'] = True
    elif change == 'window_claim': row['within_window_execution_verified'] = True
    elif change == 'fill_claim': row['actual_fill_verified'] = True
    elif change == 'price': row['reference_price'] += .5
    elif change == 'capacity': row['capacity_qty'] += 1
    elif change == 'volume': row['source_volume'] += 100
    elif change == 'five_percent': row['participation_limit'] = .05
    elif change == 'trade_identity': account['trades'][1]['stock_id'] = '2317'
    elif change == 'trade_clock': account['trades'][1]['actual_fill_time'] = '10:30:00'
    elif change == 'trade_claim': account['trades'][1]['odd_tick_verified'] = True
    elif change == 'settings_claim': account['settings']['within_window_execution_verified'] = True
    elif change == 'level_volume_claim': row['price_level_volume_verified'] = True
    elif change == 'proxy_value': row['proxy_price'] += 1.
    elif change in ('source_afterhours', 'source_scope', 'source_price', 'source_date', 'source_market', 'source_legal_range'):
        original = odds.get_odd('', '', '')
        if change == 'source_afterhours': original['after_hours'] = True
        elif change == 'source_scope': original['volume_scope'] = 'ordinary_market'
        elif change == 'source_date': original['source_date'] = '2020-01-01'
        elif change == 'source_market': original['market'] = 'TPEX'
        elif change == 'source_legal_range': original.update(odd_high=115., odd_low=85.)
        else: original['odd_high'] = 90.
        odds.get_odd = lambda *args: original
    elif change == 'duplicate_order': account['orders'].append(deepcopy(row))
    elif change == 'same_day': account['tick_plans'][0]['signal_date'] = row['date']
    elif change == 'reused_budget': account['settings']['initial_cash'] = 1000.
    elif change == 'missing_child': account['orders'].pop()
    elif change == 'missing_trade': account['trades'].pop()
    elif change == 'settings': account['settings']['odd_participation'] = .05
    with pytest.raises(ValueError):
        m.audit_range_execution(*range_case)


def test_unknown_odd_clock_cannot_spend_a_board_sale_before_the_sale_is_known():
    account = dict(settings=dict(initial_cash=100.), daily=[dict(date='2024-01-02', cash=0.)], cash_ledger=[],
        trades=[dict(date='2024-01-02', channel='board', side='sell', cash_change=100.),
                dict(date='2024-01-02', channel='odd', side='buy', cash_change=-200.)])
    with pytest.raises(ValueError, match='same-day sale proceeds'):
        m.audit_proxy_cash(account)


@pytest.fixture
def range_gap_case(gap_case):
    account = gap_case[0]; settings(account, 1)
    for column in ('open','high','low'):gap_case[4][column]=100.
    window(account['tick_plans'][0]); window(account['data_gap_exclusions'][0]['original_plan'])
    return gap_case


def test_original_gap_denominator_and_locked_resources_survive_new_odd_window(range_gap_case):
    result = m.audit_range_execution(*range_gap_case)
    assert result['data_gap_children'] == 2 and result['original_planned_children'] == 2
    assert result['nonexcluded_planned_children_reconciled'] == 0
    assert result['gap_source_failures_independently_rebuilt'] is True
    assert result['all_original_planned_children_execution_evidence_complete'] is False


def test_missing_intraday_source_cannot_keep_a_successful_board_leg(range_gap_case):
    account, ticks, odds, *_ = range_gap_case
    tape = pd.DataFrame(dict(time=pd.to_timedelta(['09:02:00']), price=[100.], shares=[100_000]))
    ticks.get = lambda *args: (tape, 'source'); ticks.audit_day = lambda *args: {}
    odds.get_odd = lambda *args: None
    for item in (account['data_gap_exclusions'][0], account['orders'][0]):
        item.update(failure_reason='Missing independent intraday odd daily evidence', failure_stage='odd')
    result = m.audit_range_execution(*range_gap_case)
    assert result['data_gap_children'] == 2 and result['fills_reconciled'] == 0
    assert account['trades'] == []


def test_board_failure_cannot_be_pretended_after_a_new_source_is_usable(range_gap_case):
    account, ticks, odds, *_ = range_gap_case
    def different(*args): raise m.ReplayDataUnavailable('Another missing input')
    ticks.get = different
    with pytest.raises(ValueError, match='independently reproduced'):
        m.audit_range_execution(*range_gap_case)


@pytest.mark.parametrize('buy,sell,side', [(.5,.5,'buy'),(.7,.3,'buy'),(.5,.5,'sell'),(.7,.3,'sell')])
def test_both_price_models_and_sides_have_independent_source_reconstruction(range_case,buy,sell,side):
    account,ticks,odds,_,_,_,_,_=range_case
    settings=account['settings']; settings.update(buy_fraction=buy,sell_fraction=sell)
    plan=account['tick_plans'][0]
    plan.update(side=side,limit_price=110. if side=='buy' else 90.,odd_limit=110. if side=='buy' else 90.)
    fraction=buy if side=='buy' else sell
    account['trades']=[]
    for row in account['orders']:
        row.update(side=side,limit_price=plan['limit_price'])
        if row['channel']=='board':
            row.update(match_range_board(ticks.get('', '', '')[0],side,row['limit_price'],row['requested_qty'],1_000_000,fraction=fraction))
        else:
            row.update(match_range_odd(odds.get_odd('', '', ''),side,row['limit_price'],row['requested_qty'],fraction=fraction))
        account['trades'].append(dict(row,qty=row['filled_qty'],**m.costs(row['reference_price'],row['filled_qty'],side,'2330')))
    account['daily'][0]['cash']=m.money(settings['initial_cash']+sum(t['cash_change'] for t in account['trades']))
    result=m.audit_range_execution(*range_case,buy_fraction=buy,sell_fraction=sell)
    assert result['ordinary_daily_proxies_rebuilt']==result['intraday_daily_proxies_rebuilt']==1
    if buy==.7:
        with pytest.raises(ValueError,match='settings'):
            m.audit_range_execution(*range_case)


@pytest.mark.parametrize('change',['price','fraction','clock','allocation','fixed_volume','prior_cap','claim','source_evidence','trade_claim','window','settings_fraction'])
def test_board_proxy_tampering_is_rejected(range_case,change):
    account=range_case[0];row=account['orders'][0]
    if change=='price':row['reference_price']+=.1
    elif change=='fraction':row['price_fraction']=.7
    elif change=='clock':row['actual_fill_time']='13:30:00'
    elif change=='allocation':row['allocations']=[dict(qty=1000,price=100.5)]
    elif change=='fixed_volume':row['source_volume']+=100000
    elif change=='prior_cap':row['prior_adv_capacity_qty']+=1000
    elif change=='claim':row['board_tick_verified']=True
    elif change=='source_evidence':row['tape_evidence']={'official_status':'invented'}
    elif change=='trade_claim':account['trades'][0]['within_window_execution_verified']=True
    elif change=='window':row['expires_at']='13:25:00'
    else:account['settings']['buy_fraction']=.7
    with pytest.raises(ValueError):m.audit_range_execution(*range_case)


def test_board_quality_audit_is_repeated_independently(range_case):
    def fail(*args):raise m.ReplayDataUnavailable('Official daily conflict')
    range_case[1].audit_day=fail
    with pytest.raises(m.ReplayDataUnavailable,match='Official daily conflict'):
        m.audit_range_execution(*range_case)
