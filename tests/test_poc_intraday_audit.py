"""Daily intraday-odd scenarios must not masquerade as verified timed fills."""
from copy import deepcopy

import pandas as pd
import pytest

from skills import poc_intraday_audit as m
from skills.poc_intraday_execution import match_intraday_odd
from test_poc_executable_replay import audit_case  # noqa: F401
from test_poc_gap_audit import gap_case  # noqa: F401


def window(plan):
    plan.update(odd_order_time=m.ODD_OPEN, odd_expires_at=m.ODD_END)


def settings(account, count=0):
    account['settings'].update(execution=m.MODEL, odd_participation=.01,
        intraday_tick_verified=False, odd_tick_verified=False, actual_fill_verified=False,
        within_window_execution_verified=False, price_level_volume_verified=False,
        data_gap_policy=m.GAP_POLICY, posthoc_data_exclusion=True, excluded_event_count=count)


@pytest.fixture
def intraday_case(audit_case):
    account, ticks, odds, routes, quotes, days, corp, feeds = audit_case
    plan = account['tick_plans'][0]; window(plan); settings(account)
    account['data_gap_exclusions'] = []
    source = dict(after_hours=False, odd_high=102., odd_low=98., odd_shares=10_000,
        source_date=plan['date'], market='TWSE',
        volume_scope='intraday_odd_session', volume_unit='shares', price_unit='TWD_per_share',
        evidence_status='official_intraday_daily_table', execution_evidence='intraday_odd_daily_hl2_proxy',
        intraday_tick_verified=False, actual_fill_verified=False)
    odds.get_odd = lambda *args: deepcopy(source)
    old = account['orders'][1]
    row = {k: old[k] for k in ('date', 'stock_id', 'event_id', 'side', 'signal_date', 'channel',
        'limit_price', 'requested_qty', 'prior_avg_volume20', 'prior_avg_amount20')}
    row.update(order_time=m.ODD_OPEN, expires_at=m.ODD_END,
               **match_intraday_odd(source, 'buy', plan['odd_limit'], plan['odd_qty']))
    account['orders'][1] = row
    account['trades'][1] = dict(row, qty=row['filled_qty'], **m.costs(row['reference_price'], row['filled_qty'], 'buy', '2330'))
    account['daily'][0]['cash'] = m.money(account['settings']['initial_cash']+sum(t['cash_change'] for t in account['trades']))
    return audit_case


def test_daily_proxy_is_reconstructed_without_asserting_intraday_time_or_tick_evidence(intraday_case):
    result = m.audit_intraday_execution(*intraday_case)
    assert result['intraday_daily_proxies_rebuilt'] == 1
    assert result['chronological_board_allocations_rebuilt'] == 1
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
def test_proxy_audit_rejects_fabricated_precision_or_changed_cash_and_denominators(intraday_case, change):
    account, _, odds, *_ = intraday_case; row = account['orders'][1]
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
        m.audit_intraday_execution(*intraday_case)


def test_unknown_odd_clock_cannot_spend_a_board_sale_before_the_sale_is_known():
    account = dict(settings=dict(initial_cash=100.), daily=[dict(date='2024-01-02', cash=0.)], cash_ledger=[],
        trades=[dict(date='2024-01-02', channel='board', side='sell', cash_change=100.),
                dict(date='2024-01-02', channel='odd', side='buy', cash_change=-200.)])
    with pytest.raises(ValueError, match='same-day sale proceeds'):
        m.audit_proxy_cash(account)


@pytest.fixture
def intraday_gap_case(gap_case):
    account = gap_case[0]; settings(account, 1)
    window(account['tick_plans'][0]); window(account['data_gap_exclusions'][0]['original_plan'])
    return gap_case


def test_original_gap_denominator_and_locked_resources_survive_new_odd_window(intraday_gap_case):
    result = m.audit_intraday_execution(*intraday_gap_case)
    assert result['data_gap_children'] == 2 and result['original_planned_children'] == 2
    assert result['nonexcluded_planned_children_reconciled'] == 0
    assert result['gap_source_failures_independently_rebuilt'] is True
    assert result['all_original_planned_children_execution_evidence_complete'] is False


def test_missing_intraday_source_cannot_keep_a_successful_board_leg(intraday_gap_case):
    account, ticks, odds, *_ = intraday_gap_case
    tape = pd.DataFrame(dict(time=pd.to_timedelta(['09:02:00']), price=[100.], shares=[100_000]))
    ticks.get = lambda *args: (tape, 'source'); ticks.audit_day = lambda *args: {}
    odds.get_odd = lambda *args: None
    for item in (account['data_gap_exclusions'][0], account['orders'][0]):
        item.update(failure_reason='Missing independent intraday odd daily evidence', failure_stage='odd')
    result = m.audit_intraday_execution(*intraday_gap_case)
    assert result['data_gap_children'] == 2 and result['fills_reconciled'] == 0
    assert account['trades'] == []


def test_board_failure_cannot_be_pretended_after_a_new_source_is_usable(intraday_gap_case):
    account, ticks, odds, *_ = intraday_gap_case
    def different(*args): raise m.ReplayDataUnavailable('Another missing input')
    ticks.get = different
    with pytest.raises(ValueError, match='independently reproduced'):
        m.audit_intraday_execution(*intraday_gap_case)
