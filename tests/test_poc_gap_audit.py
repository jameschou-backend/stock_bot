"""Missing-data exclusions remain visible, unfilled and independently justified."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from skills import poc_gap_audit as m
from skills.poc_gap_execution import exclusion_order
from test_poc_executable_replay import audit_case  # noqa: F401


@pytest.fixture
def gap_case(audit_case):
    account, ticks, odds, routes, quotes, days, corp, feeds = audit_case
    plan = account['tick_plans'][0]
    plan['opening_cash'] = account['settings']['initial_cash']
    account['settings'].update(data_gap_policy=m.GAP_POLICY, posthoc_data_exclusion=True, excluded_event_count=1)
    account['daily'][0]['cash'] = 1_000_000.
    account.update(trades=[], holdings=[], cohorts=[], corporate_actions=[],
        cash_ledger=[dict(date=plan['date'], kind='initial_deposit', cash_after=1_000_000., cash_change=1_000_000.)])
    children = [dict(channel=c, requested_qty=plan[c+'_qty']) for c in ('board', 'odd')]
    gap = dict(policy=m.GAP_POLICY, date=plan['date'], stock_id='2330', side='buy', event_id='a',
        signal_date=plan['signal_date'], reason='leader_entry', original_plan=deepcopy(plan),
        excluded_children=children, excluded_channels=['board', 'odd'], filled_qty=0,
        failure_class='ReplayDataUnavailable', failure_reason='Known board data conflict', failure_stage='board',
        market='TWSE', source_verification_required_by_caller=True, retry_sell=False, posthoc_data_exclusion=True, actual_fill_verified=False, live_qualified=False,
        cash_before=1_000_000., cash_after=1_000_000., holding_qty_before=0, holding_qty_after=0,
        cash_ledger_length_before=1, trade_count_before=0, order_count_before=0,
        exit_state_before=None, exit_state_after=None)
    account['data_gap_exclusions'] = [gap]
    account['orders'] = [exclusion_order(gap)]
    account['resource_plans'] = [dict(date=plan['date'], stock_id='2330', event_id='a',
        signal_date=plan['signal_date'], spent=0, filled_qty=0, opening_cash=1_000_000.,
        planned_qty=plan['planned_qty'], budget=plan['reserved_cash'], locked_unused_before=0.,
        locked_after=m.money(plan['reserved_cash']))]
    account['slot_decisions'] = [dict(date=plan['date'], event_id='a', stock_id='2330',
        signal_date=plan['signal_date'], attempted=True, filled_qty=0, failure=None)]
    def conflict(*args):
        raise m.ReplayDataUnavailable('Known board data conflict')
    ticks.get = conflict
    for column in ('open', 'high', 'low'):
        quotes[column] = quotes['close']
    return account, ticks, odds, routes, quotes, days, corp, feeds


def test_source_gap_is_independently_rebuilt_and_disclosed_without_mutating_account(gap_case):
    original = deepcopy(gap_case[0])
    result = m.audit_gap_execution(*gap_case)
    assert gap_case[0] == original
    assert result['original_planned_children'] == 2
    assert result['nonexcluded_planned_children_reconciled'] == 0
    assert result['data_gap_children'] == 2
    assert result['gap_source_failures_independently_rebuilt'] is True
    assert result['all_original_planned_children_execution_evidence_complete'] is False
    assert result['actual_fill_verified'] is False and result['live_qualified'] is False
    assert 'all_planned_children_reconciled' not in result


@pytest.mark.parametrize('change', [
    'duplicate', 'missing_order', 'changed_quantity', 'filled', 'hidden_cash', 'rollback_cash',
    'holding', 'cohort', 'release_budget', 'release_slot', 'replacement', 'extra_budget',
    'same_day', 'limit', 'size', 'deceptive_settings', 'cash_anchor', 'order_anchor', 'trade_anchor',
    'changed_reason', 'changed_stage', 'invented_halt', 'filled_child',
])
def test_tampering_cannot_be_hidden_under_missing_data(gap_case, change):
    account = gap_case[0]; gap = account['data_gap_exclusions'][0]; plan = account['tick_plans'][0]
    if change == 'duplicate': account['data_gap_exclusions'].append(deepcopy(gap))
    elif change == 'missing_order': account['orders'] = []
    elif change == 'changed_quantity': gap['original_plan']['planned_qty'] -= 1
    elif change == 'filled': account['trades'].append(dict(date=plan['date'], stock_id='2330', qty=1000))
    elif change == 'hidden_cash': account['cash_ledger'].append(dict(date=plan['date'], stock_id='2330', kind='buy', cash_change=-100))
    elif change == 'rollback_cash': gap['cash_after'] -= 1
    elif change == 'holding': account['holdings'].append(dict(date=plan['date'], stock_id='2330', qty=1))
    elif change == 'cohort': account['cohorts'].append(dict(event_id='a'))
    elif change == 'release_budget': account['resource_plans'][0]['locked_after'] = 0
    elif change == 'release_slot': account['slot_decisions'][0]['attempted'] = False
    elif change == 'replacement': account['slot_decisions'].append(dict(date=plan['date'], event_id='b', attempts_before=[], unfilled_before=[], occupied_before=[]))
    elif change == 'extra_budget':
        extra = deepcopy(plan); extra.update(event_id='b', reserved_cash=900_000.)
        account['tick_plans'].append(extra)
    elif change == 'same_day':
        for value in (plan, gap, gap['original_plan'], account['orders'][0], account['resource_plans'][0], account['slot_decisions'][0]):
            value['signal_date'] = plan['date']
    elif change == 'limit':
        plan['limit_price'] = gap['original_plan']['limit_price'] = 109.
    elif change == 'size':
        plan['sizing_budget'] = gap['original_plan']['sizing_budget'] = 100_000.
    elif change == 'deceptive_settings': account['settings']['posthoc_data_exclusion'] = False
    elif change == 'cash_anchor': gap['cash_before'] = gap['cash_after'] = 999_999.
    elif change == 'order_anchor': gap['order_count_before'] = 1
    elif change == 'trade_anchor': gap['trade_count_before'] = 1
    elif change == 'changed_reason': gap['failure_reason'] = account['orders'][0]['failure_reason'] = 'Different failure'
    elif change == 'changed_stage': gap['failure_stage'] = account['orders'][0]['failure_stage'] = 'odd'
    elif change == 'invented_halt': account['orders'][0]['failure'] = 'official_full_session_halt'
    elif change == 'filled_child': account['orders'].append(dict(account['orders'][0], channel='board', filled_qty=1000))
    with pytest.raises(ValueError):
        m.audit_gap_execution(*gap_case)


def test_fabricated_source_problem_is_rejected_even_when_ledgers_are_zero(gap_case):
    account, ticks, odds, routes, quotes, days, corp, feeds = gap_case
    tape = pd.DataFrame(dict(time=pd.to_timedelta(['09:02:00']), price=[100.], shares=[100_000]))
    ticks.get = lambda *args: (tape, 'source')
    ticks.audit_day = lambda *args: {}
    with pytest.raises(ValueError, match='sources are usable'):
        m.audit_gap_execution(*gap_case)


def test_programming_errors_do_not_get_reclassified_as_source_failures(gap_case):
    def broken(*args): raise ValueError('oversold or malformed engine')
    gap_case[1].get = broken
    with pytest.raises(ValueError, match='malformed engine'):
        m.audit_gap_execution(*gap_case)


@pytest.mark.parametrize('exception', [False, True])
def test_missing_legal_source_is_disclosed_and_independently_reproduced(gap_case, exception):
    account, *_, feeds = gap_case
    message = 'Official limit source absent' if exception else 'Order lacks a valid preknown legal limit'
    def missing(*args):
        if exception:
            raise m.ReplayDataUnavailable(message)
        return {}
    feeds.get_limits = missing
    for item in (account['data_gap_exclusions'][0], account['orders'][0]):
        item.update(failure_reason=message, failure_stage='board')
    result = m.audit_gap_execution(*gap_case)
    assert result['gap_plans_without_verified_legal_limits'] == 1
    assert result['all_original_preplanned_limits_verified'] is False
    assert result['gap_source_failures_independently_rebuilt'] is True


def test_odd_failure_rechecks_successful_board_source_without_keeping_board_fill(gap_case):
    account, ticks, odds, *_ = gap_case
    tape = pd.DataFrame(dict(time=pd.to_timedelta(['09:02:00']), price=[100.], shares=[100_000]))
    checked = []
    ticks.get = lambda *args: (tape, 'source')
    ticks.audit_day = lambda *args: checked.append('board')
    odds.get_odd = lambda *args: None
    for item in (account['data_gap_exclusions'][0], account['orders'][0]):
        item.update(failure_reason='Missing after-hours auction evidence', failure_stage='odd')
    result = m.audit_gap_execution(*gap_case)
    assert checked == ['board'] and result['data_gap_children'] == 2
    assert account['trades'] == [] and account['daily'][0]['cash'] == 1_000_000.


def as_sell(case):
    account = case[0]; gap = account['data_gap_exclusions'][0]; plan = account['tick_plans'][0]
    plan.update(side='sell', limit_price=90., odd_limit=90., reserved_cash=0., sizing_budget=0.)
    gap.update(side='sell', reason='three_black', retry_sell=True, original_plan=deepcopy(plan),
               holding_qty_before=plan['planned_qty'], holding_qty_after=plan['planned_qty'])
    state = dict(trigger_reason='three_black', signal_date=plan['signal_date'], target_date=plan['date'])
    gap['exit_state_before'] = deepcopy(state); gap['exit_state_after'] = deepcopy(state)
    account['orders'] = [exclusion_order(gap)]
    account['resource_plans'] = []; account['slot_decisions'] = []
    account['holdings'] = [dict(date=plan['date'], stock_id='2330', event_id='a', qty=plan['planned_qty'])]
    return gap, plan


def test_unexecuted_sell_retains_shares_and_latched_exit(gap_case):
    as_sell(gap_case)
    result = m.audit_gap_execution(*gap_case)
    assert result['data_gap_sells'] == 1 and result['data_gap_buys'] == 0


@pytest.mark.parametrize('change', ['erase', 'shares', 'latch', 'delay_retry', 'new_signal'])
def test_unexecuted_sell_cannot_disappear_or_wait_for_a_new_signal(gap_case, change):
    account = gap_case[0]; gap, plan = as_sell(gap_case)
    if change == 'erase': account['holdings'] = []
    elif change == 'shares': account['holdings'][0]['qty'] -= 1
    elif change == 'latch': gap['exit_state_after']['trigger_reason'] = None
    else:
        next_day = str((pd.Timestamp(plan['date'])+pd.offsets.BDay()).date())
        account['daily'].append(dict(date=next_day, cash=1_000_000.))
        if change == 'new_signal':
            account['orders'].append(dict(date=next_day, event_id='a', side='sell', reason='three_black', signal_date=plan['date']))
    with pytest.raises(ValueError):
        m.audit_gap_records(account)


def test_partial_journal_check_does_not_claim_source_verification(gap_case):
    account = gap_case[0]
    account['daily'] = []
    result = m.audit_gap_records({'partial_journal': account}, require_complete_days=False)
    assert result['gap_source_failures_independently_rebuilt'] is False


def test_no_gap_account_still_uses_unchanged_strict_execution_auditor(audit_case):
    account = audit_case[0]
    account['settings'].update(data_gap_policy=m.GAP_POLICY, posthoc_data_exclusion=True, excluded_event_count=0)
    account['data_gap_exclusions'] = []
    result = m.audit_gap_execution(*audit_case)
    assert result['fills_reconciled'] == 2 and result['data_gap_plans'] == 0
    assert result['all_original_planned_children_execution_evidence_complete'] is True
