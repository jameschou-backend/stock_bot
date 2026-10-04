"""A single disclosed missing-data scenario must not rewrite the strict account."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from skills import poc_executable_skip as m
from skills.execution_resources import ResourceDecisions


@pytest.fixture
def engine():
    obj = m.ExplicitConflictSkipOrders()
    p = m.expected_plan()
    obj.tick_plans = [deepcopy(p)]
    obj.day_plans = {(m.SKIP_EVENT, 'buy'): deepcopy(p)}
    obj.tick_attempts = set()
    obj.holdings = {'3230': {'qty': 0}, '1569': {'qty': 10}}
    obj.orders = []; obj.trades = []; obj.cash_ledger = []
    obj.cash = 696203.01; obj.day_cost = 0.; obj.day_basis = 0.
    obj.used = {}; obj.marks = {'1569': {'price': 39.05}}
    return obj


def execute(obj, **changes):
    args = dict(day=pd.Timestamp(m.SKIP_DATE), sid=m.SKIP_STOCK, side='buy', qty=5000,
        reason='leader_entry', event_id=m.SKIP_EVENT, signal_date='2024-10-15')
    args.update(changes)
    return obj._execute_order(**args)


def test_exact_exclusion_preserves_assets_and_positive_reservations(engine):
    saved = deepcopy(engine.__dict__)
    assert execute(engine) == 0
    for name, value in saved.items():
        if name not in ('explicit_exclusions', 'orders', 'tick_attempts'):
            assert engine.__dict__[name] == value
    assert engine.tick_attempts == {'3230'}
    row = engine.orders[0]
    assert row['channel'] == 'event' and row['requested_qty'] == 4958 and row['filled_qty'] == 0
    assert row['failure'] == m.SKIP_REASON and row['failure'] != 'official_full_session_halt'
    assert engine.explicit_exclusions[0]['original_plan']['board_qty'] == 4000
    assert engine.explicit_exclusions[0]['original_plan']['odd_qty'] == 958
    assert engine.explicit_exclusions[0]['source_hash_verification_required_by_caller'] is True
    with pytest.raises(ValueError, match='already applied'):
        execute(engine)


@pytest.mark.parametrize('change', [dict(side='sell'), dict(sid='6147'),
    dict(day=pd.Timestamp('2024-10-17')), dict(event_id='another-event')])
def test_other_orders_including_sells_always_dispatch_to_strict_execution(engine, monkeypatch, change):
    received = []
    def strict(self, *args):
        received.append(args)
        raise m.pd.errors.DataError('strict evidence remains required')
    monkeypatch.setattr(m.ExecutableOrders, '_execute_order', strict)
    with pytest.raises(m.pd.errors.DataError, match='strict evidence'):
        execute(engine, **change)
    assert len(received) == 1 and engine.explicit_exclusions == [] and engine.orders == []


@pytest.mark.parametrize('field,value', [('planned_qty', 0), ('board_qty', 0), ('odd_qty', 0),
    ('side', 'sell'), ('reserved_cash', 0), ('sizing_budget', 346860),
    ('limit_price', 69.5), ('odd_limit', 69.5), ('signal_date', '2024-10-16')])
def test_changed_plan_cannot_use_the_exception(engine, field, value):
    engine.day_plans[(m.SKIP_EVENT, 'buy')][field] = value
    with pytest.raises(ValueError, match='sealed positive plan'):
        execute(engine)
    assert engine.orders == [] and not engine.tick_attempts


@pytest.mark.parametrize('change', [dict(signal_date='2024-10-16'), dict(qty=4957),
    dict(qty=True), dict(reason='staged_add')])
def test_committed_order_identity_is_not_relaxed(engine, change):
    with pytest.raises(ValueError, match='committed plan'):
        execute(engine, **change)


def test_existing_holding_or_duplicate_plan_cannot_be_erased(engine):
    engine.holdings['3230']['qty'] = 1
    with pytest.raises(ValueError, match='existing holding'):
        execute(engine)
    engine.holdings['3230']['qty'] = 0
    engine.tick_plans.append(deepcopy(engine.tick_plans[0]))
    with pytest.raises(ValueError, match='committed plan'):
        execute(engine)


def test_real_resource_wrapper_keeps_cash_and_slot_locked_for_later_candidates(engine):
    class Base:
        def order(self, *args): return self._execute_order(*args)
    class Account(ResourceDecisions, m.ExplicitConflictSkipOrders, Base):
        def prior(self, day, sid): return 63.1
    obj = Account(opening_cash_only=True, lock_slots=True, lock_unused=True)
    obj.__dict__.update(deepcopy(engine.__dict__))
    obj.slots = 3; obj.benchmark = False; obj.previous_nav = 1040577.11; obj.names = {}
    obj.opening_limit = obj.opening_remaining = 395371.51
    obj.locked_unused = 0.; obj.occupied = {'1111', '2222'}
    day = pd.Timestamp(m.SKIP_DATE)
    assert obj.order(day, '3230', 'buy', 5000, 'leader_entry', m.SKIP_EVENT, '2024-10-15') == 0
    assert obj.resource_plans[0]['spent'] == 0 and obj.resource_plans[0]['filled_qty'] == 0
    assert obj.locked_unused == 346859.04 and obj.cash == engine.cash
    assert obj.occupied == {'1111', '2222', '3230'}
    assert obj.order(day, '2330', 'buy', 5000, 'leader_entry', 'later-candidate', '2024-10-15') == 0
    assert obj.resource_plans[-1]['failure'] == 'resource_slots_locked'
    assert obj.resource_plans[-1]['available_before'] == 48512.47
    assert len(obj.explicit_exclusions) == 1


@pytest.fixture
def audit_case(engine):
    execute(engine)
    prior_days = pd.bdate_range(end='2024-10-15', periods=188)
    days = prior_days.append(pd.DatetimeIndex([pd.Timestamp(m.SKIP_DATE)]))
    prior = [dict(date=str(d.date()), cash=395371.51) for d in prior_days]
    journal = dict(daily=deepcopy(prior), trades=[], orders=[], cash_ledger=[], corporate_actions=[],
        holdings=[], cohorts=[], resource_plans=[], selection_decisions=[], tick_plans=[m.expected_plan()])
    strict = dict(completed=False, completed_sessions=188, last_date='2024-10-15',
        reason=m.STRICT_FAILURE, partial_journal=deepcopy(journal))
    account = deepcopy(journal)
    account.update(settings=dict(initial_cash=395371.51, explicit_data_exclusion_policy=m.SKIP_POLICY,
        posthoc_data_exclusion=True, excluded_event_count=1), explicit_exclusions=deepcopy(engine.explicit_exclusions),
        orders=deepcopy(engine.orders), daily=[*prior, dict(date=m.SKIP_DATE, cash=395371.51)],
        resource_plans=[dict(date=m.SKIP_DATE, stock_id='3230', event_id=m.SKIP_EVENT,
            signal_date='2024-10-15', spent=0, filled_qty=0, opening_cash=395371.51,
            planned_qty=5000, budget=m.expected_plan()['reserved_cash'], locked_unused_before=0.,
            locked_after=m.money(m.expected_plan()['reserved_cash']))],
        slot_decisions=[dict(date=m.SKIP_DATE, event_id=m.SKIP_EVENT, stock_id='3230',
            signal_date='2024-10-15', attempted=True, filled_qty=0, failure=None)])
    quotes = pd.DataFrame([dict(date=d, stock_id='3230', close=63.1, volume=1_000_000) for d in days])
    def forbidden(*args):
        raise AssertionError('Excluded order queried execution evidence')
    ticks = SimpleNamespace(get=forbidden); odds = SimpleNamespace(get_odd=forbidden)
    corp = SimpleNamespace(reference_price=lambda sid, day, price: price)
    feeds = SimpleNamespace(get_limits=lambda sid: {m.SKIP_DATE: dict(lower=56.8, upper=69.4)})
    return account, (ticks, odds, {'3230': 'TPEX'}, quotes, days, corp, feeds), strict


def test_independent_audit_preserves_input_and_does_not_certify_excluded_fills(audit_case):
    account, args, strict = audit_case
    saved = deepcopy(account)
    result = m.audit_executable_skip(account, *args, strict_partial=strict)
    assert account == saved
    assert result['explicit_excluded_children'] == 2 and result['original_planned_children'] == 2
    assert result['nonexcluded_planned_children_reconciled'] == 0
    assert result['all_original_planned_children_execution_evidence_complete'] is False
    assert result['actual_fill_verified'] is False and result['strict_prefix']['completed_sessions'] == 188
    assert 'all_planned_children_reconciled' not in result


@pytest.mark.parametrize('change', ['expanded', 'missing_order', 'filled', 'missing_plan',
    'zero_plan', 'cash', 'cohort', 'holding', 'budget', 'slot', 'replace', 'additional_budget',
    'prefix', 'prior_close', 'legal_limit', 'calendar', 'deceptive_settings'])
def test_independent_audit_refuses_tampering(audit_case, change):
    account, args, strict = audit_case
    if change == 'expanded': account['explicit_exclusions'].append(deepcopy(account['explicit_exclusions'][0]))
    elif change == 'missing_order': account['orders'] = []
    elif change == 'filled': account['trades'].append(dict(date=m.SKIP_DATE, stock_id='3230', event_id=m.SKIP_EVENT))
    elif change == 'missing_plan': account['tick_plans'] = []
    elif change == 'zero_plan': account['tick_plans'][0]['planned_qty'] = 0
    elif change == 'cash': account['cash_ledger'].append(dict(event_id=m.SKIP_EVENT, cash_change=-1))
    elif change == 'cohort': account['cohorts'].append(dict(event_id=m.SKIP_EVENT))
    elif change == 'holding': account['holdings'].append(dict(event_id=m.SKIP_EVENT, qty=1))
    elif change == 'budget': account['resource_plans'][0]['locked_after'] = 0
    elif change == 'slot': account['slot_decisions'][0]['attempted'] = False
    elif change == 'replace': account['slot_decisions'].append(dict(date=m.SKIP_DATE, event_id='next',
        attempts_before=[], unfilled_before=[], occupied_before=[]))
    elif change == 'additional_budget':
        p = deepcopy(account['tick_plans'][0]); p.update(event_id='next', reserved_cash=50000.)
        account['tick_plans'].append(p)
    elif change == 'prefix': account['daily'][0]['cash'] += 1
    elif change == 'prior_close': args[3].loc[args[3].date.eq(pd.Timestamp('2024-10-15')), 'close'] = 63.2
    elif change == 'legal_limit': args[-1].get_limits = lambda sid: {m.SKIP_DATE: dict(lower=56.8, upper=69.5)}
    elif change == 'calendar': args = (*args[:4], args[4].delete(-2), *args[5:])
    else: account['settings']['posthoc_data_exclusion'] = False
    with pytest.raises(ValueError):
        m.audit_executable_skip(account, *args, strict_partial=strict)


def test_strict_prefix_works_on_partial_journal_without_fabricating_hash_verification(audit_case):
    account, _, strict = audit_case
    for candidate in (account, {'partial_journal': account}, {'account': account}):
        result = m.verify_strict_prefix(candidate, strict)
        assert result['strict_prefix_compared'] is True
        assert result['source_hash_verification_required_by_caller'] is True
    strict['reason'] = 'some unrelated error'
    with pytest.raises(ValueError, match='original strict failure'):
        m.verify_strict_prefix(account, strict)


def test_completed_exclusion_can_be_checked_in_later_partial_account(audit_case):
    account, _, strict = audit_case
    del account['settings']
    # A later missing tape may leave an unfinished day's plans. This helper
    # verifies only the prior excluded attempt and never reports full fills.
    account['tick_plans'].append(dict(date='2024-10-17', event_id='later', side='buy', stock_id='2330'))
    result = m.audit_explicit_exclusion({'partial_journal': account}, strict)
    assert result['explicit_exclusion_resource_lock_verified'] is True
    assert result['all_original_planned_children_execution_evidence_complete'] is False
    assert 'chronological_board_allocations_rebuilt' not in result
    account['resource_plans'][0]['spent'] = 1
    with pytest.raises(ValueError, match='daily budget'):
        m.audit_explicit_exclusion(account, strict)


def test_run_discloses_the_excluded_board_order_in_completeness_denominators(engine):
    class Base:
        def run(self): return dict(settings={}, orders=[])
    class Account(m.ExplicitConflictSkipOrders, Base): pass
    obj = Account(); obj.explicit_exclusions = [m._record()]
    result = obj.run()
    assert result['settings']['posthoc_data_exclusion'] is True
    assert result['settings']['excluded_event_count'] == 1
    assert result['ordinary_volume_evidence']['excluded_positive_board_children'] == 1
    assert result['ordinary_volume_evidence']['all_original_requested_board_capacity_observed'] is False
