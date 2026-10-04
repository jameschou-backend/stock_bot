"""Data gaps must not leak a partial child fill into the account or its locks."""
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from skills import poc_gap_execution as m
from skills.million_replay import costs, money, Replay
from skills.poc_executable_odd import SCOPE
from skills.reservation_replay import ReserveDecisions
from skills.volatility_budget_replay import VolatilityBudgetReplay
from skills.execution_resources import ResourceDecisions
from skills.slot_reuse_replay import SlotReuseReplay
from skills.five_axis_replay import FiveAxisReplay


DAY = pd.Timestamp('2024-01-03')
NEXT = pd.Timestamp('2024-01-04')
SID = '2330'
EID = 'entry-2330'


class Engine(m.DataGapOrders, ReserveDecisions, VolatilityBudgetReplay):
    """Use the actual reservation/risk/cash-move MRO without fetching inputs."""
    _costs = staticmethod(costs)

    def identity(self, *args):
        return dict(status='identified', category='股票', market='TWSE')

    def official_halt(self, *args):
        return False

    def require_prior_inputs(self, *args):
        pass

    def raw(self, day, sid, key='close'):
        return 1_000_000 if key == 'volume' else 100.

    def prior(self, day, sid):
        return 100.


def auction():
    return dict(after_hours=True, odd_high=101., odd_low=101., odd_shares=10000,
        auction_price=101., volume_scope=SCOPE, volume_unit='shares',
        price_unit='TWD_per_share', auction_time='14:30:00')


@pytest.fixture
def engine():
    # The test supplies frozen market fixtures; constructors normally acquire
    # histories. Execution and cash mutation still use the production methods.
    obj = object.__new__(Engine)
    obj.data_gap_exclusions = []
    plan = dict(date=str(DAY.date()), reference_date='2024-01-02', stock_id=SID,
        event_id=EID, side='buy', signal_date='2024-01-02', order_time='09:01:00',
        expires_at='13:25:00', odd_order_time='13:40:00', odd_expires_at='14:30:00',
        planned_qty=1003, board_qty=1000, odd_qty=3, prior_reference=100.,
        limit_price=110., odd_limit=110., sizing_budget=111000., reserved_cash=111000.,
        opening_cash=1_000_000., rejection=None)
    obj.tick_plans = [deepcopy(plan)]
    obj.day_plans = {(EID, 'buy'): plan}
    obj.tick_attempts = set()
    obj.markets = {}
    obj.feeds = SimpleNamespace(get_limits=lambda sid: {
        str(d.date()): dict(lower=90., upper=110.) for d in (DAY, NEXT)})
    tape = pd.DataFrame([dict(time=pd.Timedelta('09:02:00'), price=100., shares=100000)])
    obj.ticks = SimpleNamespace(get=lambda *a: (tape.copy(), 'test-tape-sha'),
        audit_day=lambda *a: dict(own_order_fill_proven=False))
    obj.odd_feeds = SimpleNamespace(get_odd=lambda *a: auction())
    obj.volume20 = pd.DataFrame({SID: [1_000_000., 1_000_000.]}, index=[DAY, NEXT])
    obj.amount20 = obj.volume20*100
    obj.names = {}
    obj.used = defaultdict(int)
    obj.cash = 1_000_000.
    obj.holdings = {SID: dict(qty=0, event_id=EID, due_index=42)}
    obj.marks = {SID: dict(price=99., date='2024-01-02')}
    obj.day_cost = 0.
    obj.day_basis = 0.
    obj.orders = []
    obj.trades = []
    obj.cash_ledger = [dict(date='2024-01-02', kind='initial_deposit', cash_change=1_000_000., cash_after=1_000_000.)]
    obj.active_budget = obj.residual_spend_left = obj.volatility_spend_left = 111000.
    obj.opening_remaining = 1_000_000.
    obj.active_reservation = EID
    obj.reservations = {EID: dict(remaining=111000.)}
    obj.exit_states = {}
    return obj


def execute(obj, **changes):
    args = dict(day=DAY, sid=SID, side='buy', qty=1003, reason='leader_entry',
                event_id=EID, signal_date='2024-01-02')
    args.update(changes)
    return obj._execute_order(**args)


def tracked(obj):
    return {name: deepcopy(getattr(obj, name)) for name in
        (*m._SCALARS, *m._MAPS, 'orders', 'trades', 'cash_ledger', 'exit_states', 'tick_plans', 'day_plans')}


def unavailable(*args):
    raise m.ReplayDataUnavailable('Official aggregate differs from the supplied tape')


def test_healthy_order_keeps_actual_board_and_odd_cash_movements(engine):
    assert execute(engine) == 1003
    assert [(t['channel'], t['qty'], t['reference_price']) for t in engine.trades] == [
        ('board', 1000, 100.), ('odd', 3, 101.)]
    assert engine.cash == money(1_000_000+sum(t['cash_change'] for t in engine.trades))
    spent = money(1_000_000-engine.cash)
    assert all(getattr(engine, name) == money(111000-spent)
               for name in ('active_budget', 'residual_spend_left', 'volatility_spend_left'))
    assert engine.reservations[EID]['remaining'] == money(111000-spent)
    assert engine.data_gap_exclusions == []


@pytest.mark.parametrize('stage', ['board_get', 'board_audit', 'odd_missing', 'odd_get', 'odd_legal_range'])
def test_data_gap_rolls_back_entire_order_and_every_cash_wrapper(engine, stage):
    if stage == 'board_get':
        engine.ticks.get = unavailable
    elif stage == 'board_audit':
        engine.ticks.audit_day = unavailable
    elif stage == 'odd_missing':
        engine.odd_feeds.get_odd = lambda *a: None
    elif stage == 'odd_get':
        engine.odd_feeds.get_odd = unavailable
    else:
        engine.odd_feeds.get_odd = lambda *a: dict(auction(), auction_price=111., odd_high=111., odd_low=111.)
    before = tracked(engine)
    holding_ref, reservation_ref = engine.holdings[SID], engine.reservations[EID]
    assert execute(engine) == 0
    after = tracked(engine)
    assert {k: v for k, v in after.items() if k != 'orders'} == {k: v for k, v in before.items() if k != 'orders'}
    assert engine.holdings[SID] is holding_ref and engine.reservations[EID] is reservation_ref
    assert holding_ref['qty'] == 0 and reservation_ref['remaining'] == 111000.
    assert engine.tick_attempts == {SID}
    gap, = engine.data_gap_exclusions
    assert gap['failure_stage'] == ('board' if stage.startswith('board') else 'odd')
    assert gap['cash_before'] == gap['cash_after'] == 1_000_000.
    assert gap['holding_qty_before'] == gap['holding_qty_after'] == 0
    assert gap['excluded_children'] == [dict(channel='board', requested_qty=1000), dict(channel='odd', requested_qty=3)]
    assert gap['cash_ledger_length_before'] == 1
    assert gap['trade_count_before'] == gap['order_count_before'] == 0
    assert gap['retry_sell'] is False and gap['live_qualified'] is False
    assert engine.orders == [m.exclusion_order(gap)]
    assert gap['original_plan'] is not engine.tick_plans[-1]
    with pytest.raises(ValueError, match='unique committed plan'):
        execute(engine)


def test_prior_journals_and_unrelated_holdings_survive_rollback(engine):
    engine.orders.append(dict(old_order=True))
    engine.trades.append(dict(old_trade=True))
    engine.holdings['1111'] = dict(qty=42, event_id='old', due_index=30)
    engine.marks['1111'] = dict(price=500., date='2024-01-02')
    engine.used[('1111', 'board')] = 1000
    engine.tick_attempts.add('1111')
    engine.odd_feeds.get_odd = unavailable
    before = tracked(engine)
    assert execute(engine) == 0
    assert engine.orders[:-1] == before['orders'] and engine.trades == before['trades']
    assert engine.cash_ledger == before['cash_ledger']
    assert engine.holdings == before['holdings'] and engine.used == before['used']
    assert engine.tick_attempts == {'1111', SID}
    assert engine.data_gap_exclusions[0]['order_count_before'] == 1
    assert engine.data_gap_exclusions[0]['trade_count_before'] == 1


def test_unknown_dated_identity_is_disclosed_without_touching_any_child(engine):
    engine.identity = lambda *a: dict(status='unresolved', market='TWSE')
    engine.ticks.get = lambda *a: pytest.fail('Identity failure must precede tape fetching')
    assert execute(engine) == 0
    assert engine.data_gap_exclusions[0]['failure_stage'] == 'execution_context'
    assert engine.data_gap_exclusions[0]['failure_reason'] == 'Execution requires identified dated security'
    assert engine.data_gap_exclusions[0]['market'] is None


def test_sell_gap_retains_shares_exit_signal_and_retries_on_next_day(engine):
    plan = engine.day_plans.pop((EID, 'buy'))
    plan.update(side='sell', limit_price=90., odd_limit=90.)
    engine.day_plans[(EID, 'sell')] = plan
    engine.tick_plans = [deepcopy(plan)]
    engine.holdings[SID]['qty'] = 1003
    engine.exit_states[EID] = dict(trigger_reason='three_black_lower_closes', signal_date='2024-01-02', due_index=42)
    engine.odd_feeds.get_odd = unavailable
    saved_holding, saved_exit = deepcopy(engine.holdings), deepcopy(engine.exit_states)
    assert execute(engine, side='sell', reason='scheduled_exit', signal_date=None) == 0
    gap = engine.data_gap_exclusions[0]
    assert engine.holdings == saved_holding and engine.exit_states == saved_exit
    assert gap['retry_sell'] is True and gap['holding_qty_before'] == gap['holding_qty_after'] == 1003
    assert gap['exit_state_before'] == gap['exit_state_after'] == saved_exit[EID]
    assert engine.orders[-1]['reason'] == 'three_black_lower_closes'
    assert engine.orders[-1]['signal_date'] == '2024-01-02'
    retry = dict(plan, date=str(NEXT.date()), reference_date=str(DAY.date()))
    engine.day_plans[(EID, 'sell')] = retry
    engine.tick_plans.append(deepcopy(retry))
    engine.tick_attempts.clear()
    engine.used.clear()
    engine.odd_feeds.get_odd = lambda *a: auction()
    assert execute(engine, day=NEXT, side='sell', reason='scheduled_exit', signal_date=None) == 1003
    assert engine.holdings[SID]['qty'] == 0
    assert len(engine.data_gap_exclusions) == 1
    assert all(t['signal_date'] == '2024-01-02' for t in engine.trades)
    assert all(t['reason'] == 'three_black_lower_closes' for t in engine.trades)
    assert engine.cash > 1_000_000


@pytest.mark.parametrize('error', [ValueError('account invariant'), RuntimeError('program defect'), KeyError('bad-key')])
def test_program_and_accounting_errors_are_not_converted_to_skips(engine, error):
    def fail(*args):
        raise error
    engine.odd_feeds.get_odd = fail
    with pytest.raises(type(error), match=str(error).strip("'")):
        execute(engine)
    assert not engine.data_gap_exclusions


@pytest.mark.parametrize('change', [dict(qty=True), dict(qty=1002), dict(signal_date='2024-01-03'),
                                 dict(event_id='wrong'), dict(side='invalid')])
def test_execution_identity_and_plan_validation_remain_strict(engine, change):
    engine.ticks.get = unavailable
    with pytest.raises((ValueError, KeyError)):
        execute(engine, **change)
    assert not engine.data_gap_exclusions and not engine.orders


def test_mutated_or_duplicate_plan_cannot_be_excused_as_a_source_gap(engine):
    engine.ticks.get = unavailable
    engine.tick_plans.append(deepcopy(engine.tick_plans[0]))
    with pytest.raises(ValueError, match='unique committed plan'):
        execute(engine)
    assert not engine.data_gap_exclusions


def test_real_resource_and_slot_wrappers_lock_failed_order_after_odd_rollback(engine, monkeypatch):
    class Base:
        def order(self, *args):
            return self._execute_order(*args)
        cash_move = Replay.cash_move
    class Account(SlotReuseReplay, m.DataGapOrders, Base):
        _execute_order = m.DataGapOrders._execute_order
        _costs = staticmethod(costs)
        identity = Engine.identity
        official_halt = Engine.official_halt
        require_prior_inputs = Engine.require_prior_inputs
        raw = Engine.raw
        prior = Engine.prior
    # Retain real slot/resource wrappers and terminate at the supplied frozen
    # plan, before the older execution stack would rebuild a different plan.
    monkeypatch.setattr(FiveAxisReplay, 'order', Base.order)
    obj = object.__new__(Account)
    obj.__dict__.update(deepcopy(engine.__dict__))
    obj.resource_plans = []
    obj.opening_cash_only = obj.lock_slots = obj.lock_unused = True
    obj.lock_opening_slots = obj.lock_failed_slots = True
    obj.opening_members = {'1111', '2222'}
    obj.attempted_members = set()
    obj.failed_members = set()
    obj.slot_decisions = []
    obj.active_budget = None
    obj.slots = 3
    obj.benchmark = False
    obj.previous_nav = 333000.
    obj.opening_limit = obj.opening_remaining = 1_000_000.
    obj.locked_unused = 0.
    obj.occupied = {'1111', '2222'}
    obj.odd_feeds.get_odd = unavailable
    assert obj.order(DAY, SID, 'buy', 1003, 'leader_entry', EID, '2024-01-02') == 0
    assert obj.resource_plans[0]['spent'] == 0 and obj.resource_plans[0]['filled_qty'] == 0
    assert obj.locked_unused == 111000. and obj.opening_remaining == 1_000_000.
    assert obj.cash == 1_000_000. and obj.holdings[SID]['qty'] == 0
    assert obj.occupied == {'1111', '2222', SID}
    assert obj.attempted_members == obj.failed_members == {SID}
    obj.holdings['4444'] = dict(qty=0, event_id='later')
    assert obj.order(DAY, '4444', 'buy', 1003, 'leader_entry', 'later', '2024-01-02') == 0
    assert obj.resource_plans[-1]['failure'] == 'resource_slots_locked'
    assert obj.resource_plans[-1]['available_before'] == 889000.
    assert obj.slot_decisions[-1]['attempts_before'] == [SID]
    assert obj.slot_decisions[-1]['unfilled_before'] == [SID]


def test_run_reports_excluded_children_in_original_denominator(engine):
    engine.odd_feeds.get_odd = unavailable
    execute(engine)
    class Result:
        def run(self):
            return dict(settings={'odd_participation': .05}, orders=self.orders)
    class Account(m.DataGapOrders, Result):
        pass
    obj = Account()
    obj.data_gap_exclusions = engine.data_gap_exclusions
    obj.orders = engine.orders
    result = obj.run()
    assert result['settings']['data_gap_policy'] == m.GAP_POLICY
    assert result['settings']['excluded_board_children'] == result['settings']['excluded_odd_children'] == 1
    assert result['settings']['posthoc_data_exclusion'] is True
    assert result['settings']['odd_participation'] == .01
    volume = result['ordinary_volume_evidence']
    assert volume['requested_board_children'] == 0 and volume['original_requested_board_children'] == 1
    assert volume['excluded_positive_board_children'] == 1
    assert volume['all_original_requested_board_capacity_observed'] is False
    assert result['data_gap_exclusions'] == engine.data_gap_exclusions
