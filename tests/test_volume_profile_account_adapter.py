from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.five_axis_replay import FiveAxisReplay
from skills.residual_tick_replay import TickPlanning
from skills.volume_profile_account_adapter import AccountCandidateHook, ReservationPlanner
from skills.volume_profile_selection import select_candidates, UnresolvedProfile


DAY = pd.Timestamp('2024-01-03')


def event(sid='1234'):
    return dict(members=[sid], event_id=sid+'-2024-01-02', signal_date='2024-01-02')


def engine():
    return SimpleNamespace(opening_limit=1_000_000., cash=1_000_000.,
        opening_members=set(), released_residuals=set(), holdings={}, slots=3,
        previous_nav=1_000_000., residual_budget=1_000_000/3, residual_block=False,
        prior=lambda day, sid: 100.,
        amount20=pd.DataFrame([[100_000_000.]*4], index=[DAY],
                             columns=['1234','2345','3456','4567']))


def test_planner_probe_is_pure_and_locks_full_budget_not_fill():
    planner = ReservationPlanner(engine(), DAY)
    initial = planner.snapshot()
    probe = planner.probe(event())
    assert planner.snapshot() == initial
    assert probe['attempted'] and planner.consider(event()) == probe
    assert planner.cash == 666666.67
    assert planner.probe(event())['failure'] == 'opening_slots_locked'
    planner.consider(event('2345'))
    planner.consider(event('3456'))
    assert not planner.can_continue()
    assert planner.probe(event('4567'))['failure'] == 'opening_slots_locked'


def test_opening_cash_not_same_day_income_and_residual_slot_release():
    e = engine()
    e.cash += 100_000
    e.opening_members = {'4567'}
    e.released_residuals = {'4567'}
    e.holdings = {'4567': dict(qty=123)}
    p = ReservationPlanner(e, DAY)
    assert p.cash == 1_000_000
    assert not p.occupied
    assert p.probe(event('4567'))['failure'] == 'opening_slots_locked'
    e.residual_block = True
    assert not ReservationPlanner(e, DAY).can_continue()


def test_baseline_hook_is_before_tick_reservation_and_does_not_query(monkeypatch):
    # Exercise the real TickPlanning call order around the inserted MRO hook.
    class Replay(TickPlanning, AccountCandidateHook):
        def _plan(self, day, sid, side, eid, signal, qty, budget, cash, failure):
            assert self.selection_decisions, 'hook must precede reservation'
            self.captured.append((eid, budget, failure))

    monkeypatch.setattr(FiveAxisReplay, 'corporate_day', lambda self, day: 0.)
    x = object.__new__(Replay)
    x.__dict__.update(engine().__dict__)
    x.events = {DAY: [event('2345'),event('1234')]}
    before = deepcopy(x.events)
    x.selection_arm = 'original'
    x.profile_provider = lambda ev: (_ for _ in ()).throw(AssertionError('profile queried'))
    x.candidate_selector = None
    x.selection_decisions = []
    x.benchmark = False
    x.days = pd.DatetimeIndex(['2024-01-02','2024-01-03'])
    x.positions = {day:i for i,day in enumerate(x.days)}
    x._affordable = lambda qty, step, price, cash, sid: qty
    x.captured = []
    x.corporate_day(DAY)
    assert x.events == before
    assert [r[0] for r in x.captured] == [e['event_id'] for e in before[DAY]]
    assert x.selection_decisions[0]['opening_can_reserve']
    assert all(r[1] == 1_000_000/3 for r in x.captured)


def candidate_hook(monkeypatch, arm, second_profile):
    monkeypatch.setattr(FiveAxisReplay, 'corporate_day', lambda self, day: 0.)
    x = object.__new__(AccountCandidateHook)
    x.__dict__.update(engine().__dict__)
    x.events = {DAY: [dict(event('1234'),priority=2.),dict(event('2345'),priority=1.)]}
    x.selection_arm = arm
    x.selection_decisions = []
    x.candidate_selector = select_candidates
    x.profile_provider = lambda ev: (dict(available=True,poc_up=True)
        if ev['members'][0]=='1234' else second_profile)
    return x


def test_availability_fallback_discards_partial_selection_and_restores_whole_day(monkeypatch):
    x = candidate_hook(monkeypatch, 'poc_filter_available',
        dict(available=False,reason='ordinary_tape_conflict',recoverable=False))
    original = deepcopy(x.events)
    x.corporate_day(DAY)
    row = x.selection_decisions[0]
    assert x.events == original and x.cash == 1_000_000
    assert row['fallback_to_original']
    assert row['discarded_reservation_event_ids'] == ['1234-2024-01-02']
    assert row['reset_state'] == row['initial_state']


@pytest.mark.parametrize('arm,reason,recoverable', [
    ('poc_filter','ordinary_tape_conflict',False),
    ('poc_priority_available','raw_tape_unavailable',True),
    ('poc_filter_available','unclassified_bug',False),
    ('poc_filter_available','ordinary_tape_conflict',None),
])
def test_strict_and_nonallowlisted_unknowns_stop_without_reserving_live_account(monkeypatch,arm,reason,recoverable):
    x = candidate_hook(monkeypatch, arm, dict(available=False,reason=reason,recoverable=recoverable))
    original = deepcopy(x.events)
    with pytest.raises(UnresolvedProfile):
        x.corporate_day(DAY)
    assert x.cash == 1_000_000 and x.events == original
    assert not x.selection_decisions[0]['fallback_to_original']
    assert x.selection_decisions[0]['certificate']['failure_classification']['reason'] == reason
