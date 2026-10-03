"""Exhaustive small-state equivalence to full stable partition, no market data."""
from copy import deepcopy
from itertools import product
import math

import pytest

from skills.volume_profile_selection import UnresolvedProfile, select_candidates


def events(count=5):
    return [dict(event_id=f'e{i}', members=[f'{1000+i}'], signal_date='2024-01-02',
                 entry_date='2024-01-03', priority=1.-i*.01, price=100., amount=60e6)
            for i in range(count)]


class Planner:
    """The frozen attempted rule uses raw quantity, not final fill/board quantity."""
    def __init__(self, *, cash=1000000., slots=3, occupied=(), held=(), residual_block=False):
        self.cash, self.slots, self.nav = cash, slots, 1000000.
        self.occupied, self.held = set(occupied), set(held)
        self.residual_block = residual_block
        self.applied = []

    def snapshot(self):
        return dict(cash=self.cash, slots=self.slots, occupied=sorted(self.occupied),
                    held=sorted(self.held), residual_block=self.residual_block)

    def can_continue(self):
        return len(self.occupied) < self.slots and self.cash > 40 and not self.residual_block

    def probe(self, event):
        sid, price = event['members'][0], event['price']
        allocation = min(self.cash, self.nav/self.slots)
        reason = None
        if sid in self.held or sid in self.occupied or len(self.occupied) >= self.slots:
            reason = 'opening_slots_locked'
        elif self.residual_block:
            reason = 'residual_exposure_cap'
        elif not math.isfinite(event['amount']) or event['amount'] < 50e6:
            reason = 'prior_liquidity_below_50m_or_missing'
        raw_qty = max(0, math.floor((allocation-40)/(price*(1+.001425+.0045)))) if price else 0
        return dict(attempted=reason is None and raw_qty > 0, failure=reason,
                    allocation=allocation, raw_qty=raw_qty)

    def consider(self, event):
        result = self.probe(event)
        if result['attempted']:
            self.cash = round(self.cash-result['allocation'], 2)
            self.occupied.add(event['members'][0])
            self.applied.append(event['event_id'])
        return result


class Provider:
    def __init__(self, flags):
        self.flags, self.requested = flags, []

    def __call__(self, event):
        self.requested.append(event['event_id'])
        flag = self.flags[event['event_id']]
        return dict(available=True, poc_up=flag) if flag is not None else dict(
            available=False, poc_up=None, reason='missing_raw', recoverable=True)


def test_all_small_profiles_match_full_partition_and_exact_reservation_states():
    for size in range(1, 7):
        source = events(size)
        if size >= 4:
            source[1]['amount'] = 49e6
            source[3]['price'] = 1e9
        for flags in product([False, True], repeat=size):
            mapping = dict(zip([e['event_id'] for e in source], flags))
            for arm in ('poc_priority', 'poc_filter'):
                for occupied, cash in (((), 1000000.), (('old1', 'old2'), 1000000.),
                                       ((), 100000.), ((), 140.)):
                    actual, oracle = Planner(occupied=occupied, cash=cash), Planner(occupied=occupied, cash=cash)
                    ordered = ([e for e in source if mapping[e['event_id']]] +
                               ([e for e in source if not mapping[e['event_id']]] if arm == 'poc_priority' else []))
                    expected = []
                    for event in ordered:
                        if oracle.consider(event)['attempted']:
                            expected.append(event['event_id'])
                    result = select_candidates(source, arm=arm, provider=Provider(mapping), planner=actual)
                    assert [e['event_id'] for e in result['events']] == expected
                    assert actual.snapshot() == oracle.snapshot()
                    assert result['certificate']['reserved_event_ids'] == expected
                    assert result['certificate']['complete'] is True
                    assert len(result['certificate']['decisions']) == len(source)


def test_original_preserves_all_events_and_never_asks_profiles_or_reserves():
    source, planner = events(), Planner()
    initial = planner.snapshot()
    def forbidden(_):
        raise AssertionError('No profile in original arm')
    result = select_candidates(source, arm='original', provider=forbidden, planner=planner)
    assert result['events'] == source
    assert planner.snapshot() == initial and planner.applied == []


@pytest.mark.parametrize('settings', [dict(occupied=('a','b','c')), dict(cash=40.), dict(residual_block=True)])
def test_exhausted_day_requires_no_profiles(settings):
    planner = Planner(**settings)
    provider = Provider({})
    result = select_candidates(events(), arm='poc_priority', provider=provider, planner=planner)
    assert result['events'] == [] and provider.requested == []
    assert result['certificate']['complete']
    assert result['certificate']['stop_reason'] == 'resource_exhausted'
    assert all(d['profile_status'] == 'not_needed_resource_exhausted' for d in result['certificate']['decisions'])


def test_lazy_prefix_can_stop_before_unknown_tail_after_last_reserved_slot():
    source = events()
    provider = Provider(dict(e0=False, e1=True, e2=False, e3=True, e4=None))
    result = select_candidates(source, arm='poc_priority', provider=provider, planner=Planner(occupied=('old',)))
    assert [e['event_id'] for e in result['events']] == ['e1','e3']
    assert provider.requested == ['e0','e1','e2','e3']
    cert = result['certificate']
    assert cert['complete'] and not cert['profile_ranking_complete']
    assert cert['decisions'][-1]['profile_status'] == 'not_needed_resource_exhausted'


def test_unknown_suffix_blocks_false_fallback_instead_of_becoming_false():
    source = events(3)
    provider = Provider(dict(e0=False,e1=True,e2=None))
    planner = Planner()
    with pytest.raises(UnresolvedProfile) as caught:
        select_candidates(source, arm='poc_priority', provider=provider, planner=planner)
    error = caught.value
    assert error.event_id == 'e2' and error.reason == 'missing_raw' and error.recoverable is True
    assert planner.applied == ['e1']  # private planner is discardable; false e0 never reserved
    cert = error.certificate
    assert not cert['complete'] and cert['original_ordered_ids'] == ['e0','e1','e2']
    assert cert['reserved_event_ids'] == ['e1']
    assert [r['event_id'] for r in cert['requested_profiles']] == ['e0','e1','e2']
    assert cert['failure_classification']['reason'] == 'missing_raw'
    assert cert['initial_state']['cash'] == 1000000.


def test_hard_filter_unknown_is_not_silently_dropped_and_reason_is_preserved():
    def provider(event):
        return dict(available=False,poc_up=None,reason='corporate_action_or_nonconstant_price_scale',recoverable=False)
    with pytest.raises(UnresolvedProfile) as caught:
        select_candidates(events(1), arm='poc_filter', provider=provider, planner=Planner())
    assert caught.value.recoverable is False
    assert caught.value.certificate['decisions'][0]['selection_status'] == 'unresolved'


def test_proven_unreservable_candidate_needs_no_profile():
    source = events(4)
    source[0]['amount'] = 49e6
    source[1]['price'] = 1e9
    planner = Planner(held=('1002',))
    provider = Provider(dict(e3=True))
    result = select_candidates(source, arm='poc_priority', provider=provider, planner=planner)
    assert provider.requested == ['e3']
    assert [e['event_id'] for e in result['events']] == ['e3']
    assert all(d['profile_status'] == 'not_needed_unreservable' for d in result['certificate']['decisions'][:3])


def test_sub_lot_attempt_still_consumes_slot_without_looking_at_fill():
    source = events(2)
    source[0]['price'] = 10000.  # raw_qty is 33: zero board lots, but locked attempt
    provider = Provider(dict(e0=True,e1=None))
    planner = Planner(occupied=('old1','old2'))
    result = select_candidates(source, arm='poc_priority', provider=provider, planner=planner)
    assert [e['event_id'] for e in result['events']] == ['e0']
    assert provider.requested == ['e0']
    assert 0 < result['certificate']['decisions'][0]['reservation']['raw_qty'] < 1000


def test_priority_keeps_false_fallback_while_filter_leaves_resources_unused():
    source = events(3)
    flags = dict(e0=False,e1=True,e2=False)
    ranked = select_candidates(source, arm='poc_priority', provider=Provider(flags), planner=Planner())
    filtered = select_candidates(source, arm='poc_filter', provider=Provider(flags), planner=Planner())
    assert [e['event_id'] for e in ranked['events']] == ['e1','e0','e2']
    assert [e['event_id'] for e in filtered['events']] == ['e1']
    assert filtered['certificate']['final_state']['cash'] > 600000


def test_future_outcomes_and_mutating_provider_cannot_change_input_events():
    source = events(3)
    original = deepcopy(source)
    def provider(event):
        event['priority'] = -1e9
        event['future_return'] = 100.
        return dict(available=True,poc_up=True)
    first = select_candidates(source, arm='poc_priority', provider=provider, planner=Planner())
    assert source == original
    later = deepcopy(source)
    for e in later:
        e.update(future_return=-100., future_fill=0, future_exit='2099-12-31')
    second = select_candidates(later, arm='poc_priority', provider=provider, planner=Planner())
    assert first['certificate'] == second['certificate']


def test_bad_identity_priority_and_noncausal_planner_are_rejected():
    source = events(2)
    with pytest.raises(ValueError, match='original priority'):
        select_candidates(source[::-1], arm='original', provider=None, planner=Planner())
    source[0]['priority'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        select_candidates(source, arm='original', provider=None, planner=Planner())
    class BadPlanner(Planner):
        def probe(self,event):
            self.cash -= 1
            return super().probe(event)
    with pytest.raises(ValueError, match='read-only'):
        select_candidates(events(1), arm='poc_priority', provider=Provider(dict(e0=True)), planner=BadPlanner())


def test_event_id_tie_break_and_missing_half_flag_are_explicit():
    source = events(2)
    source[1]['priority'] = source[0]['priority']
    result = select_candidates(source, arm='poc_priority', provider=Provider(dict(e0=True,e1=True)), planner=Planner())
    assert result['certificate']['reserved_event_ids'] == ['e0','e1']
    with pytest.raises(UnresolvedProfile, match='profile_flag_unavailable'):
        select_candidates(events(1), arm='poc_priority', provider=lambda _:dict(available=True,poc_up=None), planner=Planner())


@pytest.mark.parametrize('wrong_identity', [dict(event_id='other'), dict(stock_id='9999'), dict(signal_date='2024-01-03')])
def test_provider_cannot_attach_another_event_or_day_profile(wrong_identity):
    provider = lambda _:dict(available=True,poc_up=True,**wrong_identity)
    with pytest.raises(ValueError, match='differs'):
        select_candidates(events(1), arm='poc_priority', provider=provider, planner=Planner())
