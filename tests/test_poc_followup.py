from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.poc_followup import (
    ConfirmedBlack, CorrelationPlanner, confirmed_black_auditor,
    pair_correlation, strength_order,
)
from skills.three_black_exit import ThreeBlackSignals


def test_confirmation_needs_four_post_entry_candles_and_independent_audit():
    days = pd.date_range('2024-01-01', periods=7)
    close = pd.DataFrame({'1234': [100, 99, 98, 97, 96, 95, 94]}, index=days)
    quotes = pd.DataFrame([dict(date=d, stock_id='1234', open=p+1,
        high=p+2, low=p-1, close=p, volume=100000) for d, p in close['1234'].items()])
    source = ThreeBlackSignals(close, quotes, days)
    confirmed = ConfirmedBlack(source)
    assert source.exits(4, 1, '1234')
    assert not confirmed.exits(4, 1, '1234')
    assert confirmed.exits(5, 1, '1234')
    account = dict(cohorts=[dict(event_id='e', stock_id='1234', entry_date='2024-01-02')],
        black_log=[dict(event_id='e', stock_id='1234', date='2024-01-05',
                        signal_date='2024-01-04', trigger=False),
                   dict(event_id='e', stock_id='1234', date='2024-01-06',
                        signal_date='2024-01-05', trigger=True)], trades=[])
    auditor = confirmed_black_auditor()
    assert auditor(account, source)['exits'] == 1
    bad = deepcopy(account); bad['black_log'][0]['trigger'] = True
    with pytest.raises(ValueError, match='differs from historical'):
        auditor(bad, source)


def test_correlation_does_not_read_future_prices_or_fill_missing_history():
    rng = np.random.default_rng(9)
    x = rng.normal(0, .02, 80)
    days = pd.date_range('2024-01-01', periods=80)
    close = pd.DataFrame({'1111': 100*np.cumprod(1+x),
                          '2222': 50*np.cumprod(1+x)}, index=days)
    assert pair_correlation(close, '1111', '2222', days[60]) == pytest.approx(1)
    close.loc[days[61]:, '2222'] = -1e10
    assert pair_correlation(close, '1111', '2222', days[60]) == pytest.approx(1)
    assert pair_correlation(close, '1111', '2222', days[59]) is None
    close.loc[days[30], '2222'] = np.nan
    assert pair_correlation(close, '1111', '2222', days[60]) is None


def test_strength_is_continuous_and_preserves_original_ties():
    events = [dict(event_id=e, members=[e], signal_date='2024-04-01', priority=p)
              for e, p in [('a', 9), ('b', 8), ('c', 7)]]
    profiles = {'a': dict(available=True, poc_before=100, poc_after=101),
                'b': dict(available=True, poc_before=10, poc_after=12),
                'c': dict(available=True, poc_before=20, poc_after=24)}
    planner = SimpleNamespace(probe=lambda e: {'attempted': True})
    rows, evidence, _ = strength_order(events, lambda e: profiles[e['event_id']], planner)
    assert [r['event_id'] for r in rows] == ['b', 'c', 'a']
    assert [r['priority'] for r in events] == [9, 8, 7]
    profiles['b'] = dict(available=False, reason='ordinary_tape_conflict')
    rows, evidence, _ = strength_order(events, lambda e: profiles[e['event_id']], planner)
    assert rows == events and evidence['applied'] is False


def test_correlation_rejection_consumes_no_cash():
    days = pd.date_range('2024-01-01', periods=61)
    changes = np.sin(np.arange(61)) * .02
    close = pd.DataFrame({'1111': np.cumprod(1+changes),
                          '2222': 2*np.cumprod(1+changes)}, index=days)
    class Planner:
        day = days[-1] + pd.Timedelta(days=1)
        engine = SimpleNamespace(exit_signals=SimpleNamespace(adjusted_close=close))
        def snapshot(self):
            return dict(held=['1111'], occupied=['1111'], cash=100000)
        def probe(self, event):
            return dict(attempted=True, failure=None, allocation=100000)
        def consider(self, event):
            raise AssertionError('Rejected candidate consumed cash')
    planner = CorrelationPlanner(Planner(), [])
    e = dict(event_id='e', members=['2222'], signal_date=str(days[-1].date()))
    before = planner.snapshot()
    assert planner.probe(e)['failure'] == 'correlation_above_080'
    assert not planner.consider(e)['attempted']
    assert planner.snapshot() == before


def test_strength_accepts_sealed_nested_and_flat_schemas_without_changing_profile():
    from skills.poc_followup_profile_schema import strength_order, strength_inputs
    nested = dict(available=True, poc_up=True, profile=dict(available=True,
        first_half=dict(available=True, poc_price=100),
        second_half=dict(available=True, poc_price=120)))
    flat = dict(available=True, poc_up=True, poc_before=100, poc_after=101)
    events = [dict(event_id=e, members=[e], signal_date='2024-04-01', priority=p)
              for e, p in [('a', 9), ('b', 8)]]
    profiles = {'a': flat, 'b': nested}
    rows, evidence, cached = strength_order(events,
        lambda e: profiles[e['event_id']], SimpleNamespace(probe=lambda e: {'attempted': True}))
    assert [r['event_id'] for r in rows] == ['b', 'a']
    assert cached == profiles
    assert 'poc_before' not in profiles['b']
    assert evidence['profile_input_formats']['b'] == ['nested_half_poc_prices']
    with pytest.raises(ValueError, match='Conflicting'):
        strength_inputs(dict(nested, poc_before=100, poc_after=110))
    malformed = deepcopy(nested); malformed['profile']['first_half']['available'] = False
    with pytest.raises(ValueError, match='unavailable half'):
        strength_inputs(malformed)
