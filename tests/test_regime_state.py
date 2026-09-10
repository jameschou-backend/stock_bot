"""Causal market-state decisions using small independent price sequences."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.regime_state import build_trend, execution_controls, gate_events


def prices(periods=128):
    return pd.Series(np.arange(periods, dtype=float) + 100,
                     index=pd.bdate_range('2021-01-01', periods=periods))


def event(trend, signal=120, entry=None, **kwargs):
    return {'event_id': 'a', 'signal_date': str(trend.index[signal].date()),
            'entry_date': str(trend.index[signal + 1 if entry is None else entry].date()),
            'members': ['2330', '1101'], 'priority': .137, **kwargs}


def test_full_observation_count_and_strict_mean_comparison():
    close = prices()
    trend = build_trend(close)
    assert trend.index.equals(close.index)
    assert set(trend.iloc[:119]['state']) == {'UNKNOWN'}
    assert trend.iloc[118]['observation_count'] == 119
    assert trend.iloc[119]['observation_count'] == 120
    assert trend.iloc[-1]['observation_count'] == 120
    assert trend.iloc[119]['ma120'] == pytest.approx(close.iloc[:120].mean())
    assert trend.iloc[119]['state'] == 'ON'
    assert trend.iloc[119]['evidence_date'] == str(close.index[119].date())
    assert trend.iloc[118]['evidence_date'] is None
    equal = build_trend(pd.Series(100., index=close.index))
    assert equal.iloc[119]['state'] == 'OFF'  # Equality is never ON.
    falling = build_trend(close.iloc[::-1].set_axis(close.index))
    assert falling.iloc[119]['state'] == 'OFF'


def test_missing_prices_do_not_consume_observations_or_become_on():
    close = prices(140)
    close.iloc[[2, 4, 6, 8, 10]] = [np.nan, 0, -1, np.inf, -np.inf]
    close.iloc[128] = np.nan
    trend = build_trend(close)
    assert trend.iloc[123]['observation_count'] == 119
    assert trend.iloc[123]['state'] == 'UNKNOWN'
    assert trend.iloc[124]['state'] == 'ON'
    assert trend.iloc[128]['state'] == 'UNKNOWN'
    assert trend.iloc[128]['evidence_date'] is None
    expected = close.iloc[:130].where(np.isfinite(close.iloc[:130]) & close.iloc[:130].gt(0)).dropna().iloc[-120:].mean()
    assert trend.iloc[129]['ma120'] == pytest.approx(expected)
    assert trend.iloc[129]['state'] == 'ON'
    assert trend.loc[close.index[[2, 4, 6, 8, 10]], 'close'].isna().all()


def test_zero_observations_and_empty_calendar():
    close = prices(8) * np.nan
    trend = build_trend(close)
    assert trend['observation_count'].eq(0).all()
    assert trend['state'].eq('UNKNOWN').all()
    assert trend['ma120'].isna().all()
    empty = build_trend(close.iloc[:0])
    assert empty.empty
    assert execution_controls(empty).empty
    assert gate_events([], empty) == ([], [])


def test_prefix_and_future_price_changes_cannot_rewrite_states_or_controls():
    close = prices(160)
    close.iloc[126] = np.nan
    original = build_trend(close)
    prefix = build_trend(close.iloc[:133])
    pd.testing.assert_frame_equal(prefix, original.iloc[:133])
    changed = close.copy()
    changed.iloc[133:] = 1.
    mutated = build_trend(changed)
    pd.testing.assert_frame_equal(original.iloc[:133], mutated.iloc[:133])
    pd.testing.assert_frame_equal(execution_controls(original).iloc[:134],
                                  execution_controls(mutated).iloc[:134])
    pd.testing.assert_frame_equal(execution_controls(prefix), execution_controls(original).iloc[:133])


@pytest.mark.parametrize('delay', [1, 2])
def test_controls_shift_market_rows_and_preserve_unknown(delay):
    close = prices()
    close.iloc[121] = np.nan
    trend = build_trend(close)
    controls = execution_controls(trend, delay=delay)
    assert controls.iloc[:delay]['state'].eq('UNKNOWN').all()
    assert all(value is None for value in controls.iloc[:delay]['decision_date'])
    assert controls.iloc[120 + delay]['state'] == 'ON'
    assert controls.iloc[121 + delay]['state'] == 'UNKNOWN'
    assert controls.iloc[121 + delay]['decision_date'] == str(close.index[121].date())
    assert controls.iloc[122 + delay]['state'] == 'ON'
    for day, decision in controls['decision_date'].items():
        assert decision is None or pd.Timestamp(decision) < day


@pytest.mark.parametrize('bad', [0, -1, 1.5, True, '1'])
def test_invalid_execution_delay(bad):
    with pytest.raises(ValueError, match='delay'):
        execution_controls(build_trend(prices()), delay=bad)


@pytest.mark.parametrize('index', [pd.to_datetime(['2021-01-02', '2021-01-01']),
                                  pd.to_datetime(['2021-01-01', '2021-01-01']),
                                  pd.to_datetime(['2021-01-01', None]),
                                  pd.to_datetime(['2021-01-01 01:00', '2021-01-02 00:00']),
                                  pd.date_range('2021-01-01', periods=2, tz='Asia/Taipei'),
                                  pd.Index(['2021-01-01', '2021-01-02'])])
def test_invalid_calendars(index):
    with pytest.raises(ValueError, match='DatetimeIndex'):
        build_trend(pd.Series([1., 2.], index=index))


def test_non_numeric_prices_and_future_dated_evidence_fail_explicitly():
    with pytest.raises(ValueError, match='numeric prices'):
        build_trend(pd.Series(['bad'], index=pd.date_range('2021-01-01', periods=1)))
    trend = build_trend(prices())
    trend.loc[trend.index[120], 'evidence_date'] = str(trend.index[121].date())
    with pytest.raises(ValueError, match='own date'):
        execution_controls(trend)
    with pytest.raises(ValueError, match='own date'):
        gate_events([event(trend)], trend)


def test_gate_checks_original_signal_and_does_not_rescore_on_entry_day():
    close = prices()
    close.iloc[121:] = 1.
    trend = build_trend(close)
    supplied = event(trend)
    original = deepcopy(supplied)
    accepted, rejected = gate_events([supplied], trend, extra_entry_delay=1)
    assert rejected == []
    assert len(accepted) == 1
    assert trend.iloc[120]['state'] == 'ON'
    assert trend.iloc[121]['state'] == 'OFF'
    assert trend.iloc[122]['state'] == 'OFF'
    assert accepted[0]['entry_date'] == str(trend.index[122].date())
    assert accepted[0]['signal_date'] == original['signal_date']
    assert accepted[0]['priority'] == original['priority']
    assert accepted[0]['members'] == original['members']
    assert accepted[0]['trend_state'] == 'ON'
    assert supplied == original
    accepted[0]['members'].append('2454')
    assert supplied == original


def test_gate_rejects_unknown_and_off_and_keeps_event_order():
    close = prices()
    close.iloc[121] = np.nan
    close.iloc[123:] = 1.
    trend = build_trend(close)
    inputs = [event(trend, signal=i, event_id=str(i)) for i in (120, 121, 118, 124)]
    accepted, rejected = gate_events(inputs, trend)
    assert [row['event_id'] for row in accepted] == ['120']
    assert [row['event_id'] for row in rejected] == ['121', '118', '124']
    assert [row['reason'] for row in rejected] == ['trend_unknown', 'trend_unknown', 'trend_off']


@pytest.mark.parametrize('change,reason', [
    ({'signal_date': 'invalid'}, 'invalid_signal_date'),
    ({'signal_date': None}, 'invalid_signal_date'),
    ({'signal_date': '2021-01-01T01:00:00'}, 'invalid_signal_date'),
    ({'signal_date': '2021-01-01T00:00:00+08:00'}, 'invalid_signal_date'),
    ({'signal_date': '2020-12-31'}, 'signal_date_outside_calendar'),
    ({'signal_date': '2029-01-01'}, 'signal_date_outside_calendar'),
    ({'signal_date': '2021-01-02'}, 'signal_date_not_trading_session'),
    ({'entry_date': None}, 'invalid_entry_date'),
    ({'entry_date': '2021-06-18'}, 'entry_date_not_after_signal'),
    ({'entry_date': '2021-06-17'}, 'entry_date_not_after_signal'),
    ({'entry_date': '2021-06-19'}, 'entry_date_not_trading_session'),
    ({'entry_date': '2029-01-01'}, 'entry_date_outside_calendar'),
    ({'entry_date': '2021-06-22'}, 'entry_date_not_next_session'),
])
def test_gate_date_rejection_ledger(change, reason):
    trend = build_trend(prices())
    supplied = event(trend, **change)
    assert supplied['signal_date'] != supplied['entry_date'] or reason == 'entry_date_not_after_signal'
    accepted, rejected = gate_events([supplied], trend)
    assert accepted == []
    assert rejected[0]['reason'] == reason


def test_gate_does_not_invent_a_later_entry_outside_calendar():
    trend = build_trend(prices())
    accepted, rejected = gate_events([event(trend, signal=126)], trend, extra_entry_delay=1)
    assert accepted == []
    assert rejected[0]['reason'] == 'entry_delay_outside_calendar'


@pytest.mark.parametrize('bad', [-1, 2, True, .5, '1'])
def test_gate_delay_is_exactly_preregistered(bad):
    with pytest.raises(ValueError, match='extra_entry_delay'):
        gate_events([], build_trend(prices()), extra_entry_delay=bad)


def test_gate_prefix_and_future_mutation_leave_earlier_acceptance_unchanged():
    close = prices(150)
    trend = build_trend(close)
    inputs = [event(trend, signal=120), event(trend, signal=122, event_id='b')]
    original = gate_events(inputs, trend, extra_entry_delay=1)
    prefix = build_trend(close.iloc[:126])
    changed = close.copy()
    changed.iloc[126:] = np.nan
    assert gate_events(inputs, prefix, extra_entry_delay=1) == original
    assert gate_events(inputs, build_trend(changed), extra_entry_delay=1) == original
