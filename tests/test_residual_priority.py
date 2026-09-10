"""Synthetic residual scores only; no portfolio or historical performance runs."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.residual_priority import rescore_events


def prices(n=190, score_end=146, alpha=.0002):
    days = pd.bdate_range('2022-01-03', periods=n)
    market = .0004 + .002 * np.sin(np.arange(n - 1) * .43)
    market[score_end - 20:score_end] = .002
    high_beta = alpha + 2. * market
    residual = alpha + market
    residual[score_end - 20:score_end] += .003
    negative = alpha + market
    negative[score_end - 20:score_end] -= .001
    frame = pd.DataFrame({'0050': np.r_[100., 100. * np.cumprod(1 + market)],
                          '1101': np.r_[100., 100. * np.cumprod(1 + high_beta)],
                          '1102': np.r_[100., 100. * np.cumprod(1 + residual)],
                          '1103': np.r_[100., 100. * np.cumprod(1 + negative)]}, index=days)
    return frame


def event(close, sid='1101', day=146, identity='e', priority=7.):
    return {'event_id': identity, 'signal_date': str(close.index[day].date()),
            'entry_date': str((close.index[day] + pd.Timedelta(days=1)).date()),
            'members': [sid], 'priority': priority, 'extra': {'keep': ['metadata']}}


def test_high_beta_market_return_is_corrected_but_idiosyncratic_gain_remains():
    close = prices()
    events = [event(close), event(close, sid='1102', identity='idiosyncratic', priority=1.)]
    result = rescore_events(close, events, close.copy())
    assert not result['rejections']
    high, own = result['diagnostics']
    assert high['beta'] == pytest.approx(2.)
    assert own['beta'] == pytest.approx(1.)
    assert high['alpha'] == pytest.approx(.0002)
    assert own['alpha'] == pytest.approx(.0002)
    assert high['residual_score'] == pytest.approx(0., abs=1e-12)
    assert own['residual_score'] == pytest.approx(.003 * 20)
    assert events[0]['priority'] > events[1]['priority']
    assert result['events'][0]['priority'] < result['events'][1]['priority']


def test_windows_have_126_fit_returns_before_20_score_returns_including_signal_day():
    close = prices()
    result = rescore_events(close, [event(close)], close.copy())
    diagnostic = result['diagnostics'][0]
    assert diagnostic['fit_start'] == str(close.index[1].date())
    assert diagnostic['fit_end'] == str(close.index[126].date())
    assert diagnostic['score_start'] == str(close.index[127].date())
    assert diagnostic['score_end'] == str(close.index[146].date())
    assert diagnostic['fit_observations'] == 126 and diagnostic['score_observations'] == 20
    changed = close.copy()
    changed.loc[close.index[146]:, '1101'] *= 1.01
    score = rescore_events(changed, [event(close)], changed.copy())['diagnostics'][0]
    assert score['alpha'] == diagnostic['alpha'] and score['beta'] == diagnostic['beta']
    assert score['residual_score'] > diagnostic['residual_score']


def test_prefix_invariance_allows_execution_date_beyond_available_price_prefix():
    close = prices()
    events = [event(close)]
    complete = rescore_events(close, events, close.copy())
    prefix = close.iloc[:147]
    assert events[0]['entry_date'] > str(prefix.index[-1].date())
    assert rescore_events(prefix, events, prefix.copy()) == complete


def test_future_price_changes_and_anomalies_do_not_change_past_score_or_rejection():
    close = prices()
    other = close.copy()
    events = [event(close)]
    before = rescore_events(close, events, other)
    close.iloc[147:, 1:] = np.nan
    other.iloc[150:, 0] *= 40
    assert rescore_events(close, events, other) == before


def test_negative_score_is_preserved_without_new_positive_score_filter():
    close = prices()
    result = rescore_events(close, [event(close, sid='1103')], close.copy())
    assert not result['rejections']
    assert result['events'][0]['priority'] == pytest.approx(-.02)


def test_input_events_and_price_frames_are_not_mutated_and_nested_metadata_is_copied():
    close = prices()
    other = close.copy()
    original_close, original_other = close.copy(), other.copy()
    events = [event(close)]
    original_events = deepcopy(events)
    result = rescore_events(close, events, other)
    assert events == original_events
    assert {k: v for k, v in result['events'][0].items() if k != 'priority'} == {
        k: v for k, v in events[0].items() if k != 'priority'}
    result['events'][0]['extra']['keep'].append('new')
    assert events == original_events
    pd.testing.assert_frame_equal(close, original_close)
    pd.testing.assert_frame_equal(other, original_other)


@pytest.mark.parametrize('basis', ['primary', 'other'])
@pytest.mark.parametrize('bad', [np.nan, 0., -1., np.inf])
def test_missing_or_nonpositive_quotes_are_rejected_in_either_basis_without_fill(basis, bad):
    close = prices()
    other = close.copy()
    (close if basis == 'primary' else other).loc[close.index[5], '1101'] = bad
    result = rescore_events(close, [event(close)], other)
    assert not result['events']
    assert result['rejections'][0]['reason'] == 'missing_prices'
    assert str(close.index[5].date()) in result['diagnostics'][0]['missing_price_dates']
    assert result['diagnostics'][0]['residual_score'] is None


@pytest.mark.parametrize('basis', ['primary', 'other'])
@pytest.mark.parametrize('sid', ['0050', '1101'])
def test_anomaly_in_extra_20_day_fit_history_is_checked_in_both_assets_and_bases(basis, sid):
    close = prices()
    other = close.copy()
    target = close if basis == 'primary' else other
    target.loc[close.index[5]:, sid] *= 1.3
    result = rescore_events(close, [event(close)], other)
    assert not result['events']
    assert result['rejections'][0]['reason'] == 'price_anomaly'
    assert str(close.index[5].date()) in result['diagnostics'][0]['anomaly_dates']


def test_small_daily_return_basis_disagreement_over_50bp_rejects_without_extreme_move():
    close = prices()
    other = close.copy()
    other.loc[close.index[5]:, '1101'] *= 1.006
    result = rescore_events(close, [event(close)], other)
    assert result['rejections'][0]['reason'] == 'price_anomaly'


def test_anomalies_before_required_window_do_not_remove_valid_event():
    close = prices(n=210, score_end=180)
    other = close.copy()
    other.loc[close.index[1], '1101'] *= 5
    result = rescore_events(close, [event(close, day=180)], other)
    assert result['diagnostics'][0]['reason'] == 'scored'


def test_flat_benchmark_rejects_without_fallback_to_original_priority():
    close = prices()
    close['0050'] = 100.
    result = rescore_events(close, [event(close)], close.copy())
    assert result['events'] == []
    assert result['rejections'][0]['priority'] == 7.
    assert result['rejections'][0]['reason'] == 'benchmark_variance_too_small'
    assert result['diagnostics'][0]['market_fit_variance'] == 0.
    assert result['diagnostics'][0]['beta'] is None


def test_variance_gate_uses_population_variance_not_centered_sum_of_squares():
    close = prices()
    market = np.array([-.5e-6, .5e-6] * 95)[:len(close) - 1]
    close['0050'] = np.r_[100., 100 * np.cumprod(1 + market)]
    result = rescore_events(close, [event(close)], close.copy())
    diagnostic = result['diagnostics'][0]
    assert 0 < diagnostic['market_fit_variance'] < 1e-12
    assert 126 * diagnostic['market_fit_variance'] > 1e-12
    assert diagnostic['reason'] == 'benchmark_variance_too_small'


def test_short_history_missing_signal_and_unknown_stock_are_explicit_rejections():
    close = prices()
    missing_signal = event(close, identity='missing_signal')
    missing_signal.update(signal_date='2022-07-30', entry_date='2022-08-01')
    events = [event(close, day=145, identity='short'), missing_signal,
              event(close, sid='9999', identity='missing_stock')]
    result = rescore_events(close, events, close.copy())
    assert [row['reason'] for row in result['rejections']] == [
        'insufficient_history', 'signal_date_not_trading_session', 'stock_not_in_prices']
    assert result['diagnostics'][0]['available_prices'] == 146


@pytest.mark.parametrize('mutate', [
    lambda e: e.__setitem__('members', ['0050']),
    lambda e: e.__setitem__('members', ['1101', '1102']),
    lambda e: e.__setitem__('members', ['12345']),
    lambda e: e.__setitem__('members', [1101]),
    lambda e: e.__setitem__('entry_date', e['signal_date']),
    lambda e: e.__setitem__('priority', float('nan')),
    lambda e: e.__setitem__('event_id', ''),
    lambda e: e.__setitem__('signal_date', e['signal_date'] + 'T01:00:00'),
])
def test_malformed_event_inputs_raise(mutate):
    close = prices()
    supplied = event(close)
    mutate(supplied)
    with pytest.raises(ValueError):
        rescore_events(close, [supplied], close.copy())


def test_duplicate_ids_unsorted_index_and_other_basis_misalignment_raise():
    close = prices()
    with pytest.raises(ValueError, match='unique'):
        rescore_events(close, [event(close), event(close)], close.copy())
    with pytest.raises(ValueError, match='ordered unique'):
        rescore_events(close.iloc[::-1], [], close.iloc[::-1])
    with pytest.raises(ValueError, match='identical'):
        rescore_events(close, [], close[close.columns[::-1]])


def test_rejected_copy_does_not_share_mutable_metadata_with_input():
    close = prices()
    supplied = event(close, day=145)
    result = rescore_events(close, [supplied], close.copy())
    result['rejections'][0]['extra']['keep'].append('changed')
    assert supplied['extra']['keep'] == ['metadata']
