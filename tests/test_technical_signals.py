"""Exact market-row windows and causal technical signals, without a provider."""
import numpy as np
import pandas as pd
import pytest

from skills.scenario_exit_replay import ExitSignals
from skills.technical_signals import TechnicalSignals


def fixture(n=80):
    days = pd.bdate_range('2021-01-04', periods=n)
    adjusted = pd.DataFrame({'0050': 100., '1101': 100.}, index=days)
    quotes = pd.DataFrame([
        dict(date=day, stock_id=sid, open=100., high=110., low=90., close=100., volume=100.)
        for day in days for sid in adjusted
    ])
    return adjusted, quotes, days


def bar(quotes, days, index, **values):
    mask = quotes.date.eq(days[index]) & quotes.stock_id.eq('1101')
    for name, value in values.items():
        quotes.loc[mask, name] = value


def pattern_fixture():
    adjusted, quotes, days = fixture()
    for index in range(10):
        bar(quotes, days, index, high=120., low=80.)
    bar(quotes, days, 20, open=121., high=122., low=120., close=121., volume=150.)
    adjusted.at[days[20], '1101'] = 121.
    return adjusted, quotes, days


def test_subclass_preserves_exit_context_contract_and_input_objects():
    adjusted, quotes, days = fixture()
    saved_adjusted, saved_quotes = adjusted.copy(deep=True), quotes.copy(deep=True)
    features = TechnicalSignals(adjusted, quotes, days)
    state = dict(entry_index=21, entry_price=100., peak_price=100.)
    assert isinstance(features, ExitSignals)
    assert features.context(25, '1101', state.copy()) == ExitSignals(adjusted, days).context(25, '1101', state.copy())
    pd.testing.assert_frame_equal(adjusted, saved_adjusted)
    pd.testing.assert_frame_equal(quotes, saved_quotes)


def test_support_requires_exact_twenty_prior_rows_and_excludes_signal_bar():
    adjusted, quotes, days = fixture()
    bar(quotes, days, 20, open=60., high=70., low=50., close=60.)
    adjusted.at[days[20], '1101'] = 60.
    features = TechnicalSignals(adjusted, quotes, days)
    before = features.technical_context(20, '1101')
    assert before['signal_index'] == 19
    assert before['support_available'] is False
    assert before['valid_history20'] == 19
    ctx = features.technical_context(21, '1101')
    assert ctx['signal_date'] == str(days[20].date())
    assert ctx['support20'] == 90.
    assert ctx['support_failure'] is True
    assert ctx['risk_available'] is False
    assert 'support_distance_not_positive' in ctx['diagnostics']
    assert features.technical_context(22, '1101')['support20'] == 50.


def test_support_equality_is_not_failure_or_positive_entry_risk():
    adjusted, quotes, days = fixture()
    bar(quotes, days, 20, open=90., low=90., close=90.)
    adjusted.at[days[20], '1101'] = 90.
    ctx = TechnicalSignals(adjusted, quotes, days).entry_context(21, '1101')
    assert ctx['support_failure'] is False
    assert ctx['risk_fraction'] is None
    assert ctx['risk_available'] is False


def test_first_execution_row_is_unknown_never_last_row():
    adjusted, quotes, days = fixture()
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(0, '1101')
    assert ctx['signal_index'] == -1
    assert ctx['signal_date'] is None
    assert ctx['adjusted_close'] is None
    assert ctx['support20'] is None
    assert ctx['pattern_pass'] is None
    assert ctx['diagnostics'] == ['no_prior_market_session']


def test_positive_entry_distance_and_raw_support_use_signal_day_factor():
    adjusted, quotes, days = fixture()
    adjusted['1101'] *= .5
    ctx = TechnicalSignals(adjusted, quotes, days).entry_context(21, '1101')
    assert ctx['adjusted_close'] == 50.
    assert ctx['support20'] == 45.
    assert ctx['support_raw'] == 90.
    assert ctx['raw_close'] == 100.
    assert ctx['adjustment_factor'] == .5
    assert ctx['risk_fraction'] == pytest.approx(.10)
    assert ctx['risk_available'] is True


def test_pattern_exact_disjoint_ten_session_ranges_breakout_and_volume():
    adjusted, quotes, days = pattern_fixture()
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    assert ctx['resistance20'] == 120.
    assert ctx['range_preceding10'] == 40.
    assert ctx['range_recent10'] == 20.
    assert ctx['volume20'] == 100.
    assert ctx['breakout20'] is True
    assert ctx['contraction10'] is True
    assert ctx['volume_expansion'] is True
    assert ctx['pattern_available'] is True
    assert ctx['pattern_pass'] is True
    assert ctx['diagnostics'] == []


@pytest.mark.parametrize('component', ['breakout', 'contraction', 'volume'])
def test_pattern_rejects_each_independent_failed_condition(component):
    adjusted, quotes, days = pattern_fixture()
    if component == 'breakout':
        adjusted.at[days[20], '1101'] = 120.
        bar(quotes, days, 20, open=120., low=120., close=120.)
    elif component == 'contraction':
        bar(quotes, days, 10, high=119., low=81.)
    else:
        bar(quotes, days, 20, volume=149.)
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    assert ctx['pattern_available'] is True
    assert ctx['pattern_pass'] is False
    assert ctx[{'breakout': 'breakout20', 'contraction': 'contraction10', 'volume': 'volume_expansion'}[component]] is False


def test_contraction_boundary_is_inclusive_and_only_prior_rows_count():
    adjusted, quotes, days = pattern_fixture()
    bar(quotes, days, 10, high=115., low=85.)
    bar(quotes, days, 20, high=1000., low=1.)
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    assert ctx['range_recent10'] == 30.
    assert ctx['range_preceding10'] == 40.
    assert ctx['contraction10'] is True
    assert ctx['pattern_pass'] is True


def test_zero_preceding_range_cannot_establish_contraction():
    adjusted, quotes, days = fixture()
    for index in range(20):
        bar(quotes, days, index, high=100., low=100.)
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    assert ctx['range_preceding10'] == 0.
    assert ctx['range_recent10'] == 0.
    assert ctx['contraction10'] is None
    assert ctx['pattern_pass'] is None
    assert 'preceding_range_not_positive' in ctx['diagnostics']


@pytest.mark.parametrize('missing', ['row', 'raw_low', 'adjusted_close', 'invalid_high'])
def test_one_missing_or_invalid_historical_bar_invalidates_exact_window(missing):
    adjusted, quotes, days = pattern_fixture()
    if missing == 'row':
        quotes = quotes.loc[~(quotes.date.eq(days[10]) & quotes.stock_id.eq('1101'))]
    elif missing == 'raw_low':
        bar(quotes, days, 10, low=np.nan)
    elif missing == 'adjusted_close':
        adjusted.at[days[10], '1101'] = np.nan
    else:
        bar(quotes, days, 10, high=99.)
    features = TechnicalSignals(adjusted, quotes, days)
    ctx = features.technical_context(21, '1101')
    assert ctx['valid_history20'] == 19
    assert ctx['support20'] is None
    assert ctx['support_failure'] is None
    assert ctx['risk_fraction'] is None
    assert ctx['pattern_available'] is False
    assert ctx['pattern_pass'] is None
    assert 'support_history_incomplete' in ctx['diagnostics']
    assert 'volume_history_incomplete' in ctx['diagnostics']
    assert pd.isna(features.adjusted_low.at[days[10], '1101'])


@pytest.mark.parametrize('values,diagnostic', [
    ({'high': np.nan}, 'raw_ohlc_missing'),
    ({'low': 101.}, 'raw_ohlc_invalid'),
    ({'open': 111.}, 'raw_ohlc_invalid'),
    ({'close': 0.}, 'raw_ohlc_invalid'),
    ({'high': np.inf}, 'raw_ohlc_invalid'),
])
def test_invalid_current_ohlc_is_diagnosed_and_cannot_generate_signal(values, diagnostic):
    adjusted, quotes, days = fixture()
    bar(quotes, days, 20, **values)
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    assert diagnostic in ctx['diagnostics']
    assert ctx['support20'] == 90.
    assert ctx['support_available'] is False
    assert ctx['support_failure'] is None
    assert ctx['pattern_pass'] is None


def test_missing_current_bar_is_unknown_despite_present_adjusted_close():
    adjusted, quotes, days = fixture()
    quotes = quotes.loc[~(quotes.date.eq(days[20]) & quotes.stock_id.eq('1101'))]
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    assert ctx['adjusted_close'] == 100.
    assert ctx['raw_close'] is None
    assert ctx['support_available'] is False
    assert 'raw_bar_missing' in ctx['diagnostics']


@pytest.mark.parametrize('volume', [np.nan, -1., np.inf])
def test_invalid_current_volume_does_not_erase_support_but_pattern_is_unknown(volume):
    adjusted, quotes, days = fixture()
    bar(quotes, days, 20, volume=volume)
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    assert ctx['support_available'] is True
    assert ctx['volume_expansion'] is None
    assert ctx['pattern_pass'] is None
    assert 'signal_volume_missing_or_invalid' in ctx['diagnostics']


def test_known_zero_volume_is_not_missing_and_zero_baseline_is_not_expansion():
    adjusted, quotes, days = fixture()
    quotes.loc[quotes.stock_id.eq('1101'), 'volume'] = 0.
    ctx = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    assert ctx['volume20'] == 0.
    assert ctx['volume_expansion'] is None
    assert 'volume_baseline_not_positive' in ctx['diagnostics']
    assert 'volume_history_incomplete' not in ctx['diagnostics']


def test_no_forward_fill_and_window_recovers_only_after_missing_bar_expires():
    adjusted, quotes, days = fixture()
    adjusted.at[days[20], '1101'] = np.nan
    features = TechnicalSignals(adjusted, quotes, days)
    assert pd.isna(features.adjustment_factor.at[days[20], '1101'])
    assert pd.isna(features.adjusted_low.at[days[20], '1101'])
    assert features.technical_context(21, '1101')['support_available'] is False
    assert features.technical_context(41, '1101')['support_available'] is False
    assert features.technical_context(42, '1101')['support_available'] is True


def test_split_adjusted_high_low_share_a_price_scale_and_raw_stop_is_converted_back():
    adjusted, quotes, days = fixture()
    split = quotes.date.ge(days[10]) & quotes.stock_id.eq('1101')
    quotes.loc[split, ['open', 'high', 'low', 'close']] *= .5
    features = TechnicalSignals(adjusted, quotes, days)
    assert features.adjusted_low['1101'].eq(90.).all()
    assert features.adjusted_high['1101'].eq(110.).all()
    ctx = features.technical_context(21, '1101')
    assert ctx['support20'] == 90.
    assert ctx['support_raw'] == 45.
    assert ctx['risk_fraction'] == pytest.approx(.1)
    assert ctx['support_failure'] is False


def test_common_adjustment_rescaling_preserves_pattern_support_failure_and_risk():
    adjusted, quotes, days = pattern_fixture()
    first = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    adjusted['1101'] *= .25
    second = TechnicalSignals(adjusted, quotes, days).technical_context(21, '1101')
    for key in ('support_raw', 'risk_fraction', 'support_failure', 'breakout20', 'contraction10', 'volume_expansion', 'pattern_pass'):
        assert first[key] == second[key]
    assert second['support20'] == first['support20'] * .25


def test_mutating_execution_day_and_all_future_data_cannot_change_any_prior_context():
    adjusted, quotes, days = pattern_fixture()
    first = TechnicalSignals(adjusted, quotes, days)
    cutoff = 21
    adjusted.loc[days[cutoff]:] *= 37.
    affected = quotes.date.ge(days[cutoff])
    quotes.loc[affected, ['open', 'high', 'low', 'close']] *= 13.
    quotes.loc[affected, 'volume'] *= 1000.
    second = TechnicalSignals(adjusted, quotes, days)
    for index in range(cutoff + 1):
        for sid in ('0050', '1101'):
            assert first.technical_context(index, sid) == second.technical_context(index, sid)


@pytest.mark.parametrize('index', [-1, 80, 1.5, True])
def test_invalid_execution_index_fails_explicitly(index):
    adjusted, quotes, days = fixture()
    with pytest.raises(ValueError, match='Execution index'):
        TechnicalSignals(adjusted, quotes, days).technical_context(index, '1101')


def test_unknown_stock_id_fails_explicitly():
    adjusted, quotes, days = fixture()
    with pytest.raises(ValueError, match='missing from adjusted matrix'):
        TechnicalSignals(adjusted, quotes, days).technical_context(21, '9999')


@pytest.mark.parametrize('problem,match', [
    ('missing_column', 'Missing raw quote columns'),
    ('duplicate', 'Duplicate raw technical bars'),
    ('nonstring_stock', 'Raw stock IDs must be strings'),
    ('boolean_price', 'Raw OHLC and volume must be real'),
    ('string_volume', 'Raw OHLC and volume must be real'),
    ('complex_price', 'Raw OHLC and volume must be real'),
    ('timezone', 'valid naive market dates'),
    ('intraday', 'valid naive market dates'),
    ('missing_date', 'valid naive market dates'),
])
def test_ambiguous_schema_fails_explicitly(problem, match):
    adjusted, quotes, days = fixture()
    if problem == 'missing_column':
        quotes = quotes.drop(columns='low')
    elif problem == 'duplicate':
        quotes = pd.concat([quotes, quotes.iloc[[0]]], ignore_index=True)
    elif problem == 'nonstring_stock':
        quotes['stock_id'] = 1101
    elif problem == 'boolean_price':
        quotes['open'] = True
    elif problem == 'string_volume':
        quotes['volume'] = '100'
    elif problem == 'complex_price':
        quotes['open'] = 100 + 0j
    elif problem == 'timezone':
        quotes['date'] = quotes.date.dt.tz_localize('Asia/Taipei')
    elif problem == 'intraday':
        quotes['date'] += pd.Timedelta(hours=1)
    else:
        quotes.loc[0, 'date'] = pd.NaT
    with pytest.raises(ValueError, match=match):
        TechnicalSignals(adjusted, quotes, days)
