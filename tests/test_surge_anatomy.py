from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from skills.surge_anatomy import features, cohort_table, causal_check


def inputs():
    days = pd.bdate_range('2021-01-04', periods=200)
    close = pd.DataFrame(100., index=days, columns=['0050', '1101', '1102'])
    close.loc[days[131:152], '1101'] = np.linspace(100, 140, 21)
    close.loc[days[152:], '1101'] = 140.
    volume = pd.DataFrame(1000000., index=days, columns=close.columns)
    companies = pd.DataFrame([dict(stock_id=sid, name=sid, market='TWSE', industry='01',
                                    listed_date=days[0]) for sid in ('1101', '1102')])
    return close, close.copy(), close.copy(), volume, companies


def test_only_future_label_changes_when_future_prices_change():
    close, quality, raw, volume, companies = inputs()
    day = close.index[130]
    before = features(close, raw, volume, companies)
    changed = close.copy()
    changed.loc[changed.index > day, '1101'] = 100.
    after = features(changed, raw, volume, companies)
    for key in before['rules']:
        pd.testing.assert_frame_equal(before['rules'][key].loc[:day], after['rules'][key].loc[:day])
    original, _ = cohort_table(close, quality, raw, volume, companies, start=str(day.date()))
    altered, _ = cohort_table(changed, changed, raw, volume, companies, start=str(day.date()))
    assert original.iloc[0]['event'] is True or original.iloc[0]['event'] == True
    assert altered.iloc[0]['event'] is False or altered.iloc[0]['event'] == False
    assert original.iloc[0]['forward_return'] == pytest.approx(.4)
    assert original.iloc[0]['entry_date'] == str(close.index[131].date())
    assert original.iloc[0]['exit_date'] == str(close.index[151].date())


@pytest.mark.parametrize('kind', ['missing', 'nonfinite', 'disagree', 'jump'])
def test_future_data_defects_stay_unknown_without_changing_candidates(kind):
    close, quality, raw, volume, companies = inputs()
    original, _ = cohort_table(close, quality, raw, volume, companies, start=str(close.index[130].date()))
    if kind == 'missing': quality.iloc[140, quality.columns.get_loc('1101')] = np.nan
    if kind == 'nonfinite': quality.iloc[140, quality.columns.get_loc('1101')] = np.inf
    if kind == 'disagree': quality.loc[quality.index[131:152], '1101'] = np.linspace(100, 115, 21)
    if kind == 'jump': quality.iloc[140, quality.columns.get_loc('1101')] *= 2
    actual, _ = cohort_table(close, quality, raw, volume, companies, start=str(close.index[130].date()))
    assert actual[['signal_date', 'stock_id']].equals(original[['signal_date', 'stock_id']])
    assert actual.iloc[0]['event'] is None and actual.iloc[0]['label_reason']


def test_low_volume_and_not_yet_listed_are_not_eligible():
    close, quality, raw, volume, companies = inputs()
    volume['1101'] = 10
    companies.loc[companies.stock_id.eq('1102'), 'listed_date'] = close.index[180]
    result = features(close, raw, volume, companies)
    assert not result['eligible'].iloc[130].any()


def test_missing_historical_session_is_not_filled_forward():
    close, _, raw, volume, companies = inputs()
    close.iloc[100, close.columns.get_loc('1101')] = np.nan
    result = features(close, raw, volume, companies)
    assert not result['eligible'].iloc[130]['1101']


def test_immature_anchors_are_counted_but_not_labelled_negative():
    args = inputs()
    table, coverage = cohort_table(*args, start=str(args[0].index[130].date()))
    assert len(coverage[~coverage.label_mature]) == 1
    assert coverage.iloc[-1]['eligible'] == 2
    assert table.signal_date.max() < coverage.iloc[-1]['signal_date']


def test_calendar_boundary_uses_label_end_and_does_not_leak_2025():
    close, quality, raw, volume, companies = inputs()
    new_index = pd.bdate_range('2024-06-03', periods=len(close))
    for frame in (close, quality, raw, volume): frame.index = new_index
    day = new_index[new_index <= '2024-12-20'][-1]
    table, _ = cohort_table(close, quality, raw, volume, companies, start=str(day.date()))
    assert set(table.loc[table.signal_date.eq(str(day.date())), 'phase']) == {'boundary'}
    assert set(table.loc[table.signal_date.ge('2025-01-01'), 'phase']) == {'replication'}


def test_both_causality_transformations_and_alignment_guard():
    close, _, raw, volume, companies = inputs()
    result = causal_check(close, raw, volume, companies, [str(close.index[130].date())])
    assert len(result) == 2 and all(row['passed'] for row in result)
    with pytest.raises(ValueError, match='aligned'):
        features(close, raw.iloc[:-1], volume, companies)


@pytest.mark.parametrize('value', [np.nan, np.inf, 0.])
def test_missing_benchmark_is_unknown_prediction_not_negative(value):
    close, _, raw, volume, companies = inputs()
    day = close.index[160]
    close.loc[close.index[140], '0050'] = value
    result = features(close, raw, volume, companies)
    assert result['eligible'].loc[day, '1101']
    assert pd.isna(result['rules']['relative_strength'].loc[day, '1101'])
    assert pd.isna(result['rules']['C_strength_pullback'].loc[day, '1101'])
    # An observed failing volume/breakout condition determines an AND even
    # when relative strength is unavailable (nullable three-valued logic).
    assert not result['rules']['A_breakout_volume_strength'].loc[day, '1101']


def test_flat_price_and_unavailable_prior_volume_preserve_unknown():
    close, _, raw, volume, companies = inputs()
    day = close.index[130]
    volume.loc[close.index[110], '1102'] = 0.
    result = features(close, raw, volume, companies)
    assert result['eligible'].loc[day, '1102']
    assert pd.isna(result['rules']['compression'].loc[day, '1102'])
    assert pd.isna(result['rules']['volume_expansion'].loc[day, '1102'])
    table, _ = cohort_table(close, close, raw, volume, companies, start=str(day.date()))
    from skills.surge_statistics import rule_statistics
    rows = rule_statistics(table, ['compression', 'volume_expansion'])
    stats = next(row for row in rows if row['scope'] == 'discovery' and row['rule'] == 'compression')
    assert stats['prediction_unknown'] > 0
    assert stats['evaluated_labels'] < stats['observed_labels']
