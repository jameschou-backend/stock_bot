import numpy as np
import pandas as pd
import pytest

from scripts.study_rally_precursors import (episode_coordinates, forward_labels, gap_prototypes,
                                           known_first, profile_coverage, summarize)


def frames(closes=(100., 110., 140., 160., 170.), opens=None):
    index = pd.date_range('2024-01-01', periods=len(closes), freq='B')
    c = pd.DataFrame({'2330': closes}, index=index)
    return dict(c=c, close=c.copy(), open=pd.DataFrame({'2330': opens or closes}, index=index),
        valid=c.gt(0), eligible=c.gt(0), volume=c*10)


def test_label_uses_next_open_and_fixed_endpoint_not_signal_close_or_peak():
    f = frames((100., 200., 120., 180., 190.), opens=(100., 120., 120., 180., 190.))
    out = forward_labels(f, 2, .3)
    assert out['gross'][0, 0] == 0
    assert not out['rally'][0, 0]  # intrawindow doubling does not pass endpoint target
    assert out['gross'][1, 0] == .5
    assert out['rally'][1, 0]


def test_missing_midpath_is_not_shortened_or_filled():
    f = frames()
    f['valid'].iloc[1, 0] = False
    out = forward_labels(f, 3, .3)
    assert out['mature'][0, 0]
    assert not out['complete'][0, 0]
    assert np.isnan(out['gross'][0, 0])
    assert not out['rally'][0, 0]


@pytest.mark.parametrize('field,value', [('volume', 0), ('eligible', None), ('open', np.nan)])
def test_bad_future_observation_remains_unknown(field, value):
    f = frames()
    if field == 'eligible':
        f[field] = f[field].astype(object)
    f[field].iloc[2, 0] = value
    assert not forward_labels(f, 3, .3)['complete'][0, 0]


def test_immature_windows_never_exit_at_last_available_close():
    out = forward_labels(frames(), 3, .3)
    assert out['mature'][:, 0].tolist() == [True, True, False, False, False]
    assert np.isnan(out['gross'][2:, 0]).all()


@pytest.mark.parametrize('h,t', [(0, .3), (True, .3), (2, 0), (2, float('nan'))])
def test_invalid_target_is_rejected(h, t):
    with pytest.raises(ValueError):
        forward_labels(frames(), h, t)


def test_first_signal_needs_known_previous_false_and_rearms_after_false():
    match = pd.DataFrame({'2330': [False, True, True, False, True, True]})
    available = pd.DataFrame({'2330': [False, True, True, True, True, True]})
    assert known_first(match, available)['2330'].tolist() == [False, False, False, False, True, False]


def test_episode_dedup_is_per_stock_and_nonoverlapping_in_entry_path():
    labels = np.zeros((9, 2), dtype=bool)
    labels[[1, 2, 3, 4, 5, 7, 8], 0] = True
    labels[[2, 3, 6], 1] = True
    assert episode_coordinates(labels, 3) == [(1, 0), (2, 1), (4, 0), (6, 1), (7, 0)]


def test_precision_denominator_includes_nonrallies_not_only_winners():
    f = frames((100., 110., 120., 140., 150.), opens=(100., 100., 100., 140., 150.))
    labels = forward_labels(f, 2, .3)
    base = np.ones((5, 1), dtype=bool)
    first = np.array([[True], [True], [True], [False], [True]])
    row = summarize(first, base, base, labels)
    assert row['first_events'] == 4
    assert row['evaluated'] == 3
    assert row['tp'] == 1
    assert row['false_positives'] == 2
    assert row['immature'] == 1
    assert row['precision'] == pytest.approx(1/3)
    assert row['profit_win_rate'] == 1  # missing rally target is not losing money


def test_unknown_coverage_excluded_from_comparison_denominator():
    labels = forward_labels(frames(), 2, .3)
    base = np.ones((5, 1), dtype=bool)
    known = np.array([[True], [False], [True], [True], [True]])
    first = np.array([[True], [False], [True], [False], [False]])
    row = summarize(first, known, base, labels)
    assert row['known_eligible_windows'] == 2
    assert row['signal_unknown_windows'] == 1


def test_future_changes_cannot_change_causal_first_events():
    first = pd.DataFrame({'2330': [False, False, True, True, False]})
    available = pd.DataFrame(True, index=first.index, columns=first.columns)
    before = known_first(first, available).iloc[:3].copy()
    first.iloc[3:, 0] = [False, True]
    pd.testing.assert_frame_equal(before, known_first(first, available).iloc[:3])


def test_poc_baseline_contains_only_actually_known_profiles():
    days = pd.date_range('2026-01-01', periods=3)
    rows = [dict(signal_date='2026-01-01', stock_id='2330', status='up', available=True),
            dict(signal_date='2026-01-02', stock_id='2330', status='unknown', available=False),
            dict(signal_date='2026-01-03', stock_id='2330', status='down', available=True)]
    out = profile_coverage(days, ['2330', '2317'], rows)
    assert out[:, 0].tolist() == [True, False, True]
    assert not out[:, 1].any()  # noncandidate false is not a known observed POC


def gap_frames():
    f = frames([100.] * 7 + [110., 114., 112.], opens=[99.] * 7 + [107., 112., 109.])
    f['h'] = f['c'] + 1
    f['l'] = f['c'] - 1
    f['l'].iloc[9, 0] = 109.
    return f


def test_gap_prototype_uses_actual_gap_and_retest_after_gap():
    out = gap_prototypes(gap_frames())
    assert out['research_true_gap_red'][0].iloc[7, 0]
    assert not out['research_gap_retest5'][0].iloc[7, 0]
    assert out['research_gap_retest5'][0].iloc[9, 0]


def test_gap_prototype_prefix_causality():
    full = gap_frames()
    prefix = {k:v.iloc[:8] for k,v in full.items()}
    expected, actual = gap_prototypes(prefix), gap_prototypes(full)
    for key in expected:
        for part in range(2):
            pd.testing.assert_frame_equal(expected[key][part], actual[key][part].iloc[:8])


def test_gap_retest_rejects_intervening_close_below_gap_floor():
    f = gap_frames()
    f['c'].iloc[8, 0] = 99.
    assert not gap_prototypes(f)['research_gap_retest5'][0].iloc[9, 0]
