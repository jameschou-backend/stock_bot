import numpy as np
import pandas as pd
import pytest

from skills.conditional_entries import CONDITION_IDS, compute_conditions


def frames(n=420, ids=('2330', '0050')):
    index = pd.bdate_range('2023-01-02', periods=n)
    def matrix(value):
        return pd.DataFrame(value, index=index, columns=list(ids))
    return dict(c=matrix(100.), h=matrix(101.), l=matrix(98.), open=matrix(99.),
                close=matrix(100.), volume=matrix(6_000_000.),
                valid=matrix(True), eligible=matrix(True))


def candle(f, i, o, h, l, c, sid='2330'):
    for key, value in dict(open=o, h=h, l=l, c=c, close=c).items():
        f[key].iloc[i, f[key].columns.get_loc(sid)] = value


def conditions(f):
    return compute_conditions(f)['conditions']


def value(out, key, field='matched', i=-1, sid='2330'):
    return bool(out[key][field].iloc[i][sid])


def fvg_frames():
    f = frames()
    candle(f, 400, 101, 112, 100, 110)
    candle(f, 401, 111, 116, 110, 114)
    for i in range(402, len(f['c'])):
        candle(f, i, 116, 117, 113, 115)
    return f


def test_seven_conditions_keep_axes_and_boolean_known_matched_masks():
    f = frames()
    out = conditions(f)
    assert tuple(out) == CONDITION_IDS
    for row in out.values():
        for key in ('known', 'matched'):
            assert row[key].index.equals(f['c'].index)
            assert row[key].columns.equals(f['c'].columns)
            assert all(dtype == bool for dtype in row[key].dtypes)
            assert not row[key].isna().any().any()
        assert not (row['matched'] & ~row['known']).any().any()
        assert row['name'] and row['definition']


def test_high400_requires_all400_sessions_including_today():
    f = frames()
    out = conditions(f)
    assert not value(out, 'high400', 'known', 398)
    assert value(out, 'high400', 'known', 399)
    assert value(out, 'high400', i=399)
    candle(f, 0, 100, 102, 99, 101)
    out = conditions(f)
    assert not value(out, 'high400', i=399)
    assert value(out, 'high400', i=400)  # old peak leaves exact400-day window


def test_high400_threshold_uses_adjusted_close_exactly_without_price_tolerance():
    f = frames()
    candle(f, 419, 98, 101, 97, 99.8)
    assert value(conditions(f), 'high400')
    candle(f, 419, 98, 101, 97, np.nextafter(99.8, -np.inf))
    assert not value(conditions(f), 'high400')


def test_course400_preserves_current_inclusive_volume_and_500m_threshold():
    f = frames()
    f['volume'].loc[:, '2330'] = 5_000_000.
    out = conditions(f)
    assert value(out, 'course400')
    f['volume'].iloc[-2, 0] = 6_000_000
    assert not value(conditions(f), 'course400')
    f['volume'].iloc[-1, 0] = 6_000_000
    assert value(conditions(f), 'course400')
    f['volume'].loc[:, '2330'] = 4_000_000.
    out = conditions(f)
    assert value(out, 'high400') and value(out, 'course400', 'known')
    assert not value(out, 'course400')


def test_course_turnover_uses_raw_not_adjusted_prices():
    f = frames()
    f['volume'].loc[:, '2330'] = 2_000_000.
    for key in ('c', 'h', 'l'):
        f[key].loc[:, '2330'] *= 10
    out = conditions(f)
    assert value(out, 'high400')
    assert not value(out, 'course400')  # raw200m; adjusted proxywouldwronglybe2bn


def test_rs20_needs_21_complete_own_and_benchmark_sessions():
    f = frames(n=30)
    candle(f, 20, 108, 112, 107, 111)
    out = conditions(f)
    assert not value(out, 'rs20', 'known', 19)
    assert value(out, 'rs20', 'known', 20)
    assert value(out, 'rs20', i=20)
    candle(f, 20, 100, 103, 99, 102, sid='0050')
    assert not value(conditions(f), 'rs20', i=20)  # 11%-2%=9%


@pytest.mark.parametrize('sid', ['2330', '0050'])
def test_rs20_invalid_midpath_is_unknown_not_return_over_compressed_dates(sid):
    f = frames(n=35)
    candle(f, 34, 115, 121, 114, 120)
    f['valid'].iloc[20, f['valid'].columns.get_loc(sid)] = False
    out = conditions(f)
    assert not value(out, 'rs20', 'known') and not value(out, 'rs20')


def test_missing_benchmark_column_leaves_rs_unknown_not_implicit_flat_market():
    f = frames(ids=('2330',))
    candle(f, 419, 115, 121, 114, 120)
    out = conditions(f)
    assert value(out, 'high400')
    assert not out['rs20']['known'].any().any()
    assert not out['high400_and_rs20']['known'].any().any()


@pytest.mark.parametrize('field,value_', [('valid', False), ('eligible', False), ('volume', 0), ('c', np.nan)])
def test_missing_observation_breaks_400day_window(field, value_):
    f = frames()
    f[field].iloc[100, 0] = value_
    out = conditions(f)
    assert not value(out, 'high400', 'known')
    assert not value(out, 'course400', 'known')
    assert not value(out, 'high400')


def test_recent_fvg_includes_today_and_exactly_previous_four_sessions():
    f = fvg_frames()
    out = conditions(f)
    assert not value(out, 'fvg_form5', i=400)
    assert all(value(out, 'fvg_form5', i=i) for i in range(401, 406))
    assert not value(out, 'fvg_form5', i=406)


def test_recent_fvg_does_not_delete_formation_when_later_invalidated():
    f = fvg_frames()
    candle(f, 402, 102, 103, 99, 100)
    out = conditions(f)
    assert value(out, 'fvg_form5', i=402)
    assert value(out, 'fvg_form5', i=405)


def test_recent_event_requires64_consecutive_valid_sessions(monkeypatch):
    f = frames(n=70)
    def fixed_events(prepared):
        return {'events': [dict(strategy_id='smc_bull_break', stock_id='2330',
            signal_date=str(prepared['c'].index[60].date()))]}
    monkeypatch.setattr('skills.conditional_entries.compute_setups', fixed_events)
    out = conditions(f)
    assert not value(out, 'smc_break5', 'known', 62)
    assert not value(out, 'smc_break5', i=62)
    assert value(out, 'smc_break5', 'known', 63)
    assert value(out, 'smc_break5', i=63)
    assert value(out, 'smc_break5', i=64)
    assert not value(out, 'smc_break5', i=65)


def test_combination_requires_both_known_even_if_first_condition_false():
    f = frames()
    candle(f, 419, 95, 100, 94, 96)
    f['valid'].iloc[418, 1] = False
    out = conditions(f)
    assert value(out, 'high400', 'known') and not value(out, 'high400')
    assert not value(out, 'rs20', 'known')
    assert not value(out, 'high400_and_rs20', 'known')
    assert not value(out, 'high400_and_rs20')


def test_combination_matches_only_shared_same_day_conditions():
    f = frames()
    candle(f, 419, 119, 121, 118, 120)
    out = conditions(f)
    assert value(out, 'high400') and value(out, 'rs20')
    assert value(out, 'high400_and_rs20')
    assert not value(out, 'high400_and_smc_break5')


def test_prefix_and_future_mutation_preserve_all_prior_filter_masks():
    f = fvg_frames()
    cutoff = 405
    prefix = {k: v.iloc[:cutoff].copy() for k,v in f.items()}
    expected, full = conditions(prefix), conditions(f)
    for key in CONDITION_IDS:
        for field in ('known', 'matched'):
            pd.testing.assert_frame_equal(expected[key][field], full[key][field].iloc[:cutoff])
    candle(f, 410, 500, 1000, 1, 900)
    f['eligible'].iloc[406, 0] = False
    changed = conditions(f)
    for key in CONDITION_IDS:
        for field in ('known', 'matched'):
            pd.testing.assert_frame_equal(expected[key][field], changed[key][field].iloc[:cutoff])


def test_zero_prefixed_codes_are_never_candidates():
    f = frames(ids=('2330', '0050', '0063'))
    out = conditions(f)
    for row in out.values():
        assert not row['matched'][['0050', '0063']].any().any()
        assert not row['known'][['0050', '0063']].any().any()


def test_float_roundoff_is_tolerated_but_real_ohlc_error_is_unknown():
    f = frames()
    f['open'].iloc[100, 0] = 100
    f['l'].iloc[100, 0] = np.nextafter(100., np.inf)
    assert value(conditions(f), 'high400', 'known')
    f['l'].iloc[100, 0] = 100.01
    assert not value(conditions(f), 'high400', 'known')


@pytest.mark.parametrize('change', ['shift_axis', 'duplicate_date', 'zero_time', 'bad_symbol'])
def test_input_axes_and_id_validation_rejects_silent_alignment(change):
    f = frames()
    if change == 'shift_axis':
        f['h'] = f['h'].iloc[1:]
    elif change == 'duplicate_date':
        for frame in f.values():
            frame.index = pd.DatetimeIndex([frame.index[0], *frame.index[:-1]])
    elif change == 'zero_time':
        for frame in f.values():
            frame.index += pd.Timedelta(hours=1)
    else:
        for frame in f.values():
            frame.columns = ['ABC', '0050']
    with pytest.raises(ValueError):
        compute_conditions(f)
