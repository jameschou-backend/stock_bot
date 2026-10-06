import json

import numpy as np
import pandas as pd
import pytest

from skills.smc_research import compute_setups


def frames(n=85):
    days = pd.bdate_range('2024-01-01', periods=n)
    def frame(value):
        return pd.DataFrame({'2330': np.full(n, value)}, index=days)
    return dict(c=frame(100.), h=frame(101.), l=frame(98.), open=frame(99.),
                close=frame(100.), volume=frame(1_000_000.),
                valid=frame(True), eligible=frame(True))


def candle(f, i, o, h, l, c):
    for key, value in dict(open=o, h=h, l=l, c=c, close=c).items():
        f[key].iloc[i, 0] = value


def date(f, i):
    return str(f['c'].index[i].date())


def events(f, strategy):
    return [x for x in compute_setups(f)['events'] if x['strategy_id'] == strategy]


def fvg_frames(n=85):
    f = frames(n)
    candle(f, 60, 101, 112, 100, 110)
    candle(f, 61, 111, 116, 110, 114)
    for i in range(62, n):
        candle(f, i, 116, 117, 113, 115)
    return f


def first_fvg(out):
    return next(x for x in out['setups'] if x['kind'] == 'fvg')


def structure_frames():
    f = frames()
    candle(f, 60, 104, 110, 100, 105)
    candle(f, 61, 105, 109, 103, 106)
    candle(f, 62, 105, 108, 104, 106)
    candle(f, 63, 108, 112, 107, 111)
    candle(f, 64, 112, 114, 111, 113)
    candle(f, 65, 115, 120, 114, 116)
    candle(f, 66, 116, 119, 115, 117)
    candle(f, 67, 115, 118, 114, 116)
    candle(f, 68, 118, 124, 117, 122)
    for i in range(69, len(f['c'])):
        candle(f, i, 120, 123, 118, 121)
    return f


def test_fvg_known_only_at_third_close_and_retest_cannot_use_formation_bar():
    f = fvg_frames()
    out = compute_setups(f)
    zone = first_fvg(out)
    assert zone['setup_date'] == date(f, 61)
    assert (zone['zone_lower'], zone['zone_upper']) == (101, 110)
    assert zone['status'] == 'expired'
    assert not events(f, 'fvg_retest')
    assert [x['signal_date'] for x in events(f, 'fvg_form')] == [date(f, 61)]


def test_retest_requires_red_close_above_zone_and_earliest_next_session():
    f = fvg_frames()
    candle(f, 62, 109, 116, 108, 115)
    out = compute_setups(f)
    zone = first_fvg(out)
    assert zone['status'] == 'retested'
    assert zone['retest_date'] == date(f, 62)
    assert zone['retest_sessions'] == 1
    assert events(f, 'fvg_retest')[0]['setup_id'] == zone['setup_id']


@pytest.mark.parametrize('replacement', [(109, 112, 102, 108), (109, 112, 102, 109), (111, 114, 105, 110)])
def test_overlap_without_red_close_above_upper_is_not_retest(replacement):
    f = fvg_frames()
    candle(f, 62, *replacement)
    assert not events(f, 'fvg_retest')


def test_close_below_floor_invalidates_even_if_later_recovers():
    f = fvg_frames()
    candle(f, 62, 102, 103, 99, 100)
    candle(f, 63, 109, 116, 108, 115)
    zone = first_fvg(compute_setups(f))
    assert zone['status'] == 'invalidated'
    assert zone['invalidation_date'] == date(f, 62)
    assert zone['retest_date'] is None
    assert not events(f, 'fvg_retest')


@pytest.mark.parametrize('field,value', [('valid', False), ('eligible', False), ('volume', 0), ('h', np.nan)])
def test_missing_or_untradeable_session_ends_cohort_without_date_compression(field, value):
    f = fvg_frames()
    f[field].iloc[62, 0] = value
    candle(f, 63, 109, 116, 108, 115)
    out = compute_setups(f)
    zone = first_fvg(out)
    assert zone['status'] == 'data_missing'
    assert zone['data_missing_date'] == date(f, 62)
    assert not events(f, 'fvg_retest')


def test_retest_on_session_ten_is_allowed_but_eleven_is_not():
    f = fvg_frames()
    candle(f, 71, 109, 116, 108, 115)
    assert first_fvg(compute_setups(f))['retest_sessions'] == 10
    g = fvg_frames()
    candle(g, 72, 109, 116, 108, 115)
    zone = first_fvg(compute_setups(g))
    assert zone['status'] == 'expired'
    assert zone['expiry_date'] == date(g, 71)
    assert not events(g, 'fvg_retest')


def test_sample_end_is_pending_not_fabricated_expiry_or_missing():
    f = fvg_frames(n=66)
    zone = first_fvg(compute_setups(f))
    assert zone['status'] == 'pending'
    assert zone['expiry_date'] is None and zone['data_missing_date'] is None


def test_every_formation_is_retained_when_multiple_zones_share_retest():
    f = fvg_frames()
    candle(f, 62, 114, 120, 113, 118)  # second FVG: 112..113
    candle(f, 63, 113, 119, 109, 118)  # touches both and closes above both
    out = compute_setups(f)
    zones = [x for x in out['setups'] if x['kind'] == 'fvg']
    assert len(zones) == 2
    assert all(x['retest_date'] == date(f, 63) for x in zones)
    e = events(f, 'fvg_retest')
    assert len(e) == 1 and len(e[0]['setup_ids']) == 2
    assert out['counts']['event_duplicates_merged'] == 1


def test_middle_close_must_break_first_high_not_merely_be_red():
    f = fvg_frames()
    candle(f, 60, 99, 112, 98, 100)
    assert not events(f, 'fvg_form')


def test_tiny_gap_below_prior_atr_threshold_is_excluded():
    f = fvg_frames()
    candle(f, 60, 100, 102, 99, 101.5)
    candle(f, 61, 101.1, 103, 101.15, 102)
    assert not events(f, 'fvg_form')


def test_unconfirmed_pivot_is_not_backdated_and_initial_is_not_bos():
    f = structure_frames()
    initial = events(f, 'smc_bull_break')[0]
    assert initial['signal_date'] == date(f, 63)
    assert initial['pivot_date'] == date(f, 60)
    assert initial['confirmed_at'] == date(f, 62)
    assert initial['structure_kind'] == 'initial'
    assert events(f, 'smc_bos')[0]['signal_date'] == date(f, 68)
    assert not events(f, 'smc_choch')


def test_equal_high_is_not_strict_pivot():
    f = structure_frames()
    f['h'].iloc[61, 0] = 110
    assert not any(x['pivot_date'] == date(f, 60) for x in events(f, 'smc_bull_break'))


def test_consumed_level_cannot_trigger_again_after_recross():
    f = structure_frames()
    candle(f, 64, 111, 111.5, 107, 109)
    candle(f, 65, 109, 112, 108, 111)
    assert len([e for e in events(f, 'smc_bull_break') if e['pivot_date'] == date(f, 60)]) == 1


def test_missing_session_resets_prior_pivot_and_direction():
    f = structure_frames()
    f['valid'].iloc[62, 0] = False
    assert not events(f, 'smc_bull_break')


def test_prefix_and_appended_future_mutations_cannot_rewrite_prior_events():
    f = structure_frames()
    cutoff = 69
    prefix = {k: v.iloc[:cutoff].copy() for k, v in f.items()}
    expected = compute_setups(prefix)['events']
    assert expected == [e for e in compute_setups(f)['events'] if e['signal_date'] <= date(f, cutoff-1)]
    candle(f, 70, 500, 700, 1, 650)
    f['eligible'].iloc[72, 0] = False
    assert expected == [e for e in compute_setups(f)['events'] if e['signal_date'] <= date(f, cutoff-1)]


def test_fvg_prefix_preserves_formation_even_when_later_invalidated():
    f = fvg_frames()
    prefix = {k: v.iloc[:62].copy() for k, v in f.items()}
    candle(f, 62, 102, 103, 99, 100)
    assert events(prefix, 'fvg_form') == events(f, 'fvg_form')
    assert first_fvg(compute_setups(prefix))['status'] == 'pending'
    assert first_fvg(compute_setups(f))['status'] == 'invalidated'


def test_benchmark_excluded_and_output_is_json_safe():
    f = fvg_frames()
    for frame in f.values():
        frame['0050'] = frame['2330']
    out = compute_setups(f)
    assert out['events'] and not any(x['stock_id'] == '0050' for x in out['events'] + out['setups'])
    json.dumps(out, allow_nan=False)


def test_turnover_uses_raw_close_not_adjusted_close():
    f = fvg_frames()
    for key in ['c', 'h', 'l']:
        f[key] *= 10
    f['volume'] /= 4  # raw turnover 25m; adjusted-price proxy would incorrectly pass
    assert not compute_setups(f)['events']


def test_common_warmup_requires_sixty_consecutive_sessions():
    f = fvg_frames()
    f['valid'].iloc[10, 0] = False
    assert not events(f, 'fvg_form')


def test_baseline_requires_previous_known_false_and_does_not_repeat():
    f = frames()
    candle(f, 59, 105, 111, 104, 110)
    f['volume'].iloc[59, 0] = 2_000_000
    assert not events(f, 'research_breakout20')  # only 59 prior valid observations
    candle(f, 60, 112, 121, 111, 120)
    f['volume'].iloc[60, 0] = 2_000_000
    assert not events(f, 'research_breakout20')  # previous match was already true
    candle(f, 61, 119, 121, 118, 120)
    candle(f, 62, 122, 131, 121, 130)
    f['volume'].iloc[62, 0] = 2_000_000
    assert [e['signal_date'] for e in events(f, 'research_breakout20')] == [date(f, 62)]


def test_misaligned_input_rejected_instead_of_silent_alignment():
    f = frames()
    f['h'] = f['h'].iloc[1:]
    with pytest.raises(ValueError, match='coordinates'):
        compute_setups(f)


def test_prior_bear_break_then_bull_break_is_choch_not_bos():
    f = frames()
    for i, bar in {
        60:(91, 100, 80, 90), 61:(89, 101, 85, 90), 62:(90, 102, 86, 91),
        63:(79, 81, 73, 75), 64:(84, 90, 80, 85), 65:(92, 100, 85, 91),
        66:(91, 99, 87, 92), 67:(92, 98, 88, 91), 68:(99, 104, 95, 102),
    }.items():
        candle(f, i, *bar)
    e = [e for e in events(f, 'smc_choch') if e['signal_date'] == date(f, 68)]
    assert len(e) == 1 and e[0]['pivot_date'] == date(f, 65)
    assert not any(e['signal_date'] == date(f, 68) for e in events(f, 'smc_bos'))


def test_sweep_is_once_per_confirmed_low_not_every_recovery_bar():
    f = frames()
    for i, bar in {
        60:(91, 100, 80, 90), 61:(89, 101, 85, 90), 62:(90, 102, 86, 91),
        63:(89, 95, 75, 91), 64:(90, 97, 74, 92),
    }.items():
        candle(f, i, *bar)
    e = [e for e in events(f, 'smc_sweep') if e['pivot_date'] == date(f, 60)]
    assert len(e) == 1 and e[0]['signal_date'] == date(f, 63)
    assert e[0]['confirmed_at'] == date(f, 62)


def test_both_side_wick_break_is_explicit_ambiguous_and_not_bull_signal():
    f = frames()
    for i, bar in {
        60:(91, 100, 80, 90), 61:(90, 101, 85, 91), 62:(91, 105, 86, 92),
        63:(90, 102, 87, 91), 64:(91, 101, 86, 92), 65:(100, 110, 70, 108),
    }.items():
        candle(f, i, *bar)
    out = compute_setups(f)
    assert out['counts']['ambiguous_structure_sessions'] >= 1
    assert not any(e['signal_date'] == date(f, 65) and e['strategy_id'].startswith('smc_') for e in out['events'])


def test_orderblock_uses_last_prior_black_candle_and_retest_is_subsequent():
    f = structure_frames()
    candle(f, 62, 107, 108, 104, 106)
    candle(f, 64, 107, 113, 106, 112)
    out = compute_setups(f)
    zone = next(z for z in out['setups'] if z['kind'] == 'orderblock')
    assert zone['origin_date'] == date(f, 62)
    assert zone['setup_date'] == date(f, 63)
    assert (zone['zone_lower'], zone['zone_upper']) == (104, 108)
    assert zone['retest_date'] == date(f, 64)
    e = events(f, 'smc_orderblock_retest')[0]
    assert e['signal_date'] == date(f, 64) and e['setup_date'] == date(f, 63)


def test_orderblock_does_not_reach_beyond_previous_ten_sessions():
    f = structure_frames()
    candle(f, 52, 100, 101, 98, 99)
    out = compute_setups(f)
    assert not any(z['kind'] == 'orderblock' and z['setup_date'] == date(f, 63) for z in out['setups'])


def test_adjusted_equal_ohlc_roundoff_does_not_create_missing_sessions():
    f = fvg_frames()
    # Raw low == close is valid; multiplication after ratio division can place
    # adjusted low one representable float above adjusted close.
    f['l'].iloc[20, 0] = np.nextafter(f['c'].iloc[20, 0], np.inf)
    f['open'].iloc[20, 0] = f['close'].iloc[20, 0]
    out = compute_setups(f)
    assert out['counts']['adjusted_ohlc_roundoff_tolerated'] == 1
    assert events(f, 'fvg_form')


def test_material_adjusted_ohlc_violation_still_breaks_sixty_session_window():
    f = fvg_frames()
    f['l'].iloc[20, 0] = f['c'].iloc[20, 0] + .01
    f['open'].iloc[20, 0] = f['close'].iloc[20, 0]
    out = compute_setups(f)
    assert out['counts']['adjusted_ohlc_roundoff_tolerated'] == 0
    assert not events(f, 'fvg_form')


def test_adjusted_roundoff_cannot_turn_raw_doji_into_red_fvg_candle():
    f = fvg_frames()
    f['open'].iloc[61, 0] = f['close'].iloc[61, 0]
    # Slightly lower factor result than separately represented adjusted close.
    f['c'].iloc[61, 0] = np.nextafter(f['c'].iloc[61, 0], np.inf)
    assert not events(f, 'fvg_form')
