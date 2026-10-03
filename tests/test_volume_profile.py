"""Price-volume facts and timing, without market data or outcome research."""
import json

import numpy as np
import pandas as pd
import pytest

from skills.volume_profile import SOURCE_KIND, build_volume_profile


SESSIONS = ["2026-09-21", "2026-09-22"]
SIGNAL = "2026-09-23"


def ticks(prices, shares=None, first_count=None):
    shares = shares if shares is not None else [1.] * len(prices)
    first_count = first_count if first_count is not None else (len(prices) + 1) // 2
    stamps = [pd.Timestamp(SESSIONS[0] if i < first_count else SESSIONS[1]) + pd.Timedelta(hours=9, minutes=i)
              for i in range(len(prices))]
    return pd.DataFrame(dict(timestamp=stamps, price=prices, shares=shares))


def build(frame, **kwargs):
    return build_volume_profile(frame, signal_date=SIGNAL, session_dates=SESSIONS,
                                source_kind=SOURCE_KIND, **kwargs)


def test_poc_and_contiguous_value_area_adjacent_tie_goes_lower_first():
    result = build(ticks([1., 2.25, 3.5, 4.25, 6.], [10, 20, 40, 20, 10]), bins=5)
    profile = result["full"]
    assert result["bin_edges"] == [1, 2, 3, 4, 5, 6]
    assert profile["bin_volumes"] == [10, 20, 40, 20, 10]
    assert profile["poc_index"] == 2 and profile["poc_price"] == 3.5
    assert profile["poc_low"] == 3 and profile["poc_high"] == 4
    assert profile["expansion_order"] == [2, 1, 3]
    assert profile["val"] == 2 and profile["vah"] == 5
    assert profile["value_area_shares"] == 80
    assert profile["value_area_fraction"] == .8


def test_equal_poc_volume_selects_lower_bin_and_terminal_max_is_included():
    result = build(ticks([1, 2, 3], [20, 10, 10]), bins=2)
    assert result["full"]["bin_volumes"] == [20, 20]
    assert result["full"]["poc_index"] == 0
    assert result["full"]["poc_price"] == 1.5
    boundaries = build(ticks([1, 2, 3, 4, 5]), bins=4)
    assert boundaries["full"]["bin_volumes"] == [1, 1, 1, 2]
    assert boundaries["full"]["total_shares"] == 5


def test_value_area_must_include_empty_bins_between_separated_clusters():
    result = build(ticks([1, 2, 5], [40, 1, 35]), bins=4, value_fraction=1.)
    assert result["full"]["bin_volumes"] == [40, 1, 0, 35]
    assert result["full"]["expansion_order"] == [0, 1, 2, 3]
    assert result["full"]["value_area_fraction"] == 1
    assert result["full"]["val"] == 1 and result["full"]["vah"] == 5


def test_full_window_and_both_halves_use_identical_bin_edges():
    result = build(ticks([1, 9], [100, 100]), bins=4)
    assert result["bin_edges"] == [1, 3, 5, 7, 9]
    assert result["first_half"]["bin_volumes"] == [100, 0, 0, 0]
    assert result["second_half"]["bin_volumes"] == [0, 0, 0, 100]
    assert result["first_half"]["poc_price"] == 2
    assert result["second_half"]["poc_price"] == 8
    assert result["poc_shift"] == 6 and result["poc_shift_fraction"] == 3
    assert result["poc_up"] is True and result["halves_comparable"] is True
    np.testing.assert_allclose(np.array(result["first_half"]["bin_volumes"]) + result["second_half"]["bin_volumes"],
                               result["full"]["bin_volumes"])


def test_identical_execution_rows_are_never_deduplicated():
    data = ticks([100, 101], [7, 3])
    repeated = pd.concat([data.iloc[:1], data.iloc[:1], data.iloc[1:]], ignore_index=True)
    profile = build(repeated)
    assert profile["input_tick_count"] == 3
    assert profile["full"]["total_shares"] == 17
    assert profile["first_half"]["total_shares"] == 14
    assert profile == build(repeated.iloc[::-1])


def test_single_traded_price_does_not_fabricate_surrounding_range():
    data = ticks([100., 100., 999.], [7., 3., 0.], first_count=1)
    result = build(data)
    assert result["bins"] == 40 and result["single_price_range"] is True
    assert result["bin_edges"] == [100.] * 41
    assert result["full"]["bin_volumes"] == [10.] + [0.] * 39
    assert result["full"]["poc_index"] == 0
    assert result["full"]["poc_price"] == result["full"]["val"] == result["full"]["vah"] == 100.
    assert result["poc_shift"] == 0 and result["poc_up"] is False
    assert result["zero_volume_tick_count"] == 1


@pytest.mark.parametrize("data,reason", [(ticks([]), "empty_ticks"), (ticks([100, 101], [0, 0]), "zero_total_shares")])
def test_empty_or_all_zero_volume_is_unavailable(data, reason):
    result = build(data)
    assert result["available"] is False and result["reason"] == reason
    assert result["bin_edges"] is None and result["poc_shift"] is None
    assert result["full"]["poc_price"] is None
    json.dumps(result, allow_nan=False)


def test_missing_trade_rows_are_reported_without_implying_source_completeness():
    result = build(ticks([100, 101], first_count=2))
    assert result["available"] is True
    assert result["first_half"]["available"] is True
    assert result["second_half"]["available"] is False
    assert result["halves_comparable"] is False and result["poc_shift"] is None
    assert result["sessions_without_tick_rows"] == [SESSIONS[1]]
    assert result["source_completeness_verified_by_module"] is False
    assert result["corporate_action_consistency_verified_by_module"] is False


@pytest.mark.parametrize("field,value,error", [
    ("price", 0., "prices"), ("price", -1., "prices"), ("price", np.nan, "prices"),
    ("price", np.inf, "prices"), ("shares", -1., "volumes"),
    ("shares", np.nan, "volumes"), ("shares", np.inf, "volumes"),
])
def test_invalid_observations_fail_without_dropping_rows(field, value, error):
    data = ticks([100., 101.])
    data.loc[0, field] = value
    with pytest.raises(ValueError, match=error):
        build(data)


@pytest.mark.parametrize("stamp", ["2026-09-23 00:00:00", "2026-09-23 09:00:00", "2027-01-01 09:00:00"])
def test_signal_day_and_future_ticks_are_strictly_forbidden(stamp):
    data = ticks([100., 101.])
    data.loc[0, "timestamp"] = pd.Timestamp(stamp)
    with pytest.raises(ValueError, match="at or after signal_date"):
        build(data)


def test_prior_but_outside_window_tick_is_not_silently_filtered():
    data = ticks([100., 101.])
    data.loc[0, "timestamp"] = pd.Timestamp("2026-09-18 09:00")
    with pytest.raises(ValueError, match="outside"):
        build(data)


def test_timezone_aware_timestamps_dates_and_numeric_epochs_are_rejected():
    data = ticks([100., 101.])
    data["timestamp"] = data.timestamp.dt.tz_localize("Asia/Taipei")
    with pytest.raises(ValueError, match="timezone-naive"):
        build(data)
    with pytest.raises(ValueError, match="naive Taipei"):
        build_volume_profile(ticks([100.]), signal_date=pd.Timestamp(SIGNAL, tz="Asia/Taipei"),
                             session_dates=SESSIONS, source_kind=SOURCE_KIND)
    data["timestamp"] = [1, 2]
    with pytest.raises(ValueError, match="numeric epochs"):
        build(data)


@pytest.mark.parametrize("dates", [SESSIONS[::-1], [SESSIONS[0]] * 2, SESSIONS[:1], SESSIONS + [SIGNAL]])
def test_market_sessions_must_be_unique_ordered_and_even(dates):
    with pytest.raises(ValueError, match="unique, ordered"):
        build_volume_profile(ticks([100.]), signal_date=SIGNAL, session_dates=dates, source_kind=SOURCE_KIND)


def test_profile_sessions_cannot_include_signal_day_even_without_its_ticks():
    with pytest.raises(ValueError, match="strictly before"):
        build_volume_profile(ticks([]), signal_date=SIGNAL, session_dates=[SESSIONS[1], SIGNAL], source_kind=SOURCE_KIND)


@pytest.mark.parametrize("kind", ["daily_ohlcv", "uniform_daily_volume_proxy", "intraday_odd_lot", None])
def test_daily_volume_approximations_and_wrong_markets_are_unsupported(kind):
    with pytest.raises(ValueError, match="authentic regular-board"):
        build_volume_profile(ticks([100., 101.]), signal_date=SIGNAL, session_dates=SESSIONS, source_kind=kind)


@pytest.mark.parametrize("params", [dict(bins=0), dict(bins=True), dict(bins=2.5),
    dict(value_fraction=0), dict(value_fraction=1.1), dict(value_fraction=np.nan), dict(value_fraction=True)])
def test_invalid_algorithm_parameters_fail(params):
    with pytest.raises(ValueError):
        build(ticks([100., 101.]), **params)


def test_share_sum_overflow_is_explicit_instead_of_nonfinite_profile():
    with pytest.raises(ValueError, match="Aggregated trade shares"):
        build(ticks([100., 101.], [1e308, 1e308]))


def test_deterministic_random_profiles_conserve_shares_and_keep_contiguous_area():
    generator = np.random.default_rng(3003)
    for _ in range(20):
        data = ticks(generator.integers(100, 120, size=100).astype(float),
                     generator.integers(0, 1000, size=100).astype(float))
        result = build(data)
        assert result == build(data.sample(frac=1., random_state=3))
        assert result["full"]["total_shares"] == data.shares.sum()
        for key in ("full", "first_half", "second_half"):
            profile = result[key]
            low, high = profile["value_area_low_index"], profile["value_area_high_index"]
            assert set(profile["expansion_order"]) == set(range(low, high + 1))
            assert profile["value_area_fraction"] >= .7
            assert sum(profile["bin_volumes"][low:high + 1]) == profile["value_area_shares"]
        json.dumps(result, allow_nan=False)
