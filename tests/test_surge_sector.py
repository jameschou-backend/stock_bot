from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.surge_sector import sector_features


def inputs(count=10, periods=105):
    days = pd.bdate_range("2023-01-02", periods=periods)
    ids = [f"{1001 + i:04d}" for i in range(count)]
    close = pd.DataFrame(100., index=days, columns=["0050", *ids])
    for i, sid in enumerate(ids):
        close[sid] = 100 * np.power(1.001 + (i % 3) * .0001, np.arange(periods))
    raw = close.copy()
    volume = pd.DataFrame(1_000_000., index=days, columns=close.columns)
    companies = pd.DataFrame({"stock_id": ids, "listed_date": days[0]})
    members = pd.DataFrame({"stock_id": ids[:5], "industry": "sector-a"})
    return close, raw, volume, companies, members


def result_for(args, *, sid="1001", pos=80, industry="sector-a"):
    frame = sector_features(*args, [args[0].index[pos]])
    return frame.loc[frame.stock_id.eq(sid) & frame.industry.eq(industry)].iloc[0]


def test_target_excluded_breadth_and_fixed_peer_turnover_share():
    close, raw, volume, companies, members = inputs()
    # Four peers gain turnover share in the final five days. The target's
    # unchanged volume is excluded from both numerator and denominator.
    volume.loc[close.index[76:81], ["1002", "1003", "1004", "1005"]] *= 2
    args = (close, raw, volume, companies, members)
    row = result_for(args)
    assert row.expected_peers == row.observed_peers == 4
    assert row.peer_ids == ["1002", "1003", "1004", "1005"]
    assert row.peer_coverage == row.peer_breadth == 1
    assert row.peer_median_excess20 > 0
    assert bool(row.breadth_confirmed) and bool(row.turnover_confirmed) and bool(row.sector_confirmed)
    assert row.membership_point_in_time == False
    amount = raw * volume
    denominator = amount.loc[close.index[56:81], companies.stock_id.drop(0)].sum(axis=1)
    daily = amount.loc[close.index[56:81], row.peer_ids].sum(axis=1) / denominator
    assert row.share5 == pytest.approx(daily.iloc[-5:].mean())
    assert row.share_previous20 == pytest.approx(daily.iloc[:20].mean())
    assert row.share_multiple == pytest.approx(row.share5 / row.share_previous20)


def test_target_explosion_and_target_missing_history_cannot_change_own_peer_features():
    args = inputs()
    original = result_for(args)
    changed = deepcopy(args)
    day = args[0].index[80]
    changed[0].loc[day, "1001"] *= 1000
    changed[1].loc[day, "1001"] *= 1e30
    changed[2].loc[day, "1001"] *= 1e30
    actual = result_for(changed)
    pd.testing.assert_series_equal(actual, original)
    missing = deepcopy(args)
    for frame in missing[:3]:
        frame.loc[:day, "1001"] = np.nan
    pd.testing.assert_series_equal(result_for(missing), original)


def test_truncating_or_mutating_future_preserves_all_past_results():
    args = inputs()
    dates = [args[0].index[65], args[0].index[80]]
    original = sector_features(*args, dates)
    truncated = (*[frame.loc[:dates[-1]].copy() for frame in args[:3]], *args[3:])
    pd.testing.assert_frame_equal(original, sector_features(*truncated, dates))
    changed = deepcopy(args)
    for frame in changed[:3]:
        frame.loc[frame.index > dates[-1]] *= 1e20
    pd.testing.assert_frame_equal(original, sector_features(*changed, dates))


def test_missing_stock_column_stays_in_expected_and_market_denominator():
    args = list(inputs())
    missing_id = "1005"
    for i in range(3):
        args[i] = args[i].drop(columns=missing_id)
    row = result_for(args)
    assert row.expected_peers == 4 and row.observed_peers == 3
    assert row.peer_coverage == .75
    assert row.market_min_coverage25 == pytest.approx(8 / 9)
    assert pd.isna(row.breadth_confirmed) and pd.isna(row.turnover_confirmed)
    # A target with no own column remains an auditable row; its own missing
    # column cannot reduce the coverage of its four valid peers.
    absent_target = result_for(args, sid=missing_id)
    assert absent_target.observed_peers == 4 and absent_target.peer_coverage == 1
    assert absent_target.market_min_coverage25 == 1


@pytest.mark.parametrize("value", [np.nan, np.inf, 0., -1.])
def test_invalid_benchmark_endpoint_preserves_unknown_price_rule(value):
    args = list(inputs())
    args[0].loc[args[0].index[60], "0050"] = value
    row = result_for(args)
    assert row.observed_peers == 4
    assert pd.isna(row.peer_breadth) and pd.isna(row.peer_median_excess20)
    assert pd.isna(row.breadth_confirmed)
    assert row.turnover_confirmed == False
    assert row.sector_confirmed == False  # False AND unknown is known false.


def test_unlisted_members_excluded_and_prelisting_prices_cannot_qualify():
    args = list(inputs())
    args[3].loc[args[3].stock_id.eq("1005"), "listed_date"] = args[0].index[90]
    rows = sector_features(*args, [args[0].index[80]])
    assert "1005" not in set(rows.stock_id)
    row = rows.loc[rows.stock_id.eq("1001")].iloc[0]
    assert row.expected_peers == row.observed_peers == 3
    args[3].loc[args[3].stock_id.eq("1005"), "listed_date"] = args[0].index[70]
    row = result_for(args)
    assert row.expected_peers == 4 and row.observed_peers == 3
    assert pd.isna(row.breadth_confirmed)


def test_member_duplicates_deduplicate_and_overlapping_sectors_stay_separate():
    args = list(inputs())
    original = sector_features(*args, [args[0].index[80]])
    extra = pd.DataFrame({"stock_id": ["1001", "1006", "1007", "1008", "0050"], "industry": "sector-b"})
    args[4] = pd.concat([args[4], args[4], extra], ignore_index=True)
    actual = sector_features(*args, [args[0].index[80]])
    pd.testing.assert_frame_equal(actual.loc[actual.industry.eq("sector-a")].reset_index(drop=True), original)
    second = actual.loc[actual.industry.eq("sector-b") & actual.stock_id.eq("1001")].iloc[0]
    assert second.expected_peers == 3 and second.peer_ids == ["1006", "1007", "1008"]
    assert not actual.stock_id.eq("0050").any()


def test_peer_is_fixed_over_turnover_window_and_bad_history_is_not_rotated_in():
    args = list(inputs(count=25))
    args[4] = pd.DataFrame({"stock_id": ["1001", "1002", "1003", "1004", "1005", "1006"], "industry": "sector-a"})
    day = args[0].index[80]
    # 1006 had 24 usable days but is unavailable at T. Coverage remains 4/5;
    # it must contribute zero to the entire fixed-peer 25-day numerator.
    args[2].loc[day, "1006"] = 0
    row = result_for(args)
    assert row.observed_peers == 4 and row.peer_coverage == .8
    assert "1006" not in row.peer_ids and bool(row.breadth_confirmed)
    amount = args[1] * args[2]
    market = amount.loc[args[0].index[56:81], args[3].stock_id.drop(0)].sum(axis=1)
    expected = amount.loc[args[0].index[56:81], row.peer_ids].sum(axis=1) / market
    assert row.share_previous20 == pytest.approx(expected.iloc[:20].mean())
    assert row.share5 == pytest.approx(expected.iloc[-5:].mean())


def test_market_coverage_must_pass_on_each_of_25_days():
    args = list(inputs(count=25))
    # Two missing non-sector stocks make the market coverage 22/24 on one day.
    args[1].loc[args[0].index[60], ["1024", "1025"]] = np.nan
    row = result_for(args)
    assert row.breadth_confirmed == True
    assert row.market_min_coverage25 == pytest.approx(22 / 24)
    assert pd.isna(row.turnover_confirmed) and pd.isna(row.sector_confirmed)
    assert pd.isna(row.share5)


def test_insufficient_history_and_fewer_than_three_peers_are_unknown():
    args = inputs()
    early = result_for(args, pos=59)
    assert early.observed_peers == 0 and pd.isna(early.sector_confirmed)
    sparse = (*args[:4], args[4].iloc[:3])
    row = result_for(sparse)
    assert row.observed_peers == 2 and pd.isna(row.breadth_confirmed)
    assert pd.isna(row.turnover_confirmed)


@pytest.mark.parametrize("defect", ["jump", "missing_price", "zero_volume"])
def test_peer_historical_defects_remove_peer_from_all_evidence(defect):
    args = list(inputs())
    if defect == "jump":
        args[0].loc[args[0].index[50], "1005"] *= 1.2
    elif defect == "missing_price":
        args[0].loc[args[0].index[30], "1005"] = np.nan
    else:
        args[2].loc[args[0].index[60], "1005"] = 0
    row = result_for(args)
    assert row.observed_peers == 3 and "1005" not in row.peer_ids
    assert pd.isna(row.breadth_confirmed)


def test_unknown_member_listing_date_and_alignment_fail_explicitly():
    args = list(inputs())
    args[4] = pd.concat([args[4], pd.DataFrame([{"stock_id": "9999", "industry": "sector-a"}])])
    with pytest.raises(ValueError, match="listing date"):
        sector_features(*args, [args[0].index[80]])
    args = list(inputs())
    args[2] = args[2].iloc[:-1]
    with pytest.raises(ValueError, match="align"):
        sector_features(*args, [args[0].index[80]])


def test_empty_membership_returns_stable_nullable_schema():
    args = inputs()
    result = sector_features(*args[:4], args[4].iloc[:0], [args[0].index[80]])
    assert result.empty and "peer_ids" in result
    assert str(result.sector_confirmed.dtype) == "boolean"
