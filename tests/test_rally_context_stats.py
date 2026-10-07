import json

import numpy as np
import pandas as pd
import pytest

from skills.rally_context_stats import nonoverlapping_events, summarize_filters


def event(event_id="one", **changes):
    row = {
        "cohort": "original_red", "event_id": event_id,
        "stock_id": "2330", "signal_date": "2024-01-02", "signal_index": 1,
        "entry_date": "2024-01-03", "exit_date": "2024-01-30", "horizon": 20,
        "mature": True, "complete": True, "gross_return": 0.4, "net_return": 0.39,
        "benchmark_net_return": 0.1, "mfe": 0.6, "mae": -0.1,
        "threshold": 0.3, "filter_a": True, "filter_b": True,
    }
    row.update(changes)
    return row


def record(rows, *, population="raw", cohort="original_red", period="all", filter_id="filter_a"):
    summary = summarize_filters(pd.DataFrame(rows), ["filter_a", "filter_b"])
    # Require actual JSON compatibility: no numpy scalars, NaN, or Infinity.
    json.dumps(summary, allow_nan=False)
    matches = [r for r in summary["records"] if
               r["population"] == population and r["cohort"] == cohort
               and r["period"] == period and r["filter_id"] == filter_id]
    assert len(matches) == 1
    return matches[0]


def test_cooldown_selects_before_outcomes_or_filters_and_is_input_order_independent():
    rows = [
        event("a", complete=False, mature=False, net_return=np.nan, filter_a=None),
        event("b", signal_date="2024-01-03", signal_index=2),
        event("c", signal_date="2024-01-30", signal_index=20),
        event("d", signal_date="2024-01-31", signal_index=21),
        event("e", signal_date="2024-02-01", signal_index=22),
    ]
    original = pd.DataFrame(rows)
    selected = nonoverlapping_events(original)
    assert selected.event_id.tolist() == ["a", "d"]
    rows[0].update(complete=True, mature=True, net_return=-0.9, filter_a=False)
    changed = pd.DataFrame(rows).sample(frac=1, random_state=10)
    assert nonoverlapping_events(changed).event_id.tolist() == ["a", "d"]
    assert pd.isna(original.loc[0, "net_return"])


def test_cooldown_is_per_cohort_horizon_stock_and_deterministic_ties():
    rows = [
        event("z"), event("a"), event("b", cohort="legacy_course_breakout"),
        event("c", horizon=60), event("d", stock_id="2303"),
    ]
    result = nonoverlapping_events(pd.DataFrame(rows))
    assert set(result.event_id) == {"a", "b", "c", "d"}
    assert "z" not in result.event_id.tolist()


@pytest.mark.parametrize("change, message", [
    ({"event_id": "one"}, "Duplicate"),
    ({"event_id": "two", "signal_date": "2024-01-01", "signal_index": 2}, "disagrees"),
    ({"event_id": "two", "signal_date": "2024-01-03"}, "multiple signal dates"),
    ({"event_id": "two", "signal_index": 2}, "multiple signal indices"),
])
def test_invalid_identity_or_signal_order_is_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        nonoverlapping_events(pd.DataFrame([event(), event(**change)]))


def test_filter_known_baseline_and_profitable_non_rally_are_not_losses():
    rows = [
        event("a", stock_id="2301", gross_return=.5, net_return=.48),
        event("b", stock_id="2302", gross_return=.1, net_return=.08, filter_a=False),
        event("c", stock_id="2303", gross_return=.01, net_return=-.01, filter_a=False),
        event("d", stock_id="2304", gross_return=-.1, net_return=-.12),
        event("e", stock_id="2305", gross_return=.8, net_return=.78, filter_a=None),
        event("f", stock_id="2306", gross_return=.02, net_return=0),
    ]
    result = record(rows)
    assert result["baseline"]["n"] == 5
    assert result["baseline"]["win"] == 2
    assert result["baseline"]["loss"] == 2
    assert result["baseline"]["breakeven"] == 1
    assert result["baseline"]["rally_count"] == 1
    assert result["baseline"]["mean_net"] == pytest.approx(.086)
    assert result["pass"]["n"] == 3
    assert result["reject"]["n"] == 2
    assert result["rally_retention"] == 1
    assert result["positive_return_removal"] == .5
    assert result["loss_removal"] == .5
    assert result["counts"]["filter_unknown_n"] == 1
    assert result["counts"]["filter_known_candidate_n"] == 5
    # A different filter has its own matched baseline.
    assert record(rows, filter_id="filter_b")["baseline"]["n"] == 6


def test_benchmark_missing_does_not_remove_outcome_or_enter_excess_denominator():
    rows = [
        event("a", stock_id="2301", net_return=.3, benchmark_net_return=.1),
        event("b", stock_id="2302", net_return=-.1, benchmark_net_return=np.nan,
              mfe=np.nan, mae=np.nan),
    ]
    stats = record(rows)["baseline"]
    assert stats["n"] == 2
    assert stats["mean_net"] == pytest.approx(.1)
    assert stats["benchmark_paired_n"] == 1
    assert stats["mean_excess"] == pytest.approx(.2)
    assert stats["mfe_n"] == stats["mae_n"] == 1


def test_same_signal_date_pairs_date_means_not_event_weighted_means():
    rows = [
        event("a", stock_id="2301", net_return=.4),
        event("b", stock_id="2302", net_return=.2),
        event("c", stock_id="2303", net_return=0, filter_a=False),
        event("d", stock_id="2304", signal_date="2024-01-03", signal_index=2,
              entry_date="2024-01-04", net_return=.1),
        event("e", stock_id="2305", signal_date="2024-01-03", signal_index=2,
              entry_date="2024-01-04", net_return=.2, filter_a=False),
        # This large one-sided date must not enter the paired result.
        event("f", stock_id="2306", signal_date="2024-01-04", signal_index=3,
              entry_date="2024-01-05", net_return=9),
    ]
    paired = record(rows)["same_signal_date"]
    assert paired["interpretation"] == "descriptive_noncausal"
    assert paired["n_dates"] == 2
    assert paired["pass_event_n"] == 3
    assert paired["reject_event_n"] == 2
    assert paired["mean_net_pass"] == pytest.approx(.2)
    assert paired["mean_net_reject"] == pytest.approx(.1)
    assert paired["delta_mean_net"] == pytest.approx(.1)
    assert paired["reason"] is None


def test_one_market_regime_value_per_date_has_no_paired_effect():
    rows = [
        event("a"),
        event("b", stock_id="2302", signal_date="2024-01-03", signal_index=2,
              entry_date="2024-01-04", filter_a=False),
    ]
    paired = record(rows)["same_signal_date"]
    assert paired["n_dates"] == 0
    assert paired["delta_mean_net"] is None
    assert paired["reason"] == "no_dates_with_both_groups"


def test_cross_year_is_all_period_only_and_cooldown_does_not_restart():
    rows = [
        event("a", signal_date="2024-12-30", signal_index=240,
              entry_date="2024-12-31", exit_date="2025-01-28"),
        event("b", signal_date="2025-01-02", signal_index=242,
              entry_date="2025-01-03", exit_date="2025-01-31"),
    ]
    assert record(rows)["baseline"]["n"] == 2
    year = record(rows, period="2024")
    assert year["counts"]["candidate_n"] == 1
    assert year["counts"]["boundary_n"] == 1
    assert year["baseline"]["n"] == 0
    assert record(rows, period="2025")["baseline"]["n"] == 1
    assert record(rows, population="nonoverlapping", period="2025")["baseline"]["n"] == 0
    assert record(rows, population="nonoverlapping")["baseline"]["n"] == 1


def test_missing_immature_boundary_and_unknown_counts_remain_visible():
    rows = [
        event("a", stock_id="2301", mature=False, complete=False,
              entry_date=None, exit_date=None, gross_return=np.nan, net_return=np.nan),
        event("b", stock_id="2302", complete=False, exit_date=None, net_return=np.nan),
        event("c", stock_id="2303", complete=False, filter_a=False),
        event("d", stock_id="2304", complete=False, filter_a=None),
        event("e", stock_id="2305", filter_a=None),
        event("f", stock_id="2306"),
    ]
    result = record(rows)
    counts = result["counts"]
    assert counts["candidate_n"] == 6
    assert counts["immature_n"] == 1
    assert counts["missing_future_n"] == 3
    assert counts["complete_in_period_n"] == 2
    assert counts["filter_known_n"] == 1
    assert counts["filter_unknown_n"] == 1
    assert counts["filter_unknown_candidate_n"] == 2
    assert counts["missing_future_by_filter"] == {"pass": 1, "reject": 1, "unknown": 1}
    assert result["baseline"]["n"] == 1


def test_null_dates_all_unknown_and_empty_groups_are_json_safe():
    result = record([event(mature=False, complete=False, entry_date=None,
                           exit_date=None, net_return=np.nan, gross_return=np.nan,
                           filter_a=pd.NA)])
    assert result["baseline"]["n"] == 0
    assert result["baseline"]["mean_net"] is None
    assert result["loss_removal"] is None
    empty = pd.DataFrame([event()]).iloc[:0]
    assert summarize_filters(empty, ["filter_a"])["records"] == []


@pytest.mark.parametrize("bad", ["False", "True", 0, 1])
def test_nonboolean_filter_is_not_silently_coerced(bad):
    with pytest.raises(ValueError, match="must contain booleans"):
        summarize_filters(pd.DataFrame([event(filter_a=bad)]), ["filter_a"])


def test_two_cohorts_and_horizons_are_never_pooled():
    rows = [event("a"), event("b", cohort="legacy_course_breakout", net_return=-.3),
            event("c", horizon=60, threshold=.5, net_return=.1)]
    summary = summarize_filters(pd.DataFrame(rows), ["filter_a"])
    records = [r for r in summary["records"] if r["population"] == "raw" and r["period"] == "all"]
    assert len(records) == 3
    assert all(r["baseline"]["n"] == 1 for r in records)
    assert record(rows, cohort="legacy_course_breakout")["baseline"]["mean_net"] == -.3


@pytest.mark.parametrize("changes, message", [
    ({"entry_date": "2024-01-02"}, "must follow"),
    ({"exit_date": "2024-01-01"}, "cannot precede"),
    ({"signal_date": "2024-02-30"}, "invalid"),
    ({"threshold": 0}, "positive finite"),
    ({"net_return": "missing"}, "nonnumeric"),
])
def test_invalid_dates_or_values_are_explicit_errors(changes, message):
    with pytest.raises(ValueError, match=message):
        summarize_filters(pd.DataFrame([event(**changes)]), ["filter_a"])
