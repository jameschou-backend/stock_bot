from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from skills.surge_statistics import matched_controls, rule_statistics


def row(sid, event=False, rule=False, **overrides):
    value = dict(stock_id=sid, signal_date="2023-01-03", entry_date="2023-01-04", exit_date="2023-02-01",
                 market="TWSE", industry="24", adv20=100_000_000., liquidity_bin=2,
                 forward_return=.4 if event is True else .02 if event is False else None,
                 event=event, phase="discovery", all_eligible=True, rule=rule)
    value.update(overrides)
    return value


def statistic(rows, rule="rule", scope="all"):
    return next(item for item in rule_statistics(pd.DataFrame(rows), ["all_eligible", "rule"])
                if item["rule"] == rule and item["scope"] == scope)


def test_unknown_labels_and_predictions_are_not_false_negatives():
    rows = [row("1001", True, True), row("1002", False, True), row("1003", True, False),
            row("1004", False, False), row("1005", None, True), row("1006", True, None),
            row("1007", False, np.nan), row("1008", None, pd.NA)]
    result = statistic(rows)
    assert [result[key] for key in ("tp", "fp", "fn", "tn")] == [1, 1, 1, 1]
    assert result["total_eligible"] == 8
    assert result["observed_labels"] == 6 and result["unknown_labels"] == 2
    assert result["triggers"] == 3 and result["outcome_unknown_triggers"] == 1
    assert result["prediction_unknown"] == 3 and result["evaluated_labels"] == 4
    assert result["known_event_prediction_unknown"] == 1
    assert result["known_nonevent_prediction_unknown"] == 1
    assert result["precision"] == result["recall"] == result["false_positive_rate"] == .5
    assert result["false_discovery_rate"] == .5 and result["baseline_rate"] == .5
    assert result["lift"] == 1
    assert result["precision_lower_bound"] == 1 / 3
    assert result["precision_upper_bound"] == 2 / 3
    assert result["lift_ci_low"] is None
    json.dumps(result, allow_nan=False)


def test_all_eligible_baseline_and_no_trigger_are_defined():
    rows = [row("1001", True), row("1002", False), row("1003", None)]
    base = statistic(rows, "all_eligible")
    assert base["precision"] == base["baseline_rate"] == .5
    assert base["lift"] == 1 and base["recall"] == 1
    result = statistic(rows)
    for key in ("precision", "lift", "false_discovery_rate", "precision_lower_bound", "precision_upper_bound"):
        assert result[key] is None
    assert result["recall"] == 0


def test_phase_and_exit_year_exclude_boundary_and_crossing_discovery():
    rows = [row("1001", True, True, signal_date="2022-12-19", entry_date="2022-12-20", exit_date="2023-01-20"),
            row("1002", True, True, signal_date="2024-12-20", entry_date="2024-12-23", exit_date="2025-01-24", phase="boundary"),
            row("1003", False, True, signal_date="2025-01-03", entry_date="2025-01-06", exit_date="2025-02-03", phase="replication")]
    output = rule_statistics(pd.DataFrame(rows), ["rule"])
    by_scope = {item["scope"]: item for item in output}
    assert set(by_scope) == {"all", "discovery", "replication", "2023", "2025"}
    assert by_scope["all"]["total_eligible"] == 2
    assert by_scope["2025"]["total_eligible"] == 1
    assert by_scope["all"]["boundary_rows_excluded"] == 1
    bad = deepcopy(rows)
    bad[1]["phase"] = "discovery"
    with pytest.raises(ValueError, match="finish"):
        rule_statistics(pd.DataFrame(bad), ["rule"])


def test_block_bootstrap_is_deterministic_and_not_individual_stock_resampling():
    rows = []
    for i, signal in enumerate(pd.date_range("2023-01-02", periods=6, freq="MS")):
        dates = dict(signal_date=signal.strftime("%Y-%m-%d"),
                     entry_date=(signal + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                     exit_date=(signal + pd.Timedelta(days=21)).strftime("%Y-%m-%d"))
        rows.extend([row("1001", True, True, **dates), row("1002", False, i % 2 == 0, **dates),
                     row("1003", False, False, **dates)])
    first = statistic(rows)
    assert first["bootstrap_dates"] == 6 and first["bootstrap_valid_replicates"] == 1000
    assert first["lift_ci_low"] <= first["lift"] <= first["lift_ci_high"]
    shuffled = pd.DataFrame(rows).sample(frac=1, random_state=72).to_dict("records")
    second = statistic(shuffled)
    assert first == second
    # Duplicate every date block's contents with different stock ids. Whole-date
    # bootstrap uncertainty must stay identical, unlike an IID stock bootstrap.
    doubled = rows + [{**item, "stock_id": "x" + item["stock_id"]} for item in rows]
    doubled_result = statistic(doubled)
    for key in ("lift", "lift_ci_low", "lift_ci_high", "bootstrap_dates"):
        assert doubled_result[key] == first[key]
    assert statistic(rows[:12])["lift_ci_low"] is None


def test_zero_events_and_empty_scopes_are_json_safe():
    result = statistic([row("1001", False, True)])
    assert result["baseline_rate"] == 0 and result["lift"] is None
    empty = statistic([row("1001", False, True)], scope="replication")
    assert empty["total_eligible"] == 0 and empty["precision"] is None
    json.dumps([result, empty], allow_nan=False)


@pytest.mark.parametrize("field,value", [("rule", 1), ("event", "true"), ("forward_return", float("inf")),
                                        ("entry_date", "2023-01-03"), ("phase", "test")])
def test_invalid_observation_values_are_rejected(field, value):
    observation = row("1001", True, True)
    observation[field] = value
    with pytest.raises(ValueError):
        statistic([observation])


def test_known_labels_require_finite_return_and_duplicate_keys_fail():
    with pytest.raises(ValueError, match="Known outcomes"):
        statistic([row("1001", True, True, forward_return=np.nan)])
    with pytest.raises(ValueError, match="Duplicate"):
        statistic([row("1001"), row("1001")])


def test_matching_uses_exact_stratum_nearest_log_adv_and_deterministic_ties():
    rows = [row("9001", True, True), row("1004", False, False, adv20=101_000_000.),
            row("1002", False, True, adv20=100_000_000.), row("1001", False, False, adv20=100_000_000.),
            row("1003", False, False, adv20=200_000_000.),
            row("1010", False, False, market="TPEX"), row("1011", False, False, industry="25"),
            row("1012", False, False, liquidity_bin=3), row("1013", None, False)]
    pairs, summary = matched_controls(pd.DataFrame(rows), ["rule", "adv20"])
    assert pairs["control_stock_id"].tolist() == ["1001", "1002", "1004"]
    assert pairs["case_id"].tolist() == ["2023-01-03:9001"] * 3
    assert pairs["diff_rule"].tolist() == [1., 0., 1.]
    assert summary["by_scope"]["all"]["matched_cases"] == 1
    assert summary["by_scope"]["all"]["feature_differences"]["rule"]["mean_pair_difference"] == 2 / 3
    assert summary["matching_with_replacement"] and not summary["independent_pairs"]
    shuffled = pd.DataFrame(rows).sample(frac=1, random_state=18)
    other, other_summary = matched_controls(shuffled, ["rule", "adv20"])
    pd.testing.assert_frame_equal(pairs, other)
    assert summary == other_summary
    json.dumps(summary, allow_nan=False)
    json.dumps(pairs.to_dict("records"), allow_nan=False)


def test_matching_does_not_fallback_reuses_controls_and_excludes_boundaries():
    rows = [row("9001", True, True), row("9002", True, False), row("1001", False, None),
            row("9003", True, True, industry="99"), row("9004", True, True, industry=None),
            row("9005", True, True, phase="boundary"),
            row("9006", True, True, signal_date="2025-01-03", entry_date="2025-01-06", exit_date="2025-02-03", phase="replication")]
    pairs, summary = matched_controls(pd.DataFrame(rows), ["rule"])
    assert pairs["control_stock_id"].tolist() == ["1001", "1001"]
    assert pairs["diff_rule"].tolist() == [None, None]
    all_rows = summary["by_scope"]["all"]
    assert all_rows["total_cases"] == 5 and all_rows["matched_cases"] == 2 and all_rows["unmatched_cases"] == 3
    assert all_rows["reused_control_observations"] == 1
    assert summary["by_scope"]["replication"]["unmatched_cases"] == 1
    assert all_rows["feature_differences"]["rule"]["mean_case_difference"] is None
    json.dumps(pairs.to_dict("records"), allow_nan=False)


def test_no_events_or_no_controls_produces_empty_pairs_with_summary():
    for event in (True, False, None):
        pairs, summary = matched_controls(pd.DataFrame([row("1001", event)]), ["rule"])
        assert pairs.empty and "diff_rule" in pairs
        assert summary["by_scope"]["all"]["matched_cases"] == 0
        assert summary["by_scope"]["all"]["unmatched_cases"] == int(event is True)
        json.dumps(summary, allow_nan=False)
