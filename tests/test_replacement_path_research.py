"""Causal replacement decisions and proportional two-leg accounting."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.research_replacement_path_20261003 import (candidate_groups, comparison,
    join_outcome, select_replacement)
from skills.independent_signals import net_unit_return
from skills.independent_three_black import ThreeBlackPath


def path_of(values=None, n=100):
    close = np.asarray(values if values is not None else [100.] * n, dtype=float)
    days = pd.bdate_range("2020-01-01", periods=len(close))
    return ThreeBlackPath(days, close.copy(), close.copy(), np.ones(len(close), dtype=bool),
        close.copy(), close + 1, close - 1, np.full(len(close), 1_000_000.), close - .5)


def event(path, signal=24, sid="A", key=None, score=.3):
    return dict(signal_id=key or f"{sid}-{signal}", stock_id=sid,
        signal_date=str(path.days[signal].date()),
        entry_date=str(path.days[signal + 1].date()) if signal + 1 < len(path.days) else None,
        priority=score)


def original_outcome(origin, path, end=40, gross=1.1):
    entry = int(path.days.get_loc(pd.Timestamp(origin["entry_date"])))
    return dict(origin, status="closed", outcome="profit", gross_return=gross - 1,
        net_return=net_unit_return(gross), unrealized_net_return=None,
        holding_days=end - entry, holding_days_inclusive=end - entry + 1,
        exit_date=str(path.days[end].date()), exit_trigger_date=str(path.days[end - 1].date()),
        observed_end_date=str(path.days[end].date()), exit_reason="three_black", data_issue=None)


def test_replaces_on_fifth_close_and_only_once_with_best_other_stock():
    path, benchmark = path_of(), path_of()
    origin = event(path)
    choices = [event(path, 28, "B", score=9.), event(path, 29, "A", score=99.),
               event(path, 29, "C", key="second", score=.2),
               event(path, 29, "B", key="first", score=.2), event(path, 30, "D", score=999.)]
    decision = select_replacement(origin, path, benchmark, candidate_groups(choices))
    assert decision["replacement_action"] == "replace"
    assert decision["replacement_target_signal_id"] == "first"
    assert decision["replacement_held_sessions"] == 5
    assert decision["replacement_decision_date"] == str(path.days[29].date())
    assert decision["replacement_execution_date"] == str(path.days[30].date())
    assert decision["replacement_incumbent_close_return"] == 0


def test_exact_gap_boundary_and_nonprofitable_gross_trigger():
    path, benchmark = path_of(), path_of()
    origin = event(path)
    at_boundary = candidate_groups([event(path, 29, "B", score=.1)])
    assert select_replacement(origin, path, benchmark, at_boundary)["replacement_action"] == "replace"
    below = candidate_groups([event(path, 29, "B", score=.099)])
    assert select_replacement(origin, path, benchmark, below)["replacement_action"] == "unchanged"
    values = np.full(100, 100.)
    values[29] = 101.
    profitable = path_of(values)
    assert select_replacement(origin, profitable, benchmark, at_boundary)["replacement_action"] == "unchanged"


@pytest.mark.parametrize("reason", ["loss12", "three_black", "time63"])
def test_original_exit_has_priority_over_same_close_replacement(reason):
    values = np.full(100, 100.)
    index = 87 if reason == "time63" else 29
    if reason == "loss12":
        values[index] = 87.
    elif reason == "three_black":
        values[27:30] = [99., 98., 97.]
    path = path_of(values)
    if reason == "three_black":
        opened = path.opened.copy()
        opened[27:30] = values[27:30] + .5
        path = replace(path, opened=opened)
    decision = select_replacement(event(path), path, path_of(), candidate_groups([event(path, index, "B", score=9.)]))
    assert decision["replacement_action"] == "unchanged"
    assert decision["replacement_original_exit_reason"] == reason
    assert decision["replacement_target_signal_id"] is None


def test_future_prices_scores_and_outcomes_do_not_change_selected_target():
    path, benchmark = path_of(), path_of()
    origin = event(path)
    candidates = [event(path, 29, "B", score=.2), event(path, 40, "C", score=3.)]
    expected = select_replacement(origin, path, benchmark, candidate_groups(candidates))
    values = path.close.copy()
    values[30:] *= 10
    future_changed = path_of(values)
    changed = [dict(candidates[0], net_return=-.99, status="unresolved"),
               dict(candidates[1], priority=999., net_return=999.)]
    assert select_replacement(origin, future_changed, benchmark, candidate_groups(changed)) == expected
    short = path_of(path.close[:30])
    truncated_candidate = event(short, 29, "B", score=.2)
    truncated = select_replacement(origin, short, path_of(n=30), candidate_groups([truncated_candidate]))
    assert truncated["replacement_target_signal_id"] == expected["replacement_target_signal_id"]
    assert truncated["replacement_decision_date"] == expected["replacement_decision_date"]
    assert truncated["replacement_action"] == "pending_replacement"
    assert truncated["replacement_execution_date"] is None


def test_missing_relative_history_is_unknown_not_later_reselection():
    values = np.full(100, 100.)
    values[9] = np.nan
    path = path_of(values)
    decision = select_replacement(event(path), path, path_of(), candidate_groups([
        event(path, 29, "B", score=.2), event(path, 45, "C", score=.3)]))
    assert decision["replacement_action"] == "unknown"
    assert decision["replacement_issue"] == "incumbent_relative20_price_or_identity"
    assert decision["replacement_target_signal_id"] is None


def test_sell_old_then_buy_new_counts_both_full_leg_costs():
    values = np.full(100, 100.)
    values[29:] = 95.
    path, benchmark = path_of(values), path_of()
    origin, target = event(path), event(path, 29, "B", score=.2)
    decision = select_replacement(origin, path, benchmark, candidate_groups([target]))
    baseline = original_outcome(origin, path)
    target_outcome = original_outcome(target, path, end=50, gross=1.2)
    row = join_outcome(baseline, decision, path, {target["signal_id"]: target_outcome})
    expected = (1 + net_unit_return(.95)) * (1 + net_unit_return(1.2)) - 1
    assert row["variant_status"] == "closed"
    assert row["variant_net_return"] == pytest.approx(expected, abs=1e-15)
    assert row["variant_net_return"] < net_unit_return(.95 * 1.2)
    assert row["variant_holding_days"] == 25
    assert row["variant_exit_date"] == target_outcome["exit_date"]
    assert row["net_return"] == baseline["net_return"]


@pytest.mark.parametrize("missing_leg", ["old_sale", "new_stock"])
def test_failed_selected_leg_is_preserved_and_never_falls_back(missing_leg):
    path = path_of()
    origin, target, alternative = event(path), event(path, 29, "B", score=.2), event(path, 29, "C", score=.15)
    decision = select_replacement(origin, path, path_of(), candidate_groups([target, alternative]))
    target_outcome = original_outcome(target, path)
    if missing_leg == "old_sale":
        volume = path.volume.copy()
        volume[30] = 0
        path = replace(path, volume=volume)
    else:
        target_outcome.update(status="unresolved", net_return=None, gross_return=None, data_issue="missing_or_invalid_price")
    row = join_outcome(original_outcome(origin, path), decision, path,
        {target["signal_id"]: target_outcome, alternative["signal_id"]: original_outcome(alternative, path)})
    assert row["replacement_target_signal_id"] == target["signal_id"]
    assert row["variant_status"] == "unresolved"
    assert row["variant_net_return"] is None
    assert row["variant_data_issue"]


def test_pending_new_path_stays_unrealized_and_unchanged_preserves_baseline():
    path = path_of()
    origin, target = event(path), event(path, 29, "B", score=.2)
    decision = select_replacement(origin, path, path_of(), candidate_groups([target]))
    baseline, target_outcome = original_outcome(origin, path), original_outcome(target, path)
    target_outcome.update(status="open", net_return=None, unrealized_net_return=net_unit_return(1.1),
                          exit_date=None, exit_trigger_date=None, exit_reason=None)
    row = join_outcome(baseline, decision, path, {target["signal_id"]: target_outcome})
    assert row["variant_status"] == "open"
    assert row["variant_net_return"] is None
    assert row["variant_unrealized_net_return"] is not None
    unchanged = select_replacement(origin, path, path_of(), {})
    copied = join_outcome(baseline, unchanged, path, {})
    assert copied["variant_net_return"] == baseline["net_return"]
    assert copied["variant_status"] == baseline["status"]


def test_unknown_comparison_does_not_assign_zero_and_retention_uses_new_result():
    path = path_of()
    origin = event(path)
    base = original_outcome(origin, path, gross=1.5)
    decision = select_replacement(origin, path, path_of(), {})
    first = join_outcome(base, decision, path, {})
    second = dict(first, signal_id="second", variant_status="unresolved", variant_net_return=None,
                  variant_data_issue="missing")
    first["variant_net_return"] = .2
    result = comparison([first, second])
    assert result["original_closed_opportunities"] == 2
    assert result["paired_closed_opportunities"] == 1
    assert result["full_original_opportunity_mean"] is None
    assert result["unknown_never_assumed_zero"] is True
    assert result["retained_winner_count"] == 1
    assert result["retained_return30_count"] == 0
    assert result["unknown_or_unfinished_original_return30"] == 1


def test_latest_original_signal_has_no_fake_entry_or_replacement():
    path = path_of()
    decision = select_replacement(event(path, 99), path, path_of(), {})
    assert decision["replacement_action"] == "not_entered"
    assert decision["replacement_execution_date"] is None
