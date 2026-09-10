"""Synthetic contexts only: no price source, backtest or portfolio return run."""
from copy import deepcopy

import pytest

from skills.exit_policy import (
    MODE_LABELS, MODES, PHASE_LABELS, REASON_LABELS, TOLERANCE, decide_exit,
)


def context(**changes):
    row = {"held_sessions": 10, "has_signal": True, "entry_return": .05,
           "peak_return": .10, "peak_drawdown": -.04, "below_ma20_two": False,
           "relative20": .01, "market_off_two": False, "strong_trend": False}
    return row | changes


@pytest.mark.parametrize("mode", MODES)
def test_no_trigger_holds_and_returns_exact_schema(mode):
    assert decide_exit(context(), mode) == {
        "exit": False, "reason": None, "phase": "holding", "extend": False}


@pytest.mark.parametrize("mode", MODES)
def test_exact_63_deadline_without_strength(mode):
    assert not decide_exit(context(held_sessions=62), mode)["exit"]
    result = decide_exit(context(held_sessions=63), mode)
    assert result == {"exit": True, "reason": "time63_weak" if mode in (
        "trend126", "adaptive") else "time63", "phase": "exiting", "extend": False}


def test_fixed63_ignores_valid_price_conditions_but_not_time():
    row = context(entry_return=-.9, peak_return=.5, peak_drawdown=-.9,
                  below_ma20_two=True, relative20=-.9, market_off_two=True)
    assert not decide_exit(row, "fixed63")["exit"]
    assert decide_exit(row | {"held_sessions": 63}, "fixed63")["reason"] == "time63"


@pytest.mark.parametrize("value", [-.5, -.12, -.12 + TOLERANCE / 2])
def test_loss12_inclusive_threshold(value):
    assert decide_exit(context(entry_return=value), "loss12")["reason"] == "loss12"


def test_loss_tolerance_is_small_and_unavailable_return_is_not_loss():
    assert not decide_exit(context(entry_return=-.12 + 2*TOLERANCE), "loss12")["exit"]
    assert not decide_exit(context(entry_return=None), "loss12")["exit"]


@pytest.mark.parametrize("peak,drawdown", [(.2, -.12), (.2 - TOLERANCE/2, -.12 + TOLERANCE/2), (1., -.8)])
def test_trailing_requires_20_percent_arm_then_12_percent_peak_drop(peak, drawdown):
    assert decide_exit(context(peak_return=peak, peak_drawdown=drawdown), "trail20_12")["reason"] == "trailing12"


@pytest.mark.parametrize("peak,drawdown", [(.2-2*TOLERANCE, -.5), (.3, -.12+2*TOLERANCE), (None, -.5), (.3, None)])
def test_trailing_does_not_invent_missing_arm_or_peak_drop(peak, drawdown):
    assert not decide_exit(context(peak_return=peak, peak_drawdown=drawdown), "trail20_12")["exit"]


def test_trailing_arm_is_peak_since_entry_not_current_return():
    first = decide_exit(context(entry_return=.4, peak_return=.4, peak_drawdown=0.), "trail20_12")
    later = decide_exit(context(entry_return=.19, peak_return=.4, peak_drawdown=-.15), "trail20_12")
    assert first["phase"] == "trailing_armed" and not first["exit"]
    assert later["reason"] == "trailing12"
    # No entry-loss shortcut was added to this single-variable comparison.
    assert not decide_exit(context(entry_return=-.3, peak_return=.1), "trail20_12")["exit"]


@pytest.mark.parametrize("mode,flag,reason", [
    ("weak20", "below_ma20_two", "trend_break"),
    ("market_weak", "market_off_two", "market_weak"),
])
def test_weakness_requires_both_confirmation_and_strict_underperformance(mode, flag, reason):
    assert decide_exit(context(**{flag: True, "relative20": -.01}), mode)["reason"] == reason
    for relative in (0., .01, None):
        assert not decide_exit(context(**{flag: True, "relative20": relative}), mode)["exit"]
    assert not decide_exit(context(relative20=-.2), mode)["exit"]
    assert decide_exit(context(**{flag: True, "relative20": -1e-14}), mode)["reason"] == reason


def test_trend126_does_not_add_an_early_exit():
    row = context(held_sessions=62, entry_return=-.8, peak_return=.5,
                  peak_drawdown=-.8, below_ma20_two=True, relative20=-.5,
                  market_off_two=True)
    assert not decide_exit(row, "trend126")["exit"]


@pytest.mark.parametrize("mode", ["trend126", "adaptive"])
def test_extensions_stop_at_126_even_if_strength_continues(mode):
    assert decide_exit(context(held_sessions=62, strong_trend=True), mode)["extend"] is False
    for age in (63, 64, 125):
        assert decide_exit(context(held_sessions=age, strong_trend=True), mode) == {
            "exit": False, "reason": None, "phase": "extended", "extend": True}
    for age in (126, 200):
        result = decide_exit(context(held_sessions=age, strong_trend=True), mode)
        assert result["reason"] == "hard_time126" and result["extend"] is False
    assert decide_exit(context(held_sessions=64), mode)["reason"] == "time63_weak"


def test_adaptive_reason_priority_is_fixed_and_not_strength_dependent():
    row = context(held_sessions=80, entry_return=-.2, peak_return=.5,
                  peak_drawdown=-.2, below_ma20_two=True, relative20=-.5,
                  market_off_two=True, strong_trend=True)
    assert decide_exit(row, "adaptive")["reason"] == "loss12"
    row["entry_return"] = .1
    assert decide_exit(row, "adaptive")["reason"] == "trailing12"
    row["peak_drawdown"] = -.05
    assert decide_exit(row, "adaptive")["reason"] == "market_weak"
    row["market_off_two"] = False
    assert decide_exit(row, "adaptive")["reason"] == "trend_break"
    row["below_ma20_two"] = False
    assert decide_exit(row, "adaptive")["extend"] is True
    row["strong_trend"] = False
    assert decide_exit(row, "adaptive")["reason"] == "time63_weak"
    row.update(held_sessions=126, entry_return=-.5)
    assert decide_exit(row, "adaptive")["reason"] == "hard_time126"


@pytest.mark.parametrize("mode", MODES)
def test_invalid_signal_cannot_use_stale_fields_or_today_price_to_exit_or_extend(mode):
    row = context(has_signal=False, entry_return=-.9, peak_return=.9,
                  peak_drawdown=-.9, below_ma20_two=True, relative20=-.9,
                  market_off_two=True, strong_trend=True)
    row["today_execution_price"] = 1000.  # Not a permitted replacement signal.
    assert decide_exit(row, mode) == {"exit": False, "reason": None, "phase": "no_signal", "extend": False}
    result = decide_exit(row | {"held_sessions": 63}, mode)
    assert result["exit"] and not result["extend"]
    assert result["reason"] == ("time63_weak" if mode in ("trend126", "adaptive") else "time63")


@pytest.mark.parametrize("mode", MODES)
def test_explicit_missing_price_metrics_do_not_trigger_conditions(mode):
    row = context(entry_return=None, peak_return=None, peak_drawdown=None,
                  relative20=None, below_ma20_two=True, market_off_two=True)
    assert not decide_exit(row, mode)["exit"]


@pytest.mark.parametrize("key", list(context()))
def test_missing_context_key_is_an_explicit_error_even_for_fixed_baseline(key):
    row = context()
    del row[key]
    with pytest.raises(ValueError, match=key):
        decide_exit(row, "fixed63")


@pytest.mark.parametrize("key", ["entry_return", "peak_return", "peak_drawdown", "relative20"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "0.2", True, object()])
def test_malformed_numeric_fields_are_rejected_instead_of_truthiness_or_fallback(key, value):
    with pytest.raises(ValueError, match=key):
        decide_exit(context(has_signal=False, **{key: value}), "adaptive")


@pytest.mark.parametrize("key", ["has_signal", "below_ma20_two", "market_off_two", "strong_trend"])
@pytest.mark.parametrize("value", [None, 0, 1, "False", float("nan")])
def test_boolean_fields_must_be_actual_booleans(key, value):
    with pytest.raises(ValueError, match=key):
        decide_exit(context(**{key: value}), "adaptive")


@pytest.mark.parametrize("age", [-1, True, 63., "63", None])
def test_age_is_a_nonnegative_integer_session_count(age):
    with pytest.raises(ValueError, match="held_sessions"):
        decide_exit(context(held_sessions=age), "fixed63")
    assert not decide_exit(context(held_sessions=0), "fixed63")["exit"]


@pytest.mark.parametrize("mode", ["", "trailing", "ADAPTIVE", None, [], 63])
def test_unknown_mode_is_not_silently_changed(mode):
    with pytest.raises(ValueError, match="mode"):
        decide_exit(context(), mode)


def test_non_dictionary_context_is_rejected():
    with pytest.raises(ValueError, match="context"):
        decide_exit([], "fixed63")


def test_pure_calls_preserve_inputs_do_not_latch_orders_and_have_complete_labels():
    row = context(entry_return=-.5, metadata={"note": ["original"]})
    before = deepcopy(row)
    result = decide_exit(row, "adaptive")
    result["reason"] = "caller-modified"
    assert row == before
    assert decide_exit(row, "adaptive")["reason"] == "loss12"
    assert not decide_exit(context(), "adaptive")["exit"]
    assert set(MODE_LABELS) == set(MODES)
    assert all(REASON_LABELS.values()) and all(PHASE_LABELS.values())
