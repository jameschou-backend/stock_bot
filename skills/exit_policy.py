"""Pure decisions for the preregistered scenario-based exit comparison.

The caller supplies only already available signals. This module reads no price
source and does not calculate realised profit, orders, fills or portfolio NAV.
An exit is a decision to request execution; an unfilled exit must remain latched
in the execution engine, even if a later call would decide to hold.
"""
from __future__ import annotations

import math
from numbers import Integral, Real


TOLERANCE = 1e-12
MODES = ("fixed63", "loss12", "trail20_12", "weak20", "market_weak", "trend126", "adaptive")
MODE_LABELS = {
    "fixed63": "固定持有63日",
    "loss12": "跌12%停損",
    "trail20_12": "漲20%後回落12%停利",
    "weak20": "個股趨勢轉弱出場",
    "market_weak": "大盤與個股同步轉弱出場",
    "trend126": "強勢延長，最長126日",
    "adaptive": "依虧損、趨勢與大盤調整出場",
}
REASON_LABELS = {
    "time63": "持有滿63日",
    "loss12": "入場價格訊號下跌12%",
    "trailing12": "上漲20%後，距持有期高點回落12%",
    "trend_break": "連續跌破20日均線且弱於0050",
    "market_weak": "大盤連續轉弱且個股弱於0050",
    "time63_weak": "持有滿63日，未確認強勢延長條件",
    "hard_time126": "達到126日最長持有期限",
}
PHASE_LABELS = {
    "holding": "持有觀察",
    "trailing_armed": "已啟動移動停利",
    "extended": "強勢延長一天",
    "no_signal": "訊號不可用，未新增價格判斷",
    "exiting": "提出出場，等待可成交",
}
_BOOLEAN_FIELDS = ("has_signal", "below_ma20_two", "market_off_two", "strong_trend")
_RETURN_FIELDS = ("entry_return", "peak_return", "peak_drawdown", "relative20")
_REQUIRED_FIELDS = frozenset(("held_sessions", *_BOOLEAN_FIELDS, *_RETURN_FIELDS))


def _validate(context: dict, mode: str) -> None:
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if not isinstance(context, dict):
        raise ValueError("context must be a dict with all required signal fields")
    missing = _REQUIRED_FIELDS.difference(context)
    if missing:
        raise ValueError(f"context is missing required fields: {', '.join(sorted(missing))}")
    held = context["held_sessions"]
    if isinstance(held, bool) or not isinstance(held, Integral) or held < 0:
        raise ValueError("held_sessions must be a nonnegative integer market-session count")
    for field in _BOOLEAN_FIELDS:
        if not isinstance(context[field], bool):
            raise ValueError(f"{field} must be a bool, not a truthy value")
    for field in _RETURN_FIELDS:
        value = context[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        ):
            raise ValueError(f"{field} must be a finite real number or explicit None")


def _result(reason: str | None = None, *, phase: str = "holding", extend: bool = False) -> dict:
    return {"exit": reason is not None, "reason": reason,
            "phase": "exiting" if reason is not None else phase, "extend": extend}


def decide_exit(context: dict, mode: str) -> dict:
    """Decide from a lagged signal context; never infer missing market data.

    ``held_sessions`` is execution index minus entry index, so the entry day is
    zero. ``has_signal`` means the caller has the permissible prior signal, not
    today's execution price. The caller must validate its timestamp and source.
    ``below_ma20_two`` requires two adjacent valid closes after entry.
    ``peak_return`` is the maximum since entry and must never reset: once 20%
    has been reached, the trailing rule remains armed. These returns are price
    signals and must not be presented as the account's cost-inclusive P&L.

    All fields are required, including for fixed63. Explicit None means that
    numeric condition is unavailable; NaN, infinity and malformed fields fail
    loudly. A missing signal disables every price-based rule and extension,
    while the independent session deadlines still apply. Inclusive +/-12%
    and +20% comparisons tolerate 1e-12 floating point noise; relative20 < 0
    remains a strict comparison. The 126-session hard deadline takes priority
    over the adaptive early-exit reasons at that age.

    Extra caller metadata is ignored. Neither the context nor any caller-owned
    state is changed; permanent arms and pending orders remain caller-owned.
    """
    _validate(context, mode)
    held, available = context["held_sessions"], context["has_signal"]
    extension_mode = mode in ("trend126", "adaptive")
    if extension_mode and held >= 126:
        return _result("hard_time126")

    peak = context["peak_return"]
    armed = available and peak is not None and peak >= .20 - TOLERANCE
    if available:
        entry, drawdown = context["entry_return"], context["peak_drawdown"]
        relative = context["relative20"]
        weaker = relative is not None and relative < 0
        if mode in ("loss12", "adaptive") and entry is not None and entry <= -.12 + TOLERANCE:
            return _result("loss12")
        if (mode in ("trail20_12", "adaptive") and armed
                and drawdown is not None and drawdown <= -.12 + TOLERANCE):
            return _result("trailing12")
        if mode in ("market_weak", "adaptive") and context["market_off_two"] and weaker:
            return _result("market_weak")
        if mode in ("weak20", "adaptive") and context["below_ma20_two"] and weaker:
            return _result("trend_break")

    if held >= 63:
        if extension_mode:
            if available and context["strong_trend"]:
                return _result(phase="extended", extend=True)
            return _result("time63_weak")
        return _result("time63")
    if not available:
        return _result(phase="no_signal")
    if armed and mode in ("trail20_12", "adaptive"):
        return _result(phase="trailing_armed")
    return _result()
