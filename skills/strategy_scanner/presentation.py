"""Standalone research-scanner HTML with non-executable JSON embedding."""
from collections.abc import Mapping
from datetime import date
import json
from pathlib import Path


TEMPLATE = Path(__file__).resolve().parents[2] / "ui/multi_strategy_scanner.html"
MARKER = "__SCAN_DATA__"


def render_html(payload: Mapping) -> str:
    """Render a scan, never upgrade it to a trade instruction or live result.

    Unknown/missing per-strategy results remain visible as unknown in the UI.
    The scanner owns financial/schema validation; this boundary checks the
    presentation contract and escapes characters which can end a script block.
    """
    if not isinstance(payload, Mapping) or payload.get("schema") != "multi_strategy_scan_v1":
        raise ValueError("Expected multi_strategy_scan_v1 scanner payload")
    if payload.get("live_qualified") is not False:
        raise ValueError("Research scanner requires explicit live_qualified=false")
    if not isinstance(payload.get("strategies"), list) or not isinstance(payload.get("days"), list):
        raise ValueError("Scanner strategies and days must be lists")
    for key in ("start", "end", "source_end"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError("Scanner requires a nonempty " + key)
        if date.fromisoformat(payload[key]).isoformat() != payload[key]:
            raise ValueError("Scanner dates must use YYYY-MM-DD")
    if not payload["start"] <= payload["end"] <= payload["source_end"]:
        raise ValueError("Scanner dates exceed the declared source coverage")
    encoded = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for literal, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                             ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(literal, escaped)
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(MARKER) != 1:
        raise ValueError("Scanner template must contain exactly one data marker")
    return template.replace(MARKER, encoded)
