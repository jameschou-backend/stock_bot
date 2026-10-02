"""Small causal/data-contract tests; no research trials or external data access."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts.export_signal_explorer import (PRICE_COLUMNS, build_payload,
    build_signal_rows, json_for_script, pack_prices, render_html, unpack_price)


def fixture_data():
    days = pd.DatetimeIndex(["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06"])
    adjusted = pd.DataFrame({"2330": [100., 101., 102., 103., 104.]}, index=days)
    eligible = pd.DataFrame({"2330": True}, index=days)
    raw = np.array([200., 202., 204., 103., 104.])
    quotes = pd.DataFrame(dict(stock_id="2330", date=days, open=raw - 1,
                               high=raw + 2, low=raw - 2, close=raw, volume=1000.))
    rows, features = [], []
    for i in (2, 4):
        day = str(days[i].date())
        key = "signal-" + day
        entry = str(days[i + 1].date()) if i + 1 < len(days) else None
        rows.append(dict(signal_id=key, stock_id="2330", name="台積電", market="TWSE",
            signal_date=day, entry_date=entry, daily_rank=1, daily_candidate_count=1,
            rank_priority=.2, rank_stock_return20=.3, rank_benchmark_return20=.1,
            status="closed" if entry else "not_entered", net_return=.5 if entry else None))
        features.append(dict(event_id=key, stock_id="2330", signal_date=day, entry_date=entry,
            relative20=.2, volume_ratio=2., previous60_high_adjusted=99.,
            signal_close_adjusted=adjusted.iloc[i, 0], signal_close_raw=raw[i],
            mean20_turnover=70e6, median20_turnover=60e6, trend_state="ON"))
    return pd.DataFrame(rows), pd.DataFrame(features), adjusted, adjusted.copy(), quotes, eligible


def test_all_market_days_and_all_signals_survive_without_outcomes():
    data = fixture_data()
    payload = build_payload(*data, data_as_of="2026-01-06")
    assert [d["signal_count"] for d in payload["days"]] == [1, 0, 1]
    assert payload["days"][1] == {"date": "2026-01-05", "signal_count": 0, "signal_ids": []}
    assert payload["metadata"]["warmup_sessions"] == 2
    assert payload["metadata"]["signal_count"] == 2
    assert payload["metadata"]["outcomes_included"] is False
    assert all(s["signal_date"] >= "2026-01-01" for s in payload["signals"])
    assert payload["stocks"]["2330"]["prices"][0][0] == "2025-12-30"
    assert payload["signals"][-1]["entry_date"] is None
    for row in payload["signals"]:
        assert not {"status", "exit_date", "net_return", "entry_price", "mfe"}.intersection(row)


def test_future_outcome_changes_do_not_affect_t0_signal_rows():
    ranks, features, adjusted, *_ = fixture_data()
    original = build_signal_rows(ranks, features, adjusted.index, 2026, "2026-01-06")
    changed = ranks.copy()
    changed["net_return"] = [-.99, 999.]
    changed["status"] = ["unresolved", "closed"]
    changed["exit_date"] = "2099-01-01"
    assert build_signal_rows(changed, features, adjusted.index, 2026, "2026-01-06") == original


def test_same_factor_adjusts_every_ohlc_and_marker_exactly():
    data = fixture_data()
    payload = build_payload(*data, data_as_of="2026-01-06")
    assert payload["price_columns"] == PRICE_COLUMNS
    bars = [unpack_price(row) for row in payload["stocks"]["2330"]["prices"]]
    signal = payload["signals"][0]
    before, after = bars[2:4]
    assert before["adjustment_factor"] == .5
    assert after["adjustment_factor"] == 1.
    for bar in bars:
        for field in ("open", "high", "low"):
            assert bar[field] == bar["raw_" + field] * bar["adjustment_factor"]
    assert before["close"] == signal["signal_close_adjusted"] == 102.
    assert before["raw_close"] == signal["signal_close_raw"] == 204.
    assert all(bar["quality_issue"] is None for bar in bars)


def test_future_price_mutation_cannot_change_earlier_candles_or_signals():
    ranks, features, a, b, quotes, eligible = fixture_data()
    original = pack_prices(a.index, quotes, a["2330"], b["2330"], eligible["2330"])[0]
    changed_a, changed_b, changed_q = a.copy(), b.copy(), quotes.copy()
    changed_a.iloc[-1, 0] *= 5
    changed_b.iloc[-1, 0] *= 5
    changed_q.loc[changed_q.index[-1], ["open", "high", "low", "close"]] *= 5
    changed = pack_prices(a.index, changed_q, changed_a["2330"], changed_b["2330"], eligible["2330"])[0]
    assert changed[:-1] == original[:-1]
    truncated = pack_prices(a.index[:-1], quotes, a["2330"], b["2330"], eligible["2330"])[0]
    assert truncated == original[:-1]


@pytest.mark.parametrize("case,issue", [
    ("missing", "missing_or_invalid_price"),
    ("inverted", "raw_ohlc_conflict"),
    ("zero_volume", "missing_or_nonpositive_volume"),
    ("identity", "historical_identity_or_eligibility"),
    ("adjustment", "daily_adjustment_conflict"),
])
def test_bad_bars_are_explicit_and_not_forward_filled(case, issue):
    _, _, a, b, quotes, eligible = fixture_data()
    if case == "missing":
        quotes = quotes.drop(index=2)
    elif case == "inverted":
        quotes.loc[2, "high"] = 1.
    elif case == "zero_volume":
        quotes.loc[2, "volume"] = 0.
    elif case == "identity":
        eligible.iloc[2, 0] = False
    else:
        b.iloc[2, 0] *= 2
    packed, counts = pack_prices(a.index, quotes, a["2330"], b["2330"], eligible["2330"])
    assert packed[2][8] == issue
    assert counts[issue] >= 1
    if case == "missing":
        assert packed[2][1:7] == [None] * 6


def test_signal_join_fails_on_changed_marker_or_incomplete_daily_population():
    data = fixture_data()
    altered = data[1].copy()
    altered.loc[0, "signal_close_adjusted"] = 999.
    with pytest.raises(ValueError, match="marker price basis"):
        build_payload(data[0], altered, *data[2:], data_as_of="2026-01-06")
    ranks = data[0].copy()
    ranks.loc[0, "daily_candidate_count"] = 2
    with pytest.raises(ValueError, match="Incomplete same-day"):
        build_signal_rows(ranks, data[1], data[2].index, 2026, "2026-01-06")


def test_json_is_safe_inside_inert_script_without_losing_original_text():
    data = {"name": "</script><script>alert(1)</script><!--&\u2028\u2029"}
    encoded = json_for_script(data)
    assert "<" not in encoded and "&" not in encoded and "\u2028" not in encoded
    assert json.loads(encoded) == data
    template = '<script id="signal-data" type="application/json"><!-- SIGNAL_DATA --></script>'
    html = render_html(template, data)
    assert html.count("</script>") == 1
    assert "<!-- SIGNAL_DATA -->" not in html
    with pytest.raises(ValueError, match="exactly one"):
        render_html(template + template, data)
