from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skills.strategy_scanner.public_rules import (
    PUBLIC_CATALOG, PUBLIC_IDS, _ema, _inputs, _wilder, add_public_rules,
)


def frames(close, *, volume=None, amount=None, high=None, low=None):
    values = np.asarray(close, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    index = pd.bdate_range("2024-01-02", periods=len(values))
    columns = [str(2300 + j) for j in range(values.shape[1])]
    def wide(x):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        return pd.DataFrame(np.broadcast_to(arr, values.shape).copy(), index=index, columns=columns)
    c = wide(values)
    v = wide(1_000_000 if volume is None else volume)
    return dict(c=c, h=wide(values + 1 if high is None else high),
                l=wide(values - 1 if low is None else low), v=v,
                a=c*v if amount is None else wide(amount), close=c.copy(), open=c.copy(),
                valid=pd.DataFrame(True, index=index, columns=columns),
                eligible=pd.DataFrame(True, index=index, columns=columns))


def evaluate(f):
    z, rules = {}, {}
    def add(sid, match, fields, rule, *, known=None):
        available = f["valid"].eq(True) & f["eligible"].eq(True)
        for field in fields:
            assert field in z, (sid, field)
            available &= np.isfinite(z[field])
        if known is not None:
            available &= known
        assert sid not in rules
        rules[sid] = dict(match=match, known=available, fields=fields, rule=rule)
    add_public_rules(f, z, add)
    return z, rules


def final_match(rules, sid):
    row = rules[sid]
    return bool(row["known"].iloc[-1, 0] and row["match"].iloc[-1, 0])


def test_metadata_has_primary_sources_and_explicit_non_live_conventions():
    assert len(PUBLIC_IDS) == 12 == len(set(PUBLIC_IDS))
    assert PUBLIC_IDS == tuple(row["id"] for row in PUBLIC_CATALOG)
    root = Path(__file__).resolve().parents[1]
    for row in PUBLIC_CATALOG:
        assert row["status"] == "active" and row["kind"] == "entry"
        assert row["version"] == "scanner_public_v1_20261005"
        assert row["params"]["amount20_min"] == 50_000_000
        assert row["source_urls"]
        assert all(url.startswith(("https://chartschool.stockcharts.com/", "https://www.bollingerbands.com/")) for url in row["source_urls"])
        assert all((root / source).is_file() for source in row["source_paths"])
        assert row["live_qualified"] is False and row["returns_inherited"] is False
        assert row["signal_timing"] == "T_close_confirmed__earliest_next_market_session"


def test_sma_seeded_ema_and_wilder_restart_after_missing_not_ewm_skipna():
    f = pd.DataFrame({"one": [1., 2., 3., 4., np.nan, 10., 20., 30., 40.]})
    ema = _ema(f, 3)["one"]
    wilder = _wilder(f, 3)["one"]
    assert ema.iloc[:2].isna().all()
    assert ema.iloc[2] == 2 and ema.iloc[3] == 3
    assert wilder.iloc[2] == 2
    assert wilder.iloc[3] == pytest.approx(8/3)
    assert ema.iloc[4:7].isna().all()
    assert ema.iloc[7] == 20 and ema.iloc[8] == 30
    assert wilder.iloc[8] == pytest.approx(80/3)


def test_smoothing_columns_have_independent_missing_history():
    f = pd.DataFrame({"a": [1., 2., 3., 4., 5.], "b": [10., np.nan, 10., 20., 30.]})
    result = _ema(f, 3)
    assert result["a"].iloc[-1] == 4
    assert result["b"].iloc[:4].isna().all()
    assert result["b"].iloc[-1] == 20


def test_wilder_rsi_matches_independent_published_seed_example():
    # Wilder/StockCharts seed example: first 14 changes give RSI 70.464135...
    c = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
         45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
    z, _ = evaluate(frames(c))
    assert z["pub_rsi14"].iloc[:14, 0].isna().all()
    assert z["pub_rsi14"].iloc[14, 0] == pytest.approx(70.4641350211)
    assert z["pub_rsi14"].iloc[15, 0] == pytest.approx(66.2496185536)


def test_macd_signal_waits_for_26_then_9_seed_inputs():
    z, rules = evaluate(frames(np.arange(100., 150.)))
    assert z["pub_macd"].iloc[:25, 0].isna().all()
    assert z["pub_macd"].iloc[25, 0] == pytest.approx(7.)
    assert z["pub_macd_signal"].iloc[:33, 0].isna().all()
    assert z["pub_macd_signal"].iloc[33, 0] == pytest.approx(7.)
    assert not rules["macd12_26_cross"]["known"].iloc[33, 0]
    assert rules["macd12_26_cross"]["known"].iloc[34, 0]


def test_adx_seed_and_true_range_are_based_on_observed_previous_close():
    z, _ = evaluate(frames(np.arange(100., 150.)))
    assert z["pub_atr14"].iloc[:14, 0].isna().all()
    assert z["pub_atr14"].iloc[14, 0] == pytest.approx(2.)
    assert z["pub_plus_di14"].iloc[14, 0] == pytest.approx(50.)
    assert z["pub_minus_di14"].iloc[14, 0] == pytest.approx(0.)
    assert z["pub_adx14"].iloc[:27, 0].isna().all()
    assert z["pub_adx14"].iloc[27, 0] == pytest.approx(100.)


def test_ichimoku_cloud_at_t_uses_values_computed_26_sessions_earlier():
    c = np.arange(100., 220.)
    z, _ = evaluate(frames(c))
    t = 90
    historical_close = c[t-26]
    expected_conversion = historical_close - 4
    expected_base = historical_close - 12.5
    assert z["pub_cloud_a"].iloc[t, 0] == pytest.approx((expected_conversion+expected_base)/2)
    assert z["pub_cloud_b"].iloc[t, 0] == pytest.approx(historical_close-25.5)
    assert z["pub_cloud_b"].iloc[:77, 0].isna().all()
    assert z["pub_cloud_top"].iloc[77, 0] == max(z["pub_cloud_a"].iloc[77, 0], z["pub_cloud_b"].iloc[77, 0])


def test_obv_restarts_without_bridging_a_missing_session():
    f = frames(np.arange(100., 180.))
    f["valid"].iloc[40, 0] = False
    z, rules = evaluate(f)
    assert np.isnan(z["pub_obv"].iloc[40, 0])
    assert z["pub_obv"].iloc[41, 0] == 0
    assert z["pub_obv"].iloc[42, 0] == 1_000_000
    assert not rules["obv20_breakout"]["known"].iloc[41:62, 0].any()
    assert rules["obv20_breakout"]["known"].iloc[62, 0]


def test_cmf_numeric_flow_crosses_zero_without_claiming_actual_net_inflow():
    c = np.r_[np.full(30, 100.), 101.]
    f = frames(c, high=np.full(31, 101.), low=np.full(31, 99.))
    z, rules = evaluate(f)
    assert z["pub_cmf20"].iloc[-2, 0] == 0
    assert z["pub_cmf20"].iloc[-1, 0] == pytest.approx(.05)
    assert final_match(rules, "cmf20_cross")


def test_mfi_one_sided_values_are_known_but_no_direction_remains_unknown():
    z, _ = evaluate(frames(np.arange(150., 110., -1)))
    assert z["pub_mfi14"].iloc[-1, 0] == 0
    flat, flat_rules = evaluate(frames(np.full(50, 100.)))
    assert flat["pub_mfi14"].iloc[-1, 0] is not None
    assert np.isnan(flat["pub_mfi14"].iloc[-1, 0])
    assert not flat_rules["mfi14_reclaim20"]["known"].iloc[-1, 0]
    c = np.r_[np.arange(150., 110., -1), 112.]
    volume = np.r_[np.full(40, 1_000_000.), 10_000_000.]
    z, rules = evaluate(frames(c, volume=volume))
    # The 14-direction window contains the last 13 down days plus the new up day.
    expected = 100*(112*10_000_000)/(112*10_000_000+sum(c[-14:-1])*1_000_000)
    assert z["pub_mfi14"].iloc[-1, 0] == pytest.approx(expected)
    assert final_match(rules, "mfi14_reclaim20")


def test_zero_price_ranges_do_not_silently_become_flat_valid_oscillators():
    f = frames(np.full(180, 100.), high=np.full(180, 100.), low=np.full(180, 100.))
    z, rules = evaluate(f)
    for feature in ("pub_slow_k", "pub_cmf20", "pub_rsi14", "pub_mfi14", "pub_adx14"):
        assert np.isnan(z[feature].iloc[-1, 0]), feature
    for sid in ("stochastic14_3_cross", "cmf20_cross", "rsi14_reclaim30", "mfi14_reclaim20", "adx14_di_cross"):
        assert not rules[sid]["known"].iloc[-1, 0], sid


def test_cross_rules_trigger_on_crossing_not_every_subsequent_day_above():
    c = np.r_[np.full(180, 100.), 110., 110.]
    _, rules = evaluate(frames(c))
    for sid in ("ma20_60_cross", "bollinger_squeeze_breakout", "keltner20_breakout", "atr14_volatility_breakout"):
        assert rules[sid]["known"].iloc[-2, 0], sid
        assert rules[sid]["match"].iloc[-2, 0], sid
        assert not rules[sid]["match"].iloc[-1, 0], sid


def test_all_rules_enforce_liquidity_and_preserve_missing_amount_unknown():
    f = frames(np.r_[np.full(180, 100.), 110.], amount=np.full(181, 40_000_000.))
    _, rules = evaluate(f)
    assert all(not r["match"].iloc[-1, 0] for r in rules.values())
    f["a"].iloc[-3, 0] = np.nan
    _, rules = evaluate(f)
    assert all(not r["known"].iloc[-1, 0] for r in rules.values())


@pytest.mark.parametrize("bad_field,bad_value", [("valid", False), ("eligible", False), ("c", np.nan), ("v", np.nan)])
def test_bad_session_breaks_recursive_history_and_is_not_filled(bad_field, bad_value):
    f = frames(np.arange(100., 230.))
    f[bad_field].iloc[90, 0] = bad_value
    z, rules = evaluate(f)
    assert z["pub_macd"].iloc[90:116, 0].isna().all()
    assert not rules["macd12_26_cross"]["known"].iloc[90:125, 0].any()
    assert all(not r["known"].iloc[90, 0] for r in rules.values())


@pytest.mark.parametrize("cutoff", [50, 160, 260])
def test_append_or_replace_future_cannot_change_previous_features_or_signals(cutoff):
    rng = np.random.default_rng(83)
    c = 150 + rng.normal(size=(320, 3)).cumsum(axis=0)
    f = frames(c)
    f["valid"].iloc[90, 1] = False
    full_z, full_rules = evaluate(f)
    before = {key: value.iloc[:cutoff].copy() for key, value in f.items()}
    short_z, short_rules = evaluate(before)
    changed = deepcopy(f)
    for key in ("c", "h", "l", "close", "open"):
        changed[key].iloc[cutoff:] *= 3
    changed_z, changed_rules = evaluate(changed)
    for key in short_z:
        pd.testing.assert_frame_equal(short_z[key], full_z[key].iloc[:cutoff])
        pd.testing.assert_frame_equal(short_z[key], changed_z[key].iloc[:cutoff])
    for sid in PUBLIC_IDS:
        for key in ("known", "match"):
            pd.testing.assert_frame_equal(short_rules[sid][key], full_rules[sid][key].iloc[:cutoff])
            pd.testing.assert_frame_equal(short_rules[sid][key], changed_rules[sid][key].iloc[:cutoff])


def test_public_rules_register_every_id_and_do_not_mutate_inputs_or_shared_features():
    f = frames(np.linspace(100, 200, 180))
    before = deepcopy(f)
    z, rules = evaluate(f)
    assert tuple(rules) == PUBLIC_IDS
    assert all(name.startswith("pub_") for name in z)
    for key in f:
        pd.testing.assert_frame_equal(f[key], before[key])
    with pytest.raises(ValueError, match="already registered"):
        add_public_rules(f, z, lambda *_: None)


def test_misaligned_or_duplicate_axes_fail_instead_of_silent_alignment():
    f = frames(np.full(40, 100.))
    f["v"] = f["v"].iloc[1:]
    with pytest.raises(ValueError, match="share the complete session calendar"):
        evaluate(f)
    f = frames(np.full(40, 100.))
    for frame in f.values():
        frame.index = pd.Index([0]*len(frame))
    with pytest.raises(ValueError, match="ordered and unique"):
        evaluate(f)


@pytest.mark.parametrize("raw_close,boundary", [(15.2, "high"), (7.3, "low")])
def test_real_adjustment_factor_roundoff_preserves_valid_history(raw_close, boundary):
    adjusted_close = 123.45
    factor = adjusted_close / raw_close
    raw_high = raw_close if boundary == "high" else raw_close + 1
    raw_low = raw_close if boundary == "low" else raw_close - 1
    adjusted_high, adjusted_low = raw_high*factor, raw_low*factor
    if boundary == "high":
        assert adjusted_high < adjusted_close  # 123.44999999999999
    else:
        assert adjusted_low > adjusted_close  # 123.45000000000002
    f = frames(np.full(100, adjusted_close), high=np.full(100, adjusted_high),
               low=np.full(100, adjusted_low))
    f["close"][:] = raw_close
    f["open"][:] = raw_close
    c, h, l, _, _ = _inputs(f)
    assert h.ge(c).all().all() and l.le(c).all().all()
    assert (h if boundary == "high" else l).iloc[-1, 0] == adjusted_close
    z, rules = evaluate(f)
    assert z["pub_ma60"].iloc[-1, 0] == pytest.approx(adjusted_close)
    assert rules["ma20_60_cross"]["known"].iloc[-1, 0]
    assert np.isfinite(z["pub_atr14"].iloc[-1, 0])


@pytest.mark.parametrize("field,offset", [("h", -.01), ("l", .01)])
def test_roundoff_allowance_does_not_accept_real_ohlc_contradiction(field, offset):
    f = frames(np.full(100, 123.45))
    f[field].iloc[80, 0] = 123.45 + offset
    c, h, l, _, _ = _inputs(f)
    assert np.isnan(c.iloc[80, 0])
    assert np.isnan(h.iloc[80, 0]) and np.isnan(l.iloc[80, 0])
    z, rules = evaluate(f)
    assert np.isnan(z["pub_ma60"].iloc[-1, 0])
    assert not rules["ma20_60_cross"]["known"].iloc[-1, 0]
