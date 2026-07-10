"""P&L 口徑收斂（BACKTEST_ADJ_FROM_DB）+ personal-baseline 機制測試。

2026-07-10 personal-baseline 第三臂前置工程：
1. `resolve_pnl_convention`：BACKTEST_ADJ_FROM_DB 優先於 BACKTEST_ADJ_PRICE_PARQUET，
   皆未設時 raw_close（單一決策點，優先序不可静默改變）。
2. `_overlay_adj_prices_from_db`：P&L 直接用 DB price_adjust_factors 還原
   （raw × factor），per-stock ffill/bfill 缺日語義對齊 build_features.apply_adj_factors
   （與 label/特徵同源，消除混口徑）；空 factor 表必須 raise（禁 silent fallback）。
3. `_precompute_maxprice_eligible_map`：個股原始收盤價上限（close_raw 口徑，
   比照 backtest_rotation --max-price：0 < px <= max，過濾只套進場候選）。
4. `_simulate_period(slippage_multiplier=...)`：悲觀敏感度滑價倍率在唯一消費點縮放，
   涵蓋 tiered map，且不 mutate 呼叫端 dict。
5. 生產預設保護：slippage_multiplier=1.0 / max_stock_price=0.0（不改變既有基準行為）。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.backtest import (
    PNL_CONVENTION_ADJ_DB,
    PNL_CONVENTION_ADJ_PARQUET,
    PNL_CONVENTION_RAW,
    WalkForwardConfig,
    _overlay_adj_prices_from_db,
    _precompute_maxprice_eligible_map,
    _simulate_period,
    resolve_pnl_convention,
)


# ── 1. resolve_pnl_convention 優先序 ─────────────────────────────────────────

def test_resolve_priority_db_over_parquet():
    assert resolve_pnl_convention("1", "some.parquet") == PNL_CONVENTION_ADJ_DB


def test_resolve_parquet_when_db_unset():
    assert resolve_pnl_convention(None, "some.parquet") == PNL_CONVENTION_ADJ_PARQUET
    assert resolve_pnl_convention("", "some.parquet") == PNL_CONVENTION_ADJ_PARQUET
    assert resolve_pnl_convention("0", "some.parquet") == PNL_CONVENTION_ADJ_PARQUET


def test_resolve_raw_when_both_unset():
    assert resolve_pnl_convention(None, None) == PNL_CONVENTION_RAW
    assert resolve_pnl_convention("", "") == PNL_CONVENTION_RAW
    assert resolve_pnl_convention("off", "  ") == PNL_CONVENTION_RAW


def test_resolve_env_flag_truthy_variants():
    for v in ("1", "true", "TRUE", "yes", "on", " 1 "):
        assert resolve_pnl_convention(v, None) == PNL_CONVENTION_ADJ_DB, v
    for v in ("0", "false", "no", "off", "", None):
        assert resolve_pnl_convention(v, None) == PNL_CONVENTION_RAW, v


# ── 2. _overlay_adj_prices_from_db ───────────────────────────────────────────

def _price_df_two_stocks():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(4)]
    rows = []
    for sid, base in (("1001", 100.0), ("2002", 50.0)):
        for i, d in enumerate(days):
            px = base + i
            rows.append({
                "stock_id": sid, "trading_date": d,
                "open": px - 1, "high": px + 1, "low": px - 2,
                "close": px, "volume": 1000 + i,
            })
    return pd.DataFrame(rows), days


def test_overlay_db_applies_factor_to_ohlc_keeps_volume_and_close_raw():
    price_df, days = _price_df_two_stocks()
    factor_df = pd.DataFrame([
        {"stock_id": "1001", "trading_date": d, "adj_factor": 0.5} for d in days
    ] + [
        {"stock_id": "2002", "trading_date": d, "adj_factor": 1.0} for d in days
    ])
    out = _overlay_adj_prices_from_db(price_df, factor_df)

    s1 = out[out["stock_id"] == "1001"].sort_values("trading_date")
    raw1 = price_df[price_df["stock_id"] == "1001"].sort_values("trading_date")
    for col in ("open", "high", "low", "close"):
        np.testing.assert_allclose(
            s1[col].to_numpy(dtype=float),
            raw1[col].to_numpy(dtype=float) * 0.5,
        )
    # volume 保留 raw（流動性語義不動）
    np.testing.assert_array_equal(s1["volume"].to_numpy(), raw1["volume"].to_numpy())
    # close_raw = 原始收盤價（max_stock_price 過濾口徑）
    np.testing.assert_allclose(
        s1["close_raw"].to_numpy(dtype=float), raw1["close"].to_numpy(dtype=float)
    )
    # factor=1.0 的股票不變
    s2 = out[out["stock_id"] == "2002"].sort_values("trading_date")
    raw2 = price_df[price_df["stock_id"] == "2002"].sort_values("trading_date")
    np.testing.assert_allclose(
        s2["close"].to_numpy(dtype=float), raw2["close"].to_numpy(dtype=float)
    )


def test_overlay_db_internal_missing_day_ffills_not_one():
    """factor 內部缺日必須 per-stock ffill（fillna(1.0) 會製造單日假跳動）。"""
    price_df, days = _price_df_two_stocks()
    # 1001 缺 days[1] / days[2] 的 factor → 應沿用 days[0] 的 0.8
    factor_df = pd.DataFrame([
        {"stock_id": "1001", "trading_date": days[0], "adj_factor": 0.8},
        {"stock_id": "1001", "trading_date": days[3], "adj_factor": 0.9},
        {"stock_id": "2002", "trading_date": days[0], "adj_factor": 1.0},
    ])
    out = _overlay_adj_prices_from_db(price_df, factor_df)
    s1 = out[out["stock_id"] == "1001"].sort_values("trading_date")
    raw1 = price_df[price_df["stock_id"] == "1001"].sort_values("trading_date")
    expect_factor = np.array([0.8, 0.8, 0.8, 0.9])
    np.testing.assert_allclose(
        s1["close"].to_numpy(dtype=float),
        raw1["close"].to_numpy(dtype=float) * expect_factor,
    )


def test_overlay_db_leading_gap_bfills():
    """最早 raw 日早於最早 factor 日 → bfill 用最早已知 factor。"""
    price_df, days = _price_df_two_stocks()
    factor_df = pd.DataFrame([
        {"stock_id": "1001", "trading_date": days[2], "adj_factor": 0.7},
        {"stock_id": "1001", "trading_date": days[3], "adj_factor": 0.7},
        {"stock_id": "2002", "trading_date": days[0], "adj_factor": 1.0},
    ])
    out = _overlay_adj_prices_from_db(price_df, factor_df)
    s1 = out[out["stock_id"] == "1001"].sort_values("trading_date")
    raw1 = price_df[price_df["stock_id"] == "1001"].sort_values("trading_date")
    np.testing.assert_allclose(
        s1["close"].to_numpy(dtype=float),
        raw1["close"].to_numpy(dtype=float) * 0.7,
    )


def test_overlay_db_stock_without_factor_gets_one():
    """整檔無 factor（無 adj 下市股）→ 1.0 未還原，不得丟列。"""
    price_df, days = _price_df_two_stocks()
    factor_df = pd.DataFrame([
        {"stock_id": "1001", "trading_date": d, "adj_factor": 0.5} for d in days
    ])
    out = _overlay_adj_prices_from_db(price_df, factor_df)
    s2 = out[out["stock_id"] == "2002"].sort_values("trading_date")
    raw2 = price_df[price_df["stock_id"] == "2002"].sort_values("trading_date")
    assert len(s2) == len(raw2)
    np.testing.assert_allclose(
        s2["close"].to_numpy(dtype=float), raw2["close"].to_numpy(dtype=float)
    )


def test_overlay_db_empty_factor_raises():
    """空 factor 表必須 raise（禁 silent fallback 回 raw close 混口徑）。"""
    price_df, _ = _price_df_two_stocks()
    with pytest.raises(ValueError, match="price_adjust_factors"):
        _overlay_adj_prices_from_db(price_df, pd.DataFrame())
    with pytest.raises(ValueError, match="price_adjust_factors"):
        _overlay_adj_prices_from_db(price_df, None)


def test_overlay_db_preserves_row_order_and_trading_date():
    """不動 trading_date 值/型別與列順序（下游 rebalance 日期比對、row index 依賴）。"""
    price_df, _ = _price_df_two_stocks()
    price_df = price_df.sample(frac=1.0, random_state=7).reset_index(drop=True)  # 打亂
    factor_df = pd.DataFrame([
        {"stock_id": sid, "trading_date": d, "adj_factor": 1.0}
        for sid in ("1001", "2002")
        for d in sorted(price_df["trading_date"].unique())
    ])
    out = _overlay_adj_prices_from_db(price_df, factor_df)
    # 列順序 / stock_id / trading_date 完全一致
    np.testing.assert_array_equal(
        out["stock_id"].to_numpy(), price_df["stock_id"].to_numpy()
    )
    assert list(out["trading_date"]) == list(price_df["trading_date"])
    assert isinstance(out["trading_date"].iloc[0], date)


# ── 3. _precompute_maxprice_eligible_map ─────────────────────────────────────

def test_maxprice_map_threshold_inclusive_and_excludes_nonpositive():
    d = date(2025, 1, 6)
    df = pd.DataFrame([
        {"stock_id": "1001", "trading_date": d, "close": 33.0},   # = 門檻 → 合格
        {"stock_id": "1002", "trading_date": d, "close": 33.5},   # > 門檻 → 排除
        {"stock_id": "1003", "trading_date": d, "close": 10.0},   # 合格
        {"stock_id": "1004", "trading_date": d, "close": 0.0},    # 0 價 → 排除
        {"stock_id": "1005", "trading_date": d, "close": -5.0},   # 負價 → 排除
    ])
    m = _precompute_maxprice_eligible_map(df, 33.0)
    assert m[d] == {"1001", "1003"}


def test_maxprice_map_uses_close_raw_when_present():
    """有 close_raw（adj overlay 後）時必須用原始價，不可用還原價。"""
    d = date(2025, 1, 6)
    df = pd.DataFrame([
        # 還原價 20（<=33）但原始價 100（>33）→ 買不起一張 → 排除
        {"stock_id": "1001", "trading_date": d, "close": 20.0, "close_raw": 100.0},
        # 還原價 50（>33）但原始價 30（<=33）→ 買得起 → 合格
        {"stock_id": "1002", "trading_date": d, "close": 50.0, "close_raw": 30.0},
    ])
    m = _precompute_maxprice_eligible_map(df, 33.0)
    assert m[d] == {"1002"}


def test_maxprice_map_disabled_at_zero():
    d = date(2025, 1, 6)
    df = pd.DataFrame([{"stock_id": "1001", "trading_date": d, "close": 10.0}])
    assert _precompute_maxprice_eligible_map(df, 0.0) == {}
    assert _precompute_maxprice_eligible_map(df, -1.0) == {}
    assert _precompute_maxprice_eligible_map(pd.DataFrame(), 33.0) == {}


# ── 4. _simulate_period slippage_multiplier ──────────────────────────────────

def _deterministic_price_df(n_days=30):
    """單調上漲的兩檔股票（遠離 clip 區，滑價差可精確驗證）。"""
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for sid, base in (("1001", 100.0), ("2002", 60.0)):
        for i, d in enumerate(days):
            px = base * (1 + 0.002 * i)
            rows.append({
                "stock_id": sid, "trading_date": d,
                "open": px, "high": px * 1.005, "low": px * 0.995,
                "close": px, "volume": 10000,
            })
    return pd.DataFrame(rows), days


def test_slippage_multiplier_scales_tiered_map_exactly():
    df, days = _deterministic_price_df()
    picks = pd.DataFrame({"stock_id": ["1001", "2002"], "score": [0.9, 0.8]})
    tiered = {"1001": 0.010, "2002": 0.006}
    kwargs = dict(
        stoploss_pct=0.0, transaction_cost_pct=0.00585, entry_delay_days=0,
        enable_slippage=False, clip_loss_pct=-0.50, tiered_slippage_map=tiered,
    )
    base = _simulate_period(picks, df, days[5], days[25], **kwargs)
    pess = _simulate_period(picks, df, days[5], days[25], slippage_multiplier=1.5, **kwargs)

    # trades_log 滑價欄位精確 ×1.5
    base_slip = {t["stock_id"]: t["slippage_pct"] for t in base["trades_log"]}
    pess_slip = {t["stock_id"]: t["slippage_pct"] for t in pess["trades_log"]}
    for sid, s in tiered.items():
        assert base_slip[sid] == pytest.approx(s, abs=1e-12)
        assert pess_slip[sid] == pytest.approx(s * 1.5, abs=1e-12)

    # 組合報酬精確少掉等權平均的額外滑價（(0.010+0.006)/2 × 0.5）
    extra = (0.010 + 0.006) / 2 * 0.5
    assert base["return"] - pess["return"] == pytest.approx(extra, abs=1e-10)

    # 不 mutate 呼叫端 tiered map
    assert tiered == {"1001": 0.010, "2002": 0.006}


def test_slippage_multiplier_one_is_noop():
    df, days = _deterministic_price_df()
    picks = pd.DataFrame({"stock_id": ["1001"], "score": [0.9]})
    kwargs = dict(
        stoploss_pct=0.0, transaction_cost_pct=0.00585, entry_delay_days=0,
        enable_slippage=False, clip_loss_pct=-0.50,
        tiered_slippage_map={"1001": 0.010},
    )
    a = _simulate_period(picks, df, days[5], days[25], **kwargs)
    b = _simulate_period(picks, df, days[5], days[25], slippage_multiplier=1.0, **kwargs)
    a.pop("_stoploss_time"), b.pop("_stoploss_time")
    assert a == b


# ── 5. 生產預設保護 ──────────────────────────────────────────────────────────

def test_production_defaults_unchanged():
    """新參數預設值不改變既有基準行為（基準 v2.x 可重現性）。"""
    cfg = WalkForwardConfig()
    assert cfg.slippage_multiplier == 1.0
    assert cfg.max_stock_price == 0.0
