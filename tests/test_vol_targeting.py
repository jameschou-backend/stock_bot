"""Volatility Targeting 單元測試。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.vol_targeting import (
    annualize_vol,
    compute_vol_scaler_for_picks,
    estimate_portfolio_vol,
    vol_scaler,
)


# ──────────────────────────────────────────────
# annualize_vol
# ──────────────────────────────────────────────

class TestAnnualizeVol:
    def test_zero_vol(self):
        assert annualize_vol(np.zeros(100)) == 0.0

    def test_known_vol(self):
        rng = np.random.default_rng(0)
        daily = rng.normal(0, 0.02, 500)  # 2% daily std
        v = annualize_vol(daily)
        # annualized = 0.02 * sqrt(252) ≈ 0.317
        assert 0.30 < v < 0.34

    def test_too_few(self):
        assert annualize_vol(np.array([0.01])) == 0.0
        assert annualize_vol(np.array([])) == 0.0


# ──────────────────────────────────────────────
# vol_scaler
# ──────────────────────────────────────────────

class TestVolScaler:
    def test_at_target_returns_one(self):
        assert vol_scaler(0.15, target_vol=0.15) == 1.0

    def test_above_target_scales_down(self):
        # vol=0.30, target=0.15 → scaler = 0.5
        assert vol_scaler(0.30, target_vol=0.15) == 0.5

    def test_below_target_capped(self):
        # vol=0.05, target=0.15, cap=1.0 → 1.0（not 3.0）
        assert vol_scaler(0.05, target_vol=0.15, cap=1.0) == 1.0

    def test_allow_leverage_if_cap_high(self):
        assert vol_scaler(0.05, target_vol=0.15, cap=2.0) == 2.0  # min(2.0, 3.0) = 2.0

    def test_zero_vol_fallback(self):
        assert vol_scaler(0.0) == 1.0
        assert vol_scaler(float("nan")) == 1.0
        assert vol_scaler(-0.01) == 1.0  # 負值 fallback


# ──────────────────────────────────────────────
# estimate_portfolio_vol
# ──────────────────────────────────────────────

def _make_panel(picks, n_days=60, vol_daily=0.02, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2024, 1, 1)
    for sid in picks:
        walk = 100 + np.cumsum(rng.normal(0, vol_daily * 100, n_days))
        for d_idx, c in enumerate(walk):
            rows.append({"stock_id": sid, "trading_date": start + timedelta(days=d_idx),
                         "close": float(c)})
    return pd.DataFrame(rows)


class TestEstimatePortfolioVol:
    def test_equal_weight_default(self):
        panel = _make_panel(["A", "B", "C"], n_days=60, vol_daily=0.02)
        rb = date(2024, 2, 28)
        v = estimate_portfolio_vol(["A", "B", "C"], panel, rb, lookback_days=50)
        # vol ~ 0.02 * sqrt(252) / sqrt(3) ≈ 0.183 (與 random walk 等權多元化效果一致)
        assert 0.10 < v < 0.35

    def test_empty_returns_zero(self):
        v = estimate_portfolio_vol([], pd.DataFrame(columns=["stock_id", "trading_date", "close"]),
                                    date(2024, 1, 1))
        assert v == 0.0

    def test_no_history_returns_zero(self):
        panel = _make_panel(["A"], n_days=10)
        rb = date(2025, 1, 1)  # 遠後於 panel
        v = estimate_portfolio_vol(["A"], panel, rb, lookback_days=30)
        assert v == 0.0

    def test_short_lookback_returns_zero(self):
        # 只有 3 天資料 → < 6 returns required → 0
        panel = _make_panel(["A", "B"], n_days=4)
        v = estimate_portfolio_vol(["A", "B"], panel, date(2024, 1, 6), lookback_days=10)
        assert v == 0.0

    def test_custom_weights(self):
        panel = _make_panel(["A", "B"], n_days=60, seed=1)
        rb = date(2024, 2, 28)
        v_equal = estimate_portfolio_vol(["A", "B"], panel, rb, lookback_days=50)
        v_concentrated = estimate_portfolio_vol(
            ["A", "B"], panel, rb, lookback_days=50,
            weights={"A": 0.9, "B": 0.1},
        )
        # 集中權重 → diversification 較少 → vol 較高
        # 但因 random walk 隨機，僅作 sanity check（不為 0）
        assert v_equal > 0 and v_concentrated > 0


# ──────────────────────────────────────────────
# compute_vol_scaler_for_picks
# ──────────────────────────────────────────────

class TestComputeVolScalerForPicks:
    def test_returns_dict_with_keys(self):
        panel = _make_panel(["A", "B", "C"], n_days=60, vol_daily=0.05, seed=2)
        rb = date(2024, 2, 28)
        result = compute_vol_scaler_for_picks(
            ["A", "B", "C"], panel, rb, target_vol=0.15, lookback_days=50,
        )
        for k in ("scaler", "realized_vol", "target_vol", "cash_share"):
            assert k in result
        assert 0 <= result["scaler"] <= 1
        assert 0 <= result["cash_share"] <= 1
        assert abs(result["scaler"] + result["cash_share"] - 1.0) < 1e-9

    def test_high_vol_period_reduces_position(self):
        panel = _make_panel(["A", "B"], n_days=60, vol_daily=0.10, seed=3)  # 高 vol
        rb = date(2024, 2, 28)
        result = compute_vol_scaler_for_picks(
            ["A", "B"], panel, rb, target_vol=0.15, lookback_days=50,
        )
        # 10% daily vol → annualized ~1.5+ → scaler << 1
        assert result["scaler"] < 0.5

    def test_low_vol_period_keeps_full(self):
        panel = _make_panel(["A", "B"], n_days=60, vol_daily=0.003, seed=4)  # 極低 vol
        rb = date(2024, 2, 28)
        result = compute_vol_scaler_for_picks(
            ["A", "B"], panel, rb, target_vol=0.15, lookback_days=50,
        )
        # annualized ~0.05 → cap 1.0 saturates
        assert result["scaler"] == 1.0
