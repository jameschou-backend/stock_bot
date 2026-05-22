"""Stage 7.3 Kelly Criterion 單元測試。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.kelly import (
    compute_kelly_weights_for_picks,
    compute_realized_vol,
    kelly_weights,
    percentile_to_expected_return,
)


class TestComputeRealizedVol:
    def test_known_vol(self):
        rng = np.random.default_rng(42)
        # daily vol = 0.02 → annualized ≈ 0.317
        rets = rng.normal(0, 0.02, 252)
        v = compute_realized_vol(rets)
        assert 0.25 < v < 0.40

    def test_zero_vol(self):
        rets = np.zeros(60)
        v = compute_realized_vol(rets)
        assert v == 0.0

    def test_too_few(self):
        rets = np.array([0.01, -0.005, 0.002])
        v = compute_realized_vol(rets, min_periods=20)
        assert np.isnan(v)


class TestPercentileToExpectedReturn:
    def test_endpoints(self):
        out = percentile_to_expected_return(np.array([0.0, 1.0]), er_low=0.05, er_high=0.35)
        assert out[0] == pytest.approx(0.05)
        assert out[1] == pytest.approx(0.35)

    def test_clipping(self):
        out = percentile_to_expected_return(np.array([-0.1, 1.5]))
        assert out[0] == pytest.approx(0.05)
        assert out[1] == pytest.approx(0.35)


class TestKellyWeights:
    def test_equal_inputs_equal_weights(self):
        mu = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        sigma = np.array([0.30, 0.30, 0.30, 0.30, 0.30])
        w = kelly_weights(mu, sigma, per_stock_cap=1.0)
        assert np.allclose(w, 0.20)
        assert w.sum() == pytest.approx(1.0)

    def test_higher_mu_more_weight(self):
        mu = np.array([0.10, 0.20, 0.30, 0.40])
        sigma = np.full(4, 0.30)
        w = kelly_weights(mu, sigma, per_stock_cap=1.0)
        assert w[3] > w[2] > w[1] > w[0]
        assert w.sum() == pytest.approx(1.0)

    def test_lower_sigma_more_weight(self):
        mu = np.full(4, 0.20)
        sigma = np.array([0.50, 0.40, 0.30, 0.20])
        w = kelly_weights(mu, sigma, per_stock_cap=1.0)
        # 最低 sigma 應該最高權重
        assert w[3] > w[0]
        assert w.sum() == pytest.approx(1.0)

    def test_per_stock_cap_enforced(self):
        # 5 stocks × cap 0.30 → sum 上限 1.50 充裕滿配 1.0
        mu = np.array([0.50, 0.05, 0.05, 0.05, 0.05])
        sigma = np.full(5, 0.20)
        w = kelly_weights(mu, sigma, per_stock_cap=0.30)
        # 第一檔被 cap 到 0.30
        assert w[0] == pytest.approx(0.30)
        assert w.max() <= 0.30 + 1e-9
        assert w.sum() == pytest.approx(1.0)

    def test_half_kelly_normalized_same_as_full(self):
        # 因為 normalize，half-kelly 與 full-kelly weights 應該完全相同
        mu = np.array([0.10, 0.20, 0.30])
        sigma = np.array([0.30, 0.20, 0.25])
        w_full = kelly_weights(mu, sigma, half_kelly=False, per_stock_cap=1.0)
        w_half = kelly_weights(mu, sigma, half_kelly=True, per_stock_cap=1.0)
        assert np.allclose(w_full, w_half)

    def test_zero_sigma_handled(self):
        mu = np.array([0.20, 0.20])
        sigma = np.array([0.0, 0.30])
        w = kelly_weights(mu, sigma, per_stock_cap=1.0)
        # 0 sigma → degenerate to 0 weight，剩下全給另一個
        assert w[0] == pytest.approx(0.0) or w[1] > 0.99
        assert w.sum() == pytest.approx(1.0)

    def test_empty(self):
        w = kelly_weights(np.array([]), np.array([]))
        assert len(w) == 0

    def test_all_negative_falls_back_equal(self):
        mu = np.array([-0.05, -0.10, -0.15])
        sigma = np.array([0.20, 0.20, 0.20])
        w = kelly_weights(mu, sigma)
        # 全負 → min_weight=0 → 等權 fallback
        assert np.allclose(w, 1.0 / 3.0)


class TestComputeKellyWeightsForPicks:
    @pytest.fixture
    def synthetic_prices(self):
        rng = np.random.default_rng(0)
        rows = []
        for sid in ["2330", "2454", "2308", "1101", "2002"]:
            base = 100.0
            for d in pd.date_range("2025-01-01", "2025-04-30"):
                base *= (1 + rng.normal(0, 0.02))
                rows.append({"stock_id": sid, "trading_date": d.date(), "close": base})
        return pd.DataFrame(rows)

    def test_returns_dict_with_keys(self, synthetic_prices):
        out = compute_kelly_weights_for_picks(
            pick_stock_ids=["2330", "2454", "2308"],
            pick_scores=[0.9, 0.5, 0.1],
            price_df=synthetic_prices,
            rb_date=date(2025, 4, 30),
        )
        assert set(out.keys()) == {"weights", "mu", "sigma", "cap_hit", "fallback"}
        assert len(out["weights"]) == 3
        assert sum(out["weights"].values()) == pytest.approx(1.0)
        assert out["fallback"] is False

    def test_empty_picks(self, synthetic_prices):
        out = compute_kelly_weights_for_picks(
            pick_stock_ids=[],
            pick_scores=[],
            price_df=synthetic_prices,
            rb_date=date(2025, 4, 30),
        )
        assert out["weights"] == {}

    def test_no_price_data_fallback(self):
        out = compute_kelly_weights_for_picks(
            pick_stock_ids=["2330", "2454"],
            pick_scores=[0.9, 0.5],
            price_df=pd.DataFrame(columns=["stock_id", "trading_date", "close"]),
            rb_date=date(2025, 4, 30),
        )
        assert out["fallback"] is True
        # 等權
        assert all(v == 0.5 for v in out["weights"].values())

    def test_cap_enforced(self, synthetic_prices):
        out = compute_kelly_weights_for_picks(
            pick_stock_ids=["2330", "2454", "2308", "1101", "2002"],
            pick_scores=[10.0, 0.1, 0.1, 0.1, 0.1],  # 第一檔遠高於其他
            price_df=synthetic_prices,
            rb_date=date(2025, 4, 30),
            per_stock_cap=0.25,
        )
        for w in out["weights"].values():
            assert w <= 0.25 + 1e-6  # 浮點誤差容忍

    def test_higher_score_gets_more_weight_when_vols_similar(self, synthetic_prices):
        out = compute_kelly_weights_for_picks(
            pick_stock_ids=["2330", "2454", "2308"],
            pick_scores=[2.0, 1.0, 0.5],
            price_df=synthetic_prices,
            rb_date=date(2025, 4, 30),
            per_stock_cap=1.0,
        )
        # vol 接近時 score 高的權重應較大
        # 但 vol 也會影響，所以只能說至少 high-score 不會明顯墊底
        w_sorted = sorted(out["weights"].items(), key=lambda x: -x[1])
        top_sid = w_sorted[0][0]
        # 至少 top-1 應該是 2330（最高 score）或第二高的
        assert top_sid in {"2330", "2454"}
