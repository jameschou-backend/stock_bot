"""HRP（Hierarchical Risk Parity）單元測試。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.hrp import (
    _get_cluster_variance,
    correlation_distance,
    hrp_weights,
    hrp_weights_for_picks,
    hrp_weights_from_cov,
    quasi_diagonal_order,
)


# ──────────────────────────────────────────────
# Distance metric
# ──────────────────────────────────────────────

class TestCorrelationDistance:
    def test_perfect_positive_correlation_is_zero(self):
        corr = np.array([[1.0, 1.0], [1.0, 1.0]])
        d = correlation_distance(corr)
        np.testing.assert_allclose(d, np.zeros((2, 2)))

    def test_no_correlation_is_invsqrt2(self):
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        d = correlation_distance(corr)
        # d[0,1] = sqrt(0.5 * (1 - 0)) = sqrt(0.5) ≈ 0.707
        assert abs(d[0, 1] - np.sqrt(0.5)) < 1e-9

    def test_perfect_negative_correlation_is_one(self):
        corr = np.array([[1.0, -1.0], [-1.0, 1.0]])
        d = correlation_distance(corr)
        assert abs(d[0, 1] - 1.0) < 1e-9

    def test_rejects_non_square(self):
        with pytest.raises(ValueError):
            correlation_distance(np.array([[1.0, 0.5, 0.3]]))


# ──────────────────────────────────────────────
# Cluster variance
# ──────────────────────────────────────────────

class TestClusterVariance:
    def test_single_asset_returns_own_variance(self):
        cov = np.array([[4.0, 1.0], [1.0, 9.0]])
        # Single asset: inverse var weights = 1.0, cluster_var = var
        assert abs(_get_cluster_variance(cov, [0]) - 4.0) < 1e-9
        assert abs(_get_cluster_variance(cov, [1]) - 9.0) < 1e-9

    def test_two_assets_inverse_var_weighted(self):
        # 兩個 asset variance 相同，corr=0 → inverse-var weights 各 0.5
        # cluster_var = 0.5² × 4 + 0.5² × 4 = 2
        cov = np.array([[4.0, 0.0], [0.0, 4.0]])
        result = _get_cluster_variance(cov, [0, 1])
        assert abs(result - 2.0) < 1e-9


# ──────────────────────────────────────────────
# hrp_weights_from_cov
# ──────────────────────────────────────────────

class TestHRPWeightsFromCov:
    def test_two_assets_equal_var_equal_weight(self):
        """兩個 asset variance 相同 → 各 50%。"""
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = hrp_weights_from_cov(cov, [0, 1])
        np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-9)

    def test_two_assets_unequal_var_inverse_weighted(self):
        """variance 不同 → 低 var asset 拿較多權重。"""
        cov = np.array([[1.0, 0.0], [0.0, 9.0]])
        w = hrp_weights_from_cov(cov, [0, 1])
        # asset 0 var=1, asset 1 var=9
        # alloc[left=0] = 1 - var(left)/(var(left)+var(right)) = 1 - 1/10 = 0.9
        assert w[0] > w[1]
        assert abs(w[0] - 0.9) < 1e-9
        assert abs(w[1] - 0.1) < 1e-9

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        N = 8
        # Random PSD covariance
        A = rng.normal(0, 1, (N, N))
        cov = A @ A.T + np.eye(N) * 0.1
        sort_order = list(range(N))
        w = hrp_weights_from_cov(cov, sort_order)
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= 0).all()


# ──────────────────────────────────────────────
# Quasi-diagonal sort
# ──────────────────────────────────────────────

class TestQuasiDiagonalOrder:
    def test_three_assets(self):
        """3 個 asset 經過 linkage → quasi-diagonal order 是 0/1/2 的某個 permutation。"""
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        # 距離：0-1 相近，2 遠
        dist = np.array([
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.9],
            [0.9, 0.9, 0.0],
        ])
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        order = quasi_diagonal_order(link)
        assert sorted(order) == [0, 1, 2]
        # 0/1 應相鄰，2 在邊上
        idx0, idx1, idx2 = order.index(0), order.index(1), order.index(2)
        assert abs(idx0 - idx1) == 1


# ──────────────────────────────────────────────
# hrp_weights end-to-end
# ──────────────────────────────────────────────

class TestHRPWeights:
    def test_correlated_pair_share_lower_weight(self):
        """3 個 asset：A 跟 B 高度相關，C 獨立 → A/B cluster 內 share 較少 weight。"""
        rng = np.random.default_rng(0)
        T = 200
        # A, B 高相關（共同 latent factor）
        latent_ab = rng.normal(0, 1, T)
        noise_a = rng.normal(0, 0.3, T)
        noise_b = rng.normal(0, 0.3, T)
        # C 獨立
        c = rng.normal(0, 1, T)
        returns_df = pd.DataFrame({
            "A": latent_ab + noise_a,
            "B": latent_ab + noise_b,
            "C": c,
        })
        w = hrp_weights(returns_df)
        # C 應該拿大概 50%（與 A+B 平衡），A/B 各約 25%
        wc, wab = w[2], w[0] + w[1]
        assert abs(wab - 0.5) < 0.15, f"A+B 應約 50%，got {wab}"
        assert abs(wc - 0.5) < 0.15, f"C 應約 50%，got {wc}"

    def test_singleton_returns_full_weight(self):
        df = pd.DataFrame({"A": np.random.normal(0, 1, 50)})
        w = hrp_weights(df)
        assert w.shape == (1,) and w[0] == 1.0

    def test_empty(self):
        df = pd.DataFrame()
        w = hrp_weights(df)
        assert w.shape == (0,)

    def test_low_min_periods_fallback(self):
        """所有 asset 都不足 min_periods → 等權 fallback。"""
        df = pd.DataFrame({
            "A": [0.01, 0.02, np.nan],
            "B": [0.01, np.nan, 0.03],
        })
        w = hrp_weights(df, min_periods=100)  # 100 > T
        np.testing.assert_allclose(w, [0.5, 0.5])

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(1)
        returns_df = pd.DataFrame(rng.normal(0, 1, (100, 5)), columns=list("ABCDE"))
        w = hrp_weights(returns_df)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= 0).all()


# ──────────────────────────────────────────────
# Backtest-friendly wrapper
# ──────────────────────────────────────────────

class TestHRPForPicks:
    def _make_panel(self, picks: list, n_days=50):
        rng = np.random.default_rng(42)
        rows = []
        start = date(2024, 1, 1)
        for sid in picks:
            walk = 100 + np.cumsum(rng.normal(0.05, 1, n_days))
            for d_idx, c in enumerate(walk):
                rows.append({"stock_id": sid, "trading_date": start + timedelta(days=d_idx),
                             "close": float(c)})
        return pd.DataFrame(rows)

    def test_single_pick(self):
        panel = self._make_panel(["A"])
        rb = date(2024, 2, 20)
        w = hrp_weights_for_picks(["A"], panel, rb)
        assert w == {"A": 1.0}

    def test_empty_picks(self):
        w = hrp_weights_for_picks([], pd.DataFrame(columns=["stock_id", "trading_date", "close"]),
                                   date(2024, 1, 1))
        assert w == {}

    def test_weights_sum_to_one(self):
        picks = ["A", "B", "C"]
        panel = self._make_panel(picks)
        rb = date(2024, 2, 20)
        w = hrp_weights_for_picks(picks, panel, rb, lookback_days=30)
        assert abs(sum(w.values()) - 1.0) < 1e-6
        for sid in picks:
            assert sid in w

    def test_no_history_falls_back_to_equal(self):
        picks = ["A", "B"]
        panel = self._make_panel(picks, n_days=5)
        rb = date(2024, 6, 1)  # 遠後於 panel，lookback 內無資料
        w = hrp_weights_for_picks(picks, panel, rb, lookback_days=30)
        # 應該 fallback 等權
        assert abs(sum(w.values()) - 1.0) < 1e-6
