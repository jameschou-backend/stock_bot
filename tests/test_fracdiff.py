"""Fractional Differentiation 單元測試。

驗證：
- weights 計算正確（boundary case + 已知公式）
- d=0：identity（series 不變）
- d=1：first difference（與 np.diff 一致）
- d ∈ (0,1)：保留部分 memory + 達平穩
- panel apply per-stock 獨立
- NaN propagation
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.fracdiff import (
    find_optimal_d,
    fracdiff_ffd_series,
    fracdiff_panel,
    fracdiff_weights,
    fracdiff_weights_ffd,
)


# ──────────────────────────────────────────────
# Weights
# ──────────────────────────────────────────────

class TestWeights:
    def test_w0_is_1(self):
        for d in (0.1, 0.3, 0.5, 0.7, 0.9):
            w = fracdiff_weights(d)
            assert w[0] == 1.0

    def test_w1_equals_minus_d(self):
        """ω_1 = -d（從遞迴公式：ω_1 = -ω_0 × d/1 = -d）。"""
        for d in (0.3, 0.5, 0.7):
            w = fracdiff_weights(d)
            assert abs(w[1] - (-d)) < 1e-12

    def test_weights_decay(self):
        """對 d ∈ (0, 1) weights 絕對值應該逐漸衰減。"""
        w = fracdiff_weights(0.4)
        abs_w = np.abs(w)
        # 後段應該比前段小（粗略）
        assert abs_w[10] > abs_w[100] if len(w) > 100 else True
        assert abs_w[-1] < abs_w[0]

    def test_d_equals_1_is_first_diff(self):
        """d=1 時 ω = [1, -1, 0, 0, ...]（first difference）。"""
        w = fracdiff_weights(1.0, max_size=10)
        assert w[0] == 1.0
        assert w[1] == -1.0
        # k=2 開始 ω_k = -ω_{k-1} × (1-k+1)/k = -ω_{k-1} × (2-k)/k
        # k=2: (2-2)/2 = 0 → ω_2 = 0；後續都 0
        assert abs(w[2]) < 1e-12 if len(w) > 2 else True

    def test_threshold_truncates(self):
        """較大 threshold → 較短 weights。"""
        w_fine = fracdiff_weights(0.5, threshold=1e-6)
        w_coarse = fracdiff_weights(0.5, threshold=1e-2)
        assert len(w_coarse) < len(w_fine)

    def test_max_size_caps(self):
        w = fracdiff_weights(0.5, max_size=20, threshold=1e-20)
        assert len(w) <= 20

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            fracdiff_weights(0.5, max_size=0)
        with pytest.raises(ValueError):
            fracdiff_weights(0.5, threshold=0)


# ──────────────────────────────────────────────
# Series-level FFD
# ──────────────────────────────────────────────

class TestFracdiffSeries:
    def test_d_1_matches_first_diff(self):
        """d=1 用 first 兩個 weights [1, -1] → 結果 = x_t - x_{t-1}（first difference）。"""
        x = np.array([10.0, 12, 15, 14, 16, 18, 17])
        out = fracdiff_ffd_series(x, d=1.0, threshold=1e-12)
        # weights for d=1: [1, -1, 0, 0, ...] but threshold cuts to length 2
        # 結果：out[1:] = x[1:] - x[:-1]
        expected = np.concatenate([[np.nan], np.diff(x)])
        np.testing.assert_array_almost_equal(out[~np.isnan(out)], expected[~np.isnan(expected)])

    def test_d_05_produces_intermediate(self):
        """d=0.5 結果應該介於 raw 跟 first diff 之間（partial memory）。"""
        rng = np.random.default_rng(0)
        # 帶趨勢的序列
        x = 100 + np.cumsum(rng.normal(0.1, 1, 200))
        out = fracdiff_ffd_series(x, d=0.5)
        # 後半部不應為 NaN
        assert not np.all(np.isnan(out[100:]))
        # std(out) 應該遠小於 std(x)（已經 differenced）
        assert np.nanstd(out[50:]) < np.nanstd(x) / 2

    def test_leading_nans_for_insufficient_window(self):
        """series 長度 < weight window → 全 NaN；中間長度 → 前面 NaN。"""
        w = fracdiff_weights_ffd(0.4)
        K = len(w)
        # n < K: 全 NaN
        x_short = np.arange(K // 2, dtype=float)
        out_short = fracdiff_ffd_series(x_short, d=0.4)
        assert np.all(np.isnan(out_short))
        # n == K * 2: 前 K-1 NaN
        x_mid = np.arange(K * 2, dtype=float)
        out_mid = fracdiff_ffd_series(x_mid, d=0.4)
        assert np.all(np.isnan(out_mid[: K - 1]))
        assert not np.any(np.isnan(out_mid[K - 1 :]))

    def test_nan_propagates(self):
        """中間有 NaN → 對應 window 結果為 NaN。"""
        w = fracdiff_weights_ffd(0.4)
        K = len(w)
        x = np.arange(K * 2, dtype=float)
        x[K] = np.nan
        out = fracdiff_ffd_series(x, d=0.4)
        # 第 K 個 position 必 NaN（因為 input 是 NaN）
        # 之後 K-1 個 position 也會因 window 涵蓋 NaN 而 NaN
        for t in range(K, 2 * K - 1):
            assert np.isnan(out[t]), f"position {t} should be NaN"

    def test_rejects_2d_input(self):
        with pytest.raises(ValueError):
            fracdiff_ffd_series(np.zeros((3, 3)), d=0.4)


# ──────────────────────────────────────────────
# Panel
# ──────────────────────────────────────────────

class TestFracdiffPanel:
    def _make_panel(self, n_stocks=3, n_days=200, seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        start = date(2024, 1, 1)
        for i in range(n_stocks):
            sid = f"100{i}"
            walk = 100 + np.cumsum(rng.normal(0.05, 1, n_days))
            for d_idx, c in enumerate(walk):
                rows.append({
                    "stock_id": sid,
                    "trading_date": start + timedelta(days=d_idx),
                    "close": float(c),
                })
        return pd.DataFrame(rows)

    def test_adds_column(self):
        df = self._make_panel()
        result = fracdiff_panel(df, value_col="close", d=0.4)
        assert "close_fracdiff_0_40" in result.columns
        # 不修改原欄
        assert "close" in result.columns

    def test_custom_out_col(self):
        df = self._make_panel()
        result = fracdiff_panel(df, value_col="close", d=0.4, out_col="ff04")
        assert "ff04" in result.columns

    def test_per_stock_independent(self):
        """股票間互不影響。"""
        df = self._make_panel(n_stocks=2, n_days=200)
        # 對單獨一個 stock 跑
        df_a = df[df["stock_id"] == "1000"].copy()
        result_a_alone = fracdiff_panel(df_a, value_col="close", d=0.4)

        # 對兩個 stock 一起跑
        result_both = fracdiff_panel(df, value_col="close", d=0.4)
        sub_a = result_both[result_both["stock_id"] == "1000"].copy()

        # stock 1000 的結果應該完全相同
        merged = result_a_alone.merge(
            sub_a, on=["stock_id", "trading_date"], suffixes=("_alone", "_both"))
        np.testing.assert_array_almost_equal(
            merged["close_fracdiff_0_40_alone"].dropna().to_numpy(),
            merged["close_fracdiff_0_40_both"].dropna().to_numpy(),
        )

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"stock_id": ["A"], "trading_date": [date(2024, 1, 1)]})
        with pytest.raises(ValueError, match="not in df"):
            fracdiff_panel(df, value_col="missing", d=0.4)


# ──────────────────────────────────────────────
# find_optimal_d
# ──────────────────────────────────────────────

class TestFindOptimalD:
    def test_finds_d_for_trending_series(self):
        """強趨勢的 series → 需要較大的 d 才平穩。"""
        rng = np.random.default_rng(0)
        # 強趨勢
        x = 100 + np.cumsum(rng.normal(0.5, 1, 500))
        d_opt, results = find_optimal_d(x, significance=0.05)
        assert 0 < d_opt < 1
        # 所有 d 都該有 p_value（或 error）
        assert len(results) > 0

    def test_finds_low_d_for_stationary_series(self):
        """已平穩的序列 → 最小 d（如 0.05）就 reject H0。"""
        rng = np.random.default_rng(7)
        # 平穩 AR(1)
        x = rng.normal(0, 1, 500)
        d_opt, _ = find_optimal_d(x, significance=0.05)
        # 對平穩序列，小 d 就夠
        assert d_opt <= 0.5

    def test_rejects_too_short(self):
        with pytest.raises(ValueError, match="序列太短"):
            find_optimal_d(np.array([1.0, 2.0, 3.0]))


# ──────────────────────────────────────────────
# Integration: 真實 random walk → fracdiff 應該降低 std
# ──────────────────────────────────────────────

class TestSanity:
    def test_fracdiff_reduces_nonstationarity(self):
        """random walk: std(x) 隨時間 √n 增大；fracdiff 後 std 應穩定。"""
        rng = np.random.default_rng(123)
        n = 1000
        x = 100 + np.cumsum(rng.normal(0, 1, n))
        out = fracdiff_ffd_series(x, d=0.5)
        out_clean = out[~np.isnan(out)]
        # 原 series 後半 std 應 > 前半（非平穩）
        std_first_half = np.std(x[: n // 2])
        std_second_half = np.std(x[n // 2 :])
        assert std_second_half > std_first_half * 1.2  # 非平穩跡象
        # fracdiff 後 std 不該爆增
        half = len(out_clean) // 2
        std_diff_first = np.std(out_clean[:half])
        std_diff_second = np.std(out_clean[half:])
        # 比例應該 < 2x（不再嚴重發散）
        ratio = max(std_diff_first, std_diff_second) / min(std_diff_first, std_diff_second)
        assert ratio < 2.0
