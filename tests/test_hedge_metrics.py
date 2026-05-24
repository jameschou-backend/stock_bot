"""Stage 10.6 beta-hedge helper tests。"""
from __future__ import annotations

import numpy as np
import pytest

from skills.backtest import compute_hedged_metrics


def _make_result(port_rets, bench_rets):
    return {
        "periods": [
            {"return": p, "benchmark_return": b}
            for p, b in zip(port_rets, bench_rets)
        ]
    }


class TestHedgedMetrics:
    def test_zero_hedge_equals_unhedged(self):
        rng = np.random.default_rng(42)
        port = rng.normal(0.03, 0.06, 60)
        bench = rng.normal(0.01, 0.05, 60)
        r = _make_result(port.tolist(), bench.tolist())
        m = compute_hedged_metrics(r, hedge_ratio=0.0)
        # cum should match np.prod(1+port)-1
        assert abs(m["hedged_cum"] - (np.prod(1 + port) - 1)) < 1e-9

    def test_full_hedge_neutralizes_bench(self):
        # 完全與大盤同步的策略 + full hedge → 純零報酬
        bench = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        port = bench.copy()
        r = _make_result(port.tolist(), bench.tolist())
        m = compute_hedged_metrics(r, hedge_ratio=1.0)
        assert abs(m["hedged_cum"]) < 1e-9

    def test_keys_present(self):
        r = _make_result([0.02, 0.01, -0.01], [0.01, 0.005, -0.005])
        m = compute_hedged_metrics(r, hedge_ratio=0.5)
        for k in ("hedge_ratio", "ols_beta", "ols_corr",
                  "hedged_sharpe", "hedged_mdd", "hedged_calmar",
                  "hedged_cum", "hedged_annual"):
            assert k in m

    def test_empty_periods_returns_minimal(self):
        m = compute_hedged_metrics({"periods": []}, hedge_ratio=0.5)
        assert m["hedge_ratio"] == 0.5

    def test_beta_estimate_close_to_1_for_identical(self):
        bench = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005])
        port = bench * 1.5
        r = _make_result(port.tolist(), bench.tolist())
        m = compute_hedged_metrics(r, hedge_ratio=0.0)
        # OLS beta of 1.5×bench on bench should be 1.5
        assert abs(m["ols_beta"] - 1.5) < 0.01

    def test_mdd_non_positive(self):
        rng = np.random.default_rng(0)
        port = rng.normal(0.02, 0.08, 100)
        bench = rng.normal(0.005, 0.04, 100)
        r = _make_result(port.tolist(), bench.tolist())
        m = compute_hedged_metrics(r, hedge_ratio=0.5)
        assert m["hedged_mdd"] <= 0
