"""統計紀律工具化（2026-07-10 總體檢缺陷 6 規則 1）：

1. paired block-bootstrap Sharpe 95% CI（skills/statistics.py）：形狀 / 確定性 / 配對性
2. trial registry append（scripts/run_backtest.py）
3. compute_statistics_block：回測 result → 統計區塊
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from skills.statistics import BootstrapSharpeCI, paired_block_bootstrap_sharpe_ci


def _synthetic_returns(n=120, mean=0.02, std=0.06, seed=0):
    return np.random.default_rng(seed).normal(mean, std, n)


# ── 1. bootstrap CI：形狀 ───────────────────────────────────────────────────


class TestBootstrapShape:
    def test_returns_dataclass_with_expected_fields(self):
        r = _synthetic_returns()
        res = paired_block_bootstrap_sharpe_ci(r, block_size=6, n_boot=200)
        assert isinstance(res, BootstrapSharpeCI)
        assert res.n_observations == 120
        assert res.block_size == 6
        assert res.n_boot == 200
        assert res.ci_level == 0.95
        assert math.isfinite(res.sharpe_observed)
        assert res.ci_low < res.ci_high
        # 未提供 benchmark → excess 欄位為 None
        assert res.excess_sharpe_observed is None
        assert res.excess_ci_low is None and res.excess_ci_high is None

    def test_observed_sharpe_matches_backtest_convention(self):
        """sharpe_observed 必須與 skills/backtest.py summary 口徑逐位一致。"""
        r = _synthetic_returns()
        res = paired_block_bootstrap_sharpe_ci(r, n_boot=10)
        rf_monthly = (1 + 0.015) ** (1 / 12) - 1
        expected = ((r - rf_monthly).mean() / r.std()) * np.sqrt(12)  # ddof=0
        assert res.sharpe_observed == pytest.approx(float(expected), abs=1e-12)

    def test_ci_brackets_observed_for_wellbehaved_data(self):
        """常態月報酬 n=120：95% CI 應涵蓋 observed（bootstrap percentile 基本性質）。"""
        r = _synthetic_returns()
        res = paired_block_bootstrap_sharpe_ci(r)
        assert res.ci_low <= res.sharpe_observed <= res.ci_high

    def test_benchmark_enables_excess_fields(self):
        r = _synthetic_returns(seed=1)
        b = _synthetic_returns(mean=0.008, std=0.05, seed=2)
        res = paired_block_bootstrap_sharpe_ci(r, b, n_boot=200)
        assert res.excess_sharpe_observed is not None
        assert res.excess_ci_low < res.excess_ci_high

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError):
            paired_block_bootstrap_sharpe_ci(np.array([0.01]))
        with pytest.raises(ValueError):
            paired_block_bootstrap_sharpe_ci(_synthetic_returns(), block_size=0)
        with pytest.raises(ValueError):
            paired_block_bootstrap_sharpe_ci(_synthetic_returns(), n_boot=0)
        with pytest.raises(ValueError):
            paired_block_bootstrap_sharpe_ci(
                _synthetic_returns(n=120), _synthetic_returns(n=60)
            )


# ── 2. bootstrap CI：確定性（固定 seed）───────────────────────────────────


class TestBootstrapDeterminism:
    def test_same_seed_identical_ci(self):
        r = _synthetic_returns()
        b = _synthetic_returns(seed=9)
        r1 = paired_block_bootstrap_sharpe_ci(r, b, seed=42)
        r2 = paired_block_bootstrap_sharpe_ci(r, b, seed=42)
        assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)
        assert (r1.excess_ci_low, r1.excess_ci_high) == (r2.excess_ci_low, r2.excess_ci_high)

    def test_different_seed_differs(self):
        r = _synthetic_returns()
        r1 = paired_block_bootstrap_sharpe_ci(r, seed=42, n_boot=300)
        r2 = paired_block_bootstrap_sharpe_ci(r, seed=43, n_boot=300)
        assert (r1.ci_low, r1.ci_high) != (r2.ci_low, r2.ci_high)


# ── 3. bootstrap CI：配對性（paired）───────────────────────────────────────


class TestBootstrapPairing:
    def test_benchmark_identical_to_returns_gives_degenerate_excess(self):
        """benchmark ≡ returns → 主動報酬全為 0；若配對正確，每個 replicate 的
        excess Sharpe 都是 0（std=0 → 約定回傳 0），CI 退化為 [0, 0]。
        若誤用獨立索引重抽兩序列，excess 不為 0，此測試會抓到。"""
        r = _synthetic_returns()
        res = paired_block_bootstrap_sharpe_ci(r, r.copy())
        assert res.excess_sharpe_observed == 0.0
        assert res.excess_ci_low == 0.0
        assert res.excess_ci_high == 0.0

    def test_half_benchmark_excess_equals_no_rf_sharpe_of_returns(self):
        """benchmark = 0.5×returns → active = 0.5×returns，Sharpe 尺度不變（×0.5 為
        二的冪次，fp 精確）→ excess Sharpe 分佈應與「returns 本身、rf=0」的
        bootstrap 分佈逐位相同（同 seed → 同 block 索引）。配對壞掉（兩序列
        獨立重抽）時此恆等式不成立。"""
        r = _synthetic_returns()
        paired = paired_block_bootstrap_sharpe_ci(r, 0.5 * r, seed=42, n_boot=500)
        solo_no_rf = paired_block_bootstrap_sharpe_ci(
            r, None, seed=42, n_boot=500, risk_free_rate=0.0
        )
        assert paired.excess_sharpe_observed == solo_no_rf.sharpe_observed
        assert paired.excess_ci_low == solo_no_rf.ci_low
        assert paired.excess_ci_high == solo_no_rf.ci_high


# ── 4. trial registry（scripts/run_backtest.py）────────────────────────────


class TestTrialRegistry:
    def test_append_creates_file_and_counts(self, tmp_path):
        from scripts.run_backtest import append_trial_registry, registry_trial_count

        reg = tmp_path / "trial_registry.jsonl"
        assert registry_trial_count(reg) == 0

        rec = {
            "timestamp": "2026-07-10T12:00:00",
            "command": "scripts/run_backtest.py --months 120 --production-baseline",
            "sharpe": 0.647,
            "months": 120,
            "data_snapshot_summary": {"prices": {"rows": 5186575, "max_date": "2026-07-09"}},
        }
        assert append_trial_registry(rec, reg) == 1
        assert append_trial_registry({**rec, "sharpe": 0.65}, reg) == 2
        assert registry_trial_count(reg) == 2

        lines = reg.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["sharpe"] == 0.647
        assert parsed["months"] == 120
        assert "command" in parsed and "timestamp" in parsed

    def test_historical_base_constant(self):
        """DSR n_trials = registry 行數 + 80（honest_baseline 慣例，出處見常數註解）。"""
        from scripts.run_backtest import HISTORICAL_TRIALS_BASE

        assert HISTORICAL_TRIALS_BASE == 80


# ── 5. compute_statistics_block ─────────────────────────────────────────────


class TestStatisticsBlock:
    def _fake_result(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.015, 0.06, n)
        bench = rng.normal(0.007, 0.05, n)
        return {
            "periods": [
                {"return": float(r), "benchmark_return": float(b)}
                for r, b in zip(rets, bench)
            ]
        }

    def test_block_structure_and_determinism(self):
        from scripts.run_backtest import compute_statistics_block

        result = self._fake_result()
        s1 = compute_statistics_block(result, n_trials=85, risk_free_rate=0.015)
        s2 = compute_statistics_block(result, n_trials=85, risk_free_rate=0.015)
        assert s1 == s2  # 固定 seed → deterministic

        ci = s1["sharpe_ci_95"]
        assert ci["block_size"] == 6
        assert ci["n_boot"] == 1000
        assert ci["ci_low"] < ci["ci_high"]
        assert ci["n_observations"] == 60

        dsr = s1["deflated_sharpe"]
        assert dsr["n_trials"] == 85
        assert 0.0 <= dsr["p_value"] <= 1.0
        assert dsr["n_observations"] == 60
        assert "trial_registry(5) + historical_base(80)" == dsr["n_trials_source"]

    def test_returns_none_on_insufficient_periods(self):
        from scripts.run_backtest import compute_statistics_block

        assert compute_statistics_block({"periods": []}, 80, 0.015) is None
        assert (
            compute_statistics_block(
                {"periods": [{"return": 0.01, "benchmark_return": 0.0}]}, 80, 0.015
            )
            is None
        )
