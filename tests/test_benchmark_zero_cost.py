"""benchmark 成本口徑修復（2026-07-10 總體檢缺陷 3）行為鎖定測試。

舊行為：benchmark 每期扣 transaction_cost_pct（隱含每月 100% 周轉，120 期複利
拖累 ×~0.495，虛增策略超額約 +190pp）。
新約定：benchmark 預設零成本（buy-and-hold 近似）；敏感度分析用 benchmark_tc_pct
（CLI --benchmark-tc）顯式指定。
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.backtest import (
    BacktestPipeline,
    WalkForwardConfig,
    benchmark_cost_convention,
    resolve_benchmark_tc,
)


def _make_pipeline(**wf_kwargs) -> BacktestPipeline:
    wf = WalkForwardConfig(**wf_kwargs)
    return BacktestPipeline(config=SimpleNamespace(), db_session=None, wf_config=wf)


# ── 1. 預設口徑：零成本 ─────────────────────────────────────────────────────


def test_walkforward_config_defaults_zero_cost():
    wf = WalkForwardConfig()
    assert wf.benchmark_with_cost is False, "benchmark_with_cost 預設必須為 False（zero_cost）"
    assert wf.benchmark_tc_pct == 0.0


def test_pipeline_default_benchmark_tc_is_zero():
    pipe = _make_pipeline(transaction_cost_pct=0.0058425)
    assert pipe.benchmark_tc == 0.0


# ── 2. resolve_benchmark_tc 優先序 ──────────────────────────────────────────


@pytest.mark.parametrize(
    "tc_pct, with_cost, expected",
    [
        (0.0, False, 0.0),          # 預設：zero_cost
        (0.005, False, 0.005),      # 顯式敏感度分析值
        (0.0, True, 0.0058425),     # DEPRECATED 舊口徑（顯式 opt-in 才會走到）
        (0.003, True, 0.003),       # 顯式值優先於舊旗標
    ],
)
def test_resolve_benchmark_tc_precedence(tc_pct, with_cost, expected):
    assert resolve_benchmark_tc(tc_pct, with_cost, 0.0058425) == pytest.approx(expected)


def test_benchmark_cost_convention_label():
    assert benchmark_cost_convention(0.0) == "zero_cost"
    assert benchmark_cost_convention(0.0058425) == "per_period_tc"


# ── 3. _compute_benchmark_return：唯一扣成本點行為 ──────────────────────────


def _benchmark_pipe(**wf_kwargs) -> BacktestPipeline:
    pipe = _make_pipeline(**wf_kwargs)
    rb, ex = date(2024, 1, 2), date(2024, 1, 31)
    pipe.price_df = pd.DataFrame(
        {
            "stock_id": ["1101", "1101", "1102", "1102"],
            "trading_date": [rb, ex, rb, ex],
            "close": [100.0, 110.0, 100.0, 90.0],
        }
    )
    return pipe


def test_benchmark_return_zero_cost_by_default():
    """預設：等權 (+10%, -10%) → 0，不再被扣 0.58425%。"""
    pipe = _benchmark_pipe(transaction_cost_pct=0.0058425)
    bm = pipe._compute_benchmark_return(date(2024, 1, 2), date(2024, 1, 31))
    assert bm == pytest.approx(0.0, abs=1e-12)


def test_benchmark_return_explicit_tc_subtracted():
    """--benchmark-tc 0.006 → 每期等權報酬扣 0.006（敏感度分析路徑）。"""
    pipe = _benchmark_pipe(transaction_cost_pct=0.0058425, benchmark_tc_pct=0.006)
    assert pipe.benchmark_tc == pytest.approx(0.006)
    bm = pipe._compute_benchmark_return(date(2024, 1, 2), date(2024, 1, 31))
    assert bm == pytest.approx(-0.006)


def test_benchmark_return_legacy_opt_in_still_reproducible():
    """舊實驗重現路徑：顯式 benchmark_with_cost=True → 扣 transaction_cost_pct。"""
    pipe = _benchmark_pipe(transaction_cost_pct=0.0058425, benchmark_with_cost=True)
    bm = pipe._compute_benchmark_return(date(2024, 1, 2), date(2024, 1, 31))
    assert bm == pytest.approx(-0.0058425)


# ── 4. CLI --benchmark-tc 預設值 ────────────────────────────────────────────


def test_cli_benchmark_tc_default_zero():
    """run_backtest.py 的 --benchmark-tc 預設 0（zero_cost 約定）。

    argparse 定義在 main() 內（直接呼叫會真的跑回測），故以 source 檢查鎖定預設值。
    """
    import inspect

    import scripts.run_backtest as rb_cli

    src = inspect.getsource(rb_cli.main)
    assert '"--benchmark-tc"' in src
    assert 'dest="benchmark_tc"' in src
    arg_block = src.split('"--benchmark-tc"', 1)[1].split("help=")[0]
    assert "type=float, default=0.0" in arg_block, "--benchmark-tc 預設值必須為 0.0（zero_cost）"
