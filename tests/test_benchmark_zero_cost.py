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
    bm, raw = pipe._compute_benchmark_return(date(2024, 1, 2), date(2024, 1, 31))
    assert bm == pytest.approx(0.0, abs=1e-12)
    assert raw == pytest.approx(0.0, abs=1e-12)


def test_benchmark_return_explicit_tc_subtracted():
    """--benchmark-tc 0.006 → 報表口徑扣 0.006；market_return_raw 永遠零成本。"""
    pipe = _benchmark_pipe(transaction_cost_pct=0.0058425, benchmark_tc_pct=0.006)
    assert pipe.benchmark_tc == pytest.approx(0.006)
    bm, raw = pipe._compute_benchmark_return(date(2024, 1, 2), date(2024, 1, 31))
    assert bm == pytest.approx(-0.006)
    assert raw == pytest.approx(0.0, abs=1e-12)  # 訊號口徑不受 tc 影響


def test_benchmark_return_legacy_opt_in_still_reproducible():
    """舊實驗重現路徑：顯式 benchmark_with_cost=True → 扣 transaction_cost_pct。"""
    pipe = _benchmark_pipe(transaction_cost_pct=0.0058425, benchmark_with_cost=True)
    bm, raw = pipe._compute_benchmark_return(date(2024, 1, 2), date(2024, 1, 31))
    assert bm == pytest.approx(-0.0058425)
    assert raw == pytest.approx(0.0, abs=1e-12)


# ── 3.5 口徑/訊號解耦（P1 修復）：market filter 不受 benchmark_tc 影響 ──────


TIERS = [(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)]


def _period(raw: float, tc: float) -> dict:
    """模擬 period_results 一期：benchmark_return 為報表口徑、market_return_raw 零成本。"""
    return {"benchmark_return": raw - tc, "market_return_raw": raw}


def _mk_filter_pipe(**wf_kwargs) -> BacktestPipeline:
    pipe = _make_pipeline(market_filter_tiers=TIERS, **wf_kwargs)
    return pipe


def _picks_df(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({"stock_id": [f"{1000 + i}" for i in range(n)],
                         "score": [float(n - i) for i in range(n)]})


def test_market_filter_reads_raw_not_report_convention():
    """filter 讀 market_return_raw：報表口徑落 tier 帶內、raw 在帶外 → 不得觸發降倉。"""
    pipe = _mk_filter_pipe(benchmark_tc_pct=0.0058425)
    # raw = -4.7%（> -5% 門檻，不應降倉）；報表口徑 -5.28%（若誤讀會觸發 0.5 tier）
    periods = [_period(-0.047, 0.0058425)]
    picks, skip, mult = pipe._apply_market_filter_tiers(_picks_df(), periods, date(2024, 2, 1), 30)
    assert skip is False
    assert mult == pytest.approx(1.0)
    assert len(picks) == 30


def test_market_filter_triggers_on_raw_regardless_of_tc():
    """raw 越過門檻時，無論 benchmark_tc 為何都必須觸發同一 tier。"""
    for tc in (0.0, 0.0058425):
        pipe = _mk_filter_pipe(benchmark_tc_pct=tc)
        periods = [_period(-0.06, tc)]  # raw -6% < -5% → tier ×0.5
        picks, skip, mult = pipe._apply_market_filter_tiers(
            _picks_df(), periods, date(2024, 2, 1), 30
        )
        assert skip is False
        assert mult == pytest.approx(0.5), f"tc={tc} 時 tier 判定不一致"
        assert len(picks) == 15


def test_market_filter_invariant_to_benchmark_tc_end_to_end():
    """端到端不變量：同一價格資料，benchmark_tc=0 與 >0 的 filter 決策完全相同
    （picks 路徑不因報表口徑而變），而報表 benchmark_return 相差恰為 tc。"""
    rb, ex = date(2024, 1, 2), date(2024, 1, 31)
    price_df = pd.DataFrame(
        {
            "stock_id": ["1101", "1101", "1102", "1102"],
            "trading_date": [rb, ex, rb, ex],
            "close": [100.0, 90.0, 100.0, 98.0],  # 等權 raw = -6%
        }
    )
    decisions, reports, raws = [], [], []
    for tc in (0.0, 0.0058425):
        pipe = _mk_filter_pipe(benchmark_tc_pct=tc)
        pipe.price_df = price_df
        bm, raw = pipe._compute_benchmark_return(rb, ex)
        periods = [{"benchmark_return": bm, "market_return_raw": raw}]
        picks, skip, mult = pipe._apply_market_filter_tiers(
            _picks_df(), periods, date(2024, 2, 1), 30
        )
        decisions.append((skip, mult, len(picks)))
        reports.append(bm)
        raws.append(raw)
    assert decisions[0] == decisions[1], "benchmark_tc>0 改變了策略臂持倉決策（口徑耦合回歸）"
    assert raws[0] == pytest.approx(raws[1])
    assert reports[0] - reports[1] == pytest.approx(0.0058425)


def test_market_filter_fallback_to_benchmark_return_for_legacy_periods():
    """舊 period dict（無 market_return_raw）相容：fallback 讀 benchmark_return。"""
    pipe = _mk_filter_pipe()
    periods = [{"benchmark_return": -0.06}]
    _, skip, mult = pipe._apply_market_filter_tiers(_picks_df(), periods, date(2024, 2, 1), 30)
    assert skip is False
    assert mult == pytest.approx(0.5)


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
