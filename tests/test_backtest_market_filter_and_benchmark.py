"""行為鎖定（characterization）測試：漸進式大盤過濾 + benchmark 流動性一致性。

2026-07-03 架構審計補測試缺口（P0/P1）：
1. market_filter_tiers 漸進降倉：-6%/-12%/-16% → 乘數 0.5/0.25/0.10；
   market_filter_min_positions=2 下限補位。
2. benchmark 流動性一致性：大盤基準套用與策略相同的 min_avg_turnover 門檻。

以現行正確行為為準；用小型合成資料直接呼叫 helper / 純函數。
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.backtest import BacktestPipeline, WalkForwardConfig, select_market_filter_tier

PROD_TIERS = [(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)]


def _make_pipeline(**wf_kwargs) -> BacktestPipeline:
    """建立最小 BacktestPipeline（不接 DB、不 prepare），只測 helper method。"""
    wf = WalkForwardConfig(**wf_kwargs)
    cfg = SimpleNamespace()  # helper 內用 getattr(config, ..., default)，空 namespace 即走預設
    return BacktestPipeline(config=cfg, db_session=None, wf_config=wf)


def _make_picks(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": [f"{1000 + i}" for i in range(n)],
            "score": [float(n - i) for i in range(n)],
        }
    )


# ── 1. tier 選擇純函數 ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prev_bm, expected_thr, expected_mult",
    [
        (-0.06, -0.05, 0.5),    # 淺層 tier
        (-0.12, -0.10, 0.25),   # 中層 tier
        (-0.16, -0.15, 0.10),   # 深層 tier
        (-0.04, None, 1.0),     # 未觸發
        (0.03, None, 1.0),      # 上漲不觸發
        (-0.05, None, 1.0),     # 邊界：嚴格小於，恰等於門檻不觸發
    ],
)
def test_select_market_filter_tier(prev_bm, expected_thr, expected_mult):
    thr, mult = select_market_filter_tier(prev_bm, PROD_TIERS)
    assert thr == expected_thr
    assert mult == expected_mult


# ── 2. _apply_market_filter_tiers：漸進降倉行為 ─────────────────────────────


@pytest.mark.parametrize(
    "prev_bm, expected_len, expected_mult",
    [
        (-0.06, 15, 0.5),   # 30 × 0.5
        (-0.12, 7, 0.25),   # int(30 × 0.25)
        (-0.16, 3, 0.10),   # int(30 × 0.10)
        (-0.04, 30, 1.0),   # 未觸發，不降倉
    ],
)
def test_market_filter_tiers_progressive_reduction(prev_bm, expected_len, expected_mult):
    pipe = _make_pipeline(
        topn=30, market_filter_tiers=PROD_TIERS, market_filter_min_positions=2,
    )
    picks = _make_picks(30)
    period_results = [{"benchmark_return": prev_bm}]

    new_picks, skip, mult = pipe._apply_market_filter_tiers(
        picks, period_results, date(2024, 6, 3), effective_topn=30,
    )
    assert skip is False
    assert mult == expected_mult
    assert len(new_picks) == expected_len
    # 降倉時應保留分數最高的前段（picks.head）
    assert list(new_picks["stock_id"]) == list(picks["stock_id"].head(expected_len))


def test_market_filter_tiers_first_period_no_history_no_reduction():
    """第一期（無前期 benchmark）不觸發過濾。"""
    pipe = _make_pipeline(topn=30, market_filter_tiers=PROD_TIERS)
    picks = _make_picks(30)
    new_picks, skip, mult = pipe._apply_market_filter_tiers(
        picks, [], date(2024, 6, 3), effective_topn=30,
    )
    assert skip is False and mult == 1.0 and len(new_picks) == 30


def test_market_filter_tiers_zero_multiplier_holds_cash():
    """乘數 0 的 tier → 全現金（market_filter_skip=True）。"""
    pipe = _make_pipeline(topn=30, market_filter_tiers=[(-0.05, 0.0)])
    picks = _make_picks(30)
    _, skip, mult = pipe._apply_market_filter_tiers(
        picks, [{"benchmark_return": -0.06}], date(2024, 6, 3), effective_topn=30,
    )
    assert skip is True
    assert mult == 0.0


# ── 3. market_filter_min_positions=2 下限補位 ───────────────────────────────


def test_min_positions_backfills_from_day_feat():
    pipe = _make_pipeline(topn=30, market_filter_min_positions=2)
    picks = _make_picks(1)  # 過濾後只剩 1 檔 < min_positions=2
    day_feat = pd.DataFrame(
        {
            "stock_id": ["1000", "2000", "2001", "2002"],
            "score": [10.0, 9.0, 8.0, 7.0],
        }
    )
    new_picks = pipe._apply_min_positions(picks, day_feat, date(2024, 6, 3), market_filter_skip=False)
    assert len(new_picks) == 2
    # 補位取 day_feat 中未持有、score 最高者（2000）
    assert list(new_picks["stock_id"]) == ["1000", "2000"]


def test_min_positions_not_applied_when_skip_or_enough():
    pipe = _make_pipeline(topn=30, market_filter_min_positions=2)
    day_feat = _make_picks(10)
    # 全現金期不補位
    picks1 = _make_picks(0)
    assert len(pipe._apply_min_positions(picks1, day_feat, date(2024, 6, 3), market_filter_skip=True)) == 0
    # 已達下限不補位
    picks2 = _make_picks(3)
    assert len(pipe._apply_min_positions(picks2, day_feat, date(2024, 6, 3), market_filter_skip=False)) == 3


# ── 4. benchmark 流動性一致性 ───────────────────────────────────────────────


def _make_benchmark_pipeline(min_avg_turnover: float) -> BacktestPipeline:
    pipe = _make_pipeline(topn=30, min_avg_turnover=min_avg_turnover, benchmark_with_cost=False)
    rb, ex = date(2024, 1, 2), date(2024, 1, 31)
    # A +10%、B -10%（皆流動）；C +200%（不流動，應被 benchmark 排除）
    pipe.price_df = pd.DataFrame(
        {
            "stock_id": ["1101", "1101", "1102", "1102", "1103", "1103"],
            "trading_date": [rb, ex, rb, ex, rb, ex],
            "close": [100.0, 110.0, 100.0, 90.0, 10.0, 30.0],
        }
    )
    pipe.liquidity_eligible_map = {rb: {"1101", "1102"}}
    pipe.benchmark_tc = 0.0
    return pipe


def test_benchmark_applies_same_liquidity_threshold_as_strategy():
    """benchmark universe 套用與策略相同的流動性門檻（不流動股不進大盤基準）。"""
    pipe = _make_benchmark_pipeline(min_avg_turnover=0.5)
    bm = pipe._compute_benchmark_return(date(2024, 1, 2), date(2024, 1, 31))
    # 只含 1101(+10%) 與 1102(-10%) → 等權均值 0；若誤含 1103(+200%) 會變 +66.67%
    assert bm == pytest.approx(0.0, abs=1e-12)


def test_benchmark_without_liquidity_filter_includes_all():
    """門檻=0 時 benchmark 不過濾（對照組，確認上面測試差異來自流動性門檻）。"""
    pipe = _make_benchmark_pipeline(min_avg_turnover=0.0)
    bm = pipe._compute_benchmark_return(date(2024, 1, 2), date(2024, 1, 31))
    assert bm == pytest.approx((0.10 - 0.10 + 2.0) / 3)
