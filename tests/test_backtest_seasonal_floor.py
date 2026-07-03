"""行為鎖定測試：季節性降倉 topN floor 統一（config.seasonal_topn_floor）。

2026-07-03 調查結論：
- 生產路徑（enable_complex_filter=False + --seasonal-filter）走 baseline 分支，
  floor 讀 config.seasonal_topn_floor（預設 5，與 daily_pick 一般情況一致）。
- complex_filter 分支的 apply_seasonal_topn_reduction 傳 topn_floor=1 是「刻意」：
  實際下限由緊接其後的「topN 絕對下限保護」統一執行
  （極端空頭 3 檔，否則 config.seasonal_topn_floor），與 daily_pick 的 _dp_min_topn 一致。
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from skills.backtest import BacktestPipeline, WalkForwardConfig


def _make_pipeline(cfg: SimpleNamespace, **wf_kwargs) -> BacktestPipeline:
    wf = WalkForwardConfig(**wf_kwargs)
    return BacktestPipeline(config=cfg, db_session=None, wf_config=wf)


def _day_feat() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": [f"{1000 + i}" for i in range(10)],
            "score": [float(10 - i) for i in range(10)],
        }
    )


def _default_cfg(floor: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        seasonal_weak_months=(3, 10),
        seasonal_topn_multiplier=0.5,
        seasonal_topn_floor=floor,
    )


# ── 生產路徑（baseline 分支）────────────────────────────────────────────────


def test_seasonal_reduction_weak_month_production_path():
    """3 月：topN 30 → 15（×0.5，未觸 floor）。"""
    pipe = _make_pipeline(_default_cfg(), topn=30, enable_seasonal_filter=True)
    _, eff, cash, empty = pipe._apply_market_regime_filter(_day_feat(), date(2024, 3, 1), 30)
    assert (eff, cash, empty) == (15, 0.0, False)


def test_seasonal_reduction_floor_from_config_default_5():
    """3 月 topN=8：×0.5=4 < floor 5 → 5（config.seasonal_topn_floor 預設）。"""
    pipe = _make_pipeline(_default_cfg(), topn=8, enable_seasonal_filter=True)
    _, eff, _, _ = pipe._apply_market_regime_filter(_day_feat(), date(2024, 3, 1), 8)
    assert eff == 5


def test_seasonal_reduction_floor_respects_config_override():
    """config.seasonal_topn_floor=3 時：8 × 0.5 = 4 ≥ 3 → 4（floor 不再硬編 5）。"""
    pipe = _make_pipeline(_default_cfg(floor=3), topn=8, enable_seasonal_filter=True)
    _, eff, _, _ = pipe._apply_market_regime_filter(_day_feat(), date(2024, 3, 1), 8)
    assert eff == 4


def test_seasonal_reduction_not_applied_in_normal_month():
    pipe = _make_pipeline(_default_cfg(), topn=30, enable_seasonal_filter=True)
    _, eff, _, _ = pipe._apply_market_regime_filter(_day_feat(), date(2024, 6, 3), 30)
    assert eff == 30


def test_seasonal_reduction_disabled_flag():
    pipe = _make_pipeline(_default_cfg(), topn=30, enable_seasonal_filter=False)
    _, eff, _, _ = pipe._apply_market_regime_filter(_day_feat(), date(2024, 3, 1), 30)
    assert eff == 30


# ── complex_filter 分支：floor=1 + 絕對下限保護 = 等效 config floor ─────────


def test_complex_filter_seasonal_floor_unified_with_config():
    """complex 分支 3 月 topN=8：seasonal 先降到 4（floor=1 刻意不擋），
    再由「topN 絕對下限保護」抬回 config.seasonal_topn_floor=5。"""
    pipe = _make_pipeline(_default_cfg(), topn=8, enable_complex_filter=True)
    _, eff, _, _ = pipe._apply_market_regime_filter(_day_feat(), date(2024, 3, 1), 8)
    assert eff == 5


def test_complex_filter_floor_respects_config_override():
    pipe = _make_pipeline(_default_cfg(floor=3), topn=8, enable_complex_filter=True)
    _, eff, _, _ = pipe._apply_market_regime_filter(_day_feat(), date(2024, 3, 1), 8)
    # 8 × 0.5 = 4 ≥ floor 3 → 4
    assert eff == 4
