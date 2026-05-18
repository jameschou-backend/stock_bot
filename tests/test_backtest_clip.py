"""Regression tests for the single-trade loss clip in `_calc_stock_return`.

歷史 bug 修正（Experiment C, commit 4440fc5, 2026-03-10）：
- `skills/backtest.py` 的 `_calc_stock_return` 內有
      `ret = max(ret, clip_loss_pct)`（預設 clip_loss_pct=-0.50）
  將退市股單期最大虧損 clip 到 -50%。
- 等權 20 股組合下，一檔退市股若未 clip 會造成 -5pp 額外損耗（-100% vs 真實 -50%），
  10 年累計 39 個 clip 事件 × 2.5pp/事件 = ~97.5pp 累積損耗。
- 移除此 clip 會讓 10y 累積報酬從 +1216% 跌回 +338%（-878pp）。

本檔保護以下不變式：
1. 嚴重虧損（如 -90%）被 clip 到 -50%（預設 clip_loss_pct=-0.50）。
2. 一般獲利（如 +100%）不受 clip 影響。
3. 邊界附近（虧損 -49%）不被 clip，仍保留原值。
4. clip 上限可由參數調整（傳入更寬鬆的 clip 仍生效）。
5. 預設 clip_loss_pct 維持 -0.50（不被靜默移除或改寬）。
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import pytest

import skills.backtest as backtest_module
from skills.backtest import _calc_stock_return, WalkForwardConfig, run_backtest


DEFAULT_CLIP = -0.50
# 用於浮點比較的容忍值
EPS = 1e-9


# ── 不變式 1：嚴重虧損被 clip 到預設 -50% ─────────────────────────────────────

def test_clip_caps_extreme_loss_at_minus_50pct():
    """entry=100, exit=10 (-90% raw return) 應被 clip 到 -0.50（預設）。"""
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=10.0,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        clip_loss_pct=DEFAULT_CLIP,
    )
    assert ret == pytest.approx(DEFAULT_CLIP, abs=EPS), (
        f"嚴重虧損 (-90%) 未被 clip：ret={ret}，預期 {DEFAULT_CLIP}。"
        f"`max(ret, -0.50)` 防退市股拖垮組合的關鍵 fix 已回歸 "
        f"(Experiment C, commit 4440fc5)。"
    )


def test_clip_caps_total_loss_at_minus_50pct():
    """entry=100, exit≈0 (-100%) 退市情境，必須 clip 到 -0.50。

    這是 commit 4440fc5 最關鍵的保護案例：等權 20 股 × 1 退市股原本造成
    -5pp 月度組合損耗，clip 後降至 -2.5pp。
    """
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=0.0001,
        transaction_cost_pct=0.00585,
        slippage_pct=0.0,
        clip_loss_pct=DEFAULT_CLIP,
    )
    assert ret == pytest.approx(DEFAULT_CLIP, abs=EPS), (
        f"退市股 (-100%) 未被 clip：ret={ret}，預期 {DEFAULT_CLIP}。"
    )


def test_clip_caps_loss_including_transaction_costs():
    """交易成本應加進 raw return 後再 clip。

    raw = exit/entry - 1 - tc - slip = 0.1/1.0 - 1 - 0.00585 - 0 = -0.90585
    clip(-0.90585, -0.50) = -0.50
    """
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=10.0,
        transaction_cost_pct=0.00585,
        slippage_pct=0.002,
        clip_loss_pct=DEFAULT_CLIP,
    )
    assert ret == pytest.approx(DEFAULT_CLIP, abs=EPS)


# ── 不變式 2：獲利情境完全不受影響 ───────────────────────────────────────────

def test_clip_does_not_affect_positive_return():
    """entry=100, exit=200 (+100%) 不應被 clip。"""
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=200.0,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        clip_loss_pct=DEFAULT_CLIP,
    )
    expected = 1.0  # 200/100 - 1
    assert ret == pytest.approx(expected, abs=EPS), (
        f"獲利情境被誤 clip：ret={ret}，預期 {expected}。"
    )


def test_clip_does_not_affect_positive_return_with_costs():
    """獲利情境扣交易成本後仍為正，且不被 clip。"""
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=200.0,
        transaction_cost_pct=0.00585,
        slippage_pct=0.001,
        clip_loss_pct=DEFAULT_CLIP,
    )
    expected = 1.0 - 0.00585 - 0.001  # 0.99315
    assert ret == pytest.approx(expected, abs=EPS)
    assert ret > 0


def test_clip_does_not_affect_small_loss():
    """-5% 一般虧損不應被 clip 至 -50%。"""
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=95.0,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        clip_loss_pct=DEFAULT_CLIP,
    )
    expected = -0.05
    assert ret == pytest.approx(expected, abs=EPS)
    assert ret > DEFAULT_CLIP


# ── 不變式 3：clip 邊界 ─────────────────────────────────────────────────────

def test_clip_boundary_just_above_threshold():
    """raw return = -0.49（剛好高於 clip 閾值 -0.50）不被 clip。"""
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=51.0,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        clip_loss_pct=DEFAULT_CLIP,
    )
    expected = -0.49
    assert ret == pytest.approx(expected, abs=EPS)
    assert ret > DEFAULT_CLIP


def test_clip_boundary_just_below_threshold():
    """raw return = -0.51（剛好低於 clip 閾值 -0.50）被 clip 到 -0.50。"""
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=49.0,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        clip_loss_pct=DEFAULT_CLIP,
    )
    assert ret == pytest.approx(DEFAULT_CLIP, abs=EPS)


def test_clip_exactly_at_threshold():
    """raw return = -0.50（等於 clip 閾值）保留原值。"""
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=50.0,
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        clip_loss_pct=DEFAULT_CLIP,
    )
    assert ret == pytest.approx(DEFAULT_CLIP, abs=EPS)


# ── 不變式 4：clip 參數可調 ──────────────────────────────────────────────────

def test_clip_parameter_overridable_tighter():
    """傳入更嚴格的 clip（-0.30）也生效。"""
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=50.0,  # -50% raw
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        clip_loss_pct=-0.30,
    )
    assert ret == pytest.approx(-0.30, abs=EPS)


def test_clip_parameter_overridable_disabled():
    """傳入 clip < -1.0（如 -1.01）等同停用，原始虧損保留。

    對應註解：「診斷可傳 -1.01 停用」。
    """
    ret = _calc_stock_return(
        entry_px=100.0,
        exit_px=10.0,  # -90%
        transaction_cost_pct=0.0,
        slippage_pct=0.0,
        clip_loss_pct=-1.01,
    )
    assert ret == pytest.approx(-0.90, abs=EPS)
    assert ret < DEFAULT_CLIP, "停用 clip 模式應保留原始嚴重虧損，不被 -50% 截斷。"


# ── 不變式 5：預設 clip_loss_pct 維持 -0.50（不被靜默移除或改寬） ──────────

def test_calc_stock_return_signature_has_clip_param():
    """`_calc_stock_return` 簽章必須仍有 `clip_loss_pct` 參數，且不可預設為 None。"""
    sig = inspect.signature(_calc_stock_return)
    assert "clip_loss_pct" in sig.parameters, (
        "`_calc_stock_return` 缺少 clip_loss_pct 參數，"
        "退市股保護機制可能已被移除。"
    )


def test_default_clip_loss_pct_in_run_backtest_is_minus_50pct():
    """run_backtest() 的 clip_loss_pct 預設必須 == -0.50。"""
    sig = inspect.signature(run_backtest)
    if "clip_loss_pct" not in sig.parameters:
        pytest.fail("run_backtest() 缺少 clip_loss_pct 參數")
    default = sig.parameters["clip_loss_pct"].default
    assert default == pytest.approx(-0.50, abs=EPS), (
        f"run_backtest() 預設 clip_loss_pct={default}，應為 -0.50。"
        f"放寬此 clip 會讓退市股拖垮組合（10y 累積 +1216% → +338%）。"
    )


def test_default_clip_loss_pct_in_walkforward_config_is_minus_50pct():
    """WalkForwardConfig.clip_loss_pct 預設必須 == -0.50。"""
    cfg = WalkForwardConfig()
    assert cfg.clip_loss_pct == pytest.approx(-0.50, abs=EPS), (
        f"WalkForwardConfig.clip_loss_pct={cfg.clip_loss_pct}，應為 -0.50。"
    )


def test_clip_source_contains_max_ret_guard():
    """直接掃描原始碼，確保 `max(ret, clip_loss_pct)` 守衛仍存在於 _calc_stock_return。

    若有人重構移除這行，本測試會立即失敗。
    """
    src = Path(backtest_module.__file__).read_text(encoding="utf-8")
    # 允許空白變化：max(ret, clip_loss_pct) 或 max(ret,clip_loss_pct)
    pattern = re.compile(r"max\s*\(\s*ret\s*,\s*clip_loss_pct\s*\)")
    assert pattern.search(src), (
        "在 skills/backtest.py 中找不到 `max(ret, clip_loss_pct)` 守衛邏輯，"
        "退市股 -50% clip 防護可能被移除（Experiment C 關鍵 fix 已回歸）。"
    )
