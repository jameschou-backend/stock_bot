"""Stage 1.6 fix: 驗證 resolve_liquidity_threshold_twd() 單位統一行為。

footgun 來源：config.min_avg_turnover 單位是「億元」，多處 caller 寫 `* 1e8` 換算成「元」。
helper 集中此轉換並明示單位。
"""
from __future__ import annotations

from types import SimpleNamespace

from skills.risk import resolve_liquidity_threshold_twd


def test_uses_min_amt_20_when_positive():
    """min_amt_20 已是「元」，> 0 時直接回。"""
    cfg = SimpleNamespace(min_amt_20=1e8, min_avg_turnover=999.0)
    assert resolve_liquidity_threshold_twd(cfg) == 1e8


def test_falls_back_to_min_avg_turnover_billion():
    """min_amt_20=0 → 走 min_avg_turnover × 1e8。0.5 億 = 5e7 元。"""
    cfg = SimpleNamespace(min_amt_20=0.0, min_avg_turnover=0.5)
    assert resolve_liquidity_threshold_twd(cfg) == 5e7


def test_zero_threshold_when_both_zero():
    cfg = SimpleNamespace(min_amt_20=0.0, min_avg_turnover=0.0)
    assert resolve_liquidity_threshold_twd(cfg) == 0.0


def test_handles_none_attributes():
    """getattr(...) 回 None 也不該炸（or 0.0 保底）。"""
    cfg = SimpleNamespace(min_amt_20=None, min_avg_turnover=None)
    assert resolve_liquidity_threshold_twd(cfg) == 0.0


def test_handles_missing_attributes():
    """完全沒設兩個屬性也不該 AttributeError。"""
    cfg = SimpleNamespace()
    assert resolve_liquidity_threshold_twd(cfg) == 0.0


def test_min_amt_20_negative_falls_back():
    """負值不該被當成有效門檻（行為等同未設）。"""
    cfg = SimpleNamespace(min_amt_20=-1.0, min_avg_turnover=0.5)
    # 目前 helper 用 `> 0` 判斷，負值會 fall back 到 min_avg_turnover × 1e8
    assert resolve_liquidity_threshold_twd(cfg) == 5e7
