"""train_ranker 對齊修正的行為測試（2026-07-03 P1-5 / P2-4）。

- resolve_train_end：交易日制 label buffer cutoff 的邊界行為（含交易日不足 fallback）。
- _liquidity_sample_weight：與 backtest --liq-weighted 相同的權重公式。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from skills.train_ranker import (
    LABEL_HORIZON_BUFFER_DAYS,
    _liquidity_sample_weight,
    resolve_train_end,
)


def _dates(n: int) -> list:
    """n 個連續「交易日」（工作日近似即可，函數只看序列位置）。"""
    base = date(2024, 1, 1)
    return [base + timedelta(days=i) for i in range(n)]


# ── resolve_train_end：交易日制 cutoff 邊界 ─────────────────────────────────


def test_buffer_constant_is_20():
    assert LABEL_HORIZON_BUFFER_DAYS == 20


def test_train_end_enough_trading_days():
    """交易日充足：取倒數第 (buffer+1) 個交易日。"""
    tds = _dates(100)
    assert resolve_train_end(tds, tds[-1]) == tds[-21]


def test_train_end_boundary_exactly_buffer_plus_one():
    """恰好 buffer+1 個交易日：仍走交易日制，取第一個交易日。"""
    tds = _dates(21)
    assert resolve_train_end(tds, tds[-1]) == tds[0]


def test_train_end_boundary_exactly_buffer_falls_back_to_calendar():
    """恰好 buffer 個交易日（不足）：fallback 至 max_label_date - 20 日曆天。"""
    tds = _dates(20)
    assert resolve_train_end(tds, tds[-1]) == tds[-1] - timedelta(days=20)


def test_train_end_empty_dates_falls_back_to_calendar():
    max_d = date(2024, 6, 30)
    assert resolve_train_end([], max_d) == max_d - timedelta(days=20)


def test_train_end_fallback_is_conservative():
    """fallback（日曆天）必早於或等於交易日制 cutoff —— 洩漏方向安全。

    20 日曆天涵蓋的交易日 ≤ 20，故 fallback 的 train_end 只會更早（更保守）。
    以稀疏交易日序列驗證：若把 fallback 值當交易日制用，不會晚於交易日制結果。
    """
    tds = _dates(25)
    trading_cutoff = resolve_train_end(tds, tds[-1])
    calendar_cutoff = tds[-1] - timedelta(days=LABEL_HORIZON_BUFFER_DAYS)
    assert calendar_cutoff <= trading_cutoff


# ── _liquidity_sample_weight：與 backtest --liq-weighted 公式一致 ───────────


def test_liq_weight_matches_backtest_formula():
    amt = pd.Series([0.0, 1e8, 1e9, np.nan, -5.0])
    w = _liquidity_sample_weight(amt)
    # 對照 backtest.py 實作：fillna(0).clip(lower=0) → log1p → 除以平均
    ref = np.log1p(amt.fillna(0).clip(lower=0).values.astype(float))
    ref = ref / ref.mean()
    np.testing.assert_allclose(w, ref)
    assert w.mean() == pytest.approx(1.0)


def test_liq_weight_monotonic_in_turnover():
    w = _liquidity_sample_weight(pd.Series([1e6, 1e8, 1e10]))
    assert w[0] < w[1] < w[2]


def test_liq_weight_all_zero_no_division():
    """全 0（無 amt_20 資料）不除以 0，回傳全 0 權重。"""
    w = _liquidity_sample_weight(pd.Series([0.0, 0.0, np.nan]))
    np.testing.assert_allclose(w, np.zeros(3))
