"""PEAD 事件臂引擎 invariant 測試（docs/prereg_pead_arm_20260711.md §4 鎖定）。

鎖定：deadline 公式、無 lookahead（entry > deadline）、出場 = entry+20 交易日、
      YoY 同月去年定義、成本套用方向、benchmark 零成本口徑。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.backtest_pead import (
    HOLD_TRADING_DAYS,
    MONTHLY_FILING_DEADLINE_DAY,
    ROUND_TRIP_TAX_FEE,
    compute_deadline,
    compute_yoy_columns,
    resolve_entry_exit,
    summarize,
)
from skills.odd_lot_costs import odd_lot_round_trip_cost


# ── deadline 公式（§2）───────────────────────────────────────────────────────
def test_compute_deadline_next_month_day10():
    # 營收月 M 存於 M-01；deadline = (M+1) 月 10 日
    assert compute_deadline(date(2026, 6, 1)) == date(2026, 7, 10)
    assert compute_deadline(date(2026, 1, 1)) == date(2026, 2, 10)


def test_compute_deadline_year_rollover():
    # 12 月營收 → 次年 1 月 10 日
    assert compute_deadline(date(2025, 12, 1)) == date(2026, 1, 10)
    assert compute_deadline(date(2025, 12, 1)).day == MONTHLY_FILING_DEADLINE_DAY


# ── 無 lookahead 鐵律（§4）：entry 嚴格 > deadline ──────────────────────────────
def _trading_days(start: str, n: int) -> np.ndarray:
    """產生 n 個工作日（近似交易日序列，週末剔除）。"""
    days = pd.bdate_range(start=start, periods=n)
    return np.sort(days.values)


def test_entry_strictly_after_deadline():
    tds = _trading_days("2026-06-01", 120)
    deadline = date(2026, 7, 10)  # 假設營收月 2026-06
    entry, exit_ = resolve_entry_exit(deadline, tds, hold_days=HOLD_TRADING_DAYS)
    assert entry is not None
    # 鐵律：進場嚴格晚於申報截止（不早於 deadline 可用資訊）
    assert entry > deadline


def test_entry_is_first_trading_day_ge_deadline_plus_one():
    tds = _trading_days("2026-06-01", 120)
    deadline = date(2026, 7, 10)
    entry, exit_ = resolve_entry_exit(deadline, tds, hold_days=HOLD_TRADING_DAYS)
    # 手算：第一個 >= deadline 的交易日，再 +1 交易日
    dl64 = np.datetime64(deadline)
    first_ge = int(np.searchsorted(tds, dl64, side="left"))
    expect_entry = pd.Timestamp(tds[first_ge + 1]).date()
    assert entry == expect_entry


def test_exit_is_entry_plus_20_trading_days():
    tds = _trading_days("2026-06-01", 120)
    deadline = date(2026, 7, 10)
    entry, exit_ = resolve_entry_exit(deadline, tds, hold_days=HOLD_TRADING_DAYS)
    dl64 = np.datetime64(deadline)
    first_ge = int(np.searchsorted(tds, dl64, side="left"))
    entry_idx = first_ge + 1
    expect_exit = pd.Timestamp(tds[entry_idx + HOLD_TRADING_DAYS]).date()
    assert exit_ == expect_exit


def test_resolve_entry_exit_insufficient_future_days_returns_none():
    # deadline 靠近序列尾端 → 未來交易日不足 20 → None（不可 lookahead 補）
    tds = _trading_days("2026-06-01", 30)
    deadline = date(2026, 7, 10)
    assert resolve_entry_exit(deadline, tds, hold_days=HOLD_TRADING_DAYS) is None


def test_entry_when_deadline_is_itself_a_trading_day():
    # deadline 當天就是交易日：searchsorted 'left' 命中該日 → entry 仍為其後第 1 個交易日
    tds = _trading_days("2026-07-01", 120)
    deadline = pd.Timestamp(tds[5]).date()  # 保證是交易日
    entry, _ = resolve_entry_exit(deadline, tds, hold_days=HOLD_TRADING_DAYS)
    assert entry == pd.Timestamp(tds[6]).date()
    assert entry > deadline


# ── YoY 同月去年定義（§1）────────────────────────────────────────────────────
def test_compute_yoy_same_month_prior_year():
    df = pd.DataFrame({
        "stock_id": ["1234"] * 13,
        "revenue_month": pd.date_range("2024-01-01", periods=13, freq="MS"),
        "revenue_current_month": [100] * 12 + [130],  # 2025-01 = 130 vs 2024-01 = 100
    })
    out = compute_yoy_columns(df)
    jan2025 = out[out["revenue_month"] == pd.Timestamp("2025-01-01")]
    assert len(jan2025) == 1
    assert jan2025["revenue_yoy"].iloc[0] == pytest.approx(0.30)


def test_compute_yoy_drops_rows_without_prior_year():
    # 只有 12 個月 → 沒有任何一列有「同月去年」→ 全部 drop
    df = pd.DataFrame({
        "stock_id": ["1234"] * 12,
        "revenue_month": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "revenue_current_month": [100] * 12,
    })
    out = compute_yoy_columns(df)
    assert out.empty


def test_compute_yoy_zero_prior_year_excluded():
    # 同月去年為 0 → 分母無效 → 該列被 drop（避免 inf）
    df = pd.DataFrame({
        "stock_id": ["1234"] * 13,
        "revenue_month": pd.date_range("2024-01-01", periods=13, freq="MS"),
        "revenue_current_month": [0] + [100] * 11 + [50],  # 2024-01=0 → 2025-01 yoy 無效
    })
    out = compute_yoy_columns(df)
    assert (out["revenue_month"] != pd.Timestamp("2025-01-01")).all()


def test_compute_yoy_accel_is_diff_of_consecutive_yoy():
    df = pd.DataFrame({
        "stock_id": ["1234"] * 14,
        "revenue_month": pd.date_range("2024-01-01", periods=14, freq="MS"),
        # 2024-01=100, 2025-01=110 (yoy=0.10); 2024-02=100, 2025-02=121 (yoy=0.21)
        "revenue_current_month": [100, 100] + [100] * 10 + [110, 121],
    })
    out = compute_yoy_columns(df).sort_values("revenue_month")
    feb2025 = out[out["revenue_month"] == pd.Timestamp("2025-02-01")]
    # accel = yoy(2025-02) − yoy(2025-01) = 0.21 − 0.10 = 0.11
    assert feb2025["revenue_yoy_accel"].iloc[0] == pytest.approx(0.11, abs=1e-9)


# ── 成本套用方向（§5 Arm B）───────────────────────────────────────────────────
def test_odd_lot_cost_positive_and_net_below_gross():
    # 微型股層來回成本 > 0，套上後 net < gross
    cost = ROUND_TRIP_TAX_FEE + odd_lot_round_trip_cost(
        amt_20=5e6, trade_date=date(2024, 1, 15), premium_mult=1.5,
    )
    assert cost > 0
    gross = 0.05
    net = gross - cost
    assert net < gross


def test_pessimistic_mult_increases_cost():
    base = odd_lot_round_trip_cost(amt_20=5e6, trade_date=date(2024, 1, 15), premium_mult=1.0)
    pess = odd_lot_round_trip_cost(amt_20=5e6, trade_date=date(2024, 1, 15), premium_mult=1.5)
    assert pess > base


# ── summary 口徑（zero-cost benchmark / excess）───────────────────────────────
def test_summarize_excess_return_is_strategy_minus_benchmark():
    periods = [
        {"entry_date": "2024-01-15", "exit_date": "2024-02-15", "return": 0.05, "benchmark_return": 0.02},
        {"entry_date": "2024-02-15", "exit_date": "2024-03-15", "return": -0.01, "benchmark_return": 0.01},
        {"entry_date": "2024-03-15", "exit_date": "2024-04-15", "return": 0.03, "benchmark_return": 0.00},
    ]
    s = summarize(periods)
    # excess_return = 策略累積 − benchmark 累積
    assert s["excess_return"] == pytest.approx(s["total_return"] - s["benchmark_total_return"], abs=1e-6)
    assert s["n_periods"] == 3
