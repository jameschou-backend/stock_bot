"""PEAD rank/winsorize transform 臂 invariant 測試
（docs/prereg_pead_rank_arm_20260711.md §1 鎖定）。

鎖定：基期效應剔除（rev_prior_year floor）、winsorize（單調、clip 到 [P1,P99]）、
      rank 選股不選極端 yoy 毒尾、tiered slippage 分層、cost_mode 報酬單調遞減。
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.backtest_pead import (
    HOLD_TRADING_DAYS,
    REVENUE_BASE_FLOOR_TWD,
    TIERED_SLIP_LARGE,
    TIERED_SLIP_MID,
    TIERED_SLIP_SMALL,
    compute_deadline,
    resolve_entry_exit,
    run_cohorts,
    tiered_slippage_round_trip,
    winsorize_series,
)


# ══════════════════════════════════════════════════════════════════════════════
# (b) winsorize（§1b）：clip 到 [P1,P99]、單調保序（對 rank 選股 no-op）
# ══════════════════════════════════════════════════════════════════════════════
def test_winsorize_clips_to_percentile_bounds():
    s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 100.0, -50.0])
    w = winsorize_series(s, 1.0, 99.0)
    assert w.max() == pytest.approx(np.percentile(s, 99.0))
    assert w.min() == pytest.approx(np.percentile(s, 1.0))
    # 極端值被壓，界內中位值不變
    assert w.max() < s.max()
    assert w.min() > s.min()
    mid = w[(s >= 0.1) & (s <= 0.5)]
    assert (mid == s[(s >= 0.1) & (s <= 0.5)]).all()


def test_winsorize_is_monotone_preserves_rank():
    # winsorize 保序 → top-N by rank 成員不變（§1b 誠實聲明：對 rank 選股 no-op）
    s = pd.Series([5.0, 1.0, 100.0, 3.0, 0.5, 2.0])
    w = winsorize_series(s, 1.0, 99.0)
    assert list(np.argsort(w.values)) == list(np.argsort(s.values))


def test_winsorize_empty_series_noop():
    s = pd.Series([np.nan, np.nan])
    w = winsorize_series(s, 1.0, 99.0)
    assert w.isna().all()


# ══════════════════════════════════════════════════════════════════════════════
# 參考臂 tiered slippage（§4）：整股分層來回滑價
# ══════════════════════════════════════════════════════════════════════════════
def test_tiered_slippage_tiers():
    assert tiered_slippage_round_trip(6e8) == TIERED_SLIP_LARGE   # >= 5 億
    assert tiered_slippage_round_trip(2e8) == TIERED_SLIP_MID     # 1~5 億
    assert tiered_slippage_round_trip(5e6) == TIERED_SLIP_SMALL   # < 1 億（微型）
    # 缺失 / 非有限 → 中型 fallback（保守，與 backtest.py 一致）
    assert tiered_slippage_round_trip(float("nan")) == TIERED_SLIP_MID
    assert tiered_slippage_round_trip(0.0) == TIERED_SLIP_MID
    assert tiered_slippage_round_trip(-1.0) == TIERED_SLIP_MID


# ══════════════════════════════════════════════════════════════════════════════
# run_cohorts 端到端合成測試：transform 機制
# ══════════════════════════════════════════════════════════════════════════════
def _synthetic(specs, revenue_month=date(2024, 1, 1)):
    """建最小合成 (signals, adj_prices, exclude_ids)。

    specs: [{stock_id, yoy, rev_prior_year, ret, turnover_20}]。
    每檔 entry 價=100、exit 價=100×(1+ret)；同一 cohort（revenue_month）。
    """
    tds = pd.bdate_range("2024-01-15", "2024-05-31")
    trading_days = np.sort(tds.values)
    deadline = compute_deadline(revenue_month)
    entry_date, exit_date = resolve_entry_exit(deadline, trading_days, hold_days=HOLD_TRADING_DAYS)

    signals = pd.DataFrame([{
        "stock_id": s["stock_id"],
        "revenue_month": revenue_month,
        "revenue_yoy": s["yoy"],
        "revenue_yoy_accel": 0.0,
        "rev_prior_year": s["rev_prior_year"],
    } for s in specs])

    xd = pd.Timestamp(exit_date).normalize()
    rows = []
    for s in specs:
        exit_px = 100.0 * (1.0 + s["ret"])
        for td in tds:
            tdn = pd.Timestamp(td).normalize()
            rows.append({
                "stock_id": s["stock_id"],
                "trading_date": tdn,
                "close": exit_px if tdn >= xd else 100.0,
                "adj_close": exit_px if tdn >= xd else 100.0,
                "turnover_20": s["turnover_20"],
            })
    adj_prices = pd.DataFrame(rows)
    return signals, adj_prices, set()


def _healthy(n=5, base=1e8, turnover=5e6):
    """n 檔基期充足（> floor）的健康股，yoy 遞減、報酬 +5%。"""
    return [
        {"stock_id": f"100{i}", "yoy": 0.6 - 0.1 * i,
         "rev_prior_year": base, "ret": 0.05, "turnover_20": turnover}
        for i in range(n)
    ]


def test_base_floor_excludes_small_base_stock():
    # 基期 < 1000 萬（含 pump）→ transform 後從 pool 剔除（§1a）
    pump = {"stock_id": "9001", "yoy": 500.0, "rev_prior_year": 1_000_000,  # 100 萬 < floor
            "ret": -0.50, "turnover_20": 5e6}
    signals, adj_prices, excl = _synthetic(_healthy(5) + [pump])

    _, _ = run_cohorts(signals, adj_prices, excl, topn=3, signal_transform=False, cost_mode="none")
    periods_no, _ = run_cohorts(signals, adj_prices, excl, topn=3,
                                signal_transform=False, cost_mode="none")
    periods_tf, _ = run_cohorts(signals, adj_prices, excl, topn=3,
                                signal_transform=True, cost_mode="none")

    assert periods_no[0]["n_universe"] == 6
    assert periods_no[0]["n_pool"] == 6           # 無 transform：pump 留在 pool
    assert periods_tf[0]["n_pool"] == 5           # transform：基期 100 萬 < 1000 萬 → 剔除
    assert REVENUE_BASE_FLOOR_TWD == 10_000_000.0


def test_rank_selection_avoids_extreme_yoy_toxic_tail():
    # 診斷的失敗機制：絕對/rank top-N 會選中「基期≈0 → yoy 爆量」的毒尾微型股。
    # transform 後 pump 被基期 floor 剔除 → 選股不含毒尾 → 報酬轉正。
    pump = {"stock_id": "9001", "yoy": 500.0, "rev_prior_year": 500_000,  # 基期≈0
            "ret": -0.50, "turnover_20": 5e6}
    signals, adj_prices, excl = _synthetic(_healthy(5) + [pump])

    periods_no, _ = run_cohorts(signals, adj_prices, excl, topn=3,
                                signal_transform=False, cost_mode="none")
    periods_tf, _ = run_cohorts(signals, adj_prices, excl, topn=3,
                                signal_transform=True, cost_mode="none")

    # 無 transform：yoy=500 排第一 → 選中 pump → −50% 拖累 → 報酬為負
    assert periods_no[0]["return"] < 0
    # transform：pump 剔除 → top-3 皆健康股 (+5%) → 報酬 = +5%
    assert periods_tf[0]["return"] == pytest.approx(0.05, abs=1e-9)
    assert periods_tf[0]["return"] > periods_no[0]["return"]


def test_cost_mode_returns_are_monotone_decreasing():
    # gross > tiered net > oddlot net（成本為正；微型股 oddlot ×1.5 > 整股 tiered）
    signals, adj_prices, excl = _synthetic(_healthy(5, turnover=5e6))
    kw = dict(topn=3, signal_transform=True, min_avg_turnover_yi=0.0033)

    r_none, _ = run_cohorts(signals, adj_prices, excl, cost_mode="none", **kw)
    r_tier, _ = run_cohorts(signals, adj_prices, excl, cost_mode="tiered", **kw)
    r_odd, _ = run_cohorts(signals, adj_prices, excl,
                           cost_mode="oddlot", odd_lot_premium_mult=1.5, **kw)

    assert r_none[0]["return"] > r_tier[0]["return"] > r_odd[0]["return"]


def test_sanity_gate_pool_ic_present_after_transform():
    # transform 後 pool 診斷（sanity gate）欄位存在且可算（此合成單 cohort 僅檢結構）
    signals, adj_prices, excl = _synthetic(_healthy(5))
    _, diag = run_cohorts(signals, adj_prices, excl, topn=3,
                          signal_transform=True, cost_mode="none")
    assert "pool_long_short_returns" in diag
    assert "pool_ic_values" in diag


def test_invalid_cost_mode_raises():
    signals, adj_prices, excl = _synthetic(_healthy(5))
    with pytest.raises(ValueError):
        run_cohorts(signals, adj_prices, excl, topn=3, cost_mode="bogus")
