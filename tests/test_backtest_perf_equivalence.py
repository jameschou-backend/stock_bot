"""backtest 效能索引路徑 vs 原全表掃描路徑的等價性鎖定測試（2026-07-03 健檢發現 1/3/4/9）。

鐵律：所有效能改動必須行為零改變。此檔以隨機資料鎖定：
  - _DateRangeIndex 選列（含列序）與 boolean mask 完全一致
  - _liquidity_ok_mask 與原 Python 迴圈完全一致
  - _compute_benchmark_return pivot 快路徑與舊全表掃描路徑數值 exact 相等
  - _simulate_period stock_row_index 路徑與 boolean mask 路徑結果 exact 相等
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.backtest import (
    BacktestPipeline,
    WalkForwardConfig,
    _DateRangeIndex,
    _liquidity_ok_mask,
    _liquidity_pairs_frame,
    _simulate_period,
)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def _random_price_df(rng, n_stocks=8, n_days=60, shuffle=True):
    sids = [f"{1000+i}" for i in range(n_stocks)]
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(n_days)]
    rows = []
    for sid in sids:
        px = 50.0 + rng.random() * 100
        for d in days:
            px *= 1 + rng.normal(0, 0.02)
            rows.append({
                "stock_id": sid, "trading_date": d,
                "open": px, "high": px * 1.01, "low": px * 0.99,
                "close": px, "volume": int(rng.integers(1000, 99999)),
            })
    df = pd.DataFrame(rows)
    if shuffle:  # 模擬非排序輸入（DuckDB 讀取順序不保證）
        df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    return df


def test_date_range_index_matches_boolean_mask(rng):
    df = _random_price_df(rng)
    idx = _DateRangeIndex(df["trading_date"])
    d1, d2 = date(2025, 1, 10), date(2025, 2, 3)

    # eq
    expect = df[df["trading_date"] == d1]
    got = df.iloc[idx.eq_positions(d1)]
    pd.testing.assert_frame_equal(got, expect, check_exact=True)

    # <= (end_incl)
    expect = df[df["trading_date"] <= d2]
    got = df.iloc[idx.positions(end_incl=d2)]
    pd.testing.assert_frame_equal(got, expect, check_exact=True)

    # < (end_excl)
    expect = df[df["trading_date"] < d2]
    got = df.iloc[idx.positions(end_excl=d2)]
    pd.testing.assert_frame_equal(got, expect, check_exact=True)

    # 閉區間
    expect = df[(df["trading_date"] >= d1) & (df["trading_date"] <= d2)]
    got = df.iloc[idx.positions(start=d1, end_incl=d2)]
    pd.testing.assert_frame_equal(got, expect, check_exact=True)


def test_liquidity_mask_matches_python_loop(rng):
    df = _random_price_df(rng, n_stocks=6, n_days=30)
    days = sorted(df["trading_date"].unique())
    # 隨機 eligible map（部分日期缺席）
    liq_map = {}
    for d in days[::2]:
        liq_map[d] = set(rng.choice([f"{1000+i}" for i in range(6)], size=3, replace=False))

    expect = np.array([
        str(sid) in liq_map.get(td, set())
        for sid, td in zip(df["stock_id"].astype(str).values,
                           pd.to_datetime(df["trading_date"]).dt.date.values)
    ])
    got = _liquidity_ok_mask(df["stock_id"], df["trading_date"], _liquidity_pairs_frame(liq_map))
    np.testing.assert_array_equal(got, expect)


def _make_pipeline(price_df, min_avg_turnover=0.0, liq_map=None):
    wf = WalkForwardConfig(topn=5, min_avg_turnover=min_avg_turnover, benchmark_with_cost=False)
    pipe = BacktestPipeline(config=SimpleNamespace(), db_session=None, wf_config=wf)
    pipe.price_df = price_df
    pipe.liquidity_eligible_map = liq_map or {}
    pipe.benchmark_tc = 0.0
    return pipe


def test_benchmark_pivot_path_equals_scan_path(rng):
    df = _random_price_df(rng)
    days = sorted(df["trading_date"].unique())
    rb, ex = days[5], days[40]
    liq_map = {rb: {f"{1000+i}" for i in range(5)}}

    for mat, lmap in [(0.0, {}), (0.5, liq_map)]:
        pipe_old = _make_pipeline(df, mat, lmap)          # 無 pivot → 舊全表掃描路徑
        pipe_new = _make_pipeline(df, mat, lmap)
        pipe_new._bm_close_pivot = df.pivot_table(index="trading_date", columns="stock_id", values="close")
        pipe_new._bm_col_str = pd.Index([str(c) for c in pipe_new._bm_close_pivot.columns])

        old = pipe_old._compute_benchmark_return(rb, ex)
        new = pipe_new._compute_benchmark_return(rb, ex)
        assert new == old  # 浮點 exact，不是近似


def test_benchmark_pivot_path_zero_price_and_missing(rng):
    """0 價格排除（Bug-2）與缺日回 0.0 的行為在快路徑上保持一致。"""
    df = _random_price_df(rng, n_stocks=4, n_days=10)
    days = sorted(df["trading_date"].unique())
    rb, ex = days[0], days[-1]
    # 注入 0 價
    df.loc[(df["stock_id"] == "1000") & (df["trading_date"] == rb), "close"] = 0.0

    pipe_old = _make_pipeline(df)
    pipe_new = _make_pipeline(df)
    pipe_new._bm_close_pivot = df.pivot_table(index="trading_date", columns="stock_id", values="close")
    pipe_new._bm_col_str = pd.Index([str(c) for c in pipe_new._bm_close_pivot.columns])
    assert pipe_new._compute_benchmark_return(rb, ex) == pipe_old._compute_benchmark_return(rb, ex)
    # 不存在的日期 → 兩路徑皆 (0.0, 0.0)（報表口徑, 零成本 raw）
    ghost = date(2030, 1, 1)
    assert pipe_new._compute_benchmark_return(ghost, ex) == (0.0, 0.0)
    assert pipe_old._compute_benchmark_return(ghost, ex) == (0.0, 0.0)


def test_simulate_period_stock_index_equals_mask_path(rng):
    df = _random_price_df(rng)
    days = sorted(df["trading_date"].unique())
    entry, exit_ = days[10], days[35]
    picks = pd.DataFrame({
        "stock_id": ["1001", "1003", "1005", "9999"],  # 9999 不存在 → 兩路徑皆忽略
        "score": [0.9, 0.8, 0.7, 0.6],
    })
    stock_rows = {str(k): v for k, v in df.groupby("stock_id").indices.items()}

    kwargs = dict(
        stoploss_pct=-0.07, transaction_cost_pct=0.00585, entry_delay_days=0,
        enable_slippage=False, clip_loss_pct=-0.50,
    )
    old = _simulate_period(picks, df, entry, exit_, **kwargs)
    new = _simulate_period(picks, df, entry, exit_, stock_row_index=stock_rows, **kwargs)
    old.pop("_stoploss_time"), new.pop("_stoploss_time")
    assert new == old  # dict 全比對（含 stock_returns 浮點 exact）
