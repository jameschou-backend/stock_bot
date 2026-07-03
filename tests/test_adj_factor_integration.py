"""2026-07-03 健檢 P0-2 / P1-1 回歸測試：adj factor 整合。

P1-1：factor 內部缺日若 fillna(1.0)，會在 factor<1 區段中間製造單日假跳動。
P0-2：adj_close 與未還原 open/high/low 混用，ATR/KD/CMF 等混價特徵在歷史段
      產生物理不可能的值（2016 年 ATR% 中位數 42.8%、KD 均值 -281）。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from skills.build_features import _compute_features, apply_adj_factors


# ── P1-1：apply_adj_factors 缺日語義 ─────────────────────────────


def _price_frame(dates, close=100.0):
    return pd.DataFrame(
        {
            "stock_id": ["1101"] * len(dates),
            "trading_date": dates,
            "close": [close] * len(dates),
        }
    )


def test_internal_gap_ffill_no_fake_jump():
    """內部缺日沿用前值：定價股票的 adj_close 必須是常數（舊行為：缺日跳回 raw）。"""
    dates = pd.date_range("2018-03-12", periods=5, freq="B")
    df = _price_frame(dates)
    factor = pd.DataFrame(
        {
            "stock_id": ["1101"] * 4,
            "trading_date": [dates[0], dates[1], dates[3], dates[4]],  # 缺 dates[2]
            "adj_factor": [0.75] * 4,
        }
    )
    out = apply_adj_factors(df, factor)
    assert np.allclose(out["adj_close"].values, 75.0), (
        f"adj_close={out['adj_close'].tolist()} —— 內部缺日被填 1.0 會出現 100.0 假跳（P1-1 迴歸）"
    )


def test_leading_gap_bfill():
    """leading gap 用最早已知 factor 回填。"""
    dates = pd.date_range("2018-03-12", periods=4, freq="B")
    df = _price_frame(dates)
    factor = pd.DataFrame(
        {
            "stock_id": ["1101"] * 2,
            "trading_date": [dates[2], dates[3]],
            "adj_factor": [0.8, 0.8],
        }
    )
    out = apply_adj_factors(df, factor)
    assert np.allclose(out["adj_close"].values, 80.0)


def test_trailing_gap_ffill():
    """trailing gap（sponsor 快照後新交易日）沿用最後 factor，序列連續。"""
    dates = pd.date_range("2026-06-22", periods=4, freq="B")
    df = _price_frame(dates)
    factor = pd.DataFrame(
        {
            "stock_id": ["1101"] * 2,
            "trading_date": [dates[0], dates[1]],
            "adj_factor": [0.98, 0.98],
        }
    )
    out = apply_adj_factors(df, factor)
    assert np.allclose(out["adj_close"].values, 98.0)


def test_stock_without_any_factor_stays_raw():
    """整檔無 factor（無 adj 的下市股）回退 1.0，且不受其他股影響。"""
    dates = pd.date_range("2018-03-12", periods=3, freq="B")
    df = pd.concat([_price_frame(dates), _price_frame(dates).assign(stock_id="9999")])
    factor = pd.DataFrame(
        {
            "stock_id": ["1101"] * 3,
            "trading_date": list(dates),
            "adj_factor": [0.5] * 3,
        }
    )
    out = apply_adj_factors(df, factor)
    assert np.allclose(out[out["stock_id"] == "1101"]["adj_close"].values, 50.0)
    assert np.allclose(out[out["stock_id"] == "9999"]["adj_close"].values, 100.0)
    assert (out[out["stock_id"] == "9999"]["factor_missing"] == 1).all()


def test_adj_ohlc_columns_generated():
    """含 open/high/low 時必須產生同 factor 的 adj_open/high/low。"""
    dates = pd.date_range("2018-03-12", periods=3, freq="B")
    df = _price_frame(dates)
    df["open"] = 99.0
    df["high"] = 102.0
    df["low"] = 98.0
    factor = pd.DataFrame(
        {"stock_id": ["1101"] * 3, "trading_date": list(dates), "adj_factor": [0.5] * 3}
    )
    out = apply_adj_factors(df, factor)
    assert np.allclose(out["adj_open"].values, 49.5)
    assert np.allclose(out["adj_high"].values, 51.0)
    assert np.allclose(out["adj_low"].values, 49.0)


# ── P0-2：混價特徵必須用還原 OHLC ────────────────────────────────


def _make_dividend_stock_df(periods: int = 60, ex_idx: int = 30) -> pd.DataFrame:
    """模擬高配息股：還原序列平滑 ~100，除息日前 raw 價是還原價的 2 倍。

    舊行為（high/low/open 用 raw、close 用 adj）在除息日前段：
    high≈202 vs prev_close(adj)≈100 → TR≈100 → ATR% 爆表；KD/CMF 超出值域。
    """
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(7)
    adj_close = 100.0 + np.cumsum(rng.normal(0, 0.5, periods))
    factor = np.where(np.arange(periods) < ex_idx, 0.5, 1.0)
    raw_close = adj_close / factor
    df = pd.DataFrame(
        {
            "stock_id": ["2882"] * periods,
            "trading_date": dates,
            "open": raw_close * 0.995,
            "high": raw_close * 1.01,
            "low": raw_close * 0.99,
            "close": raw_close,
            "adj_close": adj_close,
            "adj_open": adj_close * 0.995,
            "adj_high": adj_close * 1.01,
            "adj_low": adj_close * 0.99,
            "volume": [1000] * periods,
            "foreign_net": [0] * periods,
            "trust_net": [0] * periods,
            "dealer_net": [0] * periods,
        }
    )
    return df


def test_mixed_price_features_use_adj_ohlc():
    """ATR%/KD/CMF/trend_persistence 必須在正常值域（P0-2 迴歸）。"""
    df = _make_dividend_stock_df()
    out = _compute_features(df, use_adjusted_price=True).sort_values("trading_date")
    last = out.iloc[-1]

    # 平滑 ±1% 序列的 ATR% 應在個位數百分比；混價舊行為會 >20%
    assert last["atr_14_pct"] < 0.10, (
        f"atr_14_pct={last['atr_14_pct']:.3f} —— high/low 未還原時 TR 被灌成價格級距（P0-2 迴歸）"
    )
    if "kd_d" in out.columns:
        kd = out["kd_d"].dropna()
        assert ((kd >= -1) & (kd <= 101)).all(), f"kd_d 超出 0-100 值域：{kd.min()}~{kd.max()}"
    if "cmf_20" in out.columns:
        cmf = out["cmf_20"].dropna()
        assert ((cmf >= -1.5) & (cmf <= 1.5)).all(), f"cmf_20 超出 ±1 值域：{cmf.min()}~{cmf.max()}"
    if "trend_persistence" in out.columns:
        tp = out["trend_persistence"].dropna()
        # adj close > adj open（+0.5% 序列）恆成立 → 全段應接近 1；
        # 舊行為前段 adj_close(100) < raw_open(199) 恆 False
        assert tp.iloc[-1] > 0.9, f"trend_persistence={tp.iloc[-1]} —— open 未還原（P0-2 迴歸）"
