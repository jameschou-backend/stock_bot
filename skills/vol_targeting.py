"""Stage 7.2: Volatility Targeting（總部位 vs 現金的動態縮放）

動機：你既有 market_filter_tiers 看大盤近 N 日報酬決定持倉比例，但這是
「大盤 proxy」。Vol Targeting 直接看你**自己的** picks 組合的實際波動率：
  - 估計 portfolio 預期 vol（從 picks 歷史 returns + 等權假設）
  - 跟 target vol（如 15% 年化）比
  - 若 portfolio vol > target → 縮減總部位（持現金 cushion）
  - 若 vol < target → 保持 100%（不加槓桿，避免 over-leverage 風險）

預期效益：
  - 高波動期自動避險（如 covid 2020 / 2018 中美貿易戰爆發）
  - 不依賴大盤指標，純看自己 picks 的 realized vol
  - 跟 market_filter_tiers 可能 complementary（大盤平穩但 picks 高 vol 時生效）

設計原則：opt-in pure module；介面適合 backtest 在 simulate phase 套用 scaler。
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def annualize_vol(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """從 daily returns 算年化 vol。"""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    return float(arr.std(ddof=1) * np.sqrt(periods_per_year))


def estimate_portfolio_vol(
    picks: Iterable[str],
    price_panel: pd.DataFrame,
    rb_date,
    lookback_days: int = 60,
    weights: Optional[dict] = None,
    annualize: bool = True,
) -> float:
    """估計 picks 組合的（等權 or 自訂權重）波動率，從歷史 returns。

    Args:
        picks: stock_id iterable
        price_panel: 含 stock_id / trading_date / close
        rb_date: 估算日；用 [rb_date - lookback_days, rb_date) 的 returns
        lookback_days: 回溯天數
        weights: {stock_id: weight}，None 則等權
        annualize: True 則回年化（×√252），False 回日波動

    Returns:
        portfolio 年化 vol (or daily vol)；少於 2 個 valid stock 或 < 5 returns → 0.0
    """
    picks = list(picks)
    if not picks:
        return 0.0
    start = rb_date - timedelta(days=lookback_days)
    sub = price_panel[
        (price_panel["stock_id"].isin(picks)) &
        (price_panel["trading_date"] >= start) &
        (price_panel["trading_date"] < rb_date)
    ].copy()
    if sub.empty:
        return 0.0
    sub["close"] = pd.to_numeric(sub["close"], errors="coerce")
    sub = sub.dropna(subset=["close"])
    sub = sub.sort_values(["stock_id", "trading_date"])
    wide_px = sub.pivot(index="trading_date", columns="stock_id", values="close")
    if wide_px.shape[1] < 1 or len(wide_px) < 6:
        return 0.0
    returns = wide_px.pct_change(fill_method=None).dropna(how="all")
    if returns.empty:
        return 0.0

    # 套用 weights
    if weights is None:
        w = np.ones(returns.shape[1]) / returns.shape[1]
    else:
        w_list = []
        for c in returns.columns:
            w_list.append(float(weights.get(c, 0)))
        total = sum(w_list)
        w = np.asarray(w_list) / total if total > 0 else np.ones(len(w_list)) / len(w_list)

    # Portfolio returns time series
    port_returns = (returns.fillna(0).to_numpy() @ w)
    factor = np.sqrt(252) if annualize else 1.0
    if len(port_returns) < 2:
        return 0.0
    return float(np.nanstd(port_returns, ddof=1) * factor)


def vol_scaler(
    realized_vol: float,
    target_vol: float = 0.15,
    cap: float = 1.0,
    floor: float = 0.0,
) -> float:
    """根據 realized vol 跟 target vol 計算總部位 scaler。

    - vol = 0 (or NaN) → 1.0 fallback
    - scaler = min(cap, target / realized)
    - clamp to [floor, cap]，預設 [0, 1]（不加槓桿）

    Args:
        realized_vol: 估計年化 vol（如 0.30）
        target_vol: 目標年化 vol（預設 15%）
        cap: scaler 上限（預設 1.0，不加槓桿）
        floor: scaler 下限（預設 0.0，允許全現金）
    """
    if not np.isfinite(realized_vol) or realized_vol <= 0:
        return 1.0
    raw = target_vol / realized_vol
    return float(np.clip(raw, floor, cap))


def compute_vol_scaler_for_picks(
    picks: Iterable[str],
    price_panel: pd.DataFrame,
    rb_date,
    target_vol: float = 0.15,
    lookback_days: int = 60,
    cap: float = 1.0,
    floor: float = 0.0,
    weights: Optional[dict] = None,
) -> dict:
    """One-shot helper：給 picks + rb_date，回傳 scaler + diagnostic dict。

    Returns:
        dict with `scaler`, `realized_vol`, `target_vol`, `cash_share` (= 1 - scaler).
    """
    rv = estimate_portfolio_vol(
        picks, price_panel, rb_date,
        lookback_days=lookback_days, weights=weights,
    )
    s = vol_scaler(rv, target_vol=target_vol, cap=cap, floor=floor)
    return {
        "scaler": s,
        "realized_vol": rv,
        "target_vol": target_vol,
        "cash_share": float(1.0 - s),
    }
