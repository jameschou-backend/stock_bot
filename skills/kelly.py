"""Stage 7.3: Kelly Criterion fractional position sizing。

對每月 top-N picks 不再等權，依連續凱利公式調整相對比例：

    f_i ∝ μ_i / σ²_i

其中：
  - μ_i：第 i 檔股票的 expected return（從 model score percentile 線性校正）
  - σ²_i：個股 60d 日報酬 variance（annualized）

設計選擇：
  1. **Half-Kelly**：實證上 full Kelly 對 μ 估計誤差非常敏感，half-Kelly (×0.5)
     在 Sharpe 損失極小（理論上 -25% growth）但 drawdown 大幅改善（Thorp 1969）。
  2. **Per-stock cap**：sum(f_i)=1 保持滿倉（總現金由 vol-target 控制），
     但個股最大權重 cap（預設 0.10）防止過度集中。
  3. **與 7.2 vol-target 互補**：Kelly 改變 picks 之間相對比例，vol-target 改變
     總現金部位。兩者堆疊應該有 multiplicative benefit。
  4. **Pure module**：不依賴 backtest 內部狀態，便於獨立測試 / quick eval。

文獻：
  - Kelly 1956 "A New Interpretation of Information Rate"
  - Thorp 1969 "Optimal Gambling Systems for Favorable Games"
  - MacLean, Thorp, Ziemba 2010 "The Kelly Capital Growth Investment Criterion"
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def compute_realized_vol(
    returns: np.ndarray,
    min_periods: int = 20,
    annualize: bool = True,
) -> float:
    """計算日報酬序列的 realized volatility（已年化）。

    Args:
        returns: 日報酬陣列（小數，非 %）
        min_periods: 最少樣本數，不足回傳 NaN
        annualize: True → ×√252

    Returns:
        annualized vol (e.g. 0.30 = 30%)，樣本不足回 nan
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < min_periods:
        return float("nan")
    sigma = float(returns.std(ddof=1))
    if annualize:
        sigma *= np.sqrt(252.0)
    return sigma


def percentile_to_expected_return(
    score_pct: np.ndarray,
    er_low: float = 0.05,
    er_high: float = 0.35,
) -> np.ndarray:
    """將 cross-sectional score percentile (0~1) 線性映射成 expected annual return。

    經驗校正：top-20 picks 的歷史年化 ≈ +43%，平均 Sharpe ~1.1 → top picks 約 25-40% er。
    用 [er_low=5%, er_high=35%] 線性內插：percentile=0 → 5%，percentile=1 → 35%。

    這只是粗略校正，主要目的是讓 score 較高者 μ 較大，從而 Kelly 配比較重。
    """
    pct = np.clip(np.asarray(score_pct, dtype=float), 0.0, 1.0)
    return er_low + (er_high - er_low) * pct


def kelly_weights(
    expected_returns: np.ndarray,
    realized_vols: np.ndarray,
    half_kelly: bool = True,
    per_stock_cap: float = 0.10,
    min_weight: float = 0.0,
) -> np.ndarray:
    """從每檔 μ_i, σ_i 算 Kelly fractional weights。

    Steps:
      1. raw_f_i = μ_i / σ_i²        # full Kelly
      2. 半凱利：raw_f_i *= 0.5
      3. 負 raw_f_i → 設為 min_weight（不做空）
      4. normalize 到 sum=1
      5. cap：individual ≤ per_stock_cap，超出的重新分配

    Args:
        expected_returns: 每檔 μ_i（小數，已年化）
        realized_vols: 每檔 σ_i（小數，已年化）
        half_kelly: True → ×0.5 降低估計誤差敏感
        per_stock_cap: 個股最大權重 (0.10 = 10%)
        min_weight: 最小權重（負 Kelly 或 NaN 退化值），預設 0

    Returns:
        np.ndarray shape (n,)，sum=1（除非 cap 太小無法滿配）
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(realized_vols, dtype=float)
    n = len(mu)
    if n == 0:
        return np.zeros(0)
    if len(sigma) != n:
        raise ValueError(f"length mismatch: μ={n}, σ={len(sigma)}")

    # σ² 為 0 或 NaN 時退化為等權
    sigma2 = sigma ** 2
    valid = (sigma2 > 1e-12) & np.isfinite(mu) & np.isfinite(sigma2)
    raw = np.where(valid, mu / np.maximum(sigma2, 1e-12), 0.0)
    if half_kelly:
        raw = raw * 0.5

    # 不做空：負 Kelly → min_weight
    raw = np.maximum(raw, min_weight)

    # 若全部 ≤ 0 → 退化為等權
    s = raw.sum()
    if s <= 1e-12:
        return np.full(n, 1.0 / n)

    w = raw / s

    # individual cap：迴圈分配剩餘權重
    for _ in range(10):
        excess_mask = w > per_stock_cap
        if not excess_mask.any():
            break
        excess = (w[excess_mask] - per_stock_cap).sum()
        w[excess_mask] = per_stock_cap
        # 將 excess 平均分給未 cap 的
        free_mask = ~excess_mask
        if not free_mask.any():
            # 所有 stocks 都 cap，無法滿配 → 維持並 break
            break
        w[free_mask] += excess / free_mask.sum()
    # 浮點誤差修正
    w = np.clip(w, 0.0, 1.0)
    if w.sum() > 1.0:
        w = w / w.sum()
    return w


def compute_kelly_weights_for_picks(
    pick_stock_ids: Sequence[str],
    pick_scores: Sequence[float],
    price_df: pd.DataFrame,
    rb_date,
    lookback_days: int = 60,
    half_kelly: bool = True,
    per_stock_cap: float = 0.10,
    er_low: float = 0.05,
    er_high: float = 0.35,
) -> Dict[str, object]:
    """對給定 picks 計算 Kelly fractional weights（含失敗 fallback）。

    Args:
        pick_stock_ids: top-N picks 的 stock_id list
        pick_scores: 對應的 model score（越大越好）
        price_df: 全市場價格 DataFrame（須含 stock_id, trading_date, close）
        rb_date: 再平衡日（不含當日的價格用於估 vol）
        lookback_days: realized vol 回溯天數
        half_kelly: True → 半凱利
        per_stock_cap: 個股最大權重
        er_low / er_high: percentile→expected return 線性映射端點

    Returns:
        dict with keys:
          - weights: dict[stock_id] = weight (sum=1)
          - mu: dict[stock_id] = expected return
          - sigma: dict[stock_id] = realized vol (annualized)
          - cap_hit: list of stock_ids that hit per_stock_cap
          - fallback: bool — True 表示退化為等權（資料不足等）
    """
    n = len(pick_stock_ids)
    if n == 0:
        return {"weights": {}, "mu": {}, "sigma": {}, "cap_hit": [], "fallback": False}

    pick_stock_ids = [str(s) for s in pick_stock_ids]
    # 等權 fallback
    eq = {s: 1.0 / n for s in pick_stock_ids}
    fallback_result = {
        "weights": eq, "mu": {}, "sigma": {}, "cap_hit": [], "fallback": True,
    }

    if price_df is None or len(price_df) == 0:
        return fallback_result

    # 確認必要欄位
    if not {"stock_id", "trading_date", "close"}.issubset(price_df.columns):
        return fallback_result

    pdf = price_df.copy()
    pdf["stock_id"] = pdf["stock_id"].astype(str)
    pdf["trading_date"] = pd.to_datetime(pdf["trading_date"]).dt.date

    rb_d = pd.to_datetime(rb_date).date() if not hasattr(rb_date, "year") else rb_date
    # 嚴格 <：rb_date 當天的價格不用於計算 vol
    from datetime import timedelta as _td
    start = rb_d - _td(days=int(lookback_days) * 2 + 30)
    sub = pdf[(pdf["trading_date"] >= start) & (pdf["trading_date"] < rb_d)]
    if sub.empty:
        return fallback_result

    sigmas = np.full(n, np.nan)
    for i, sid in enumerate(pick_stock_ids):
        s_px = sub[sub["stock_id"] == sid].sort_values("trading_date")
        if len(s_px) < 20:
            continue
        # 取最近 lookback_days 筆
        s_px = s_px.tail(lookback_days)
        rets = s_px["close"].pct_change().dropna().values
        sigmas[i] = compute_realized_vol(rets, min_periods=20)

    # 若 >50% picks 資料不足 → 退回等權
    invalid = np.isnan(sigmas).sum()
    if invalid > n // 2:
        return fallback_result

    # NaN sigma 用中位數填補
    valid_mask = ~np.isnan(sigmas)
    if not valid_mask.any():
        return fallback_result
    med = float(np.median(sigmas[valid_mask]))
    sigmas = np.where(valid_mask, sigmas, med)

    # score → percentile → expected return
    scores = np.asarray(pick_scores, dtype=float)
    from scipy.stats import rankdata
    pct = (rankdata(scores) - 1) / max(len(scores) - 1, 1) if len(scores) > 1 else np.array([0.5])
    mu = percentile_to_expected_return(pct, er_low=er_low, er_high=er_high)

    w = kelly_weights(
        expected_returns=mu, realized_vols=sigmas,
        half_kelly=half_kelly, per_stock_cap=per_stock_cap,
    )
    cap_hit = [sid for sid, wi in zip(pick_stock_ids, w) if wi >= per_stock_cap - 1e-9]
    return {
        "weights": {sid: float(wi) for sid, wi in zip(pick_stock_ids, w)},
        "mu": {sid: float(m) for sid, m in zip(pick_stock_ids, mu)},
        "sigma": {sid: float(s) for sid, s in zip(pick_stock_ids, sigmas)},
        "cap_hit": cap_hit,
        "fallback": False,
    }
