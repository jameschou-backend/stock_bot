"""Fractionally Differentiated Features（López de Prado AFML Ch 5）

問題：股票 close price 非平穩（有趨勢），無法直接餵 ML model；
但 log return（d=1 完全差分）保留 0 memory，丟掉「動能 / 慣性」這類訊號。

Fractional Differentiation 取 d ∈ (0, 1) — 保留部分 memory 又達到平穩：
    (1 - B)^d 用 binomial series expand：
    ỹ_t = Σ_{k=0}^K ω_k · x_{t-k}
    其中 ω_0 = 1, ω_k = ω_{k-1} · -(d - k + 1) / k

兩種實作策略：
  1. **Expanding window**: 對序列從頭累積 weights（無 lookback 限制，但末端 weight 微小）
  2. **Fixed-width FFD (推薦)**: 用累積權重 < threshold 截斷成固定 window，
                                 每個位置都使用相同數量的 lookback → 樣本間 IID 假設更合理

本模組以 FFD 為主，效能用 numpy convolve（O(N log N)）。

預期用法：
    - 對 close price apply FFD with d=0.3/0.4/0.5 → 加入 features
    - 對 cumulative volume / amt_20 等也可以 apply
    - **加 features 後，build_features 不需要動 — Stage 4.3 用 opt-in CLI 產出 parquet**

文獻：
    Marcos López de Prado, "Advances in Financial Machine Learning" 2018, Ch 5
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Weight computation
# ──────────────────────────────────────────────

def fracdiff_weights(d: float, max_size: int = 10000, threshold: float = 1e-5) -> np.ndarray:
    """計算 fractional diff 權重 ω_0, ω_1, ... 直到 |ω_K| < threshold 或達 max_size。

    ω_0 = 1
    ω_k = -ω_{k-1} · (d - k + 1) / k       (k >= 1)

    Note: 對 d ∈ (0, 1) 權重會逐漸衰減；d ≈ 1 衰減慢需要大 max_size。

    Args:
        d: differentiation order ∈ (0, 1) 通常
        max_size: 權重數量上限
        threshold: 絕對值低於此即停止累積

    Returns:
        ω 陣列，shape = (K,)，K <= max_size
    """
    if max_size < 1:
        raise ValueError("max_size >= 1")
    if threshold <= 0:
        raise ValueError("threshold > 0")
    weights = [1.0]
    for k in range(1, max_size):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    return np.asarray(weights, dtype=np.float64)


def fracdiff_weights_ffd(d: float, threshold: float = 1e-3) -> np.ndarray:
    """Fixed-width FFD weights：用較寬鬆 threshold（如 1e-3）截出較短 window。

    比 fracdiff_weights 的 1e-5 預設更短，適合產出固定長度的 lookback window。
    """
    return fracdiff_weights(d, max_size=10000, threshold=threshold)


# ──────────────────────────────────────────────
# Apply to 1D series
# ──────────────────────────────────────────────

def fracdiff_ffd_series(
    series: np.ndarray,
    d: float,
    threshold: float = 1e-3,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """對 1D series 套 Fixed-width Fractional Difference。

    回傳長度同 input；前 K-1 個位置因 window 不足填 NaN。
    NaN 在中間的位置也會傳播（任一 weight 對應 NaN → 結果 NaN）。

    Args:
        series: 1D array
        d: differentiation order
        threshold: weight cutoff（小越多 weights）
        weights: 預先算好的 weights（重複呼叫時避免重複算）

    Returns:
        same-length array
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("series must be 1-D")
    if weights is None:
        weights = fracdiff_weights_ffd(d, threshold=threshold)
    K = len(weights)
    n = len(series)
    if n < K:
        # 樣本長度不足 weights window
        return np.full(n, np.nan)

    out = np.full(n, np.nan)
    # 對每個位置 t（t >= K-1），計算 Σ_{k=0..K-1} ω_k · series[t-k]
    for t in range(K - 1, n):
        window = series[t - K + 1 : t + 1][::-1]  # 反轉成 [x_t, x_{t-1}, ...]
        if np.any(np.isnan(window)):
            out[t] = np.nan
            continue
        out[t] = float(np.dot(weights, window))
    return out


# ──────────────────────────────────────────────
# Apply to panel (per-stock)
# ──────────────────────────────────────────────

def fracdiff_panel(
    df: pd.DataFrame,
    value_col: str,
    d: float,
    out_col: Optional[str] = None,
    threshold: float = 1e-3,
    group_col: str = "stock_id",
    date_col: str = "trading_date",
) -> pd.DataFrame:
    """對 panel data 按 stock_id 分組 apply FFD。

    輸出新增一欄（預設名 `{value_col}_fracdiff_{d}`，可自訂）；不修改原欄。

    Args:
        df: 需含 group_col, date_col, value_col
        value_col: 要做 FFD 的欄位（如 'close'）
        d: differentiation order
        out_col: 輸出欄名；None 則自動命名
        threshold: weights cutoff
        group_col: 分組欄（預設 'stock_id'）
        date_col: 排序欄（預設 'trading_date'）
    """
    if value_col not in df.columns:
        raise ValueError(f"{value_col} not in df")
    if group_col not in df.columns:
        raise ValueError(f"{group_col} not in df")

    if out_col is None:
        out_col = f"{value_col}_fracdiff_{d:.2f}".replace(".", "_")

    # 預算 weights 一次（每個 stock 共用）
    weights = fracdiff_weights_ffd(d, threshold=threshold)

    result = df.copy()
    result[out_col] = np.nan
    # 用 sort 確保時間順序
    result = result.sort_values([group_col, date_col]).reset_index(drop=True)

    # per-stock apply
    for _, idx_group in result.groupby(group_col, sort=False, group_keys=False).groups.items():
        series = result.loc[idx_group, value_col].to_numpy()
        out = fracdiff_ffd_series(series, d, threshold=threshold, weights=weights)
        result.loc[idx_group, out_col] = out

    return result


# ──────────────────────────────────────────────
# Optimal d via ADF test
# ──────────────────────────────────────────────

def find_optimal_d(
    series: np.ndarray,
    d_grid: Optional[List[float]] = None,
    threshold: float = 1e-3,
    significance: float = 0.05,
) -> Tuple[float, dict]:
    """掃 d ∈ {0.05, 0.10, ..., 0.95} 找「最小 d 使 ADF p-value < significance」（即剛好平穩）。

    最小 d 意味著「保留最多 memory 又達到平穩」（López de Prado 推薦）。

    Args:
        series: 1D array（通常是 close price）
        d_grid: d 候選；None 用預設 0.05 ~ 0.95 step 0.05
        threshold: FFD weight cutoff
        significance: ADF reject H0 的 p-value 門檻

    Returns:
        (optimal_d, results)：optimal_d 為達到平穩的最小 d；
                              若沒有 d 達到平穩則回傳 0.95；
                              results 是 dict[d] -> dict with adf_stat, p_value
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        raise ImportError("find_optimal_d 需要 statsmodels（`pip install statsmodels`）") from exc

    series = np.asarray(series, dtype=np.float64)
    series = series[~np.isnan(series)]
    if len(series) < 30:
        raise ValueError(f"序列太短（n={len(series)}）無法做 ADF test")

    if d_grid is None:
        d_grid = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ~ 0.95

    results = {}
    optimal_d = None
    for d in d_grid:
        try:
            diffed = fracdiff_ffd_series(series, d, threshold=threshold)
            diffed_clean = diffed[~np.isnan(diffed)]
            if len(diffed_clean) < 20:
                results[d] = {"adf_stat": None, "p_value": None, "error": "too few non-NaN"}
                continue
            adf_stat, p_value, *_ = adfuller(diffed_clean, autolag="AIC")
            results[d] = {"adf_stat": float(adf_stat), "p_value": float(p_value)}
            if optimal_d is None and p_value < significance:
                optimal_d = d
        except Exception as exc:
            results[d] = {"adf_stat": None, "p_value": None, "error": str(exc)}

    if optimal_d is None:
        # 都沒平穩 → 用最大 d
        optimal_d = d_grid[-1]
        logger.warning("沒有 d 達到平穩（p < %.2f），使用 d=%.2f", significance, optimal_d)
    return optimal_d, results
