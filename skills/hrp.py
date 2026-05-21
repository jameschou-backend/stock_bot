"""Stage 7.1: Hierarchical Risk Parity（López de Prado 2016）

動機：你目前 production 用等權重（equal weight per pick），但 N 檔 picks 之間
可能高相關（同產業/相同 driver）→ 集中風險未分散。HRP 用 hierarchical
clustering 偵測相關性結構，然後 inverse-variance 加權，使：
  - 相關性高的 cluster 內部分享較少權重
  - 不相關 cluster 之間相對均衡
  - 不需要協方差矩陣求逆（比 Markowitz mean-variance 更穩定，
    對 ill-conditioned covariance 不會 blow up）

演算法（López 2016, Algorithm 1）：
  1. Distance matrix: d_ij = sqrt(0.5 * (1 - corr_ij))
  2. Hierarchical clustering（single linkage）
  3. Quasi-diagonalization：依 cluster tree leaf 順序重排
  4. Recursive bisection：依 quasi-diag 順序，每次切兩半，
     用 inverse-variance weighting 在兩半間分配，遞迴下去

預期效益：
  - **MDD 改善**（避免單一產業/factor 集中暴露）
  - **Sharpe 邊際改善**（透過更分散的 risk allocation）
  - 對 picks 數 N >= 5 時最有效，N < 3 退化為等權

設計原則：opt-in pure module；介面跟 backtest._compute_position_weights 兼容。
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# HRP core algorithm
# ──────────────────────────────────────────────

def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """López 2016 公式：d_ij = sqrt(0.5 * (1 - corr_ij))。

    Range: 完全正相關(corr=1) → d=0；不相關(corr=0) → d≈0.71；完全負相關(corr=-1) → d=1。
    """
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square 2-D, got shape {corr.shape}")
    # clamp 避免浮點誤差
    corr = np.clip(corr, -1.0, 1.0)
    return np.sqrt(0.5 * (1.0 - corr))


def quasi_diagonal_order(linkage_matrix: np.ndarray) -> List[int]:
    """從 scipy.cluster.hierarchy.linkage 結果取 quasi-diagonal sort order。

    輸入 linkage shape (N-1, 4)，每 row 是一個 merge。
    回傳 leaf indices 的順序，使最終 distance matrix（按此 order 重排後）
    呈準對角化（相似的 cluster 排在一起）。
    """
    link = np.asarray(linkage_matrix)
    # N+ N-1 个 nodes（N leaves + N-1 internal nodes），internal node id 從 N 開始
    n = link.shape[0] + 1
    # Recursive 展開最後一個 internal node 對應 cluster
    # Use López 2016 implementation idiom
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # 最後一個 cluster 包含的 leaf 數
    while sort_ix.max() >= n:
        # Internal node → 展開為其 children
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n]
        i = df0.index
        j = (df0.values - n).astype(int)  # 對應 linkage row（須為 int）
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return [int(x) for x in sort_ix.tolist()]


def _get_cluster_variance(cov_matrix: np.ndarray, indices: Sequence[int]) -> float:
    """計算 cluster (indices) 的 inverse-variance weighted variance：
    σ² = w' Σ w，其中 w = 1/diag(Σ) 正規化。
    """
    cov_sub = cov_matrix[np.ix_(indices, indices)]
    # Inverse variance weighting within cluster
    inv_var = 1.0 / np.diag(cov_sub)
    w = inv_var / inv_var.sum()
    return float(w @ cov_sub @ w)


def hrp_weights_from_cov(
    cov_matrix: np.ndarray, sort_order: Sequence[int]
) -> np.ndarray:
    """Recursive bisection: 依 sort_order 切割 + inverse-variance weighting。

    Args:
        cov_matrix: NxN 協方差矩陣
        sort_order: quasi-diagonal sort 後的 index 順序

    Returns:
        np.ndarray shape (N,)，weights 對應原 index（非 sorted）, 加總 = 1
    """
    n = cov_matrix.shape[0]
    if n != len(sort_order):
        raise ValueError(f"cov size {n} != sort_order length {len(sort_order)}")

    weights = np.ones(n, dtype=float)
    clusters = [list(sort_order)]  # 初始：所有 indices 一個 cluster

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            # 切兩半
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            var_left = _get_cluster_variance(cov_matrix, left)
            var_right = _get_cluster_variance(cov_matrix, right)
            # Inverse-variance allocation
            alloc = 1.0 - var_left / (var_left + var_right)  # left 拿多少權重比例
            for i in left:
                weights[i] *= alloc
            for i in right:
                weights[i] *= (1.0 - alloc)
            new_clusters.extend([left, right])
        clusters = new_clusters

    # Normalize（理論上加總應該 = 1，浮點誤差校正）
    weights = weights / weights.sum()
    return weights


def hrp_weights(
    returns_df: pd.DataFrame,
    min_periods: int = 30,
) -> np.ndarray:
    """從 returns panel 計算 HRP weights。

    Args:
        returns_df: shape (T, N)，columns 是 stock_id，rows 是 trading_date
        min_periods: 要求每股至少多少筆資料才參與 cluster

    Returns:
        np.ndarray shape (N,) 對應 returns_df.columns 順序，加總 = 1
    """
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise ImportError("hrp_weights 需要 scipy.cluster.hierarchy") from exc

    if returns_df.empty or returns_df.shape[1] < 2:
        # 1 個 stock → 100%；空表 → 空
        n = returns_df.shape[1]
        return np.full(n, 1.0 / max(n, 1)) if n > 0 else np.array([])

    # 過濾條件：min_periods 不足 OR std == 0（後者會讓 corr 出現 NaN）
    valid_cols = []
    for c in returns_df.columns:
        s = returns_df[c]
        if s.notna().sum() < min_periods:
            continue
        if s.fillna(0).std(ddof=1) < 1e-12:  # 幾乎為 0，無波動 → 排除
            continue
        valid_cols.append(c)
    if len(valid_cols) < 2:
        n = returns_df.shape[1]
        return np.full(n, 1.0 / n)

    returns_valid = returns_df[valid_cols].fillna(0).to_numpy()
    corr = np.corrcoef(returns_valid, rowvar=False)
    cov = np.cov(returns_valid, rowvar=False)

    # NaN/inf 防呆：剔除 corr 含 NaN 的列/行（罕見但發生過）
    if not np.all(np.isfinite(corr)):
        bad = np.where(~np.all(np.isfinite(corr), axis=1))[0]
        if len(bad):
            keep_mask = np.ones(len(valid_cols), dtype=bool)
            keep_mask[bad] = False
            if keep_mask.sum() < 2:
                n = returns_df.shape[1]
                return np.full(n, 1.0 / n)
            valid_cols = [valid_cols[i] for i in range(len(valid_cols)) if keep_mask[i]]
            returns_valid = returns_valid[:, keep_mask]
            corr = np.corrcoef(returns_valid, rowvar=False)
            cov = np.cov(returns_valid, rowvar=False)

    # Distance + hierarchical clustering
    dist = correlation_distance(corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    # 額外保險：dist 內任何 NaN / inf 都置 0（不影響 cluster 結構）
    dist = np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0)
    dist_condensed = squareform(dist, checks=False)
    link = linkage(dist_condensed, method="single")
    sort_order = quasi_diagonal_order(link)
    weights_valid = hrp_weights_from_cov(cov, sort_order)

    # 對應回原 returns_df.columns 順序
    full_weights = np.zeros(returns_df.shape[1])
    col_to_idx = {c: i for i, c in enumerate(returns_df.columns)}
    for k, col in enumerate(valid_cols):
        full_weights[col_to_idx[col]] = weights_valid[k]

    if full_weights.sum() > 0:
        full_weights = full_weights / full_weights.sum()
    return full_weights


# ──────────────────────────────────────────────
# Backtest-friendly wrapper
# ──────────────────────────────────────────────

def hrp_weights_for_picks(
    picks: List[str],
    price_panel: pd.DataFrame,
    rb_date,
    lookback_days: int = 60,
    min_periods: int = 30,
) -> dict:
    """對 backtest 的 picks list + price_panel 計算 HRP weights。

    Args:
        picks: 要持有的股票 id list
        price_panel: 含 stock_id / trading_date / close（或 adj_close）
        rb_date: 再平衡日；用 rb_date 之前 lookback_days 的 returns 算 corr
        lookback_days: 回溯天數
        min_periods: 個股最少 returns 筆數

    Returns:
        {stock_id: weight}，加總 = 1。少於 2 檔 → 等權 fallback。
    """
    if not picks:
        return {}
    if len(picks) == 1:
        return {picks[0]: 1.0}

    # 篩選 lookback 期間
    from datetime import timedelta
    start = rb_date - timedelta(days=lookback_days)
    sub = price_panel[
        (price_panel["stock_id"].isin(picks)) &
        (price_panel["trading_date"] >= start) &
        (price_panel["trading_date"] < rb_date)
    ].copy()
    if sub.empty:
        n = len(picks)
        return {p: 1.0 / n for p in picks}

    sub["close"] = pd.to_numeric(sub["close"], errors="coerce")
    sub = sub.dropna(subset=["close"]).sort_values(["stock_id", "trading_date"])

    # Pivot to wide returns
    wide_px = sub.pivot(index="trading_date", columns="stock_id", values="close")
    returns = wide_px.pct_change().dropna(how="all")

    weights_arr = hrp_weights(returns, min_periods=min_periods)
    cols = list(returns.columns)
    weight_map = {col: float(weights_arr[i]) for i, col in enumerate(cols)}

    # 對沒有 returns 的 picks 補等權 share
    missing = [p for p in picks if p not in weight_map]
    if missing:
        # 對 missing 拿剩餘權重的均分（或等權 fallback）
        n_have = sum(weight_map.values())
        residual = max(0.0, 1.0 - n_have)
        if residual > 1e-9:
            per = residual / len(missing)
            for p in missing:
                weight_map[p] = per
        else:
            # 全分配給 have；missing 0
            for p in missing:
                weight_map[p] = 0.0
    # 最終 normalize
    total = sum(weight_map.values())
    if total > 0:
        weight_map = {k: v / total for k, v in weight_map.items()}
    return weight_map
