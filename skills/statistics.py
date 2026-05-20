"""統計嚴謹度評估工具：Deflated Sharpe / PBO / CPCV

防止「100+ 次回測實驗」造成的 selection bias：
- **Deflated Sharpe Ratio (DSR)**：把觀察到的 Sharpe 折扣掉「N 次試驗中最大值」的期望值，
  得到「真實 alpha」的單尾顯著性 p-value。
- **PBO (Probability of Backtest Overfitting)**：將樣本切 S 段，所有 S/2 vs S/2 組合
  測試「train 期最佳策略在 test 期排名」，計算落到 median 以下的比例。
- **CPCV (Combinatorial Purged Cross-Validation)**：比 walk-forward 嚴格的 CV，
  test 區段周圍加 embargo 避免邊界 forward leakage。

參考文獻：
- Bailey & López de Prado 2014, "The Deflated Sharpe Ratio"
- Bailey, Borwein, López de Prado, Zhu 2014, "The Probability of Backtest Overfitting"
- López de Prado 2018, "Advances in Financial Machine Learning" Ch 7

所有 helper 為 pure functions，無 DB/網路依賴，方便單元測試。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant，用於 expected max of N iid normals 公式
EULER_MASCHERONI: float = 0.5772156649015329


# ──────────────────────────────────────────────
# Deflated Sharpe Ratio
# ──────────────────────────────────────────────

@dataclass
class DSRResult:
    """Deflated Sharpe Ratio 計算結果。"""
    sr_observed: float
    sr_expected_under_null: float  # H0 (no skill) 下 N 次試驗 Sharpe 最大值的期望
    n_trials: int
    n_observations: int
    p_value: float  # P(true SR > 0)
    is_significant_5pct: bool

    def __str__(self) -> str:
        verdict = "SIGNIFICANT" if self.is_significant_5pct else "NOT significant"
        return (
            f"DSR: SR_observed={self.sr_observed:.3f}, "
            f"SR_null={self.sr_expected_under_null:.3f}, "
            f"n_trials={self.n_trials}, n_obs={self.n_observations}, "
            f"p={self.p_value:.4f} → {verdict} @ 5%"
        )


def expected_max_sharpe_under_null(
    n_trials: int,
    sr_estimates_std: float = 1.0,
) -> float:
    """N 次無 alpha 試驗中，最大 Sharpe 的期望值（Bailey & López de Prado）。

    E[max{SR}] ≈ sqrt(Var[SR]) * ((1 - γ) Φ^-1(1 - 1/N) + γ Φ^-1(1 - 1/(N e)))

    其中 γ = 0.5772... (Euler-Mascheroni)，Φ^-1 是 normal quantile。
    """
    if n_trials < 1:
        raise ValueError("n_trials >= 1")
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return sr_estimates_std * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    sr_observed: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_estimates_std: float = 1.0,
) -> DSRResult:
    """Deflated Sharpe Ratio（Bailey & López de Prado 2014, eq 10）。

    在「跑過 N 個策略候選」的事實下，把觀察到的 Sharpe 折扣，得到單尾 p-value
    （>0.95 = 在 5% 顯著水準下確有正 alpha）。

    Args:
        sr_observed: 觀察到的策略 Sharpe ratio（年化或同單位）
        n_trials: 跑過幾個策略候選（用來校正 selection bias）。**這個數字越大，
                  DSR 折扣越多**。對 stock_bot 來說大約 50-100。
        n_observations: 真實樣本數（建議用月頻收益的月數，即 backtest 月份數）
        skewness: returns 的偏度（normal=0）
        kurtosis: returns 的峰度（normal=3，**不是 excess kurtosis**）
        sr_estimates_std: 所有 trial 的 SR 標準差；不知道時用 1.0 是保守上限

    Returns:
        DSRResult with `p_value` 與 `is_significant_5pct` 旗標
    """
    if n_observations < 2:
        raise ValueError("n_observations >= 2")

    sr_0 = expected_max_sharpe_under_null(n_trials, sr_estimates_std)

    # DSR 主公式（含非正態修正項）
    numerator = (sr_observed - sr_0) * math.sqrt(n_observations - 1)
    denom_sq = 1 - skewness * sr_observed + ((kurtosis - 1) / 4) * (sr_observed ** 2)
    denominator = math.sqrt(max(denom_sq, 1e-12))

    z_dsr = numerator / denominator
    p_value = float(stats.norm.cdf(z_dsr))

    return DSRResult(
        sr_observed=sr_observed,
        sr_expected_under_null=sr_0,
        n_trials=n_trials,
        n_observations=n_observations,
        p_value=p_value,
        is_significant_5pct=(p_value > 0.95),
    )


# ──────────────────────────────────────────────
# Probability of Backtest Overfitting (PBO)
# ──────────────────────────────────────────────

@dataclass
class PBOResult:
    pbo: float  # 0~1，越低越好
    n_combinations: int
    overfit_count: int
    n_strategies: int
    n_samples: int

    def __str__(self) -> str:
        return (
            f"PBO: {self.pbo:.1%} ({self.overfit_count}/{self.n_combinations} "
            f"combinations overfit) | N strategies={self.n_strategies}, "
            f"T samples={self.n_samples}"
        )


def probability_of_backtest_overfit(
    returns_matrix: np.ndarray,
    n_splits: int = 16,
) -> PBOResult:
    """PBO (Bailey, Borwein, López de Prado, Zhu 2014)。

    將 T 時間樣本切 S 段，對所有 C(S, S/2) 個「訓練 S/2 / 測試 S/2」組合：
    1. 算出每個策略在訓練期的 Sharpe
    2. 找出訓練期最佳策略
    3. 看該策略在測試期排名 (rank percentile)
    4. 落到 50% 以下 = overfit

    PBO = overfit 次數 / 總 combinations。

    Args:
        returns_matrix: shape (T, N)，T 時間樣本 × N 策略候選 returns
        n_splits: 切 S 段（必為偶數，建議 16）

    Returns:
        PBOResult；pbo < 0.5 = 沒有系統性 overfit
    """
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T, N)")
    if n_splits % 2 != 0 or n_splits < 4:
        raise ValueError("n_splits must be even and >= 4")

    T, N = returns_matrix.shape
    if T < n_splits:
        raise ValueError(f"T={T} < n_splits={n_splits}")
    if N < 2:
        raise ValueError(f"need >= 2 strategies, got N={N}")

    S = n_splits
    chunk_size = T // S
    chunks = [returns_matrix[i * chunk_size : (i + 1) * chunk_size] for i in range(S)]

    overfits = 0
    total = 0
    half = S // 2

    for train_groups in combinations(range(S), half):
        test_groups = tuple(i for i in range(S) if i not in train_groups)
        train_returns = np.concatenate([chunks[i] for i in train_groups], axis=0)
        test_returns = np.concatenate([chunks[i] for i in test_groups], axis=0)

        # Sharpe per strategy on train / test
        train_mean = train_returns.mean(axis=0)
        train_std = train_returns.std(axis=0, ddof=1) + 1e-12
        train_sharpe = train_mean / train_std

        test_mean = test_returns.mean(axis=0)
        test_std = test_returns.std(axis=0, ddof=1) + 1e-12
        test_sharpe = test_mean / test_std

        best_idx = int(np.argmax(train_sharpe))
        test_ranks = stats.rankdata(test_sharpe) / N
        if test_ranks[best_idx] < 0.5:
            overfits += 1
        total += 1

    return PBOResult(
        pbo=overfits / total if total > 0 else 0.0,
        n_combinations=total,
        overfit_count=overfits,
        n_strategies=N,
        n_samples=T,
    )


# ──────────────────────────────────────────────
# CPCV — Combinatorial Purged Cross-Validation
# ──────────────────────────────────────────────

def cpcv_splits(
    n_samples: int,
    n_groups: int = 12,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> Iterator[Tuple[List[int], List[int]]]:
    """CPCV splits（López de Prado 2018, Ch 7）。

    將時間序列切 K 等分，從中選 k 組當 test，其餘當 train。每個 test 區段
    周圍加 embargo（與 train 隔絕一段距離）避免邊界 forward leakage。

    Total combinations = C(K, k)。預設 (12, 2) = 66 個 fold。

    Args:
        n_samples: 總樣本數
        n_groups: 切 K 組
        n_test_groups: 每次 test 用 k 組（典型 1-2）
        embargo_pct: 每側 embargo 區域占總樣本的比例（0.01 = 1%）

    Yields:
        (train_indices, test_indices) tuples
    """
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups < n_groups")
    if not (0 <= embargo_pct < 0.5):
        raise ValueError("embargo_pct in [0, 0.5)")

    chunk = n_samples // n_groups
    embargo = max(1, int(n_samples * embargo_pct)) if embargo_pct > 0 else 0

    for test_groups in combinations(range(n_groups), n_test_groups):
        test_idx: List[int] = []
        test_boundaries: List[Tuple[int, int]] = []
        for g in test_groups:
            start, end = g * chunk, (g + 1) * chunk
            test_idx.extend(range(start, end))
            test_boundaries.append((start, end))
        test_set = set(test_idx)

        train_idx: List[int] = []
        for i in range(n_samples):
            if i in test_set:
                continue
            in_embargo = any(
                (start - embargo) <= i < start or end <= i < (end + embargo)
                for start, end in test_boundaries
            )
            if not in_embargo:
                train_idx.append(i)

        yield train_idx, test_idx


# ──────────────────────────────────────────────
# Convenience: 從 returns 計算 annualized Sharpe + 高階動差
# ──────────────────────────────────────────────

def sharpe_from_returns(returns: np.ndarray, periods_per_year: int = 12) -> float:
    """從 returns array 計算年化 Sharpe。

    Args:
        returns: 期間收益（如 monthly returns 用 periods_per_year=12，daily 用 252）
        periods_per_year: 年化倍數
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std <= 0:
        return 0.0
    return float(mean / std * math.sqrt(periods_per_year))


def returns_moments(returns: np.ndarray) -> dict:
    """回傳 returns 的 mean / std / skew / kurt（kurt 為非 excess）。"""
    returns = np.asarray(returns, dtype=float)
    return {
        "n": int(len(returns)),
        "mean": float(returns.mean()) if len(returns) else 0.0,
        "std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "skewness": float(stats.skew(returns)) if len(returns) > 2 else 0.0,
        # scipy excess kurtosis -> 加 3 變成原始 kurtosis
        "kurtosis": float(stats.kurtosis(returns) + 3) if len(returns) > 3 else 3.0,
    }
