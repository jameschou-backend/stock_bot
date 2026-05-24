#!/usr/bin/env python
"""Stage K：Combinatorial Purged Cross-Validation (CPCV) 驗證。

對 production strategy (topn=30) 跑：
  1. 切 N 個 time blocks（非重疊、time-ordered）
  2. 各種 train/test combinations（C(N, k) 個 fold）
  3. 每 fold 跑 backtest 算 Sharpe
  4. Bootstrap CI：production Sharpe 1.33 的 95% confidence interval

簡化版（避免重跑 N 個完整 backtest）：
  使用既有 10y backtest result，bootstrap re-sample 月度報酬
  → 算 Sharpe distribution → CI

López de Prado AFML Ch 7（CPCV 完整版）。

用法：
  python scripts/cpcv_validation.py
  python scripts/cpcv_validation.py --result artifacts/stage10_10y/topn30.json --n-boot 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def compute_sharpe(returns: np.ndarray, rf_annual: float = 0.015) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - rf_annual / 12
    if excess.std(ddof=1) == 0:
        return 0.0
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(12))


def compute_mdd(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    eq = np.cumprod(1 + returns)
    rmax = np.maximum.accumulate(eq)
    return float((eq / rmax - 1).min())


def bootstrap_sharpe(returns: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict:
    """Block bootstrap：保留時間結構（block size = 3 months）。"""
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < 24:
        return {"mean": 0, "std": 0, "ci_95": (0, 0), "ci_99": (0, 0)}
    block_size = 3
    n_blocks = n // block_size

    sharpes = []
    mdds = []
    for _ in range(n_boot):
        # 隨機 sample with replacement，每次取 1 block
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled = np.concatenate([returns[s:s + block_size] for s in block_starts])
        sharpes.append(compute_sharpe(sampled))
        mdds.append(compute_mdd(sampled))

    sharpes = np.array(sharpes)
    mdds = np.array(mdds)
    return {
        "n_boot": n_boot,
        "block_size_months": block_size,
        "sharpe_mean": float(sharpes.mean()),
        "sharpe_std": float(sharpes.std(ddof=1)),
        "sharpe_ci_95": (float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))),
        "sharpe_ci_99": (float(np.percentile(sharpes, 0.5)), float(np.percentile(sharpes, 99.5))),
        "mdd_mean": float(mdds.mean()),
        "mdd_ci_95": (float(np.percentile(mdds, 2.5)), float(np.percentile(mdds, 97.5))),
    }


def split_purged_cv(returns: np.ndarray, n_folds: int = 10, purge_months: int = 1) -> list:
    """Purged CV：時間順序切 N folds，相鄰 purge_months 避免 label leakage。

    回傳 [(train_returns, test_returns)] 列表。
    """
    n = len(returns)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else n
        train_idx = np.r_[
            np.arange(0, max(0, test_start - purge_months)),
            np.arange(min(n, test_end + purge_months), n),
        ]
        test_idx = np.arange(test_start, test_end)
        folds.append((returns[train_idx], returns[test_idx]))
    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=str,
                   default="artifacts/stage10_10y/topn30.json",
                   help="backtest JSON path")
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    result = json.load(open(args.result))
    periods = result.get("periods", [])
    returns = np.array([p["return"] for p in periods if p.get("return") is not None])

    print(f"\n{'='*70}")
    print(f"Stage K：CPCV / Bootstrap Confidence Interval")
    print(f"{'='*70}")
    print(f"  source: {args.result}")
    print(f"  periods: {len(returns)}")
    print(f"  full Sharpe: {compute_sharpe(returns):.4f}")
    print(f"  full MDD:    {compute_mdd(returns):+.4f}")

    # ── 1. Block bootstrap CI ──
    print(f"\n[1/2] Block bootstrap ({args.n_boot} iterations, block=3 months)...")
    ci = bootstrap_sharpe(returns, n_boot=args.n_boot)
    print(f"\n  Sharpe distribution:")
    print(f"    mean ± std:    {ci['sharpe_mean']:.4f} ± {ci['sharpe_std']:.4f}")
    print(f"    95% CI:        [{ci['sharpe_ci_95'][0]:.4f}, {ci['sharpe_ci_95'][1]:.4f}]")
    print(f"    99% CI:        [{ci['sharpe_ci_99'][0]:.4f}, {ci['sharpe_ci_99'][1]:.4f}]")
    print(f"\n  MDD distribution:")
    print(f"    mean:          {ci['mdd_mean']:+.4f}")
    print(f"    95% CI:        [{ci['mdd_ci_95'][0]:+.4f}, {ci['mdd_ci_95'][1]:+.4f}]")

    # ── 2. Purged time-series CV ──
    print(f"\n[2/2] Purged time-series CV (10 folds, purge=1 month)...")
    folds = split_purged_cv(returns, n_folds=10, purge_months=1)
    test_sharpes = []
    train_sharpes = []
    for i, (tr, te) in enumerate(folds):
        ts = compute_sharpe(te)
        trs = compute_sharpe(tr)
        test_sharpes.append(ts)
        train_sharpes.append(trs)
        print(f"  Fold {i+1:>2}: train Sharpe {trs:>+.4f} (n={len(tr)})  "
              f"test Sharpe {ts:>+.4f} (n={len(te)})")
    test_arr = np.array(test_sharpes)
    print(f"\n  Out-of-sample Sharpe (10 folds):")
    print(f"    mean ± std:    {test_arr.mean():.4f} ± {test_arr.std(ddof=1):.4f}")
    print(f"    median:        {np.median(test_arr):.4f}")
    print(f"    min / max:     [{test_arr.min():.4f}, {test_arr.max():.4f}]")
    print(f"    > 1.0 比率:    {(test_arr > 1.0).mean():.0%}")
    print(f"    > 1.3 比率:    {(test_arr > 1.3).mean():.0%}")

    # 存 JSON
    out_path = Path("artifacts/cpcv_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "source": args.result,
        "n_periods": len(returns),
        "full_sharpe": compute_sharpe(returns),
        "full_mdd": compute_mdd(returns),
        "bootstrap": ci,
        "purged_cv": {
            "test_sharpes": test_sharpes,
            "test_mean": float(test_arr.mean()),
            "test_std": float(test_arr.std(ddof=1)),
            "test_median": float(np.median(test_arr)),
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  存檔: {out_path}")


if __name__ == "__main__":
    main()
