#!/usr/bin/env python
"""Stage 10.1: 多策略組合 ensemble。

跑 N 個 sub-strategy（在 topn / lookback / rebalance_freq / liquidity 維度差異化），
資金等權分配，組合 equity_curve 看是否能提升 Sharpe / 降低 MDD。

設計：
  - 不依賴新資料、不改 model 架構，純策略層級組合
  - 5 個 sub-strategy 都是「production-like」配置但各自調一個維度
  - Diversification 假設：sub 之間相關性 < 1.0 → 組合 Sharpe > 任一 sub

實作：
  - 各 sub 獨立 run_backtest，回傳 equity_curve（資金 1 的成長）
  - combined = mean(equity_curves)  ← 簡化：忽略 picks 重複（保守）
  - 計算 combined 的 Sharpe / MDD / Calmar

判定門檻：
  combined Sharpe > best sub Sharpe + 0.05
  AND combined MDD ≥ best sub MDD - 2pp

用法：
    python scripts/multi_strategy_ensemble.py --months 60     # quick eval
    python scripts/multi_strategy_ensemble.py --months 120    # 10y 驗證
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.config import load_config
from app.db import get_session
from skills.backtest import run_backtest
from skills.build_features import PRUNED_FEATURE_COLS

# ──────────────────────────────────────────────────────────────
# Sub-strategy 設計
# ──────────────────────────────────────────────────────────────
# 共同 baseline：seasonal_filter, no stoploss, market filter tiers, liq weighting, pruned features
COMMON = dict(
    stoploss_pct=0.0,
    enable_seasonal_filter=True,
    market_filter_tiers=[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)],
    market_filter_min_positions=2,
    liquidity_weighting=True,
    feature_columns=PRUNED_FEATURE_COLS,
    label_horizon_buffer=20,
)

# 各 sub 差異維度
SUB_STRATEGIES = {
    "S1_production": {**COMMON, "topn": 20, "rebalance_freq": "M"},
    "S2_concentrated": {**COMMON, "topn": 10, "rebalance_freq": "M"},
    "S3_diversified": {**COMMON, "topn": 30, "rebalance_freq": "M"},
    "S4_quarterly": {**COMMON, "topn": 20, "rebalance_freq": "Q"},
    "S5_strict_liq": {**COMMON, "topn": 20, "rebalance_freq": "M", "min_avg_turnover": 2.0},
}


def run_sub(name: str, kwargs: dict, months: int) -> dict:
    """跑單個 sub-strategy 並回傳 result（含 equity_curve）。"""
    print(f"\n=== [{name}] starting ===")
    print(f"  config: {kwargs}")
    cfg = load_config()
    with get_session() as db:
        result = run_backtest(cfg, db, backtest_months=months, **kwargs)
    s = result.get("summary", {})
    sh = s.get("sharpe_ratio")
    mdd = s.get("max_drawdown")
    cum = compute_cum(result["periods"])
    print(f"  → Sharpe {sh:.4f}  MDD {mdd:+.4f}  cum {cum:+.2%}")
    return result


def compute_cum(periods: list) -> float:
    """從 periods 算累積報酬。"""
    cum = 1.0
    for p in periods:
        r = p.get("return")
        if r is not None:
            cum *= (1 + float(r))
    return cum - 1


def equity_curve_from_periods(periods: list) -> pd.Series:
    """把 periods 的 monthly return 轉成 cumulative equity series（index=rebalance_date）。"""
    if not periods:
        return pd.Series(dtype=float)
    rows = []
    cum = 1.0
    for p in periods:
        r = p.get("return")
        if r is None:
            continue
        cum *= (1 + float(r))
        rb = p.get("rebalance_date") or p.get("date")
        rows.append({"date": pd.to_datetime(rb), "equity": cum})
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df["equity"]


def compute_combined_metrics(equity_series: pd.Series, rf: float = 0.015) -> dict:
    """從 equity series 算 Sharpe / MDD / Calmar / annualized."""
    if equity_series.empty:
        return {"sharpe": np.nan, "mdd": np.nan, "calmar": np.nan, "annual": np.nan, "cum": 0.0}
    rets = equity_series.pct_change().dropna()
    if rets.empty:
        return {"sharpe": 0.0, "mdd": 0.0, "calmar": 0.0, "annual": 0.0, "cum": 0.0}
    # monthly Sharpe annualized × sqrt(12)
    excess = rets - (rf / 12.0)
    sharpe = float(excess.mean() / excess.std() * np.sqrt(12)) if excess.std() > 0 else 0.0
    cum = float(equity_series.iloc[-1] - 1)
    n_months = len(rets)
    annual = (1 + cum) ** (12.0 / n_months) - 1 if n_months > 0 else 0.0
    # MDD
    rolling_max = equity_series.cummax()
    dd = (equity_series / rolling_max - 1)
    mdd = float(dd.min())
    calmar = annual / abs(mdd) if mdd < 0 else 0.0
    return {"sharpe": sharpe, "mdd": mdd, "calmar": calmar, "annual": annual, "cum": cum}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=60)
    p.add_argument("--strategies", type=str, default=None,
                   help="comma-separated subset of strategy names, e.g. S1_production,S2_concentrated")
    p.add_argument("--out", type=str, default="artifacts/multi_strategy_ensemble.json")
    args = p.parse_args()

    if args.strategies:
        names = args.strategies.split(",")
        configs = {n: SUB_STRATEGIES[n] for n in names if n in SUB_STRATEGIES}
    else:
        configs = SUB_STRATEGIES

    print(f"\n{'='*70}")
    print(f"Stage 10.1：多策略組合 ensemble ({args.months}mo)")
    print(f"{'='*70}")
    print(f"  Sub-strategies: {list(configs.keys())}")

    # ── 跑每個 sub ──
    results = {}
    for name, kw in configs.items():
        results[name] = run_sub(name, kw, args.months)

    # ── 收集 equity_curves + 個別 metrics ──
    sub_metrics = {}
    eq_series = {}
    for name, r in results.items():
        eq = equity_curve_from_periods(r["periods"])
        eq_series[name] = eq
        sub_metrics[name] = compute_combined_metrics(eq)

    # ── 組合 (等權平均 equity curve) ──
    if eq_series:
        eq_df = pd.DataFrame(eq_series).sort_index()
        # forward-fill NaN（quarterly vs monthly index 對齊）
        eq_df = eq_df.fillna(method="ffill")
        # outer join 後第一格 NaN 用 1.0 填補
        eq_df = eq_df.fillna(1.0)
        combined_eq = eq_df.mean(axis=1)
    else:
        combined_eq = pd.Series(dtype=float)
    combined_metrics = compute_combined_metrics(combined_eq)

    # ── 輸出 ──
    print("\n" + "=" * 70)
    print(f"{'Strategy':<22} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'Annual':>8} {'Cum':>10}")
    print("-" * 80)
    for name in configs.keys():
        m = sub_metrics[name]
        print(f"{name:<22} {m['sharpe']:>+8.4f} {m['mdd']:>+8.4f} {m['calmar']:>+8.4f} "
              f"{m['annual']:>+8.2%} {m['cum']:>+10.2%}")
    print("-" * 80)
    m = combined_metrics
    print(f"{'COMBINED (1/N)':<22} {m['sharpe']:>+8.4f} {m['mdd']:>+8.4f} {m['calmar']:>+8.4f} "
          f"{m['annual']:>+8.2%} {m['cum']:>+10.2%}")

    # 判斷 lift
    best_sub_sharpe = max(s["sharpe"] for s in sub_metrics.values() if not np.isnan(s["sharpe"]))
    best_sub_mdd = max(s["mdd"] for s in sub_metrics.values() if not np.isnan(s["mdd"]))
    delta_sharpe = combined_metrics["sharpe"] - best_sub_sharpe
    delta_mdd = combined_metrics["mdd"] - best_sub_mdd
    print(f"\n  Δ Sharpe (vs best sub) = {delta_sharpe:+.4f}")
    print(f"  Δ MDD    (vs best sub) = {delta_mdd:+.4f} pp")
    diversification_ok = delta_sharpe > 0.05 and delta_mdd > -0.02
    if diversification_ok:
        print(f"  ✅ 組合 diversification 有效")
    else:
        print(f"  ⚠️ 組合無明顯 diversification lift")

    # 存 JSON
    out = {
        "months": args.months,
        "sub_metrics": sub_metrics,
        "combined": combined_metrics,
        "delta_sharpe_vs_best_sub": delta_sharpe,
        "delta_mdd_vs_best_sub": delta_mdd,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  存檔: {args.out}")


if __name__ == "__main__":
    main()
