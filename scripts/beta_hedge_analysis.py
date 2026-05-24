#!/usr/bin/env python
"""Stage 10.6: Beta-hedge 敏感度分析（純後處理，不需重跑 backtest）。

讀現有 10y backtest result，計算多個 hedge_ratio 下：
  hedged_return_t = portfolio_return_t - hedge_ratio * benchmark_return_t

對比：
  unhedged (h=0)  Sharpe, MDD, cum
  h=0.25
  h=0.50
  h=0.75
  h=1.00 (market-neutral)

判斷：
  若 h=0.5 或 h=1.0 顯著贏 unhedged → 「移除 beta」有效，值得實作期貨對沖
  若都退化 → 策略 alpha 已含 beta capture，hedge 純損失（confirm Stage 10.5 結論）

用法：
    python scripts/beta_hedge_analysis.py artifacts/optuna_10y/baseline.json
    python scripts/beta_hedge_analysis.py artifacts/stage10_10y/topn30.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compute_metrics(returns: list, rf_monthly: float = 0.015 / 12) -> dict:
    """從 monthly returns 算 cum / Sharpe / MDD / Calmar / annual."""
    arr = np.asarray([r for r in returns if r is not None], dtype=float)
    if len(arr) == 0:
        return {"sharpe": 0, "mdd": 0, "calmar": 0, "annual": 0, "cum": 0}
    cum = float(np.prod(1 + arr) - 1)
    n = len(arr)
    annual = (1 + cum) ** (12.0 / n) - 1
    excess = arr - rf_monthly
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(12)) if excess.std() > 0 else 0
    # MDD on equity curve
    eq = np.cumprod(1 + arr)
    rolling_max = np.maximum.accumulate(eq)
    dd = (eq / rolling_max - 1)
    mdd = float(dd.min())
    calmar = annual / abs(mdd) if mdd < 0 else 0
    return {"sharpe": sharpe, "mdd": mdd, "calmar": calmar, "annual": annual, "cum": cum}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backtest_json", type=str)
    p.add_argument("--hedge-ratios", type=str, default="0.0,0.25,0.50,0.75,1.0",
                   help="comma-separated hedge ratios")
    args = p.parse_args()

    result = json.load(open(args.backtest_json))
    periods = result["periods"]

    # 收集 (portfolio_return, benchmark_return) per period
    port_rets = []
    bench_rets = []
    for p_ in periods:
        r = p_.get("return")
        b = p_.get("benchmark_return")
        if r is not None and b is not None:
            port_rets.append(float(r))
            bench_rets.append(float(b))

    n = len(port_rets)
    print(f"\n{'='*70}")
    print(f"Stage 10.6: Beta-Hedge 敏感度分析")
    print(f"{'='*70}")
    print(f"  source: {args.backtest_json}")
    print(f"  periods: {n}")

    # 估 beta（OLS slope of port on bench）
    port_arr = np.asarray(port_rets)
    bench_arr = np.asarray(bench_rets)
    if bench_arr.std() > 0:
        beta = float(np.cov(port_arr, bench_arr)[0, 1] / bench_arr.var())
        corr = float(np.corrcoef(port_arr, bench_arr)[0, 1])
        print(f"  策略 vs 大盤 OLS beta = {beta:.4f}, corr = {corr:.4f}")
    else:
        beta = 1.0
        corr = 0.0

    hedge_ratios = [float(x) for x in args.hedge_ratios.split(",")]

    print(f"\n{'hedge':>7} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'Annual':>8} {'Cum':>10}  diff (vs h=0)")
    print("-" * 80)

    base = None
    for h in hedge_ratios:
        hedged = port_arr - h * bench_arr
        m = compute_metrics(list(hedged))
        if base is None:
            base = m
            marker = ""
        else:
            d_sh = m["sharpe"] - base["sharpe"]
            d_mdd = m["mdd"] - base["mdd"]
            if d_sh > 0.05 and d_mdd > -0.02:
                marker = "  ⭐ BETTER"
            elif d_sh < -0.10:
                marker = "  ❌"
            else:
                marker = ""
        print(f"{h:>7.2f} {m['sharpe']:>+8.4f} {m['mdd']:>+8.4f} {m['calmar']:>+8.4f} "
              f"{m['annual']:>+8.2%} {m['cum']:>+10.2%}{marker}")

    # 算 alpha-only Sharpe（beta-neutral hedge）
    print(f"\n  Beta-neutral hedge (h = OLS beta = {beta:.4f}):")
    alpha_only = port_arr - beta * bench_arr
    am = compute_metrics(list(alpha_only))
    print(f"  alpha-only: Sharpe={am['sharpe']:+.4f}  MDD={am['mdd']:+.4f}  cum={am['cum']:+.2%}")
    print(f"  ratio alpha/total Sharpe = {am['sharpe']/base['sharpe']:.2%}" if base['sharpe'] > 0 else "")


if __name__ == "__main__":
    main()
