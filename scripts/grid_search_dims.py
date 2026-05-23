#!/usr/bin/env python
"""Stage 10.2: 三維 OAT grid search（topn × min_positions × retrain_freq）。

基於 Stage 10.1 的發現（topn 20→30 +Sharpe 0.18），系統性探索其他預設值是否
也藏 alpha。用 One-at-a-Time (OAT) 法避免 N×M×K combinatorial explosion。

設計：
  baseline = topn=30, min_pos=2, freq=3（today's new production）
  變動 topn:    25, 35, 40, 50         (4 個)
  變動 min_pos: 3, 5                    (2 個)
  變動 freq:    1, 6                    (2 個)
  共 9 個 sub-strategy

每 sub 跑 60mo 取 Sharpe / MDD / Calmar，挑 top-2 候選跑 10y 驗證。
全部 ~36 min serial（複用 multi_strategy_ensemble 架構）。

用法：
    python scripts/grid_search_dims.py --months 60
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

from app.config import load_config
from app.db import get_session
from skills.backtest import run_backtest
from skills.build_features import PRUNED_FEATURE_COLS

# 共同 baseline (與 production 一致)
COMMON = dict(
    stoploss_pct=0.0,
    enable_seasonal_filter=True,
    market_filter_tiers=[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)],
    liquidity_weighting=True,
    feature_columns=PRUNED_FEATURE_COLS,
    label_horizon_buffer=20,
    rebalance_freq="M",
)

# OAT 設計：baseline + 各維度單獨變動
GRID = {
    # ── topn 維度 ──
    "baseline_v2":  dict(**COMMON, topn=30, market_filter_min_positions=2, retrain_freq_months=3),
    "topn_25":      dict(**COMMON, topn=25, market_filter_min_positions=2, retrain_freq_months=3),
    "topn_35":      dict(**COMMON, topn=35, market_filter_min_positions=2, retrain_freq_months=3),
    "topn_40":      dict(**COMMON, topn=40, market_filter_min_positions=2, retrain_freq_months=3),
    "topn_50":      dict(**COMMON, topn=50, market_filter_min_positions=2, retrain_freq_months=3),
    # ── min_positions 維度 ──
    "min_pos_3":    dict(**COMMON, topn=30, market_filter_min_positions=3, retrain_freq_months=3),
    "min_pos_5":    dict(**COMMON, topn=30, market_filter_min_positions=5, retrain_freq_months=3),
    # ── retrain_freq 維度 ──
    "freq_1mo":     dict(**COMMON, topn=30, market_filter_min_positions=2, retrain_freq_months=1),
    "freq_6mo":     dict(**COMMON, topn=30, market_filter_min_positions=2, retrain_freq_months=6),
}


def compute_cum(periods):
    cum = 1.0
    for p in periods:
        r = p.get("return")
        if r is not None:
            cum *= (1 + float(r))
    return cum - 1


def run_sub(name: str, kwargs: dict, months: int) -> dict:
    print(f"\n=== [{name}] starting ===")
    diff_keys = ("topn", "market_filter_min_positions", "retrain_freq_months")
    print(f"  config diff: {dict((k, kwargs[k]) for k in diff_keys if k in kwargs)}")
    cfg = load_config()
    with get_session() as db:
        result = run_backtest(cfg, db, backtest_months=months, **kwargs)
    s = result.get("summary") or {}
    sh = s.get("sharpe_ratio") or 0.0
    mdd = s.get("max_drawdown") or 0.0
    cal = s.get("calmar_ratio") or 0.0
    cum = compute_cum(result["periods"])
    print(f"  → Sharpe {sh:+.4f}  MDD {mdd:+.4f}  Calmar {cal:+.4f}  cum {cum:+.2%}")
    return {"sharpe": sh, "mdd": mdd, "calmar": cal, "cum": cum, "config": kwargs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=60)
    p.add_argument("--out", type=str, default="artifacts/grid_search_dims.json")
    args = p.parse_args()

    print(f"\n{'='*70}")
    print(f"Stage 10.2：OAT grid search ({args.months}mo, {len(GRID)} sub)")
    print(f"{'='*70}")

    results = {}
    for name, kw in GRID.items():
        results[name] = run_sub(name, kw, args.months)

    # ── 報告 ──
    print("\n" + "=" * 80)
    print(f"{'Strategy':<18} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'Cum':>10}  diff from baseline")
    print("-" * 80)
    base = results["baseline_v2"]
    for name in GRID.keys():
        r = results[name]
        d_sh = r["sharpe"] - base["sharpe"]
        d_mdd = r["mdd"] - base["mdd"]
        marker = ""
        if name != "baseline_v2":
            if d_sh > 0.05 and d_mdd > -0.02:
                marker = "  ⭐ BETTER"
            elif d_sh < -0.05 and d_mdd < 0:
                marker = "  ❌"
            elif d_sh < -0.10:
                marker = "  ⚠️"
        print(f"{name:<18} {r['sharpe']:>+8.4f} {r['mdd']:>+8.4f} {r['calmar']:>+8.4f} {r['cum']:>+10.2%}  "
              f"ΔSharpe={d_sh:+.4f}  ΔMDD={d_mdd:+.4f}{marker}")

    # ── 挑 top-2 candidates by Sharpe ──
    sorted_by_sh = sorted(
        [(n, r) for n, r in results.items() if n != "baseline_v2"],
        key=lambda x: -x[1]["sharpe"],
    )
    print("\n  Top-2 candidates by Sharpe:")
    for name, r in sorted_by_sh[:2]:
        d_sh = r["sharpe"] - base["sharpe"]
        d_mdd = r["mdd"] - base["mdd"]
        print(f"    {name:<18}  ΔSharpe={d_sh:+.4f}  ΔMDD={d_mdd:+.4f}")
    print("\n  →下一步：對 top-2 candidates 跑 10y 驗證")

    # 存 JSON
    serializable = {n: {k: v for k, v in r.items() if k != "config"} for n, r in results.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"months": args.months, "results": serializable, "baseline": "baseline_v2"},
                  f, indent=2, default=str)
    print(f"\n  存檔: {args.out}")


if __name__ == "__main__":
    main()
