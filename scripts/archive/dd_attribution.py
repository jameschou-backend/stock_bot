#!/usr/bin/env python
"""Stage 10.3: Drawdown 解構分析。

讀 backtest result，找出 MDD 期間（peak→trough），拆解：
  1. 期間多長、哪個月
  2. stock-level loss contribution（哪幾檔造成）
  3. industry distribution（是否集中某產業）
  4. 大盤狀態（同期間大盤跌多少）
  5. 是否有共同特徵可作為未來 trigger 規避

用法：
    python scripts/dd_attribution.py artifacts/stage10_10y/topn30.json
    python scripts/dd_attribution.py artifacts/stage10_10y/topn30.json --top-losers 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.db import get_session
from app.models import Stock


def load_industries() -> dict:
    """從 DB 載入 stock_id → industry mapping。"""
    with get_session() as s:
        rows = s.query(Stock.stock_id, Stock.industry_category).all()
    return {str(sid): (ind or "未分類") for sid, ind in rows}


def compute_equity_curve(periods: list) -> list:
    """從 periods 算 cumulative equity series。"""
    cum = 1.0
    curve = []
    for p in periods:
        r = p.get("return")
        if r is None:
            continue
        cum *= (1 + float(r))
        curve.append({
            "date": p.get("rebalance_date") or p.get("date"),
            "equity": cum,
            "monthly_return": r,
            "benchmark_return": p.get("benchmark_return"),
            "stock_returns": p.get("stock_returns", {}),
            "wins": p.get("wins"),
            "losses": p.get("losses"),
        })
    return curve


def find_mdd_period(curve: list) -> dict:
    """找 MDD：peak 到 trough。"""
    peak = curve[0]["equity"]
    peak_idx = 0
    trough = peak
    trough_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, c in enumerate(curve):
        eq = c["equity"]
        if eq > peak:
            peak = eq
            peak_idx = i
        dd = eq / peak - 1
        if dd < max_dd:
            max_dd = dd
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i

    return {
        "mdd_pct": max_dd,
        "peak_idx": max_dd_peak_idx,
        "trough_idx": max_dd_trough_idx,
        "peak_date": curve[max_dd_peak_idx]["date"],
        "trough_date": curve[max_dd_trough_idx]["date"],
        "duration_months": max_dd_trough_idx - max_dd_peak_idx,
    }


def attribute_losses(curve: list, mdd_info: dict, industries: dict, top_n: int = 10) -> dict:
    """對 MDD 期間每月找 top losers + industry distribution。"""
    peak_i = mdd_info["peak_idx"]
    trough_i = mdd_info["trough_idx"]

    # 整個 MDD 期間的 stock total loss
    stock_total_pnl = defaultdict(float)  # 累積 P&L 貢獻
    stock_appearance = Counter()  # 在 MDD 期間出現次數
    monthly_summary = []

    for i in range(peak_i, trough_i + 1):
        period = curve[i]
        srs = period["stock_returns"]
        # 找該月最大 N 個 losers
        losers = sorted(srs.items(), key=lambda x: x[1])[:top_n]
        monthly_summary.append({
            "date": period["date"],
            "monthly_return": period["monthly_return"],
            "benchmark_return": period["benchmark_return"],
            "wins": period["wins"],
            "losses": period["losses"],
            "top_losers": [(sid, ret, industries.get(sid, "未分類")) for sid, ret in losers[:5]],
        })
        # 累積到 stock-level totals（按 1/N 等權，沒考慮市值）
        n_picks = len(srs) if srs else 1
        for sid, ret in srs.items():
            contribution = ret / n_picks  # 等權下，每檔對組合的貢獻
            stock_total_pnl[sid] += contribution
            stock_appearance[sid] += 1

    # 找 MDD 期間累積 PnL 最差的 N 檔
    worst_stocks = sorted(stock_total_pnl.items(), key=lambda x: x[1])[:top_n]
    worst_industries = Counter()
    for sid, _ in worst_stocks:
        worst_industries[industries.get(sid, "未分類")] += 1

    return {
        "monthly_summary": monthly_summary,
        "worst_stocks_overall": [
            {"stock_id": sid, "cum_contribution": pnl,
             "appearances": stock_appearance[sid],
             "industry": industries.get(sid, "未分類")}
            for sid, pnl in worst_stocks
        ],
        "worst_industries": dict(worst_industries.most_common(10)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backtest_json", type=str, help="path to backtest result JSON")
    p.add_argument("--top-losers", type=int, default=10)
    args = p.parse_args()

    with open(args.backtest_json) as f:
        result = json.load(f)

    print(f"\n{'='*70}")
    print(f"Stage 10.3：Drawdown Attribution Analysis")
    print(f"{'='*70}")
    print(f"source: {args.backtest_json}")

    summary = result.get("summary", {})
    print(f"\n[backtest summary]")
    print(f"  Sharpe:  {summary.get('sharpe_ratio')}")
    print(f"  MDD:     {summary.get('max_drawdown')}")
    print(f"  cum:     計算中...")

    print(f"\n[1/3] Building equity curve...")
    curve = compute_equity_curve(result["periods"])
    print(f"  total periods: {len(curve)}")
    cum = curve[-1]["equity"] - 1 if curve else 0
    print(f"  final cum:     {cum:+.2%}")

    print(f"\n[2/3] Finding MDD period...")
    mdd = find_mdd_period(curve)
    print(f"  MDD:           {mdd['mdd_pct']:+.4f} ({mdd['mdd_pct']:.2%})")
    print(f"  peak:          {mdd['peak_date']}  (idx {mdd['peak_idx']})")
    print(f"  trough:        {mdd['trough_date']}  (idx {mdd['trough_idx']})")
    print(f"  duration:      {mdd['duration_months']} months")

    print(f"\n[3/3] Loading industry mapping + computing attribution...")
    industries = load_industries()
    print(f"  loaded {len(industries)} stock→industry mappings")

    attr = attribute_losses(curve, mdd, industries, top_n=args.top_losers)

    print(f"\n{'='*70}")
    print(f"MDD 期間 monthly breakdown ({mdd['peak_date']} ~ {mdd['trough_date']})")
    print(f"{'='*70}")
    print(f"{'date':<15} {'port_ret':>10} {'bench':>10} {'wins':>5} {'losses':>7}  worst 3 picks")
    print("-" * 110)
    for m in attr["monthly_summary"]:
        worst3 = " | ".join(f"{sid}({ret:+.1%},{ind[:6]})"
                            for sid, ret, ind in m["top_losers"][:3])
        print(f"{str(m['date']):<15} {m['monthly_return']:>+10.2%} {m['benchmark_return']:>+10.2%} "
              f"{m['wins']:>5} {m['losses']:>7}  {worst3}")

    print(f"\n{'='*70}")
    print(f"MDD 期間累積貢獻最差 {args.top_losers} 檔")
    print(f"{'='*70}")
    print(f"{'stock_id':<10} {'industry':<24} {'cum_contrib':>12} {'appearances':>12}")
    print("-" * 70)
    for w in attr["worst_stocks_overall"]:
        print(f"{w['stock_id']:<10} {w['industry']:<24} {w['cum_contribution']:>+12.4%} "
              f"{w['appearances']:>12d}")

    print(f"\n{'='*70}")
    print(f"產業分布（worst {args.top_losers} 檔）")
    print(f"{'='*70}")
    for ind, n in attr["worst_industries"].items():
        print(f"  {ind:<28} {n} 檔")

    # 額外：MDD 期間大盤狀況
    bm_returns = [m["benchmark_return"] for m in attr["monthly_summary"] if m["benchmark_return"] is not None]
    if bm_returns:
        bm_cum = 1.0
        for r in bm_returns:
            bm_cum *= (1 + r)
        bm_cum -= 1
        port_cum = 1.0
        for m in attr["monthly_summary"]:
            port_cum *= (1 + m["monthly_return"])
        port_cum -= 1
        print(f"\n[MDD 期間大盤對照]")
        print(f"  策略累積: {port_cum:+.2%}")
        print(f"  大盤累積: {bm_cum:+.2%}")
        print(f"  超額:    {port_cum - bm_cum:+.2%}")

    # 存 JSON
    out_path = Path(args.backtest_json).with_name(
        Path(args.backtest_json).stem + "_dd_attribution.json")
    with open(out_path, "w") as f:
        json.dump({"mdd": mdd, "attribution": attr}, f, indent=2, default=str)
    print(f"\n  存檔: {out_path}")


if __name__ == "__main__":
    main()
