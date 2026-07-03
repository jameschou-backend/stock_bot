#!/usr/bin/env python3
"""Oracle 分析：找出每筆交易的最佳出場時機，分析損失的原因與出場訊號規律。

Usage:
    python scripts/oracle_analysis.py
    python scripts/oracle_analysis.py --input artifacts/backtest/rotation_c2.json
    python scripts/oracle_analysis.py --max-hold 30 --output artifacts/oracle_c2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import get_session
from skills import data_store


def run_oracle_analysis(
    trades: list,
    price_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    max_hold_days: int = 30,
) -> dict:
    """
    對每筆交易找出 max_hold_days 內的最高收盤價（Oracle 出場），
    計算 Oracle 報酬 vs 實際報酬，並分析 Oracle 出場日的特徵分佈。
    """
    price_df = price_df.copy()
    price_df["trading_date"] = pd.to_datetime(price_df["trading_date"]).dt.date
    price_df["stock_id"] = price_df["stock_id"].astype(str)
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")

    feat_df = feat_df.copy()
    feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
    feat_df["stock_id"] = feat_df["stock_id"].astype(str)

    # 建立 price lookup: stock_id -> sorted list of (date, close)
    price_by_stock: dict[str, pd.DataFrame] = {}
    for sid, grp in price_df.groupby("stock_id"):
        price_by_stock[sid] = grp.sort_values("trading_date")[["trading_date", "close"]].reset_index(drop=True)

    feat_by_stock: dict[str, pd.DataFrame] = {}
    for sid, grp in feat_df.groupby("stock_id"):
        feat_by_stock[sid] = grp.sort_values("trading_date").reset_index(drop=True)

    results = []
    feature_cols = [c for c in feat_df.columns if c not in {"stock_id", "trading_date", "future_ret_h"}]

    for trade in trades:
        sid = str(trade["stock_id"])
        entry_date = date.fromisoformat(trade["entry_date"])
        entry_price = float(trade["entry_price"])
        actual_exit_date = date.fromisoformat(trade["exit_date"])
        actual_ret = float(trade["realized_pnl_pct"])
        days_held = int(trade["days_held"])
        exit_reason = trade["exit_reason"]

        sp = price_by_stock.get(sid)
        if sp is None or entry_price <= 0:
            continue

        # 找 entry_date 之後 max_hold_days 個交易日的價格
        forward = sp[sp["trading_date"] > entry_date].head(max_hold_days)
        if forward.empty:
            continue

        # Oracle：最高收盤日
        peak_idx = forward["close"].idxmax()
        oracle_date = forward.loc[peak_idx, "trading_date"]
        oracle_price = float(forward.loc[peak_idx, "close"])
        oracle_ret = oracle_price / entry_price - 1

        # 實際 exit 在 forward window 的第幾天
        actual_day_in_window = (forward["trading_date"] <= actual_exit_date).sum()

        # Oracle 出場日的特徵
        sf = feat_by_stock.get(sid)
        oracle_feat = {}
        if sf is not None:
            row = sf[sf["trading_date"] == oracle_date]
            if not row.empty:
                for col in feature_cols:
                    if col in row.columns:
                        v = row[col].iloc[0]
                        oracle_feat[col] = float(v) if pd.notna(v) else None

        results.append({
            "stock_id": sid,
            "entry_date": str(entry_date),
            "entry_price": entry_price,
            "actual_exit_date": str(actual_exit_date),
            "actual_ret": actual_ret,
            "actual_days_held": days_held,
            "exit_reason": exit_reason,
            "oracle_date": str(oracle_date),
            "oracle_price": oracle_price,
            "oracle_ret": oracle_ret,
            "oracle_day_in_window": int((forward["trading_date"] <= oracle_date).sum()),
            "left_on_table": oracle_ret - actual_ret,
            "oracle_feat": oracle_feat,
        })

    if not results:
        return {"error": "no results"}

    df = pd.DataFrame(results)

    # ── 整體統計 ──
    avg_actual = df["actual_ret"].mean()
    avg_oracle = df["oracle_ret"].mean()
    avg_left   = df["left_on_table"].mean()
    median_oracle_day = df["oracle_day_in_window"].median()

    # ── Oracle 出場日分佈 ──
    bins = [0, 3, 5, 10, 15, 20, 25, 30]
    labels = ["1-3", "4-5", "6-10", "11-15", "16-20", "21-25", "26-30"]
    day_dist = pd.cut(df["oracle_day_in_window"], bins=bins, labels=labels)
    day_dist_pct = (day_dist.value_counts(normalize=True).sort_index() * 100).round(1).to_dict()

    # ── 依 exit_reason 分組的 oracle 損失 ──
    reason_stats = {}
    for reason, grp in df.groupby("exit_reason"):
        reason_stats[reason] = {
            "count": len(grp),
            "avg_actual_ret": round(grp["actual_ret"].mean() * 100, 2),
            "avg_oracle_ret": round(grp["oracle_ret"].mean() * 100, 2),
            "avg_left_on_table": round(grp["left_on_table"].mean() * 100, 2),
        }

    # ── Oracle 出場日特徵分佈（中位數）──
    oracle_feat_rows = [r["oracle_feat"] for r in results if r["oracle_feat"]]
    oracle_feat_df = pd.DataFrame(oracle_feat_rows)

    key_feats = ["rsi_14", "bias_20", "ma_alignment", "foreign_buy_streak_5",
                 "foreign_buy_consecutive_days", "vol_ratio_20", "foreign_net_5",
                 "foreign_net_20", "kd_k", "kd_d", "boll_pct", "ret_5", "ret_20"]
    feat_summary = {}
    for col in key_feats:
        if col in oracle_feat_df.columns:
            vals = oracle_feat_df[col].dropna()
            if len(vals) > 0:
                feat_summary[col] = {
                    "median": round(float(vals.median()), 4),
                    "p25":    round(float(vals.quantile(0.25)), 4),
                    "p75":    round(float(vals.quantile(0.75)), 4),
                    "mean":   round(float(vals.mean()), 4),
                }

    # ── 分析：提早出場 vs 太晚出場 ──
    df["exited_before_oracle"] = df["actual_exit_date"] < df["oracle_date"]
    early_exit_pct = df["exited_before_oracle"].mean() * 100
    # 提早出場組的損失
    early_df = df[df["exited_before_oracle"]]
    late_df  = df[~df["exited_before_oracle"]]

    # ── 打印報告 ──
    print("\n" + "=" * 65)
    print("Oracle 分析報告（最佳出場時機）")
    print("=" * 65)
    print(f"  分析筆數：{len(df)} 筆")
    print(f"\n  ── 報酬對比 ──")
    print(f"  平均實際報酬：   {avg_actual*100:+.2f}%")
    print(f"  平均 Oracle 報酬：{avg_oracle*100:+.2f}%")
    print(f"  平均未吃到報酬：  {avg_left*100:+.2f}pp")
    print(f"\n  ── 出場時機 ──")
    print(f"  提早出場（未到峰值即出）：{early_exit_pct:.1f}%")
    print(f"  Oracle 出場日中位數：第 {median_oracle_day:.0f} 天")
    print(f"\n  Oracle 出場日分佈（交易日）：")
    for k, v in day_dist_pct.items():
        bar = "█" * int(v / 2)
        print(f"    Day {k:>5}：{v:5.1f}%  {bar}")

    print(f"\n  ── 依出場原因的 Oracle 損失 ──")
    print(f"  {'原因':<20} {'筆數':>6} {'實際%':>8} {'Oracle%':>8} {'損失pp':>8}")
    print(f"  {'-'*55}")
    for reason, s in sorted(reason_stats.items(), key=lambda x: -x[1]["avg_left_on_table"]):
        print(f"  {reason:<20} {s['count']:>6} {s['avg_actual_ret']:>8.2f} {s['avg_oracle_ret']:>8.2f} {s['avg_left_on_table']:>8.2f}")

    print(f"\n  ── Oracle 出場日特徵中位數 ──")
    print(f"  （提供出場訊號設計的事實依據）")
    for col, s in feat_summary.items():
        print(f"  {col:<35} 中位數={s['median']:>8.3f}  [P25={s['p25']:.3f}, P75={s['p75']:.3f}]")

    print(f"\n  ── 提早出場 vs 持到峰值後才出 ──")
    if len(early_df) > 0:
        print(f"  提早出場組（{len(early_df)}筆）：avg 實際 {early_df['actual_ret'].mean()*100:+.2f}%，"
              f"avg Oracle {early_df['oracle_ret'].mean()*100:+.2f}%，"
              f"損失 {early_df['left_on_table'].mean()*100:+.2f}pp")
    if len(late_df) > 0:
        print(f"  持到峰值後才出（{len(late_df)}筆）：avg 實際 {late_df['actual_ret'].mean()*100:+.2f}%，"
              f"avg Oracle {late_df['oracle_ret'].mean()*100:+.2f}%，"
              f"損失 {late_df['left_on_table'].mean()*100:+.2f}pp")
    print("=" * 65)

    return {
        "n_trades": len(df),
        "avg_actual_ret": round(avg_actual * 100, 2),
        "avg_oracle_ret": round(avg_oracle * 100, 2),
        "avg_left_on_table": round(avg_left * 100, 2),
        "early_exit_pct": round(early_exit_pct, 1),
        "oracle_day_median": float(median_oracle_day),
        "oracle_day_distribution": day_dist_pct,
        "by_exit_reason": reason_stats,
        "oracle_exit_features": feat_summary,
        "trades": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle 最佳出場分析")
    parser.add_argument("--input", type=str,
                        default="artifacts/backtest/rotation_c2.json")
    parser.add_argument("--max-hold", type=int, default=30)
    parser.add_argument("--output", type=str, default="artifacts/oracle_c2.json")
    args = parser.parse_args()

    # 載入回測結果
    with open(ROOT / args.input, encoding="utf-8") as f:
        bt = json.load(f)
    trades = bt["trades_log"]
    print(f"載入 {len(trades)} 筆交易：{args.input}")

    # 載入資料
    config = load_config()
    with get_session() as session:
        from sqlalchemy import func
        from app.models import Feature

        min_d = min(date.fromisoformat(t["entry_date"]) for t in trades) - timedelta(days=60)
        max_d = max(date.fromisoformat(t["exit_date"]) for t in trades) + timedelta(days=35)

        print(f"載入價格與特徵資料：{min_d} ~ {max_d}")
        price_df = data_store.get_prices(session, min_d, max_d)
        feat_df  = data_store.get_features(session, min_d, max_d)

    result = run_oracle_analysis(trades, price_df, feat_df, max_hold_days=args.max_hold)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
