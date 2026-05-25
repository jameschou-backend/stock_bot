#!/usr/bin/env python
"""Stage 11.2 sentiment sample IC eval（用 4200 sample news）。

對既有 sentiment data 計算 sentiment-based features 跟 forward_ret_h
的 cross-sectional IC，判斷是否值得 full $80 backfill。

Features:
  sentiment_avg_5d: 5 日平均 sentiment（-1~+1）
  sentiment_net_5d: 5 日 (pos_count - neg_count) / total_count
  sentiment_strength_5d: 5 日平均 |sentiment|
  sentiment_pos_count_5d: 5 日 sentiment=+1 篇數
  sentiment_neg_count_5d: 5 日 sentiment=-1 篇數
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

from app.db import get_session


def load_sentiment_with_labels():
    with get_session() as s:
        rows = s.execute(text("""
            SELECT n.stock_id, n.news_datetime, ns.sentiment_score, ns.confidence,
                   l.future_ret_h
            FROM news_sentiment ns
            JOIN raw_stock_news n ON n.id = ns.news_id
            LEFT JOIN labels l ON l.stock_id = n.stock_id
                AND l.trading_date = DATE(n.news_datetime)
            WHERE ns.sentiment_score IS NOT NULL
        """)).fetchall()
    df = pd.DataFrame(rows, columns=[
        "stock_id", "news_datetime", "sentiment", "confidence", "future_ret_h"
    ])
    df["news_date"] = pd.to_datetime(df["news_datetime"]).dt.date
    df["future_ret_h"] = pd.to_numeric(df["future_ret_h"], errors="coerce")
    return df


def compute_per_stock_daily(df):
    """per stock × per day aggregate sentiment features."""
    g = df.groupby(["stock_id", "news_date"]).agg(
        sentiment_sum=("sentiment", "sum"),
        sentiment_count=("sentiment", "count"),
        pos_count=("sentiment", lambda x: (x == 1).sum()),
        neg_count=("sentiment", lambda x: (x == -1).sum()),
        avg_confidence=("confidence", "mean"),
    ).reset_index()
    g["sentiment_avg"] = g["sentiment_sum"] / g["sentiment_count"]
    return g


def rolling_5d_features(daily_df):
    """per stock 5 日 rolling."""
    out = []
    for sid, g in daily_df.groupby("stock_id"):
        g = g.set_index("news_date").sort_index()
        g["sentiment_avg_5d"] = g["sentiment_avg"].rolling(5, min_periods=1).mean()
        g["sentiment_net_5d"] = (g["pos_count"].rolling(5, min_periods=1).sum() -
                                 g["neg_count"].rolling(5, min_periods=1).sum()) / \
                                g["sentiment_count"].rolling(5, min_periods=1).sum().replace(0, np.nan)
        g["sentiment_pos_count_5d"] = g["pos_count"].rolling(5, min_periods=1).sum()
        g["sentiment_neg_count_5d"] = g["neg_count"].rolling(5, min_periods=1).sum()
        g["sentiment_strength_5d"] = g["sentiment_avg"].abs().rolling(5, min_periods=1).mean()
        g["stock_id"] = sid
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def cross_section_ic(feat_df, label_df, feature_col):
    sub = feat_df[["stock_id", "news_date", feature_col]].rename(
        columns={"news_date": "trading_date"}).copy()
    sub[feature_col] = pd.to_numeric(sub[feature_col], errors="coerce")
    sub["trading_date"] = pd.to_datetime(sub["trading_date"]).dt.date
    label_df = label_df.copy()
    label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
    merged = sub.merge(label_df, on=["stock_id", "trading_date"], how="inner").dropna()
    ic_by_date = {}
    for td, grp in merged.groupby("trading_date"):
        if len(grp) < 5:  # sample 小，降低門檻
            continue
        rho, _ = stats.spearmanr(grp[feature_col], grp["future_ret_h"])
        if not np.isnan(rho):
            ic_by_date[td] = rho
    if not ic_by_date:
        return {"feature": feature_col, "n_dates": 0,
                "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "pos_rate": np.nan}
    s = pd.Series(ic_by_date)
    return {
        "feature": feature_col, "n_dates": int(len(s)),
        "ic_mean": float(s.mean()), "ic_std": float(s.std()),
        "icir": float(s.mean() / s.std()) if s.std() > 0 else 0.0,
        "pos_rate": float((s > 0).mean()),
    }


def main():
    print("\n=== Sentiment Sample IC Eval (~4200 news) ===\n")

    print("[1/3] Loading sentiment + labels...")
    df = load_sentiment_with_labels()
    print(f"  total rows: {len(df):,}, unique stocks: {df['stock_id'].nunique()}")
    print(f"  date range: {df['news_date'].min()} → {df['news_date'].max()}")

    print("\n[2/3] Per-stock daily aggregate + 5d rolling...")
    daily = compute_per_stock_daily(df)
    feat_df = rolling_5d_features(daily)
    print(f"  features rows: {len(feat_df):,}")

    print("\n[3/3] Cross-section IC ...")
    # 用 raw labels (不 filter by sentiment join, 完整 forward_ret_h)
    from sqlalchemy import text
    with get_session() as s:
        min_d = feat_df["news_date"].min()
        max_d = feat_df["news_date"].max()
        rows = s.execute(text(f"""
            SELECT stock_id, trading_date, future_ret_h
            FROM labels
            WHERE trading_date BETWEEN '{min_d}' AND '{max_d}'
        """)).fetchall()
    label_df = pd.DataFrame(rows, columns=["stock_id", "trading_date", "future_ret_h"])
    label_df["future_ret_h"] = pd.to_numeric(label_df["future_ret_h"], errors="coerce")
    print(f"  labels: {len(label_df):,}")

    features = ["sentiment_avg_5d", "sentiment_net_5d",
                "sentiment_pos_count_5d", "sentiment_neg_count_5d",
                "sentiment_strength_5d"]
    print(f"\n{'feature':<28} {'n_dt':>6} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>8} {'pos%':>7}")
    print("-" * 80)
    pos_results = []
    for f in features:
        r = cross_section_ic(feat_df, label_df, f)
        marker = " ⭐" if abs(r["icir"]) >= 0.30 else ""
        print(f"{f:<28} {r['n_dates']:>6d} {r['ic_mean']:>+10.4f} "
              f"{r['ic_std']:>10.4f} {r['icir']:>+8.4f} {r['pos_rate']:>7.1%}{marker}")
        if abs(r["icir"]) >= 0.30:
            pos_results.append(r)

    print()
    if pos_results:
        print(f"✅ {len(pos_results)} sentiment features ICIR ≥ 0.30")
        print("   → 值得 full $80 backfill")
    else:
        print(f"⚠️ 沒 sentiment feature ICIR ≥ 0.30 (sample 太小或訊號弱)")
        print(f"   sample 4200 vs full 3.4M = 0.12%，結果不一定 representative")


if __name__ == "__main__":
    main()
