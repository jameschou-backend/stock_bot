#!/usr/bin/env python
"""News features IC 評估（Stage 11.1）。

對 raw_stock_news 90 天資料計算 4 個 attention features 並跑 cross-sectional
Spearman IC vs future_ret_h（20 日 forward return）。

判定門檻：|ICIR| ≥ 0.30 + coverage ≥ 50% + n_dates ≥ 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select

from app.db import get_session
from app.models import Label, RawStockNews
from skills.feature_utils import align_news_to_trading_day


def load_news(start_date, end_date):
    from datetime import datetime as _dt
    with get_session() as s:
        rows = s.execute(
            select(
                RawStockNews.stock_id,
                RawStockNews.news_datetime,
                RawStockNews.source,
            )
            .where(RawStockNews.news_datetime >= _dt.combine(start_date, _dt.min.time()))
            .where(RawStockNews.news_datetime <= _dt.combine(end_date, _dt.max.time()))
        ).all()
    return pd.DataFrame(rows, columns=["stock_id", "news_datetime", "source"])


def load_labels(start_date, end_date):
    with get_session() as s:
        rows = s.execute(
            select(Label.stock_id, Label.trading_date, Label.future_ret_h)
            .where(Label.trading_date.between(start_date, end_date))
        ).all()
    df = pd.DataFrame(rows, columns=["stock_id", "trading_date", "future_ret_h"])
    df["future_ret_h"] = pd.to_numeric(df["future_ret_h"], errors="coerce")
    return df


def compute_news_features(news_df: pd.DataFrame, trading_days=None) -> pd.DataFrame:
    """per-stock per-day → 4 features。

    trading_days 提供時，新聞以 align_news_to_trading_day 對齊到「次一可交易日」
    （消除盤後 >=13:30 / 週末新聞的同日 lookahead，與 build_features 一致）；
    否則 fallback 用日曆日（僅向後相容）。
    """
    news_df = news_df.copy()
    news_df["stock_id"] = news_df["stock_id"].astype(str)
    if trading_days is not None and len(news_df) > 0:
        news_df["date"] = align_news_to_trading_day(news_df["news_datetime"], trading_days).values
        news_df = news_df.dropna(subset=["date"])
    else:
        news_df["date"] = pd.to_datetime(news_df["news_datetime"]).dt.date

    daily = news_df.groupby(["stock_id", "date"]).agg(
        news_count=("news_datetime", "count"),
        source_diversity=("source", "nunique"),
    ).reset_index()

    out_parts = []
    for sid, g in daily.groupby("stock_id"):
        g = g.set_index("date").sort_index()
        g["news_count_5d"] = g["news_count"].rolling(5, min_periods=1).sum()
        g["news_count_20d_avg"] = g["news_count"].rolling(20, min_periods=5).mean()
        g["news_count_change_5d"] = g["news_count_5d"] / (g["news_count_20d_avg"] * 5).replace(0, np.nan)
        g["news_source_diversity_5d"] = g["source_diversity"].rolling(5, min_periods=1).sum()
        g["news_count_change_1d"] = g["news_count"] / g["news_count_5d"].replace(0, np.nan) * 5
        g["stock_id"] = sid
        g = g.reset_index().rename(columns={"date": "trading_date"})
        out_parts.append(g[["stock_id", "trading_date", "news_count_5d",
                            "news_count_change_5d", "news_source_diversity_5d",
                            "news_count_change_1d"]])
    if not out_parts:
        return pd.DataFrame()
    return pd.concat(out_parts, ignore_index=True)


def cross_section_ic(feat_df, label_df, feature_col):
    sub = feat_df[["stock_id", "trading_date", feature_col]].copy()
    sub[feature_col] = pd.to_numeric(sub[feature_col], errors="coerce")
    n_total = len(sub)
    n_valid = sub[feature_col].notna().sum()
    cov = n_valid / n_total if n_total > 0 else 0.0

    # 注意：news date 可能是日曆日，label trading_date 是交易日
    # join 時保留 inner（只取有 label 的日子）
    sub["trading_date"] = pd.to_datetime(sub["trading_date"]).dt.date
    label_df = label_df.copy()
    label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
    merged = sub.merge(label_df, on=["stock_id", "trading_date"], how="inner").dropna()
    ic_by_date = {}
    for td, grp in merged.groupby("trading_date"):
        if len(grp) < 10:
            continue
        rho, _ = stats.spearmanr(grp[feature_col], grp["future_ret_h"])
        if not np.isnan(rho):
            ic_by_date[td] = rho
    if not ic_by_date:
        return {"feature": feature_col, "coverage": cov, "n_dates": 0,
                "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "pos_rate": np.nan}
    s = pd.Series(ic_by_date)
    return {
        "feature": feature_col, "coverage": float(cov), "n_dates": int(len(s)),
        "ic_mean": float(s.mean()), "ic_std": float(s.std()),
        "icir": float(s.mean() / s.std()) if s.std() > 0 else 0.0,
        "pos_rate": float((s > 0).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=3,
                   help="evaluation window (months)")
    args = p.parse_args()

    end = date.today()
    start = end - timedelta(days=args.months * 30 + 30)  # 多 30 天 lookback

    print(f"\n{'='*70}")
    print(f"Stage 11.1：News features IC 評估")
    print(f"{'='*70}")
    print(f"期間：{start} ~ {end}")

    print("\n[1/3] 載入 news ...")
    news_df = load_news(start, end)
    print(f"  總筆數: {len(news_df):,}, stocks: {news_df['stock_id'].nunique()}")

    print("\n[2/3] 載入 labels ...")
    label_df = load_labels(start, end)
    print(f"  總筆數: {len(label_df):,}")

    print("\n[3/3] 計算 features + IC ...")
    feat_df = compute_news_features(news_df, label_df["trading_date"].tolist())
    # 排除 lookback warmup
    eval_start = end - timedelta(days=args.months * 30)
    feat_df_eval = feat_df[pd.to_datetime(feat_df["trading_date"]).dt.date >= eval_start]
    label_df_eval = label_df[pd.to_datetime(label_df["trading_date"]).dt.date >= eval_start]
    print(f"  eval features: {len(feat_df_eval):,} (stocks={feat_df_eval['stock_id'].nunique()})")

    print(f"\n{'特徵':<28} {'cov':>7} {'n_dt':>6} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>8} {'pos%':>7} {'判定'}")
    print("-" * 100)

    features = ["news_count_5d", "news_count_change_5d",
                "news_source_diversity_5d", "news_count_change_1d"]
    recoverable = []
    for f in features:
        r = cross_section_ic(feat_df_eval, label_df_eval, f)
        ok = abs(r["icir"]) >= 0.30 and r["coverage"] >= 0.50 and r["n_dates"] >= 30
        verdict = "✅ 候選" if ok else (
            "⚠️ ICIR 弱" if not abs(r["icir"]) >= 0.30 else
            "⚠️ cov 不足" if r["coverage"] < 0.50 else "⚠️ n_dt 少"
        )
        if ok:
            recoverable.append(r)
        print(f"{f:<28} {r['coverage']:>7.1%} {r['n_dates']:>6d} "
              f"{r['ic_mean']:>10.4f} {r['ic_std']:>10.4f} {r['icir']:>+8.4f} "
              f"{r['pos_rate']:>7.1%}  {verdict}")

    print("\n" + "=" * 70)
    if recoverable:
        print(f"✅ {len(recoverable)} 個 news features 達標，可加進 PRUNED：")
        for r in recoverable:
            print(f"  {r['feature']:<28} ICIR={r['icir']:+.4f}")
    else:
        print("⚠️ 無 news feature 達 ICIR 0.30 門檻")


if __name__ == "__main__":
    main()
