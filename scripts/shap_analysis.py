#!/usr/bin/env python3
"""SHAP 特徵重要性分析：訓練一個 LightGBM 模型並計算 SHAP 值，找出冗餘特徵。

Usage:
    python scripts/shap_analysis.py                    # 最近 5 年訓練資料
    python scripts/shap_analysis.py --train-years 3   # 最近 3 年
    python scripts/shap_analysis.py --liq-weighted     # 流動性加權（對應生產設定）
"""
from __future__ import annotations

import argparse
import sys
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


def main():
    parser = argparse.ArgumentParser(description="SHAP 特徵重要性分析")
    parser.add_argument("--train-years", type=int, default=5, help="使用最近幾年訓練資料（預設 5）")
    parser.add_argument("--liq-weighted", action="store_true", dest="liq_weighted",
                        help="流動性加權：sample_weight ∝ log(1+amt_20)")
    parser.add_argument("--output", type=str, default="artifacts/shap_report.txt")
    args = parser.parse_args()

    config = load_config()
    today = date.today()
    train_start = today - timedelta(days=args.train_years * 365)
    train_end = today - timedelta(days=20)   # label horizon buffer

    print(f"訓練期間: {train_start} ~ {train_end}")

    with get_session() as session:
        print("載入特徵與標籤...")
        feat_df = data_store.get_features(session, start_date=train_start, end_date=train_end)
        labels_df = data_store.get_labels(session, start_date=train_start, end_date=train_end)

    print(f"特徵: {len(feat_df):,} rows  標籤: {len(labels_df):,} rows")

    merged = feat_df.merge(
        labels_df[["stock_id", "trading_date", "future_ret_h"]],
        on=["stock_id", "trading_date"],
        how="inner",
    )
    print(f"合併後: {len(merged):,} rows")

    # 移除非特徵欄位
    meta_cols = {"stock_id", "trading_date", "future_ret_h"}
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X = merged[feature_cols].copy()
    y = merged["future_ret_h"].astype(float).values

    # 清理
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().all():
            X[col] = 0.0
        else:
            X[col] = X[col].fillna(X[col].median())

    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    y = y[valid]
    merged_valid = merged.loc[valid]

    print(f"清理後: {len(X):,} rows  特徵數: {len(feature_cols)}")

    # 流動性加權
    sample_weight = None
    if args.liq_weighted and "amt_20" in X.columns:
        amt20 = merged_valid["amt_20"].fillna(0).clip(lower=0).values.astype(float)
        liq_w = np.log1p(amt20)
        mean_w = liq_w.mean()
        if mean_w > 0:
            liq_w = liq_w / mean_w
        sample_weight = liq_w
        print(f"流動性加權：median={np.median(liq_w):.2f}  max={np.max(liq_w):.2f}")

    # 訓練 LightGBM
    import lightgbm as lgb
    print("訓練 LightGBM...")
    dtrain = lgb.Dataset(X.values, label=y, feature_name=list(X.columns),
                         weight=sample_weight)
    params = {
        "objective": "regression",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }
    model = lgb.train(params, dtrain, num_boost_round=500,
                      callbacks=[lgb.log_evaluation(period=100)])
    print("訓練完成")

    # LightGBM 內建重要性
    lgb_gain = model.feature_importance(importance_type="gain")
    lgb_split = model.feature_importance(importance_type="split")
    feat_names = model.feature_name()

    lgb_df = pd.DataFrame({
        "feature": feat_names,
        "lgb_gain": lgb_gain,
        "lgb_split": lgb_split,
    }).sort_values("lgb_gain", ascending=False)
    lgb_df["gain_pct"] = lgb_df["lgb_gain"] / lgb_df["lgb_gain"].sum() * 100
    lgb_df["cum_gain_pct"] = lgb_df["gain_pct"].cumsum()

    # SHAP 值
    print("計算 SHAP 值（最多 5000 樣本）...")
    import shap
    sample_idx = np.random.choice(len(X), min(5000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feat_names,
        "shap_mean_abs": shap_abs_mean,
    }).sort_values("shap_mean_abs", ascending=False)
    shap_df["shap_pct"] = shap_df["shap_mean_abs"] / shap_df["shap_mean_abs"].sum() * 100
    shap_df["cum_shap_pct"] = shap_df["shap_pct"].cumsum()

    # 合併
    summary = lgb_df.merge(shap_df, on="feature", how="left")

    # 輸出
    lines = []
    lines.append(f"SHAP 分析報告 (訓練期間: {train_start} ~ {train_end})")
    lines.append(f"樣本數: {len(X):,}  特徵數: {len(feature_cols)}  SHAP樣本: {len(X_sample):,}")
    lines.append(f"流動性加權: {'是' if args.liq_weighted else '否'}")
    lines.append("")
    lines.append(f"{'排名':>4}  {'特徵':<35}  {'LGB增益%':>8}  {'SHAP平均|值|%':>12}  {'LGB增益':>10}")
    lines.append("-" * 80)
    for rank, row in enumerate(summary.itertuples(), 1):
        lines.append(
            f"{rank:>4}  {row.feature:<35}  {row.gain_pct:>8.2f}%  "
            f"{row.shap_pct:>11.2f}%  {row.lgb_gain:>10.0f}"
        )

    lines.append("")
    lines.append("=== 低重要性特徵（LGB gain < 0.5% AND SHAP < 0.5%）===")
    low_imp = summary[(summary["gain_pct"] < 0.5) & (summary["shap_pct"] < 0.5)]
    for row in low_imp.itertuples():
        lines.append(f"  {row.feature:<35}  gain={row.gain_pct:.3f}%  shap={row.shap_pct:.3f}%")
    lines.append(f"共 {len(low_imp)} 個低重要性特徵")

    lines.append("")
    lines.append("=== 前 10 特徵（累積 SHAP）===")
    top10 = shap_df.head(10)
    for row in top10.itertuples():
        lines.append(f"  {row.feature:<35}  {row.shap_pct:.2f}%  (累積 {row.cum_shap_pct:.1f}%)")

    report = "\n".join(lines)
    print("\n" + report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n報告已儲存: {args.output}")

    # 儲存詳細 CSV
    csv_path = str(out_path).replace(".txt", ".csv")
    summary.to_csv(csv_path, index=False)
    print(f"詳細數據: {csv_path}")


if __name__ == "__main__":
    main()
