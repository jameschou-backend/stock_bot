#!/usr/bin/env python
"""Stage E：SHAP 深度分析 v2（不重訓，用既有 model，降低跟 backfill 競爭）。

對最近的 LightGBM ranker model 跑：
  1. Global importance（mean |SHAP|）
  2. Regime breakdown（bull vs bear，依 market_above_200ma）
  3. Redundant feature pairs（|corr| > 0.85）

輸出：
  artifacts/shap_analysis/global_importance.csv
  artifacts/shap_analysis/regime_breakdown.csv
  artifacts/shap_analysis/redundant_pairs.csv

用法：
    python scripts/shap_v2.py                    # 最新 model + 5000 sample
    python scripts/shap_v2.py --sample 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import shap
from sqlalchemy import select

from app.db import get_session
from app.models import Feature
from skills.build_features import PRUNED_FEATURE_COLS


def load_latest_model():
    models = sorted(Path("artifacts/models").glob("ranker_lgbm_*.pkl"))
    if not models:
        raise FileNotFoundError("無 ranker model")
    latest = models[-1]
    print(f"  載入 model: {latest.name}")
    obj = joblib.load(latest)
    if isinstance(obj, dict):
        return obj.get("model", obj), obj.get("feature_names", PRUNED_FEATURE_COLS)
    if isinstance(obj, tuple):
        return obj[0], obj[1] if len(obj) > 1 else PRUNED_FEATURE_COLS
    return obj, PRUNED_FEATURE_COLS


def load_features(months: int, sample: int) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=months * 30)
    print(f"  載入 features {start} ~ {end} ...")
    with get_session() as s:
        rows = s.execute(
            select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            .where(Feature.trading_date.between(start, end))
        ).all()
    if not rows:
        raise RuntimeError("無 features 資料")
    records = []
    for sid, td, fjson in rows:
        d = fjson if isinstance(fjson, dict) else json.loads(fjson)
        d["stock_id"] = sid
        d["trading_date"] = td
        records.append(d)
    df = pd.DataFrame(records)
    print(f"  總筆數: {len(df):,}, 採樣 {sample}")
    if len(df) > sample:
        df = df.sample(sample, random_state=42)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--months", type=int, default=24)
    p.add_argument("--sample", type=int, default=5000)
    args = p.parse_args()

    out_dir = Path("artifacts/shap_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nStage E：SHAP 深度分析\n{'='*70}")

    if args.model:
        obj = joblib.load(args.model)
        if isinstance(obj, dict):
            model, feature_names = obj.get("model", obj), obj.get("feature_names", PRUNED_FEATURE_COLS)
        elif isinstance(obj, tuple):
            model = obj[0]
            feature_names = obj[1] if len(obj) > 1 else PRUNED_FEATURE_COLS
        else:
            model = obj
            feature_names = PRUNED_FEATURE_COLS
    else:
        model, feature_names = load_latest_model()
    print(f"  Model feature 數: {len(feature_names)}")

    df = load_features(args.months, args.sample)

    for f in feature_names:
        if f not in df.columns:
            df[f] = 0.0
    X = df[feature_names].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\n[1/3] 計算 SHAP values ({len(X)} × {len(feature_names)})...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    print(f"\n[2/3] Global importance...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    importance = importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["pct"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum() * 100
    importance["cum_pct"] = importance["pct"].cumsum()
    importance.to_csv(out_dir / "global_importance.csv", index=False)

    print(f"\n  Top 15 by SHAP importance:")
    print(f"  {'#':>3} {'feature':<28} {'mean|SHAP|':>12} {'pct':>7} {'cum%':>8}")
    print("  " + "-" * 66)
    for i, row in importance.head(15).iterrows():
        print(f"  {i+1:>3} {row['feature']:<28} {row['mean_abs_shap']:>12.6f} "
              f"{row['pct']:>6.2f}% {row['cum_pct']:>7.2f}%")
    top_80 = importance[importance["cum_pct"] <= 80]
    print(f"\n  Top {len(top_80)} features 貢獻 80% importance")
    print(f"  剩 {len(feature_names) - len(top_80)} features 共 20%（可能剪枝候選）")

    print(f"\n[3/3] Regime breakdown（market_above_200ma）...")
    if "market_above_200ma" in X.columns:
        bull_mask = X["market_above_200ma"] > 0.5
        bear_mask = X["market_above_200ma"] <= 0.5
        print(f"  bull={bull_mask.sum()}, bear={bear_mask.sum()}")
        if bull_mask.sum() > 100 and bear_mask.sum() > 100:
            bull_imp = np.abs(shap_values[bull_mask]).mean(axis=0)
            bear_imp = np.abs(shap_values[bear_mask]).mean(axis=0)
            regime_df = pd.DataFrame({
                "feature": feature_names,
                "bull_mean_abs_shap": bull_imp,
                "bear_mean_abs_shap": bear_imp,
            })
            regime_df["bull_vs_bear"] = regime_df["bull_mean_abs_shap"] / regime_df["bear_mean_abs_shap"].replace(0, np.nan)
            regime_df = regime_df.sort_values("bear_mean_abs_shap", ascending=False)
            regime_df.to_csv(out_dir / "regime_breakdown.csv", index=False)

            print(f"\n  Top 10 in BEAR（regime-conditional 候選）：")
            print(f"  {'feature':<28} {'bear':>10} {'bull/bear':>10}")
            for _, row in regime_df.head(10).iterrows():
                bvr = row["bull_vs_bear"]
                bvr_str = f"{bvr:.2f}x" if not pd.isna(bvr) else "N/A"
                marker = ""
                if not pd.isna(bvr):
                    if bvr > 1.5:
                        marker = " ⬆ bull-heavy"
                    elif bvr < 0.67:
                        marker = " ⬇ bear-heavy"
                print(f"  {row['feature']:<28} {row['bear_mean_abs_shap']:>10.6f} {bvr_str:>10}{marker}")
        else:
            print(f"  bull/bear 樣本不足")

    print(f"\n  尋找 redundant pairs（|corr| > 0.85）...")
    corr = X.corr()
    redundant = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            c = corr.iloc[i, j]
            if abs(c) > 0.85:
                redundant.append({
                    "f1": feature_names[i], "f2": feature_names[j], "corr": float(c),
                    "f1_shap": float(mean_abs[i]), "f2_shap": float(mean_abs[j]),
                })
    if redundant:
        rd_df = pd.DataFrame(redundant)
        rd_df["abs_corr"] = rd_df["corr"].abs()
        rd_df = rd_df.sort_values("abs_corr", ascending=False)
        rd_df.to_csv(out_dir / "redundant_pairs.csv", index=False)
        print(f"  {len(redundant)} 對高相關 features")
        print(f"  Top 5：")
        for _, row in rd_df.head(5).iterrows():
            weaker = row["f1"] if row["f1_shap"] < row["f2_shap"] else row["f2"]
            print(f"    {row['f1']:<26} <-> {row['f2']:<26} corr={row['corr']:+.4f}  弱者={weaker}")
    else:
        print(f"  無高相關 pairs")

    print(f"\n  輸出:")
    print(f"    {out_dir}/global_importance.csv")
    print(f"    {out_dir}/regime_breakdown.csv")
    print(f"    {out_dir}/redundant_pairs.csv")


if __name__ == "__main__":
    main()
