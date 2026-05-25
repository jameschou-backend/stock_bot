#!/usr/bin/env python3
"""
Diagnostic: Does float32 vs float64 cause different LightGBM predictions?

Tests 3 models for the 2022-06-01 retrain, 2022-07-01 scoring:
1. Parquet (float32) - current code
2. Parquet converted to float64 before training
3. MySQL directly (ca7c05f style)
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datetime import date, timedelta

from app.config import load_config
from app.db import get_session
from skills import data_store

TARGET_RB = date(2022, 7, 1)      # 再平衡日
SCORE_RB   = date(2022, 7, 1)      # 評分日
RETRAIN_RB = date(2022, 6, 1)      # 模型訓練日
WIN_START  = date(2017, 1, 1)      # 訓練視窗起點 (近似值)
WIN_END    = date(2022, 9, 9)      # 訓練視窗終點 (rb + 90+10d)
LBL_CUTOFF = date(2022, 5, 12)     # label_horizon_buffer=20 後的截止
TRACK_SID  = "2364"

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

def train_and_score(feat_df, label_df, score_date, dtype_note):
    """Train LightGBM on (feat_df, label_df), score at score_date."""
    # ── 1. Build training set ──
    train_feat = feat_df[
        (feat_df["trading_date"] < RETRAIN_RB) &
        (feat_df["trading_date"] >= WIN_START)
    ]
    train_label = label_df[
        (label_df["trading_date"] < LBL_CUTOFF) &
        (label_df["trading_date"] >= WIN_START)
    ]
    merged = train_feat.merge(train_label, on=["stock_id", "trading_date"], how="inner")

    print(f"\n[{dtype_note}] Training rows: {len(merged):,}")
    print(f"  Date range: {merged['trading_date'].min()} ~ {merged['trading_date'].max()}")
    print(f"  Feature dtype: {feat_df.dtypes.drop(['stock_id','trading_date']).unique()}")

    meta_cols = {"stock_id", "trading_date", "future_ret_h"}
    fmat = merged.drop(columns=[c for c in meta_cols if c in merged.columns])
    fmat = fmat.replace([np.inf, -np.inf], np.nan)
    for col in fmat.columns:
        if fmat[col].isna().all():
            fmat[col] = 0
        else:
            fmat[col] = fmat[col].fillna(fmat[col].median())
    valid = fmat.notna().all(axis=1)
    fmat = fmat.loc[valid]
    merged = merged.loc[fmat.index]
    y = merged["future_ret_h"].astype(float).values

    feature_names = list(fmat.columns)
    print(f"  Feature cols: {len(feature_names)}, training rows after filter: {len(fmat):,}")

    # ── 2. Train model ──
    if HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(fmat.values, y)
    else:
        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42,
        )
        model.fit(fmat.values, y)

    # ── 3. Score on rebalance date ──
    day_feat = feat_df[feat_df["trading_date"] == SCORE_RB].copy()
    if day_feat.empty:
        for fb in range(1, 6):
            fb_date = SCORE_RB - timedelta(days=fb)
            day_feat = feat_df[feat_df["trading_date"] == fb_date].copy()
            if not day_feat.empty:
                print(f"  Fallback to {fb_date}")
                break

    score_fmat = day_feat.drop(columns=["stock_id", "trading_date"], errors="ignore")
    for col in feature_names:
        if col not in score_fmat.columns:
            score_fmat[col] = 0
    score_fmat = score_fmat[feature_names]
    score_fmat = score_fmat.replace([np.inf, -np.inf], np.nan)
    for col in score_fmat.columns:
        if score_fmat[col].isna().all():
            score_fmat[col] = 0
        else:
            score_fmat[col] = score_fmat[col].fillna(score_fmat[col].median())

    scores = model.predict(score_fmat.values)
    day_feat = day_feat.reset_index(drop=True)
    day_feat["score"] = scores

    # ── 4. Find target stock ──
    target_row = day_feat[day_feat["stock_id"].astype(str) == TRACK_SID]
    total_stocks = len(day_feat)

    if not target_row.empty:
        score_val = float(target_row["score"].iloc[0])
        rank = int((day_feat["score"] >= score_val).sum())
        print(f"\n  ► 股票 {TRACK_SID}: score={score_val:.6f}, rank={rank}/{total_stocks}")
    else:
        print(f"\n  ► 股票 {TRACK_SID}: NOT FOUND in scoring universe")

    # Print top 5
    top5 = day_feat.nlargest(5, "score")[["stock_id", "score"]]
    print(f"  Top 5: {list(zip(top5['stock_id'].tolist(), top5['score'].round(4).tolist()))}")

    return day_feat[["stock_id", "score"]].copy()


def main():
    config = load_config()

    with get_session() as session:
        print("=" * 60)
        print(f"Diagnostic: float32 vs float64 for LightGBM")
        print(f"Retrain date: {RETRAIN_RB}, Score date: {SCORE_RB}")
        print(f"Training window: {WIN_START} ~ {LBL_CUTOFF}")
        print("=" * 60)

        # ── Model A: Parquet (float32) ──
        print("\n[A] Loading from parquet (float32)...")
        feat_a = data_store.get_features(session, WIN_START, WIN_END)
        label_a = data_store.get_labels(session, WIN_START, WIN_END)
        print(f"  Feature dtype sample: {feat_a.dtypes.iloc[2]}")  # First feature col
        scores_a = train_and_score(feat_a, label_a, SCORE_RB, "A: parquet float32")

        # ── Model B: Parquet converted to float64 ──
        print("\n[B] Converting parquet float32 → float64...")
        feat_b = feat_a.copy()
        num_cols = [c for c in feat_b.columns if c not in ("stock_id", "trading_date")]
        feat_b[num_cols] = feat_b[num_cols].astype("float64")
        label_b = label_a.copy()
        scores_b = train_and_score(feat_b, label_b, SCORE_RB, "B: parquet→float64")

        # ── Model C: MySQL directly ──
        print("\n[C] Loading from MySQL directly (float64)...")
        from sqlalchemy import select
        from app.models import Feature, Label
        import json

        feat_stmt = (
            select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            .where(Feature.trading_date.between(WIN_START, WIN_END))
            .order_by(Feature.trading_date, Feature.stock_id)
        )
        label_stmt = (
            select(Label.stock_id, Label.trading_date, Label.future_ret_h)
            .where(Label.trading_date.between(WIN_START, WIN_END))
            .order_by(Label.trading_date, Label.stock_id)
        )
        raw_feat = pd.read_sql(feat_stmt, session.get_bind())
        label_c = pd.read_sql(label_stmt, session.get_bind())
        raw_feat["trading_date"] = pd.to_datetime(raw_feat["trading_date"]).dt.date
        label_c["trading_date"] = pd.to_datetime(label_c["trading_date"]).dt.date

        # Parse JSON (same as ca7c05f _parse_features)
        parsed = [json.loads(v) if isinstance(v, str) else (v if isinstance(v, dict) else {}) for v in raw_feat["features_json"]]
        parsed_df = pd.json_normalize(parsed)
        parsed_df = parsed_df.replace([np.inf, -np.inf], np.nan)
        feat_c = raw_feat[["stock_id", "trading_date"]].reset_index(drop=True)
        feat_c = pd.concat([feat_c, parsed_df.reset_index(drop=True)], axis=1)

        # Schema filter (ca7c05f style: 50% non-NaN)
        fc = [c for c in feat_c.columns if c not in ("stock_id", "trading_date")]
        thr = max(1, int(len(fc) * 0.50))
        mask = feat_c[fc].notna().sum(axis=1) >= thr
        n_drop = int((~mask).sum())
        print(f"  MySQL schema filter: dropped {n_drop:,} rows (threshold {thr} of {len(fc)} features)")
        feat_c = feat_c.loc[mask].reset_index(drop=True)

        scores_c = train_and_score(feat_c, label_c, SCORE_RB, "C: MySQL float64")

        # ── Compare ──
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)

        merged_compare = scores_a.merge(scores_b, on="stock_id", suffixes=("_A", "_B"))
        merged_compare = merged_compare.merge(scores_c, on="stock_id")
        merged_compare = merged_compare.rename(columns={"score": "score_C"})

        # Check 2364
        row = merged_compare[merged_compare["stock_id"].astype(str) == TRACK_SID]
        if not row.empty:
            sa, sb, sc = float(row["score_A"].iloc[0]), float(row["score_B"].iloc[0]), float(row["score_C"].iloc[0])
            print(f"\n{TRACK_SID} scores: A(float32)={sa:.6f}, B(float64)={sb:.6f}, C(MySQL)={sc:.6f}")
            print(f"  F+ reference: 0.115900")
            print(f"  A≈B? {abs(sa-sb)<0.001}")
            print(f"  A≈C? {abs(sa-sc)<0.01}")
            print(f"  B≈C? {abs(sb-sc)<0.01}")

        # Overall score distribution comparison
        corr_AB = np.corrcoef(merged_compare["score_A"], merged_compare["score_B"])[0, 1]
        corr_AC = np.corrcoef(merged_compare["score_A"], merged_compare["score_C"])[0, 1]
        corr_BC = np.corrcoef(merged_compare["score_B"], merged_compare["score_C"])[0, 1]
        print(f"\nCorrelation A↔B: {corr_AB:.6f}")
        print(f"Correlation A↔C: {corr_AC:.6f}")
        print(f"Correlation B↔C: {corr_BC:.6f}")

        # Rank comparison for top 100
        for scores, name in [(scores_a, "A"), (scores_b, "B"), (scores_c, "C")]:
            s_sorted = scores.sort_values("score", ascending=False).reset_index(drop=True)
            s_sorted["rank"] = s_sorted.index + 1
            target = s_sorted[s_sorted["stock_id"].astype(str) == TRACK_SID]
            if not target.empty:
                print(f"Model {name}: {TRACK_SID} rank = {int(target['rank'].iloc[0])}/{len(s_sorted)}")


if __name__ == "__main__":
    main()
