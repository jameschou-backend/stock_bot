#!/usr/bin/env python3
"""
Test: Does dropping all-NaN training features from parquet match MySQL (53-feature) results?

Model D: parquet (float32) but dropping all-NaN feature columns → should match C (MySQL)
Model E: parquet with extended training window (from 2016-03-15) → should match F+'s 0.1159
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from datetime import date, timedelta

from app.config import load_config
from app.db import get_session
from skills import data_store

TARGET_RB  = date(2022, 7, 1)
RETRAIN_RB = date(2022, 6, 1)
WIN_START  = date(2017, 1, 1)      # 5-year lookback from Jun 2022
WIN_START_FULL = date(2016, 1, 1)  # Full history (matches F+ no-lookback)
WIN_END    = date(2022, 9, 9)
LBL_CUTOFF = date(2022, 5, 12)     # label_horizon_buffer=20
SCORE_DATE = date(2022, 7, 1)
TRACK_SID  = "2364"

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import GradientBoostingRegressor

def make_model(feat_df, label_df, win_start, drop_all_nan=False, note=""):
    train_feat = feat_df[
        (feat_df["trading_date"] < RETRAIN_RB) &
        (feat_df["trading_date"] >= win_start)
    ]
    train_label = label_df[
        (label_df["trading_date"] < LBL_CUTOFF) &
        (label_df["trading_date"] >= win_start)
    ]
    merged = train_feat.merge(train_label, on=["stock_id", "trading_date"], how="inner")

    meta_cols = {"stock_id", "trading_date", "future_ret_h"}
    fmat = merged.drop(columns=[c for c in meta_cols if c in merged.columns])
    fmat = fmat.replace([np.inf, -np.inf], np.nan)

    dropped_cols = []
    if drop_all_nan:
        all_nan = [c for c in fmat.columns if fmat[c].isna().all()]
        dropped_cols = all_nan
        fmat = fmat.drop(columns=all_nan)

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

    n_feat = len(feature_names)
    n_rows = len(fmat)
    print(f"\n  [{note}] n_feat={n_feat}, n_rows={n_rows:,}, dropped_cols={dropped_cols}")

    if HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(fmat.values, y)
    else:
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
        model.fit(fmat.values, y)

    return model, feature_names


def score_model(model, feature_names, feat_df, score_date):
    day_feat = feat_df[feat_df["trading_date"] == score_date].copy()
    if day_feat.empty:
        for fb in range(1, 6):
            fb_date = score_date - timedelta(days=fb)
            day_feat = feat_df[feat_df["trading_date"] == fb_date].copy()
            if not day_feat.empty:
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

    target = day_feat[day_feat["stock_id"].astype(str) == TRACK_SID]
    total = len(day_feat)
    if not target.empty:
        sv = float(target["score"].iloc[0])
        rank = int((day_feat["score"] >= sv).sum())
        print(f"    {TRACK_SID}: score={sv:.6f}, rank={rank}/{total}")
    else:
        print(f"    {TRACK_SID}: NOT FOUND")
    return day_feat[["stock_id", "score"]].copy()


def main():
    config = load_config()
    with get_session() as session:
        print("=" * 60)
        print("Diagnostic: Feature count effect on LightGBM scores")
        print("=" * 60)

        # Load parquet once for both win ranges
        print(f"\nLoading parquet (5yr window: {WIN_START}~{WIN_END})...")
        feat_5yr = data_store.get_features(session, WIN_START, WIN_END)
        label_5yr = data_store.get_labels(session, WIN_START, WIN_END)
        print(f"  Parquet features: {len([c for c in feat_5yr.columns if c not in ('stock_id','trading_date')])} cols")

        # Check which columns are all-NaN in training window
        train_slice = feat_5yr[
            (feat_5yr["trading_date"] < RETRAIN_RB) &
            (feat_5yr["trading_date"] >= WIN_START)
        ]
        feat_cols = [c for c in feat_5yr.columns if c not in ("stock_id", "trading_date")]
        all_nan_in_train = [c for c in feat_cols if train_slice[c].isna().all()]
        print(f"  All-NaN in training window: {all_nan_in_train}")

        # Model A: parquet 56 features (current behavior)
        print("\n--- Model A: parquet 56 features (current code) ---")
        model_a, fnames_a = make_model(feat_5yr, label_5yr, WIN_START, drop_all_nan=False, note="56 feat")
        scores_a = score_model(model_a, fnames_a, feat_5yr, SCORE_DATE)

        # Model D: parquet drop all-NaN cols
        print("\n--- Model D: parquet drop all-NaN feature columns ---")
        model_d, fnames_d = make_model(feat_5yr, label_5yr, WIN_START, drop_all_nan=True, note="drop NaN cols")
        scores_d = score_model(model_d, fnames_d, feat_5yr, SCORE_DATE)

        # Model E: parquet with full history (matches F+'s no-lookback)
        FULL_WIN_END = date(2022, 9, 9)
        print(f"\nLoading full history parquet ({WIN_START_FULL}~{FULL_WIN_END})...")
        feat_full = data_store.get_features(session, WIN_START_FULL, FULL_WIN_END)
        label_full = data_store.get_labels(session, WIN_START_FULL, FULL_WIN_END)
        print(f"  Full features loaded: {len(feat_full):,} rows")

        # Model E: full history, 56 features (current code behavior with no --train-lookback)
        print("\n--- Model E: full history parquet 56 features ---")
        model_e, fnames_e = make_model(feat_full, label_full, WIN_START_FULL, drop_all_nan=False, note="full hist 56")
        scores_e = score_model(model_e, fnames_e, feat_full, SCORE_DATE)

        # Model F: full history, drop all-NaN cols
        print("\n--- Model F: full history parquet drop all-NaN cols ---")
        model_f, fnames_f = make_model(feat_full, label_full, WIN_START_FULL, drop_all_nan=True, note="full hist drop NaN")
        scores_f = score_model(model_f, fnames_f, feat_full, SCORE_DATE)

        # Compare all with F+ reference
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"F+ reference score for {TRACK_SID}: 0.115900")
        print(f"")
        for name, s in [("A: parquet 5yr 56feat", scores_a),
                        ("D: parquet 5yr drop-NaN", scores_d),
                        ("E: parquet full 56feat", scores_e),
                        ("F: parquet full drop-NaN", scores_f)]:
            row = s[s["stock_id"].astype(str) == TRACK_SID]
            if not row.empty:
                sv = float(row["score"].iloc[0])
                rank = int((s["score"] >= sv).sum())
                diff = abs(sv - 0.1159)
                print(f"  {name}: score={sv:.6f}, rank={rank}/{len(s)}, |diff from F+|={diff:.6f}")


if __name__ == "__main__":
    main()
