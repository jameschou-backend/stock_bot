#!/usr/bin/env python3
"""
Exact replication of ca7c05f MySQL full-history loading for 2022-07-01 period.
Uses _load_features_labels + _parse_and_filter_features like ca7c05f did.
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
from sqlalchemy import select

from app.config import load_config
from app.db import get_session
from app.models import Feature, Label

TRACK_SID  = "2364"
RETRAIN_RB = date(2022, 6, 1)
LBL_CUTOFF = date(2022, 5, 12)    # buffer=20
SCORE_DATE = date(2022, 7, 1)
DATA_START = date(2016, 1, 1)
DATA_END   = date(2022, 10, 9)    # rb + 100 days


def _parse_features(series: pd.Series) -> pd.DataFrame:
    """Exact ca7c05f _parse_features logic."""
    parsed = [json.loads(v) if isinstance(v, str) else (v if isinstance(v, dict) else {}) for v in series]
    return pd.json_normalize(parsed)


def _parse_and_filter_features(raw_feat_df: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    """Exact ca7c05f _parse_and_filter_features logic."""
    parsed = _parse_features(raw_feat_df["features_json"])
    parsed = parsed.replace([np.inf, -np.inf], np.nan)
    feat_df = raw_feat_df[["stock_id", "trading_date"]].reset_index(drop=True)
    feat_df = pd.concat([feat_df, parsed.reset_index(drop=True)], axis=1)
    del parsed

    if feature_columns is not None:
        _avail = [c for c in feature_columns if c in feat_df.columns]
        feat_df = feat_df[["stock_id", "trading_date"] + _avail]

    # schema filter (50% threshold)
    _fc = [c for c in feat_df.columns if c not in ("stock_id", "trading_date")]
    if _fc:
        _thr = max(1, int(len(_fc) * 0.50))
        _mask = feat_df[_fc].notna().sum(axis=1) >= _thr
        _n_drop = int((~_mask).sum())
        if _n_drop > 0:
            print(f"  [schema filter] Dropped {_n_drop:,} rows (threshold={_thr}/{len(_fc)})")
        feat_df = feat_df.loc[_mask].reset_index(drop=True)

    return feat_df


def main():
    config = load_config()
    with get_session() as session:
        print("=" * 60)
        print("Replicating ca7c05f MySQL full-history loading")
        print(f"Data: {DATA_START} ~ {DATA_END}")
        print(f"Retrain: {RETRAIN_RB}, Label cutoff: {LBL_CUTOFF}")
        print(f"Score date: {SCORE_DATE}")
        print("=" * 60)

        # Load features from MySQL (ca7c05f style)
        print("\n[1] Loading features from MySQL (full history)...")
        feat_stmt = (
            select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            .where(Feature.trading_date.between(DATA_START, DATA_END))
            .order_by(Feature.trading_date, Feature.stock_id)
        )
        raw_feat = pd.read_sql(feat_stmt, session.get_bind())
        raw_feat["trading_date"] = pd.to_datetime(raw_feat["trading_date"]).dt.date
        print(f"  Raw MySQL rows: {len(raw_feat):,}")

        # Parse features (ca7c05f style)
        print("[2] Parsing features JSON...")
        feat_df = _parse_and_filter_features(raw_feat, feature_columns=None)
        feat_df["stock_id"] = feat_df["stock_id"].astype(str)
        feat_cols = [c for c in feat_df.columns if c not in ("stock_id", "trading_date")]
        print(f"  Parsed: {len(feat_df):,} rows, {len(feat_cols)} feature cols")
        print(f"  Date range: {feat_df['trading_date'].min()} ~ {feat_df['trading_date'].max()}")

        # Load labels
        print("[3] Loading labels from MySQL...")
        label_stmt = (
            select(Label.stock_id, Label.trading_date, Label.future_ret_h)
            .where(Label.trading_date.between(DATA_START, DATA_END))
            .order_by(Label.trading_date, Label.stock_id)
        )
        label_df = pd.read_sql(label_stmt, session.get_bind())
        label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
        label_df["stock_id"] = label_df["stock_id"].astype(str)
        print(f"  Labels: {len(label_df):,} rows")

        # Build training set (ca7c05f style)
        print("\n[4] Building training set...")
        train_feat = feat_df[feat_df["trading_date"] < RETRAIN_RB]
        train_label = label_df[label_df["trading_date"] < LBL_CUTOFF]
        merged = train_feat.merge(train_label, on=["stock_id", "trading_date"], how="inner")
        print(f"  Training rows: {len(merged):,}")
        print(f"  Date range: {merged['trading_date'].min()} ~ {merged['trading_date'].max()}")

        _meta_cols = {"stock_id", "trading_date", "future_ret_h"}
        fmat = merged.drop(columns=[c for c in _meta_cols if c in merged.columns])
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

        print(f"  Feature cols: {len(fmat.columns)}, training rows after filter: {len(fmat):,}")
        feature_names = list(fmat.columns)

        # Train model
        print("[5] Training LightGBM...")
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50,
                random_state=42, n_jobs=-1, verbose=-1,
            )
            model.fit(fmat.values, y)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                              max_depth=5, subsample=0.8, random_state=42)
            model.fit(fmat.values, y)

        # Score on 2022-07-01
        print(f"\n[6] Scoring on {SCORE_DATE}...")
        day_feat = feat_df[feat_df["trading_date"] == SCORE_DATE].copy()
        if day_feat.empty:
            for fb in range(1, 6):
                day_feat = feat_df[feat_df["trading_date"] == SCORE_DATE - timedelta(days=fb)].copy()
                if not day_feat.empty:
                    print(f"  Fallback to {SCORE_DATE - timedelta(days=fb)}")
                    break
        print(f"  Stocks on score date: {len(day_feat)}")

        sfmat = day_feat.drop(columns=["stock_id", "trading_date"], errors="ignore")
        for col in feature_names:
            if col not in sfmat.columns:
                sfmat[col] = 0
        sfmat = sfmat[feature_names]
        sfmat = sfmat.replace([np.inf, -np.inf], np.nan)
        for col in sfmat.columns:
            if sfmat[col].isna().all():
                sfmat[col] = 0
            else:
                sfmat[col] = sfmat[col].fillna(sfmat[col].median())

        scores = model.predict(sfmat.values)
        day_feat = day_feat.reset_index(drop=True)
        day_feat["score"] = scores

        target = day_feat[day_feat["stock_id"].astype(str) == TRACK_SID]
        total = len(day_feat)
        if not target.empty:
            sv = float(target["score"].iloc[0])
            rank = int((day_feat["score"] >= sv).sum())
            print(f"\n  ► MySQL full history: {TRACK_SID} score={sv:.6f}, rank={rank}/{total}")
            print(f"  F+ reference: score=0.115900")
            print(f"  Diff: {abs(sv - 0.1159):.6f}")
        else:
            print(f"  {TRACK_SID} not found in scoring universe")

        # Show top 10 scores
        top10 = day_feat.nlargest(10, "score")[["stock_id", "score"]]
        print("\n  Top 10 stocks:")
        for _, row in top10.iterrows():
            marker = " ← 2364" if str(row["stock_id"]) == TRACK_SID else ""
            print(f"    {row['stock_id']}: {row['score']:.6f}{marker}")


if __name__ == "__main__":
    main()
