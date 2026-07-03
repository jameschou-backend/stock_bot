#!/usr/bin/env python3
"""
Compare parquet vs MySQL feature values to detect if data has changed since F+ was run.
Also checks feature values for 2364 specifically to find discrepancies.
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
from skills import data_store

TRACK_SID = "2364"
CHECK_DATES = [date(2022, 6, 1), date(2022, 5, 31), date(2022, 5, 1), date(2021, 12, 1)]
WIN_START = date(2019, 1, 1)
WIN_END = date(2022, 7, 5)


def main():
    with get_session() as session:
        print("=" * 70)
        print("Diagnostic: Parquet vs MySQL feature values for 2364")
        print("=" * 70)

        # Load parquet
        print("\n[1] Loading from parquet...")
        feat_parquet = data_store.get_features(session, WIN_START, WIN_END)
        feat_parquet["stock_id"] = feat_parquet["stock_id"].astype(str)

        # Load MySQL
        print("[2] Loading from MySQL...")
        stmt = (
            select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            .where(Feature.trading_date.between(WIN_START, WIN_END))
            .order_by(Feature.trading_date, Feature.stock_id)
        )
        raw = pd.read_sql(stmt, session.get_bind())
        raw["trading_date"] = pd.to_datetime(raw["trading_date"]).dt.date

        # Parse MySQL JSON
        parsed = [json.loads(v) if isinstance(v, str) else (v if isinstance(v, dict) else {}) for v in raw["features_json"]]
        parsed_df = pd.json_normalize(parsed)
        parsed_df = parsed_df.replace([np.inf, -np.inf], np.nan)
        feat_mysql = raw[["stock_id", "trading_date"]].reset_index(drop=True)
        feat_mysql = pd.concat([feat_mysql, parsed_df.reset_index(drop=True)], axis=1)
        feat_mysql["stock_id"] = feat_mysql["stock_id"].astype(str)

        print(f"  Parquet: {len(feat_parquet):,} rows, {len(feat_parquet.columns)-2} feature cols")
        print(f"  MySQL: {len(feat_mysql):,} rows, {len(feat_mysql.columns)-2} feature cols")

        # Compare for target stock
        print(f"\n[3] Feature comparison for {TRACK_SID}...")
        p_sid = feat_parquet[feat_parquet["stock_id"] == TRACK_SID].copy()
        m_sid = feat_mysql[feat_mysql["stock_id"] == TRACK_SID].copy()

        print(f"  Parquet rows: {len(p_sid)}, MySQL rows: {len(m_sid)}")
        print(f"  Parquet date range: {p_sid['trading_date'].min()} ~ {p_sid['trading_date'].max()}")
        print(f"  MySQL date range: {m_sid['trading_date'].min()} ~ {m_sid['trading_date'].max()}")

        # Find common dates and compare
        common_dates = set(p_sid["trading_date"]) & set(m_sid["trading_date"])
        print(f"  Common dates: {len(common_dates)}")

        # Check a few specific dates
        for chk_date in sorted(common_dates)[-5:]:  # Last 5 common dates
            p_row = p_sid[p_sid["trading_date"] == chk_date]
            m_row = m_sid[m_sid["trading_date"] == chk_date]
            if p_row.empty or m_row.empty:
                continue

            # Compare common feature columns
            p_feat_cols = [c for c in p_row.columns if c not in ("stock_id", "trading_date")]
            m_feat_cols = [c for c in m_row.columns if c not in ("stock_id", "trading_date")]
            common_cols = list(set(p_feat_cols) & set(m_feat_cols))

            max_diff = 0.0
            max_diff_col = ""
            for col in common_cols:
                pv = float(p_row[col].iloc[0]) if not p_row[col].isna().iloc[0] else float('nan')
                mv = float(m_row[col].iloc[0]) if not m_row[col].isna().iloc[0] else float('nan')
                if not (np.isnan(pv) or np.isnan(mv)):
                    diff = abs(pv - mv)
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_col = col

            print(f"    {chk_date}: max_diff={max_diff:.6f} (col={max_diff_col})")

        # [4] Check the feature VALUES that the model sees on 2022-07-01
        print(f"\n[4] Feature values for {TRACK_SID} on 2022-06-30 or 2022-07-01...")
        for dt_check in [date(2022, 7, 1), date(2022, 6, 30)]:
            p_row = p_sid[p_sid["trading_date"] == dt_check]
            m_row = m_sid[m_sid["trading_date"] == dt_check]
            if not p_row.empty:
                print(f"  Parquet {dt_check}: found, first few features:")
                p_feat_cols = [c for c in p_row.columns if c not in ("stock_id", "trading_date")]
                for col in p_feat_cols[:5]:
                    print(f"    {col}: {float(p_row[col].iloc[0]):.6f}")
                break
        for dt_check in [date(2022, 7, 1), date(2022, 6, 30)]:
            m_row = m_sid[m_sid["trading_date"] == dt_check]
            if not m_row.empty:
                print(f"  MySQL {dt_check}: found, first few features:")
                m_feat_cols = [c for c in m_row.columns if c not in ("stock_id", "trading_date")]
                for col in m_feat_cols[:5]:
                    print(f"    {col}: {float(m_row[col].iloc[0]):.6f}")
                break

        # [5] Check: has the features table been updated recently?
        print("\n[5] Most recent feature update times...")
        stmt2 = select(Feature.trading_date).order_by(Feature.trading_date.desc()).limit(5)
        recent_dates = session.execute(stmt2).fetchall()
        print("  Most recent feature dates in MySQL:", [str(r[0]) for r in recent_dates])

        # [6] Check label values for 2364
        print(f"\n[6] Label values for {TRACK_SID}...")
        l_stmt = select(Label.stock_id, Label.trading_date, Label.future_ret_h).where(
            (Label.stock_id == TRACK_SID) &
            (Label.trading_date.between(WIN_START, WIN_END))
        ).order_by(Label.trading_date)
        labels_mysql = pd.read_sql(l_stmt, session.get_bind())
        labels_mysql["trading_date"] = pd.to_datetime(labels_mysql["trading_date"]).dt.date

        label_parquet = data_store.get_labels(session, WIN_START, WIN_END)
        label_parquet = label_parquet[label_parquet["stock_id"].astype(str) == TRACK_SID].copy()

        print(f"  MySQL labels: {len(labels_mysql)} rows")
        print(f"  Parquet labels: {len(label_parquet)} rows")

        # Check a few label values
        if not labels_mysql.empty and not label_parquet.empty:
            for ldate in sorted(labels_mysql["trading_date"].tolist())[-3:]:
                ml = labels_mysql[labels_mysql["trading_date"] == ldate]["future_ret_h"]
                pl = label_parquet[label_parquet["trading_date"] == ldate]["future_ret_h"]
                mv = float(ml.iloc[0]) if not ml.empty else None
                pv = float(pl.iloc[0]) if not pl.empty else None
                print(f"  Date {ldate}: MySQL={mv}, Parquet={pv}")

        # [7] Quick model training with exact MySQL data for F+'s full history window
        print(f"\n[7] Training with exact MySQL data, full history...")
        from datetime import date as d
        FULL_WIN_START = d(2016, 1, 1)
        FULL_WIN_END = d(2022, 9, 9)
        RETRAIN_RB = d(2022, 6, 1)
        LBL_CUTOFF = d(2022, 5, 12)

        stmt3 = (
            select(Feature.stock_id, Feature.trading_date, Feature.features_json)
            .where(Feature.trading_date.between(FULL_WIN_START, FULL_WIN_END))
            .order_by(Feature.trading_date, Feature.stock_id)
        )
        print("  Loading MySQL full history features (may take ~2min)...")
        raw_full = pd.read_sql(stmt3, session.get_bind())
        raw_full["trading_date"] = pd.to_datetime(raw_full["trading_date"]).dt.date
        print(f"  MySQL rows: {len(raw_full):,}")

        parsed_f = [json.loads(v) if isinstance(v, str) else {} for v in raw_full["features_json"]]
        parsed_df_f = pd.json_normalize(parsed_f)
        parsed_df_f = parsed_df_f.replace([np.inf, -np.inf], np.nan)
        feat_mysql_full = raw_full[["stock_id", "trading_date"]].reset_index(drop=True)
        feat_mysql_full = pd.concat([feat_mysql_full, parsed_df_f.reset_index(drop=True)], axis=1)
        feat_mysql_full["stock_id"] = feat_mysql_full["stock_id"].astype(str)

        # Load labels
        lstmt = select(Label.stock_id, Label.trading_date, Label.future_ret_h).where(
            Label.trading_date.between(FULL_WIN_START, FULL_WIN_END)
        ).order_by(Label.trading_date, Label.stock_id)
        label_full_mysql = pd.read_sql(lstmt, session.get_bind())
        label_full_mysql["trading_date"] = pd.to_datetime(label_full_mysql["trading_date"]).dt.date
        print(f"  MySQL labels: {len(label_full_mysql):,}")

        # Schema filter
        fc = [c for c in feat_mysql_full.columns if c not in ("stock_id", "trading_date")]
        thr = max(1, int(len(fc) * 0.50))
        mask = feat_mysql_full[fc].notna().sum(axis=1) >= thr
        feat_mysql_full = feat_mysql_full[mask].reset_index(drop=True)

        # Train
        try:
            import lightgbm as lgb
            HAS_LGBM = True
        except ImportError:
            HAS_LGBM = False
            from sklearn.ensemble import GradientBoostingRegressor

        train_feat = feat_mysql_full[
            (feat_mysql_full["trading_date"] < RETRAIN_RB) &
            (feat_mysql_full["trading_date"] >= FULL_WIN_START)
        ]
        train_label = label_full_mysql[
            (label_full_mysql["trading_date"] < LBL_CUTOFF) &
            (label_full_mysql["trading_date"] >= FULL_WIN_START)
        ]
        merged = train_feat.merge(train_label, on=["stock_id", "trading_date"], how="inner")
        print(f"  Merged training rows: {len(merged):,}")

        meta = {"stock_id", "trading_date", "future_ret_h"}
        fmat = merged.drop(columns=[c for c in meta if c in merged.columns])
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
        print(f"  Feature cols: {len(feature_names)}, training rows: {len(fmat):,}")

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

        # Score on 2022-07-01
        SCORE_DATE = d(2022, 7, 1)
        day_feat = feat_mysql_full[feat_mysql_full["trading_date"] == SCORE_DATE].copy()
        if day_feat.empty:
            for fb in range(1, 6):
                day_feat = feat_mysql_full[feat_mysql_full["trading_date"] == SCORE_DATE - timedelta(days=fb)].copy()
                if not day_feat.empty:
                    print(f"  Fallback scoring date")
                    break

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
        if not target.empty:
            sv = float(target["score"].iloc[0])
            rank = int((day_feat["score"] >= sv).sum())
            print(f"\n  ► MySQL full history: {TRACK_SID} score={sv:.6f}, rank={rank}/{len(day_feat)}")
            print(f"  F+ reference: score=0.115900")
        else:
            print(f"  {TRACK_SID} not found")


if __name__ == "__main__":
    main()
