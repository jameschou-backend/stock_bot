"""訓練 Meta-Label 第二層 model（López de Prado AFML Ch 3.7）。

流程：
  1. 讀取 features parquet（artifacts/features/features_YYYY.parquet）
  2. 訓練 primary model（同 train_ranker 的 LightGBM regressor 配置）
     對訓練資料 in-sample predict 拿到 primary_score
     ⚠️ 這個是「自我打分」用作 meta 訓練特徵，不是 OOS 評估
  3. 讀取 TB labels parquet（Stage 4.1 產出）
  4. 合併 → 訓練 meta classifier
  5. 寫 artifacts/models/meta_label_{tag}.joblib

用法：
    # 預設：對 Strategy D 配置 (5d / +5 / -3) 訓練
    python scripts/train_meta_label.py --tb-labels artifacts/labels/tb_d_5_3.parquet --tag d_5_3

    # Strategy C 配置
    python scripts/train_meta_label.py --tb-labels artifacts/labels/tb_c_8_5.parquet --tag c_8_5

    # 只用 primary_score > 0 的訓練（推薦，符合 meta-label 設計）
    python scripts/train_meta_label.py --tb-labels artifacts/labels/tb_d_5_3.parquet \\
        --tag d_5_3 --only-positive-signal

只是 opt-in 實驗，不接到 daily pipeline。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_LGBM = False

from skills.feature_store import FeatureStore
from skills.meta_label import prepare_meta_training_data, train_meta_model


def _train_primary_model(features_df: pd.DataFrame, label_col: str):
    """訓練 primary regression model（用作 meta 特徵），配置同 train_ranker。"""
    feature_cols = [
        c for c in features_df.columns
        if c not in ("stock_id", "trading_date", label_col)
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]
    X = features_df[feature_cols].fillna(0)
    y = features_df[label_col].to_numpy()

    if _HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42,
        )
    model.fit(X, y)
    return model, feature_cols


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tb-labels", type=Path, required=True,
                   help="Triple-Barrier labels parquet（Stage 4.1 產出）")
    p.add_argument("--tag", type=str, required=True,
                   help="模型 tag（用於檔名 meta_label_{tag}.joblib）")
    p.add_argument("--features-parquet", type=Path,
                   default=None,
                   help="features parquet 路徑；不給則用 FeatureStore 全量")
    p.add_argument("--primary-label-col", type=str, default="future_ret_h",
                   help="primary model 的訓練 label（預設 fixed-horizon return）")
    p.add_argument("--only-positive-signal", action="store_true",
                   help="只用 primary_score > 0 的樣本訓練 meta")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/models"))
    return _run(p.parse_args())


def _run(args) -> int:
    t0 = time.monotonic()
    print(f"=== Meta-Label 訓練：tag={args.tag} ===")

    # ── 1. 讀 features ─────────────────────────
    print("  載入 features ...", end="", flush=True)
    if args.features_parquet:
        feat_df = pd.read_parquet(args.features_parquet)
    else:
        # 用 FeatureStore 全量讀（會自動找 artifacts/features/features_YYYY.parquet）
        from datetime import date as _date
        fs = FeatureStore()
        max_date = fs.get_max_date()
        if max_date is None:
            print(" ❌ FeatureStore 為空")
            return 1
        # 預設讀 10 年資料（從 2016 起）
        min_date = _date(2016, 1, 1)
        feat_df = fs.read(min_date, max_date)
    print(f" {len(feat_df):,} rows, {len([c for c in feat_df.columns if c not in ('stock_id', 'trading_date')])} feature cols")

    # ── 2. 讀 TB labels ────────────────────────
    print(f"  載入 TB labels {args.tb_labels} ...", end="", flush=True)
    tb_df = pd.read_parquet(args.tb_labels)
    print(f" {len(tb_df):,} rows")

    # ── 3. 訓練 primary model + in-sample predict ──
    # primary 用 future_ret_h 訓練（既有 labels）；為了拿 score，需要先 merge primary label
    if args.primary_label_col not in feat_df.columns:
        # 從 DB 載 labels 並 merge
        from sqlalchemy import select
        from app.db import get_session
        from app.models import Label
        print("  載入 primary labels（labels 表）...", end="", flush=True)
        with get_session() as s:
            stmt = select(Label.stock_id, Label.trading_date, Label.future_ret_h)
            label_df = pd.read_sql(stmt, s.get_bind())
        print(f" {len(label_df):,} rows")
        # 確保 trading_date 是 date 型別（feat_df 與 label_df 對齊）
        feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
        label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
        feat_df = feat_df.merge(label_df, on=["stock_id", "trading_date"], how="inner")
        print(f"  merge 後 features+label: {len(feat_df):,} rows")

    feat_df = feat_df.dropna(subset=[args.primary_label_col])
    if feat_df.empty:
        print("❌ 沒有 primary label，無法訓練 primary model")
        return 1

    print(f"  訓練 primary model ({args.primary_label_col}) ...", end="", flush=True)
    t1 = time.monotonic()
    primary_model, primary_features = _train_primary_model(feat_df, args.primary_label_col)
    t_primary = time.monotonic() - t1
    print(f" {t_primary:.1f}s, {len(primary_features)} features")

    # in-sample predict（拿 primary_score 當 meta 特徵）
    primary_score = primary_model.predict(feat_df[primary_features].fillna(0))
    score_df = pd.DataFrame({
        "stock_id": feat_df["stock_id"].values,
        "trading_date": feat_df["trading_date"].values,
        "primary_score": primary_score,
    })

    # ── 4. 準備 meta 訓練資料 ──────────────────
    # tb labels 的 trading_date 也轉成 date 型別
    tb_df["trading_date"] = pd.to_datetime(tb_df["trading_date"]).dt.date

    print(f"  prepare meta training data (only_positive={args.only_positive_signal}) ...", end="", flush=True)
    # 先按時間排序
    feat_sorted = feat_df.sort_values("trading_date").reset_index(drop=True)
    score_sorted = score_df.sort_values("trading_date").reset_index(drop=True)
    X, primary_arr, y = prepare_meta_training_data(
        feat_sorted[["stock_id", "trading_date"] + primary_features],
        score_sorted,
        tb_df,
        only_positive_signal=args.only_positive_signal,
    )
    print(f" X={X.shape}, y pos_rate={y.mean():.2%}")

    # ── 5. 訓練 meta model ────────────────────
    print(f"  訓練 meta classifier ...", end="", flush=True)
    t2 = time.monotonic()
    meta = train_meta_model(X, y, val_frac=args.val_frac)
    t_meta = time.monotonic() - t2
    print(f" {t_meta:.1f}s")

    # ── 6. 結果 ──────────────────────────────
    print()
    print("=== Validation 結果 ===")
    print(f"  train pos rate: {meta.train_pos_rate:.2%} ({meta.n_train:,} samples)")
    print(f"  val   pos rate: {meta.val_pos_rate:.2%} ({meta.n_val:,} samples)")
    print(f"  val metrics @ threshold=0.5:")
    for k, v in meta.val_metrics.items():
        if isinstance(v, float):
            print(f"    {k:<20} {v:.4f}")
        else:
            print(f"    {k:<20} {v}")

    # ── 7. 儲存 ───────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"meta_label_{args.tag}.joblib"
    joblib.dump({
        "meta_model": meta,
        "primary_model": primary_model,
        "primary_features": primary_features,
        "tag": args.tag,
        "tb_labels_source": str(args.tb_labels),
        "only_positive_signal": args.only_positive_signal,
        "trained_at": datetime.now().isoformat(),
    }, out_path)

    elapsed = time.monotonic() - t0
    print()
    print(f"✅ 儲存 {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"   總耗時 {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
