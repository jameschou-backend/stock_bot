#!/usr/bin/env python
"""IC/ICIR 分析腳本 — 驗證新增特徵的預測效力。

用法:
    python scripts/ic_analysis_new_features.py           # 分析最近 2 年
    python scripts/ic_analysis_new_features.py --months 36  # 分析最近 3 年

輸出:
    每個特徵的 IC（Spearman），ICIR = IC.mean() / IC.std()
    ICIR > 0.1 → 有效，納入模型
    ICIR < 0.1 → 無效，排除
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select

from app.config import load_config
from app.db import get_session
from app.models import Feature, Label

# 要分析的新特徵（2026-03-11 新增）
NEW_FEATURES = [
    "foreign_buy_streak",
    "volume_surge_ratio",
    "foreign_buy_intensity",
]

# 對照組：既有的相似特徵，用來評估新特徵是否真的更好
REFERENCE_FEATURES = [
    "foreign_buy_consecutive_days",  # foreign_buy_streak 的基準
    "vol_ratio_20",                  # volume_surge_ratio 的基準
    "foreign_buy_ratio_5",           # foreign_buy_intensity 的基準
]


def fetch_features(session, start_date: date, end_date: date) -> pd.DataFrame:
    """從 DB 讀取 features_json 並展開為寬表格。"""
    stmt = (
        select(Feature.stock_id, Feature.trading_date, Feature.features_json)
        .where(Feature.trading_date.between(start_date, end_date))
        .order_by(Feature.trading_date, Feature.stock_id)
    )
    rows = session.execute(stmt).fetchall()
    if not rows:
        return pd.DataFrame()

    records = []
    for stock_id, trading_date, features_json in rows:
        d = features_json if isinstance(features_json, dict) else json.loads(features_json)
        d["stock_id"] = stock_id
        d["trading_date"] = trading_date
        records.append(d)
    return pd.DataFrame(records)


def fetch_labels(session, start_date: date, end_date: date) -> pd.DataFrame:
    """從 DB 讀取標籤（future_ret_h）。"""
    stmt = (
        select(Label.stock_id, Label.trading_date, Label.future_ret_h)
        .where(Label.trading_date.between(start_date, end_date))
        .order_by(Label.trading_date, Label.stock_id)
    )
    rows = session.execute(stmt).fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "trading_date", "future_ret_h"])
    df["future_ret_h"] = pd.to_numeric(df["future_ret_h"], errors="coerce")
    return df


def compute_ic_series(feat_df: pd.DataFrame, label_df: pd.DataFrame, feature: str) -> pd.Series:
    """對每個交易日計算 Spearman IC（特徵 vs 未來報酬）。"""
    if feature not in feat_df.columns:
        return pd.Series(dtype=float)

    merged = feat_df[["stock_id", "trading_date", feature]].merge(
        label_df, on=["stock_id", "trading_date"], how="inner"
    )
    merged[feature] = pd.to_numeric(merged[feature], errors="coerce")
    merged = merged.dropna(subset=[feature, "future_ret_h"])

    ic_by_date = {}
    for td, grp in merged.groupby("trading_date"):
        if len(grp) < 10:  # 樣本太少跳過
            continue
        rho, _ = stats.spearmanr(grp[feature], grp["future_ret_h"])
        ic_by_date[td] = rho

    return pd.Series(ic_by_date)


def main():
    parser = argparse.ArgumentParser(description="IC/ICIR 分析 — 新增特徵有效性驗證")
    parser.add_argument("--months", type=int, default=24, help="回測月數（預設 24）")
    args = parser.parse_args()

    config = load_config()
    end_date = date.today()
    start_date = end_date - timedelta(days=args.months * 30)

    print(f"\n{'='*60}")
    print(f"IC/ICIR 分析")
    print(f"{'='*60}")
    print(f"分析期間: {start_date} ~ {end_date}（{args.months} 個月）")

    with get_session() as session:
        print("\n[1/3] 載入特徵資料...")
        feat_df = fetch_features(session, start_date, end_date)
        if feat_df.empty:
            print("  ❌ 特徵資料不足，請先執行 make pipeline")
            return

        n_dates = feat_df["trading_date"].nunique()
        n_stocks = feat_df["stock_id"].nunique()
        print(f"  ✓ {n_dates} 個交易日，{n_stocks} 支股票")

        # 檢查新特徵是否存在
        available_new = [f for f in NEW_FEATURES if f in feat_df.columns]
        missing_new = [f for f in NEW_FEATURES if f not in feat_df.columns]
        if missing_new:
            print(f"  ⚠️  新特徵尚未在 DB 中: {missing_new}")
            print("     請先執行 make pipeline 以觸發 schema 重算。")
            if not available_new:
                return
            print(f"     僅分析已存在的特徵: {available_new}")

        print("\n[2/3] 載入標籤資料...")
        label_df = fetch_labels(session, start_date, end_date)
        if label_df.empty:
            print("  ❌ 標籤資料不足")
            return
        print(f"  ✓ {label_df['trading_date'].nunique()} 個標籤日期")

    print("\n[3/3] 計算 IC / ICIR...")
    print(f"\n{'特徵名稱':<30} {'IC均值':>8} {'IC標準差':>10} {'ICIR':>8} {'正IC率':>8} {'判定':>8}")
    print("-" * 76)

    results = {}
    all_features = list(dict.fromkeys(available_new + REFERENCE_FEATURES))
    for feat in all_features:
        ic_series = compute_ic_series(feat_df, label_df, feat)
        if ic_series.empty:
            print(f"  {feat:<28} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>8} {'(無資料)':>8}")
            continue

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        pos_rate = (ic_series > 0).mean()

        is_new = feat in NEW_FEATURES
        if is_new:
            verdict = "✅ 有效" if abs(icir) > 0.1 else "❌ 無效"
        else:
            verdict = "(對照)"

        prefix = "★ " if is_new else "  "
        print(f"{prefix}{feat:<28} {ic_mean:>8.4f} {ic_std:>10.4f} {icir:>8.4f} {pos_rate:>8.1%} {verdict:>8}")

        results[feat] = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "pos_rate": pos_rate,
            "is_new": is_new,
            "valid": abs(icir) > 0.1 if is_new else None,
        }

    # 彙總新特徵判定
    print(f"\n{'='*60}")
    print("新特徵判定彙總")
    print(f"{'='*60}")
    valid_new = []
    invalid_new = []
    for feat in available_new:
        if feat not in results:
            continue
        r = results[feat]
        if r["valid"]:
            valid_new.append(feat)
            print(f"  ✅ {feat}: ICIR={r['icir']:.4f} → 納入模型")
        else:
            invalid_new.append(feat)
            print(f"  ❌ {feat}: ICIR={r['icir']:.4f} → 排除（< 0.1）")

    if missing_new:
        print(f"\n  ⚠️  {missing_new} 尚未計算，需先重建特徵。")

    print(f"\n建議操作：")
    if valid_new:
        print(f"  保留特徵: {valid_new}")
    if invalid_new:
        print(f"  移除特徵（從 EXTENDED_FEATURE_COLUMNS 刪除）: {invalid_new}")

    return results


if __name__ == "__main__":
    main()
