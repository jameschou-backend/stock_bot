"""Stage 5.4：把 RawPER 真實值跟 fracdiff_0_50 補進 features parquet。

背景：features parquet 已預留 pbr_ratio / dividend_yield 欄位但全 NaN
（既有 parquet 是 raw_per backfill 之前產生的，build_features 增量不回算）。
另外 close_fracdiff_0_50 是 Stage 4.3 新加，原本不在 features parquet 內。

本 script 不修改 build_features 內部邏輯，直接 post-process 既有 parquet：
  1. 載入 features_YYYY.parquet（FeatureStore）
  2. 從 raw_per 表 merge pbr_ratio / dividend_yield / earnings_yield（去掉 placeholder）
  3. 從 fracdiff_features.parquet merge close_fracdiff_0_50
  4. 寫回各年 parquet（覆蓋）

下次 build_features 跑時若 raw_per / fracdiff 持續更新，新 row 會帶上正確值
（build_features 內部已有 raw_per merge 邏輯，缺的是 fracdiff — 留待後續完整整合）。

用法：
    python scripts/enrich_features_stage5_4.py
    python scripts/enrich_features_stage5_4.py --dry-run    # 看會改什麼但不寫
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from sqlalchemy import select

from app.db import get_session
from app.models import RawPER


FEATURE_STORE_DIR = Path("artifacts/features")
FRACDIFF_PARQUET = Path("artifacts/labels/fracdiff_features.parquet")

NEW_COLS = ["pbr_ratio", "dividend_yield", "earnings_yield", "close_fracdiff_0_50"]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="只報告會改的，不實際寫")
    return p.parse_args()


def _load_per_df() -> pd.DataFrame:
    with get_session() as s:
        q = select(RawPER.stock_id, RawPER.trading_date, RawPER.per, RawPER.pbr, RawPER.dividend_yield)
        df = pd.read_sql(q, s.get_bind())
    df["stock_id"] = df["stock_id"].astype(str)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    for c in ["per", "pbr", "dividend_yield"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pbr_ratio"] = df["pbr"]
    # earnings_yield = 1/PER（clip -10~50 同 build_features 邏輯）
    df["earnings_yield"] = (1.0 / df["per"].replace(0, np.nan)).clip(-10, 50)
    return df[["stock_id", "trading_date", "pbr_ratio", "dividend_yield", "earnings_yield"]]


def _load_fracdiff_df() -> pd.DataFrame:
    fd = pd.read_parquet(FRACDIFF_PARQUET)
    fd["stock_id"] = fd["stock_id"].astype(str)
    fd["trading_date"] = pd.to_datetime(fd["trading_date"]).dt.date
    return fd[["stock_id", "trading_date", "close_fracdiff_0_50"]]


def main() -> int:
    args = _parse_args()
    t0 = time.monotonic()

    print("=== Stage 5.4 features enrich ===")
    print(f"  features dir: {FEATURE_STORE_DIR}")
    print(f"  dry_run: {args.dry_run}")
    print()

    print("  載入 raw_per ...", end="", flush=True)
    per_df = _load_per_df()
    print(f" {len(per_df):,} rows ({per_df['pbr_ratio'].notna().sum():,} pbr non-NaN)")

    print(f"  載入 fracdiff parquet ...", end="", flush=True)
    fd_df = _load_fracdiff_df()
    print(f" {len(fd_df):,} rows")
    print()

    parquet_files = sorted(FEATURE_STORE_DIR.glob("features_*.parquet"))
    print(f"找到 {len(parquet_files)} 個 features parquet")

    total_before = 0
    total_pbr_added = 0
    total_div_added = 0
    total_eyield_added = 0
    total_fd_added = 0

    for f in parquet_files:
        df = pd.read_parquet(f)
        df["stock_id"] = df["stock_id"].astype(str)
        df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
        n = len(df)
        total_before += n

        # 統計現況
        before_pbr = df["pbr_ratio"].notna().sum() if "pbr_ratio" in df.columns else 0
        before_div = df["dividend_yield"].notna().sum() if "dividend_yield" in df.columns else 0
        before_eyield = df["earnings_yield"].notna().sum() if "earnings_yield" in df.columns else 0
        before_fd = df["close_fracdiff_0_50"].notna().sum() if "close_fracdiff_0_50" in df.columns else 0

        # 一律 drop 既有欄位（用 raw_per/fracdiff 全集替換），避免 merge 變 _x / _y
        drop_cols = [c for c in ["pbr_ratio", "dividend_yield", "earnings_yield",
                                  "close_fracdiff_0_50"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        df = df.merge(per_df, on=["stock_id", "trading_date"], how="left")
        df = df.merge(fd_df, on=["stock_id", "trading_date"], how="left")

        after_pbr = df["pbr_ratio"].notna().sum()
        after_div = df["dividend_yield"].notna().sum()
        after_eyield = df["earnings_yield"].notna().sum()
        after_fd = df["close_fracdiff_0_50"].notna().sum()

        added_pbr = after_pbr - before_pbr
        added_div = after_div - before_div
        added_eyield = after_eyield - before_eyield
        added_fd = after_fd - before_fd

        total_pbr_added += added_pbr
        total_div_added += added_div
        total_eyield_added += added_eyield
        total_fd_added += added_fd

        print(f"  {f.name}: rows={n:,}, +pbr={added_pbr:>7,}, +div={added_div:>7,}, "
              f"+eyld={added_eyield:>7,}, +fd={added_fd:>7,}")

        if not args.dry_run:
            df.to_parquet(f, index=False)

    elapsed = time.monotonic() - t0
    print()
    print(f"=== 統計 ===")
    print(f"  total rows: {total_before:,}")
    print(f"  pbr_ratio added:        {total_pbr_added:,}")
    print(f"  dividend_yield added:   {total_div_added:,}")
    print(f"  earnings_yield added:   {total_eyield_added:,}")
    print(f"  close_fracdiff_0_50:    {total_fd_added:,}")
    print()
    if args.dry_run:
        print("  ⚠️ DRY RUN — 沒實際寫入。移除 --dry-run 才會更新 parquet")
    else:
        print(f"  ✅ 已更新 {len(parquet_files)} 個 parquet ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
