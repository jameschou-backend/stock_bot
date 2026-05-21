"""產出 Fractional Differentiation features parquet（opt-in 實驗用）。

針對 close price 套用多個 d 值（預設 0.3 / 0.4 / 0.5），加入 features 池。

用法：
    # 預設：對 close 套 d=[0.3, 0.4, 0.5]
    python scripts/build_fracdiff_features.py

    # 自訂 d 值
    python scripts/build_fracdiff_features.py --d-values 0.2 0.35 0.5 0.7

    # 同時跑 ADF test 找最佳 d 並印報告
    python scripts/build_fracdiff_features.py --find-optimal --probe-stocks 2330 0050 1101

輸出：artifacts/labels/fracdiff_features.parquet
欄位：stock_id, trading_date, close_fracdiff_0_30, close_fracdiff_0_40, close_fracdiff_0_50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from sqlalchemy import select

from app.db import get_session
from app.models import RawPrice
from skills.fracdiff import find_optimal_d, fracdiff_panel


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d-values", nargs="*", type=float, default=[0.3, 0.4, 0.5],
                   help="多個 d 值，每個 d 產一欄（預設 0.3 / 0.4 / 0.5）")
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/labels/fracdiff_features.parquet"))
    p.add_argument("--stock-pattern", type=str, default=r"\d{4}",
                   help="stock_id regex（預設 4 碼）")
    p.add_argument("--threshold", type=float, default=1e-3,
                   help="FFD weight cutoff（預設 1e-3）")
    p.add_argument("--find-optimal", action="store_true",
                   help="同時跑 ADF test 找最佳 d 並印報告（用 --probe-stocks）")
    p.add_argument("--probe-stocks", nargs="*", default=["2330", "0050", "1101"],
                   help="ADF 探測用股票（預設 2330/0050/1101）")
    return p.parse_args()


def _run_adf_probes(df: pd.DataFrame, stock_ids: list, threshold: float) -> None:
    """對指定股票跑 ADF test 找最佳 d。"""
    print()
    print("=== ADF 平穩性探測 ===")
    print(f"  {'stock':<8} {'optimal_d':<12} {'best p-value':<14} {'平穩 from d':<12}")
    print("  " + "-" * 50)
    for sid in stock_ids:
        sub = df[df["stock_id"] == sid].sort_values("trading_date")
        if len(sub) < 100:
            print(f"  {sid:<8} (insufficient data, {len(sub)} rows)")
            continue
        close = sub["close"].astype(float).to_numpy()
        try:
            d_opt, results = find_optimal_d(close, threshold=threshold)
            # 找 d_opt 的 p_value
            p_val = results.get(d_opt, {}).get("p_value")
            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
            # 從哪個 d 開始 p < 0.05
            stationary_ds = [d for d, r in results.items()
                             if r.get("p_value") is not None and r["p_value"] < 0.05]
            first_stationary = min(stationary_ds) if stationary_ds else "—"
            print(f"  {sid:<8} {d_opt:<12.2f} {p_str:<14} {first_stationary}")
        except Exception as exc:
            print(f"  {sid:<8} ADF failed: {exc}")


def main() -> int:
    args = _parse_args()
    t0 = time.monotonic()

    print(f"=== Fracdiff features 產出 ===")
    print(f"  d values: {args.d_values}")
    print(f"  輸出: {args.output}")

    print("  載入 raw_prices ...", end="", flush=True)
    with get_session() as s:
        q = select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close)
        q = q.order_by(RawPrice.stock_id, RawPrice.trading_date)
        df = pd.read_sql(q, s.get_bind())
    print(f" {len(df):,} rows / {df['stock_id'].nunique()} stocks")

    if args.stock_pattern:
        before = df["stock_id"].nunique()
        df = df[df["stock_id"].astype(str).str.fullmatch(args.stock_pattern)]
        print(f"  pattern '{args.stock_pattern}' filter: {before} → {df['stock_id'].nunique()} stocks ({len(df):,} rows)")

    # 確保 close 是 float + dropna
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

    if args.find_optimal:
        _run_adf_probes(df, args.probe_stocks, args.threshold)

    print()
    print(f"=== 套用 fracdiff（per-stock per-d）===")
    # 累積套用每個 d
    result = df[["stock_id", "trading_date", "close"]].copy()
    for d in args.d_values:
        t1 = time.monotonic()
        col = f"close_fracdiff_{d:.2f}".replace(".", "_")
        print(f"  d={d:.2f} ({col}) ...", end="", flush=True)
        result = fracdiff_panel(
            result, value_col="close", d=d,
            out_col=col, threshold=args.threshold,
        )
        n_valid = result[col].notna().sum()
        elapsed = time.monotonic() - t1
        print(f" {n_valid:,}/{len(result):,} non-NaN ({elapsed:.1f}s)")

    # 不輸出 close（只留 fracdiff 欄位）
    out_cols = ["stock_id", "trading_date"] + [
        f"close_fracdiff_{d:.2f}".replace(".", "_") for d in args.d_values
    ]
    out_df = result[out_cols]

    # Drop rows where ALL fracdiff cols are NaN（節省空間）
    fracdiff_cols = [c for c in out_cols if c.startswith("close_fracdiff_")]
    out_df = out_df.dropna(subset=fracdiff_cols, how="all")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)

    print()
    print(f"=== 統計 ===")
    for c in fracdiff_cols:
        s = out_df[c].dropna()
        print(f"  {c}: n={len(s):,}, mean={s.mean():.4f}, std={s.std():.4f}, "
              f"min={s.min():.2f}, max={s.max():.2f}")

    elapsed = time.monotonic() - t0
    print()
    print(f"✅ 輸出 {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"   耗時 {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
