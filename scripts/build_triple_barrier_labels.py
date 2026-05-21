"""產出 Triple-Barrier label parquet（不打 DB schema，opt-in 實驗用）。

用法：
    # 預設配置：+15% PT / -7% SL / 20 日 time barrier
    python scripts/build_triple_barrier_labels.py

    # 自訂 barrier
    python scripts/build_triple_barrier_labels.py --upper-pt 0.20 --lower-sl -0.10 --max-horizon 30

    # 限定股票（dev / smoke）
    python scripts/build_triple_barrier_labels.py --limit-stocks 10

    # 自訂輸出
    python scripts/build_triple_barrier_labels.py --output artifacts/labels/tb_15_7_20.parquet

輸出：parquet 檔，預設 `artifacts/labels/triple_barrier.parquet`
欄位：stock_id, trading_date, tb_label, tb_return, tb_exit_type, tb_exit_day_offset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from sqlalchemy import select

from app.db import get_session
from app.models import RawPrice
from skills.build_labels import triple_barrier_labels


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--upper-pt", type=float, default=0.15, help="profit-take 上界（預設 +15%%）")
    p.add_argument("--lower-sl", type=float, default=-0.07, help="stop-loss 下界（預設 -7%%）")
    p.add_argument("--max-horizon", type=int, default=20, help="time barrier 交易日數（預設 20）")
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/labels/triple_barrier.parquet"),
                   help="輸出 parquet 路徑")
    p.add_argument("--limit-stocks", type=int, default=None,
                   help="只跑前 N 支股票（測試用）")
    p.add_argument("--start-date", type=str, default=None,
                   help="只用此日期之後的 prices（YYYY-MM-DD；預設全部）")
    p.add_argument("--stock-pattern", type=str, default=r"\d{4}",
                   help="stock_id regex（預設 4 碼，與生產 universe 一致；空字串=不過濾）")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    t0 = time.monotonic()

    print(f"=== Triple-Barrier label 產出 ===")
    print(f"  PT / SL / max_horizon: +{args.upper_pt:.2%} / {args.lower_sl:+.2%} / {args.max_horizon} 天")
    print(f"  輸出: {args.output}")

    print("  載入 raw_prices ...", end="", flush=True)
    with get_session() as s:
        q = select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close)
        if args.start_date:
            q = q.where(RawPrice.trading_date >= args.start_date)
        q = q.order_by(RawPrice.stock_id, RawPrice.trading_date)
        df = pd.read_sql(q, s.get_bind())
    print(f" {len(df):,} rows / {df['stock_id'].nunique()} stocks")

    if args.stock_pattern:
        before_stocks = df["stock_id"].nunique()
        df = df[df["stock_id"].astype(str).str.fullmatch(args.stock_pattern)]
        print(f"  pattern '{args.stock_pattern}' filter: {before_stocks} → {df['stock_id'].nunique()} stocks ({len(df):,} rows)")

    if args.limit_stocks:
        keep = sorted(df["stock_id"].unique())[: args.limit_stocks]
        df = df[df["stock_id"].isin(keep)]
        print(f"  --limit-stocks {args.limit_stocks}: filter 至 {len(df):,} rows / {df['stock_id'].nunique()} stocks")

    t_load = time.monotonic() - t0
    print(f"  計算 Triple-Barrier ...", end="", flush=True)
    t1 = time.monotonic()
    tb = triple_barrier_labels(
        df,
        upper_pt=args.upper_pt,
        lower_sl=args.lower_sl,
        max_horizon=args.max_horizon,
    )
    t_compute = time.monotonic() - t1
    print(f" {len(tb):,} labels（{t_compute:.1f}s）")

    if tb.empty:
        print("❌ 沒有 label 產出（可能資料太少或 forward window 不足）")
        return 1

    # 分佈
    print()
    print("=== Label 分佈 ===")
    dist = tb["tb_label"].value_counts().sort_index()
    total = len(tb)
    for label, cnt in dist.items():
        type_name = {-1: "stop-loss (-1)", 0: "time (0)", 1: "profit-take (+1)"}[label]
        print(f"  {type_name:<25} {cnt:>10,} ({cnt/total:.1%})")

    print()
    print("=== Return 統計 ===")
    by_type = tb.groupby("tb_exit_type")["tb_return"].describe()[["count", "mean", "std", "min", "max"]]
    print(by_type.round(4).to_string())

    print()
    print("=== Exit day offset 分佈 ===")
    print(f"  Mean: {tb['tb_exit_day_offset'].mean():.1f} days")
    print(f"  Median: {tb['tb_exit_day_offset'].median():.0f} days")
    print(f"  P25 / P75: {tb['tb_exit_day_offset'].quantile(0.25):.0f} / {tb['tb_exit_day_offset'].quantile(0.75):.0f} days")

    # 寫 parquet
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tb.to_parquet(args.output, index=False)
    elapsed = time.monotonic() - t0
    print()
    print(f"✅ 已輸出 {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"   耗時 {elapsed:.1f}s（載入 {t_load:.1f}s + 計算 {t_compute:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
