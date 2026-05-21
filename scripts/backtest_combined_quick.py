"""Stage 5.3 局部驗證：fracdiff_0_50 + pbr_ratio + dividend_yield 組合效益。

雙重 inject：
  - 從 RawPER 表 → pbr_ratio + dividend_yield
  - 從 fracdiff_features.parquet → close_fracdiff_0_50

Baseline 用 PRUNED_FEATURE_COLS，treatment 用 PRUNED + 3 個新欄位。

用法：
    python scripts/backtest_combined_quick.py --months 60
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from sqlalchemy import select

from app.config import load_config
from app.db import get_session
from app.models import RawPER
from skills import data_store
from skills.backtest import run_backtest
from skills.build_features import PRUNED_FEATURE_COLS


PER_COLS = ["pbr_ratio", "dividend_yield"]
FRACDIFF_COL = "close_fracdiff_0_50"
NEW_FEATURES = PER_COLS + [FRACDIFF_COL]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=60)
    p.add_argument("--fracdiff-parquet", type=Path,
                   default=Path("artifacts/labels/fracdiff_features.parquet"))
    return p.parse_args()


def _load_per_df() -> pd.DataFrame:
    with get_session() as s:
        q = select(RawPER.stock_id, RawPER.trading_date, RawPER.pbr, RawPER.dividend_yield)
        df = pd.read_sql(q, s.get_bind())
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["pbr_ratio"] = pd.to_numeric(df["pbr"], errors="coerce")
    df["dividend_yield"] = pd.to_numeric(df["dividend_yield"], errors="coerce")
    return df[["stock_id", "trading_date", "pbr_ratio", "dividend_yield"]]


def _load_fracdiff_df(path: Path) -> pd.DataFrame:
    fd = pd.read_parquet(path)
    fd["trading_date"] = pd.to_datetime(fd["trading_date"]).dt.date
    return fd[["stock_id", "trading_date", FRACDIFF_COL]]


def _patch(per_df, fracdiff_df):
    original = data_store.get_features

    def patched(*args, **kwargs):
        feat = original(*args, **kwargs)
        if all(c in feat.columns for c in NEW_FEATURES):
            return feat
        feat = feat.merge(per_df, on=["stock_id", "trading_date"], how="left")
        feat = feat.merge(fracdiff_df, on=["stock_id", "trading_date"], how="left")
        return feat

    data_store.get_features = patched
    return original


def _run(months, feature_columns):
    config = load_config()
    with get_session() as s:
        result = run_backtest(
            config=config, db_session=s,
            backtest_months=months, retrain_freq_months=3, topn=20,
            stoploss_pct=0.0, transaction_cost_pct=0.001425,
            rebalance_freq="M", label_horizon_buffer=20,
            enable_slippage=False, feature_columns=feature_columns,
            enable_seasonal_filter=True,
            market_filter_tiers=[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)],
            market_filter_min_positions=2, liquidity_weighting=True,
            clip_loss_pct=-0.50,
        )
    return result.get("summary", {})


def main():
    args = _parse_args()
    print(f"=== Stage 5.3 combined: PER + fracdiff ===")
    print(f"  months: {args.months}")
    print(f"  new features: {NEW_FEATURES}")
    print()

    print("─── 載入注入資料 ──")
    per_df = _load_per_df()
    fd_df = _load_fracdiff_df(args.fracdiff_parquet)
    print(f"  PER: {len(per_df):,} rows ({per_df['pbr_ratio'].notna().sum():,} pbr non-NaN)")
    print(f"  fracdiff: {len(fd_df):,} rows")
    print()

    print("─── BASELINE: PRUNED_FEATURE_COLS ─────────────")
    t1 = time.monotonic()
    baseline = _run(args.months, PRUNED_FEATURE_COLS)
    print(f"  耗時 {(time.monotonic()-t1)/60:.1f} 分")

    print()
    print("─── TREATMENT: + PER + fracdiff ──")
    data_store.invalidate()
    original = _patch(per_df, fd_df)
    try:
        t2 = time.monotonic()
        treat = _run(args.months, PRUNED_FEATURE_COLS + NEW_FEATURES)
        print(f"  耗時 {(time.monotonic()-t2)/60:.1f} 分")
    finally:
        data_store.get_features = original
        data_store.invalidate()

    # 比對
    print()
    print("=" * 70)
    print(f"{'Metric':<22} {'Baseline':<14} {'+PER+fracdiff':<16} {'Δ'}")
    print("-" * 70)
    for k, label, unit in [
        ("total_return", "累積報酬", "%"),
        ("annualized_return", "年化報酬", "%"),
        ("max_drawdown", "MDD", "%"),
        ("sharpe_ratio", "Sharpe", ""),
        ("calmar_ratio", "Calmar", ""),
        ("win_rate", "勝率", "%"),
        ("profit_factor", "Profit factor", ""),
        ("total_trades", "Total trades", ""),
    ]:
        b = baseline.get(k); t = treat.get(k)
        if b is None or t is None:
            continue
        if unit == "%":
            print(f"{label:<22} {b*100:>+10.2f}%   {t*100:>+10.2f}%    {(t-b)*100:>+10.2f}%")
        else:
            d = t - b if isinstance(b, (int, float)) and isinstance(t, (int, float)) else 0
            print(f"{label:<22} {b:>11.3f}    {t:>11.3f}     {d:>+10.3f}")
    print("=" * 70)

    s_b = baseline.get("sharpe_ratio") or 0
    s_t = treat.get("sharpe_ratio") or 0
    delta = s_t - s_b
    print()
    print(f"ΔSharpe = {delta:+.3f}")
    print(f"Stage 2 標準: >= 0.3 → {'✅ 達標' if delta >= 0.3 else '△ 邊緣' if delta >= 0.1 else '❌ 不達'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
