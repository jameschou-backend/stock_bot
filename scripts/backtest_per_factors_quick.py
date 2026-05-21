"""Stage 5.3 局部驗證：把 PER value factors (pbr_ratio + dividend_yield) 加入
production model 後 Sharpe 變化。

不改 build_features.py，透過 monkey-patch `data_store.get_features` 注入
RawPER 表的兩個欄位。Baseline 用 PRUNED_FEATURE_COLS，treatment 用
PRUNED_FEATURE_COLS + ['pbr_ratio', 'dividend_yield']。

用法：
    # 60 個月（5 年）對比
    python scripts/backtest_per_factors_quick.py --months 60

    # 10 年 walk-forward
    python scripts/backtest_per_factors_quick.py --months 120
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


INJECT_COLS = ["pbr_ratio", "dividend_yield"]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=60)
    return p.parse_args()


def _load_per_df() -> pd.DataFrame:
    """從 raw_per 表載入 pbr 跟 dividend_yield，rename 對齊 FEATURE_COLUMNS 命名。"""
    with get_session() as s:
        q = select(RawPER.stock_id, RawPER.trading_date, RawPER.pbr, RawPER.dividend_yield)
        df = pd.read_sql(q, s.get_bind())
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["pbr_ratio"] = pd.to_numeric(df["pbr"], errors="coerce")
    df["dividend_yield"] = pd.to_numeric(df["dividend_yield"], errors="coerce")
    return df[["stock_id", "trading_date", "pbr_ratio", "dividend_yield"]]


def _patch_get_features(per_df: pd.DataFrame):
    """Monkey-patch data_store.get_features 注入 pbr_ratio + dividend_yield。"""
    original = data_store.get_features

    def patched(*args, **kwargs):
        feat = original(*args, **kwargs)
        if all(c in feat.columns for c in INJECT_COLS):
            return feat
        feat = feat.merge(per_df, on=["stock_id", "trading_date"], how="left")
        return feat

    data_store.get_features = patched
    return original


def _run(months: int, feature_columns) -> dict:
    config = load_config()
    with get_session() as s:
        result = run_backtest(
            config=config,
            db_session=s,
            backtest_months=months,
            retrain_freq_months=3,
            topn=20,
            stoploss_pct=0.0,
            transaction_cost_pct=0.001425,
            rebalance_freq="M",
            label_horizon_buffer=20,
            enable_slippage=False,
            feature_columns=feature_columns,
            enable_seasonal_filter=True,
            market_filter_tiers=[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)],
            market_filter_min_positions=2,
            liquidity_weighting=True,
            clip_loss_pct=-0.50,
        )
    return result.get("summary", {})


def _print_compare(baseline: dict, treatment: dict) -> None:
    print()
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline (pruned)':<22} {'+pbr+div_yield':<22} {'Δ'}")
    print("-" * 70)
    keys = [
        ("total_return", "累積報酬", "%"),
        ("annualized_return", "年化報酬", "%"),
        ("benchmark_total_return", "大盤累積", "%"),
        ("max_drawdown", "MDD", "%"),
        ("sharpe_ratio", "Sharpe", ""),
        ("calmar_ratio", "Calmar", ""),
        ("win_rate", "勝率", "%"),
        ("profit_factor", "Profit factor", ""),
        ("total_trades", "Total trades", ""),
        ("stoploss_triggered", "Stoploss triggered", ""),
    ]
    for k, label, unit in keys:
        b = baseline.get(k)
        t = treatment.get(k)
        if b is None or t is None:
            print(f"{label:<25} N/A")
            continue
        if unit == "%":
            b_s = f"{b * 100:+8.2f}%" if isinstance(b, float) else str(b)
            t_s = f"{t * 100:+8.2f}%" if isinstance(t, float) else str(t)
            delta = t - b
            delta_s = f"{delta * 100:+8.3f}%"
        else:
            b_s = f"{b:>10.3f}" if isinstance(b, float) else f"{b:>10}"
            t_s = f"{t:>10.3f}" if isinstance(t, float) else f"{t:>10}"
            delta = (t - b) if isinstance(b, (int, float)) and isinstance(t, (int, float)) else None
            delta_s = f"{delta:+10.3f}" if isinstance(delta, (int, float)) else "N/A"
        print(f"{label:<25} {b_s:<22} {t_s:<22} {delta_s}")
    print("=" * 70)


def main() -> int:
    args = _parse_args()
    print(f"=== Stage 5.3 quick eval: pbr_ratio + dividend_yield ===")
    print(f"  months: {args.months}")
    print()

    print("─── 載入 raw_per 注入資料 ──")
    t0 = time.monotonic()
    per_df = _load_per_df()
    n_pbr = per_df["pbr_ratio"].notna().sum()
    n_yield = per_df["dividend_yield"].notna().sum()
    print(f"  raw_per: {len(per_df):,} rows  pbr_ratio non-NaN: {n_pbr:,}  div_yield non-NaN: {n_yield:,}")
    print(f"  日期範圍: {per_df['trading_date'].min()} ~ {per_df['trading_date'].max()}")
    print()

    print("─── BASELINE: PRUNED_FEATURE_COLS ─────────────")
    t1 = time.monotonic()
    baseline = _run(args.months, PRUNED_FEATURE_COLS)
    print(f"  耗時 {(time.monotonic() - t1) / 60:.1f} 分")

    print()
    print("─── TREATMENT: + pbr_ratio + dividend_yield ──")
    data_store.invalidate()
    original = _patch_get_features(per_df)
    try:
        treat_cols = PRUNED_FEATURE_COLS + INJECT_COLS
        t2 = time.monotonic()
        treatment = _run(args.months, treat_cols)
        print(f"  耗時 {(time.monotonic() - t2) / 60:.1f} 分")
    finally:
        data_store.get_features = original
        data_store.invalidate()

    _print_compare(baseline, treatment)

    s_b = baseline.get("sharpe_ratio") or 0
    s_t = treatment.get("sharpe_ratio") or 0
    delta_sr = s_t - s_b
    print()
    print(f"Stage 2 判定門檻：Sharpe 提升 >= 0.3 + DSR p > 0.95")
    print(f"  ΔSharpe = {delta_sr:+.3f}")
    if delta_sr >= 0.3:
        print(f"  ✅ 達標，建議從 _PRUNE_SET 永久移除這 2 個並重跑 production training")
    elif delta_sr >= 0.1:
        print(f"  △ 邊緣，建議跑 --months 120 確認")
    else:
        print(f"  ❌ 不達標（{delta_sr:+.2f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
