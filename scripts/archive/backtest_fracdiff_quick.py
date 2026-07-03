"""Stage 5.3 局部驗證：fracdiff_0_50 加入 production model 後 Sharpe 變化。

策略：不改 build_features.py。透過 monkey-patch `data_store.get_features`
注入 fracdiff_features.parquet 的 close_fracdiff_0_50 欄位，並把
PRUNED_FEATURE_COLS + ['close_fracdiff_0_50'] 當訓練特徵集。

對照組：純 PRUNED_FEATURE_COLS（既有生產配置）。

用法：
    # 12 個月 smoke
    python scripts/backtest_fracdiff_quick.py --months 12

    # 完整 10 年 walk-forward
    python scripts/backtest_fracdiff_quick.py --months 120

兩條都會跑，輸出對照表。所有其他 backtest 參數沿用生產配置：
  --seasonal-filter --no-stoploss --liq-weighted
  --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from app.config import load_config
from app.db import get_session
from skills import data_store
from skills.backtest import run_backtest
from skills.build_features import PRUNED_FEATURE_COLS


FRACDIFF_COL = "close_fracdiff_0_50"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=12,
                   help="walk-forward 期數（生產配置月頻，12=1年，120=10年）")
    p.add_argument("--fracdiff-parquet", type=Path,
                   default=Path("artifacts/labels/fracdiff_features.parquet"))
    return p.parse_args()


def _patch_get_features(fracdiff_path: Path) -> None:
    """Monkey-patch data_store.get_features 注入 fracdiff 欄位。"""
    print(f"  載入 fracdiff: {fracdiff_path} ...", end="", flush=True)
    fd = pd.read_parquet(fracdiff_path)
    fd["trading_date"] = pd.to_datetime(fd["trading_date"]).dt.date
    # 只留 fracdiff_0_50（避免一次塞太多）
    keep_cols = ["stock_id", "trading_date", FRACDIFF_COL]
    fd = fd[keep_cols]
    print(f" {len(fd):,} rows")

    original = data_store.get_features

    def patched(*args, **kwargs):
        feat = original(*args, **kwargs)
        if FRACDIFF_COL in feat.columns:
            return feat  # 已 merge 過
        feat = feat.merge(fd, on=["stock_id", "trading_date"], how="left")
        return feat

    data_store.get_features = patched
    return original


def _run_backtest_inline(months: int, feature_columns) -> dict:
    """跑 walk-forward backtest，回傳 summary dict。

    全部沿用 scripts/run_backtest.py 的生產配置：
      --seasonal-filter --no-stoploss --liq-weighted
      --market-filter-tiers="-0.05:0.5,-0.10:0.25,-0.15:0.10" --market-filter-min-pos 2
    """
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
    print(f"{'Metric':<25} {'Baseline (pruned)':<22} {'+fracdiff_0_50':<22} {'Δ'}")
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
            b_s = f"{b * 100:+8.2f}%" if isinstance(b, float) and abs(b) < 100 else f"{b:+10.2f}"
            t_s = f"{t * 100:+8.2f}%" if isinstance(t, float) and abs(t) < 100 else f"{t:+10.2f}"
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
    print(f"=== Stage 5.3 quick eval: fracdiff_0_50 加入 production ===")
    print(f"  months: {args.months}")
    print(f"  fracdiff: {args.fracdiff_parquet}")
    print()

    print("─── BASELINE: PRUNED_FEATURE_COLS ─────────────")
    t0 = time.monotonic()
    baseline = _run_backtest_inline(args.months, PRUNED_FEATURE_COLS)
    print(f"  耗時 {(time.monotonic() - t0) / 60:.1f} 分")

    print()
    print("─── TREATMENT: PRUNED_FEATURE_COLS + fracdiff_0_50 ──")
    # Patch + invalidate cache（讓下次 get_features 重新跑 patched 版本）
    data_store.invalidate()
    original = _patch_get_features(args.fracdiff_parquet)
    try:
        treatment_features = PRUNED_FEATURE_COLS + [FRACDIFF_COL]
        t1 = time.monotonic()
        treatment = _run_backtest_inline(args.months, treatment_features)
        print(f"  耗時 {(time.monotonic() - t1) / 60:.1f} 分")
    finally:
        data_store.get_features = original
        data_store.invalidate()

    _print_compare(baseline, treatment)

    # Stage 2 判定門檻
    s_b = baseline.get("sharpe_ratio") or 0
    s_t = treatment.get("sharpe_ratio") or 0
    delta_sr = s_t - s_b
    print()
    print(f"Stage 2 判定門檻：Sharpe 提升 >= 0.3 + DSR p > 0.95")
    print(f"  ΔSharpe = {delta_sr:+.3f}")
    if delta_sr >= 0.3:
        print(f"  ✅ 達標，建議 Stage 5.3 完整整合 fracdiff 進 build_features")
    elif delta_sr >= 0.1:
        print(f"  △ 邊緣（+{delta_sr:.2f}）：12-month smoke 不確定，建議跑 --months 120 確認")
    else:
        print(f"  ❌ 不達標（+{delta_sr:.2f}），fracdiff 對 Strategy A 效益不足")
    return 0


if __name__ == "__main__":
    sys.exit(main())
