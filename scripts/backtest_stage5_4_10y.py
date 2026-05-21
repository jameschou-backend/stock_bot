"""Stage 5.4 10y WF 對照：features parquet 已 enrich（含 pbr_ratio /
dividend_yield / close_fracdiff_0_50 真實值），對比有無這 3 個新欄位。

baseline：手動排除 3 個欄位
treatment：完整 PRUNED_FEATURE_COLS（已含 3 個）

用法：
    python scripts/backtest_stage5_4_10y.py --months 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from app.config import load_config
from app.db import get_session
from skills import data_store
from skills.backtest import run_backtest
from skills.build_features import PRUNED_FEATURE_COLS


NEW_FEATURES = {"pbr_ratio", "dividend_yield", "close_fracdiff_0_50"}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=120)
    return p.parse_args()


def _run(months: int, feature_columns: list) -> dict:
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


def main() -> int:
    args = _parse_args()
    print(f"=== Stage 5.4 10y WF: PER + fracdiff 完整整合 ===")
    print(f"  months: {args.months}")
    print(f"  treatment features: {len(PRUNED_FEATURE_COLS)}")
    print(f"  new features: {sorted(NEW_FEATURES)}")
    print()

    baseline_features = [f for f in PRUNED_FEATURE_COLS if f not in NEW_FEATURES]
    treatment_features = list(PRUNED_FEATURE_COLS)
    print(f"  baseline feature count: {len(baseline_features)}")
    print(f"  treatment feature count: {len(treatment_features)}")
    print()

    print("─── BASELINE: PRUNED_FEATURE_COLS 排除新 3 欄 ───────")
    data_store.invalidate()
    t1 = time.monotonic()
    baseline = _run(args.months, baseline_features)
    print(f"  耗時 {(time.monotonic() - t1) / 60:.1f} 分")

    print()
    print("─── TREATMENT: PRUNED_FEATURE_COLS 全集（含新 3 欄）───")
    data_store.invalidate()
    t2 = time.monotonic()
    treat = _run(args.months, treatment_features)
    print(f"  耗時 {(time.monotonic() - t2) / 60:.1f} 分")

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
    if delta >= 0.3:
        print(f"  ✅ 達 Stage 2 標準（+0.3），整合進 production")
    elif delta >= 0.1:
        print(f"  △ 邊緣達標（+0.1），實質 alpha 但低於 +0.3 嚴格門檻")
    elif delta >= 0:
        print(f"  ⚠️ 微弱正改善，可能 noise")
    else:
        print(f"  ❌ 負效益")
    return 0


if __name__ == "__main__":
    sys.exit(main())
