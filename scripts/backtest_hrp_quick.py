"""Stage 7.1 quick eval：HRP vs equal-weight 倉位管理對 Strategy A 60mo。

不改 backtest.py，透過 monkey-patch BacktestPipeline._compute_position_weights
注入 HRP weight 邏輯。Baseline = position_sizing='equal'，
Treatment = HRP weights based on 60-day return correlation in picks.

用法：
    python scripts/backtest_hrp_quick.py --months 60
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

from app.config import load_config
from app.db import get_session
from skills import backtest as bt_mod
from skills import data_store
from skills.backtest import run_backtest
from skills.build_features import PRUNED_FEATURE_COLS
from skills.hrp import hrp_weights_for_picks


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=60)
    p.add_argument("--lookback", type=int, default=60,
                   help="HRP correlation lookback 天數")
    return p.parse_args()


def _patch_hrp_weights(lookback: int):
    """Monkey-patch BacktestPipeline._compute_position_weights 用 HRP。"""
    original = bt_mod.BacktestPipeline._compute_position_weights

    def patched(self, picks: pd.DataFrame, rb_date) -> pd.DataFrame:
        pick_sids = picks["stock_id"].astype(str).tolist()
        if len(pick_sids) <= 1:
            return original(self, picks, rb_date)
        weights_map = hrp_weights_for_picks(
            pick_sids, self.price_df, rb_date, lookback_days=lookback,
        )
        return pd.DataFrame(
            [{"stock_id": sid, "weight": float(weights_map.get(sid, 0))} for sid in pick_sids]
        )

    bt_mod.BacktestPipeline._compute_position_weights = patched
    return original


def _run(months: int) -> dict:
    config = load_config()
    with get_session() as s:
        result = run_backtest(
            config=config, db_session=s,
            backtest_months=months, retrain_freq_months=3, topn=20,
            stoploss_pct=0.0, transaction_cost_pct=0.001425,
            rebalance_freq="M", label_horizon_buffer=20,
            enable_slippage=False, feature_columns=list(PRUNED_FEATURE_COLS),
            enable_seasonal_filter=True,
            market_filter_tiers=[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)],
            market_filter_min_positions=2, liquidity_weighting=True,
            clip_loss_pct=-0.50,
        )
    return result.get("summary", {})


def main() -> int:
    args = _parse_args()
    print(f"=== Stage 7.1 HRP quick eval ===")
    print(f"  months: {args.months}, HRP lookback: {args.lookback}")
    print()

    print("─── BASELINE: equal weight ───────")
    data_store.invalidate()
    t1 = time.monotonic()
    baseline = _run(args.months)
    print(f"  耗時 {(time.monotonic() - t1) / 60:.1f} 分")

    print()
    print("─── TREATMENT: HRP weights ──────")
    data_store.invalidate()
    original = _patch_hrp_weights(args.lookback)
    try:
        t2 = time.monotonic()
        treatment = _run(args.months)
        print(f"  耗時 {(time.monotonic() - t2) / 60:.1f} 分")
    finally:
        bt_mod.BacktestPipeline._compute_position_weights = original
        data_store.invalidate()

    print()
    print("=" * 70)
    print(f"{'Metric':<22} {'Equal weight':<14} {'HRP':<14} {'Δ'}")
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
        b = baseline.get(k); t = treatment.get(k)
        if b is None or t is None:
            continue
        if unit == "%":
            print(f"{label:<22} {b*100:>+10.2f}%   {t*100:>+10.2f}%    {(t-b)*100:>+10.2f}%")
        else:
            d = t - b if isinstance(b, (int, float)) and isinstance(t, (int, float)) else 0
            print(f"{label:<22} {b:>11.3f}    {t:>11.3f}     {d:>+10.3f}")
    print("=" * 70)

    s_b = baseline.get("sharpe_ratio") or 0
    s_t = treatment.get("sharpe_ratio") or 0
    delta_s = s_t - s_b
    mdd_b = baseline.get("max_drawdown") or 0
    mdd_t = treatment.get("max_drawdown") or 0
    delta_mdd = (mdd_t - mdd_b) * 100
    print()
    print(f"  ΔSharpe = {delta_s:+.3f}")
    print(f"  ΔMDD    = {delta_mdd:+.2f}pp（正數 = 改善）")
    if delta_s >= 0.3:
        print(f"  ✅ Sharpe 達 +0.3 標準")
    elif delta_s >= 0.1:
        print(f"  △ Sharpe 邊緣（+{delta_s:.2f}）")
    if delta_mdd >= 3.0:
        print(f"  ✅ MDD 改善 >= 3pp，HRP 達 risk-defensive 預期")
    elif delta_mdd >= 1.0:
        print(f"  △ MDD 微改善（+{delta_mdd:.1f}pp）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
