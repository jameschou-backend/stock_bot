"""Stage 7.2 quick eval：Vol Targeting vs baseline 對 Strategy A 60mo。

不改 backtest.py，透過 monkey-patch _apply_market_regime_filter 在回傳的
cash_ratio 之上 add vol-target cash share（取較大者）。

Baseline = production cash_ratio 行為（依大盤 200MA + 5d bounce）。
Treatment = max(baseline cash_ratio, vol_target cash share)。

預期：高波動期 → vol-target 拉高 cash share → MDD 改善。

用法：
    python scripts/backtest_vol_target_quick.py --months 60 --target-vol 0.30
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
from skills.vol_targeting import compute_vol_scaler_for_picks


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=60)
    p.add_argument("--target-vol", type=float, default=0.30,
                   help="目標年化 vol（預設 30% — 台股波動性較高）")
    p.add_argument("--lookback", type=int, default=60)
    return p.parse_args()


def _patch_vol_target(target_vol: float, lookback: int):
    """Monkey-patch _apply_market_regime_filter 加 vol-target cash share。"""
    original = bt_mod.BacktestPipeline._apply_market_regime_filter

    def patched(self, day_feat, rb_date, current_topn):
        out = original(self, day_feat, rb_date, current_topn)
        if not isinstance(out, tuple) or len(out) != 4:
            return out
        filtered_df, effective_topn, cash_ratio, day_feat_empty = out
        if day_feat_empty or filtered_df is None or filtered_df.empty or effective_topn <= 0:
            return out

        # 從 filtered_df 取 top-N estimate picks
        score_col = None
        for c in ("score", "model_score", "primary_score"):
            if c in filtered_df.columns:
                score_col = c
                break
        if score_col:
            top = filtered_df.nlargest(effective_topn, score_col)
        else:
            top = filtered_df.head(effective_topn)
        pick_sids = top["stock_id"].astype(str).tolist()
        if len(pick_sids) < 2:
            return out

        try:
            vt = compute_vol_scaler_for_picks(
                pick_sids, self.price_df, rb_date,
                target_vol=target_vol, lookback_days=lookback,
            )
            extra_cash = float(vt.get("cash_share", 0.0))
            new_cash_ratio = max(cash_ratio, extra_cash)
            return filtered_df, effective_topn, new_cash_ratio, day_feat_empty
        except Exception as exc:
            # 不影響原行為
            return out

    bt_mod.BacktestPipeline._apply_market_regime_filter = patched
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
    print(f"=== Stage 7.2 Vol Targeting quick eval ===")
    print(f"  months: {args.months}, target_vol: {args.target_vol:.0%}, lookback: {args.lookback}")
    print()

    print("─── BASELINE: production cash_ratio ───")
    data_store.invalidate()
    t1 = time.monotonic()
    baseline = _run(args.months)
    print(f"  耗時 {(time.monotonic() - t1) / 60:.1f} 分")

    print()
    print("─── TREATMENT: + vol-target cash share ──")
    data_store.invalidate()
    original = _patch_vol_target(args.target_vol, args.lookback)
    try:
        t2 = time.monotonic()
        treatment = _run(args.months)
        print(f"  耗時 {(time.monotonic() - t2) / 60:.1f} 分")
    finally:
        bt_mod.BacktestPipeline._apply_market_regime_filter = original
        data_store.invalidate()

    print()
    print("=" * 70)
    print(f"{'Metric':<22} {'Baseline':<14} {'+VolTarget':<14} {'Δ'}")
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
    if delta_s >= 0.1 or delta_mdd >= 3.0:
        print(f"  ✅ 有效（Sharpe +0.1+ 或 MDD +3pp+）")
    else:
        print(f"  ⚠️ 不顯著改善")
    return 0


if __name__ == "__main__":
    sys.exit(main())
