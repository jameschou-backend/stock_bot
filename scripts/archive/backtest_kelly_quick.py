"""Stage 7.3 Quick eval：用 monkey-patch 把 backtest 等權改為 Kelly weights，
跑 60mo 對照 baseline。

判定門檻：Sharpe Δ ≥ +0.10 或 MDD Δ ≥ +3pp 才進入 10y 驗證。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.config import load_config
from app.db import get_session
from skills import backtest as bt
from skills.kelly import compute_kelly_weights_for_picks


def _make_kelly_method(orig_method, half_kelly: bool, cap: float):
    """回傳替換後的 _compute_position_weights，內部呼叫 Kelly。"""
    def kelly_method(self, picks: pd.DataFrame, rb_date) -> pd.DataFrame:
        if picks.empty:
            return orig_method(self, picks, rb_date)
        try:
            sids = picks["stock_id"].astype(str).tolist()
            scores = picks["score"].astype(float).tolist() if "score" in picks.columns else [1.0] * len(sids)
            out = compute_kelly_weights_for_picks(
                pick_stock_ids=sids,
                pick_scores=scores,
                price_df=self.price_df,
                rb_date=rb_date,
                half_kelly=half_kelly,
                per_stock_cap=cap,
            )
            if out["fallback"]:
                return orig_method(self, picks, rb_date)
            return pd.DataFrame(
                [{"stock_id": s, "weight": w} for s, w in out["weights"].items()]
            )
        except Exception as exc:
            print(f"  [kelly] fallback: {exc}")
            return orig_method(self, picks, rb_date)
    return kelly_method


def run(months: int, kelly: bool, half_kelly: bool, cap: float, label: str):
    cfg = load_config()
    with get_session() as db:
        orig = bt.BacktestPipeline._compute_position_weights
        if kelly:
            bt.BacktestPipeline._compute_position_weights = _make_kelly_method(orig, half_kelly, cap)
            print(f"  [kelly] enabled (half={half_kelly}, cap={cap})")
        try:
            result = bt.run_backtest(
                cfg, db,
                backtest_months=months,
                topn=20,
                position_sizing="equal",
                enable_seasonal_filter=True,
                stoploss_pct=0.0,
                market_filter_tiers=[(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.10)],
                market_filter_min_positions=2,
                liquidity_weighting=True,
                feature_columns=bt.PRUNED_FEATURE_COLS if hasattr(bt, "PRUNED_FEATURE_COLS") else None,
                label_horizon_buffer=20,
            )
        finally:
            bt.BacktestPipeline._compute_position_weights = orig
        s = result["summary"]
        print(f"\n=== {label} ===")
        print(f"  Sharpe:   {s.get('sharpe_ratio'):.4f}")
        print(f"  MDD:      {s.get('max_drawdown'):.4f}")
        print(f"  Calmar:   {s.get('calmar_ratio'):.4f}")
        print(f"  Win:      {s.get('win_rate'):.4f}")
        # 累積從 periods
        cum = 1.0
        for p in result["periods"]:
            r = p.get("return")
            if r is not None:
                cum *= (1 + r)
        print(f"  Cum:      {cum - 1:+.2%}")
        return s, cum - 1


def main():
    months = 60
    print(f"\n[1/2] Baseline 60mo （equal weight）...")
    base_s, base_cum = run(months, kelly=False, half_kelly=False, cap=1.0, label="Baseline")
    print(f"\n[2/2] Kelly half (half_kelly=True, cap=0.10) ...")
    kelly_s, kelly_cum = run(months, kelly=True, half_kelly=True, cap=0.10, label="Kelly Half")

    print("\n" + "=" * 70)
    print(f"{'Metric':18s} {'Baseline':>14s} {'+Kelly':>14s} {'Δ':>12s}")
    print("-" * 70)
    for k in ("sharpe_ratio", "max_drawdown", "calmar_ratio", "win_rate"):
        a, b = base_s.get(k), kelly_s.get(k)
        if a is not None and b is not None:
            print(f"{k:18s} {a:>14.4f} {b:>14.4f} {b-a:>+12.4f}")
    print(f"{'cumulative':18s} {base_cum:>14.4%} {kelly_cum:>14.4%} {kelly_cum-base_cum:>+12.4%}")
    print("=" * 70)
    dS = kelly_s.get("sharpe_ratio", 0) - base_s.get("sharpe_ratio", 0)
    dMDD = kelly_s.get("max_drawdown", 0) - base_s.get("max_drawdown", 0)
    ok = dS >= 0.10 or dMDD >= 0.03
    print(f"\n  ΔSharpe = {dS:+.4f}")
    print(f"  ΔMDD    = {dMDD:+.2%} (正數 = 改善)")
    print(f"  {'✅ 有效（進入 10y 驗證）' if ok else '⚠️ 無顯著改善（不繼續）'}")


if __name__ == "__main__":
    main()
