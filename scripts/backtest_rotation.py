#!/usr/bin/env python3
"""Strategy C: Dynamic rotation backtest.

Daily scan of model scores. Sell when a position drops out of top percentile
or exceeds max hold days. Buy the highest-scoring stock into empty slots.

Usage:
    python scripts/backtest_rotation.py
    python scripts/backtest_rotation.py --rank-threshold 0.20 --max-hold 30
    python scripts/backtest_rotation.py --config 1  # loose
    python scripts/backtest_rotation.py --config 3  # aggressive
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import get_session
from skills import data_store

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
from sklearn.ensemble import GradientBoostingRegressor


def _train_model(train_X, train_y, fast_mode=False):
    n_est = 150 if fast_mode else 500
    if _HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=n_est, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(train_X, train_y)
    else:
        model = GradientBoostingRegressor(
            n_estimators=100 if fast_mode else 300, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42,
        )
        model.fit(train_X, train_y)
    return model


class Position:
    __slots__ = ("stock_id", "entry_date", "entry_price", "score", "days_held")

    def __init__(self, stock_id: str, entry_date: date, entry_price: float, score: float):
        self.stock_id = stock_id
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.score = score
        self.days_held = 0


def run_rotation(
    config,
    db_session,
    backtest_months: int = 120,
    max_positions: int = 6,
    rank_threshold: float = 0.20,
    max_hold_days: int = 30,
    top_entry_n: int = 10,
    retrain_freq_months: int = 3,
    label_horizon_buffer: int = 20,
    transaction_cost_pct: float = 0.003,
    fast_mode: bool = False,
    market_filter_tiers: Optional[List[tuple]] = None,
    min_hold_days: int = 0,
    force_exit_threshold: Optional[float] = None,
) -> Dict:
    """
    min_hold_days: 最小持倉天數保護。持股未滿此天數時，不因排名滑落而出場。
    force_exit_threshold: 強制出場門檻（百分位，如 0.30 = top 30%）。
        持倉保護期間，若排名掉出此門檻（比 rank_threshold 更寬鬆），仍強制出場。
        預設 None 表示使用 rank_threshold * 1.5。
    """
    if force_exit_threshold is None:
        force_exit_threshold = min(rank_threshold * 1.5, 1.0)

    print("\n" + "=" * 60)
    print("Strategy C: Dynamic Rotation Backtest")
    print("=" * 60)
    print(f"  Rank threshold: top {rank_threshold:.0%} (sell if drops out)")
    print(f"  Max hold: {max_hold_days} days")
    print(f"  Min hold: {min_hold_days} days (force exit threshold: top {force_exit_threshold:.0%})")
    print(f"  Entry: top {top_entry_n} by score")
    print(f"  Max positions: {max_positions}")
    print(f"  Transaction cost: {transaction_cost_pct:.2%} per trade")

    from sqlalchemy import func
    from app.models import Feature, Label

    max_feat_date = db_session.query(func.max(Feature.trading_date)).scalar()
    min_feat_date = db_session.query(func.min(Feature.trading_date)).scalar()
    max_label_date = db_session.query(func.max(Label.trading_date)).scalar()
    data_end = min(max_feat_date, max_label_date)
    backtest_start = data_end - timedelta(days=30 * backtest_months)
    data_start = min_feat_date

    t0 = time.time()
    price_df = data_store.get_prices(db_session, data_start, data_end)
    price_df["trading_date"] = pd.to_datetime(price_df["trading_date"]).dt.date
    price_df["stock_id"] = price_df["stock_id"].astype(str)
    for col in ["close", "volume"]:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    print(f"  Prices: {len(price_df):,} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    feat_df = data_store.get_features(db_session, data_start, data_end)
    feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
    feat_df["stock_id"] = feat_df["stock_id"].astype(str)
    print(f"  Features: {len(feat_df):,} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    label_df = data_store.get_labels(db_session, data_start, data_end)
    label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
    label_df["stock_id"] = label_df["stock_id"].astype(str)
    print(f"  Labels: {len(label_df):,} rows ({time.time()-t0:.1f}s)")

    all_dates = sorted(price_df["trading_date"].unique())
    bt_dates = [d for d in all_dates if backtest_start <= d <= data_end]
    print(f"  Trading days: {len(bt_dates)}")

    _meta_cols = {"stock_id", "trading_date", "future_ret_h"}
    feat_cols = [c for c in feat_df.columns if c not in _meta_cols]

    current_model = None
    current_feat_names = feat_cols
    last_train_date = None
    retrain_interval = timedelta(days=30 * retrain_freq_months)

    equity = 10000.0
    positions: Dict[str, Position] = {}
    trades_log: List[Dict] = []
    equity_curve: List[Dict] = []
    exit_reasons = defaultdict(int)

    # Monthly benchmark for market filter
    _monthly_bm_cache: Dict[str, float] = {}
    _current_max_pos = max_positions

    _t_start = time.time()

    for day_idx, today in enumerate(bt_dates):
        # ── Retrain ──
        need_retrain = (current_model is None or
                        (last_train_date and today - last_train_date >= retrain_interval))
        if need_retrain:
            train_cutoff = today - timedelta(days=label_horizon_buffer)
            tf = feat_df[feat_df["trading_date"] < today]
            tl = label_df[label_df["trading_date"] < train_cutoff]
            if not tf.empty and not tl.empty:
                merged = tf.merge(tl, on=["stock_id", "trading_date"], how="inner")
                if len(merged) >= 1000:
                    fmat = merged.drop(columns=[c for c in _meta_cols if c in merged.columns])
                    fmat = fmat.replace([np.inf, -np.inf], np.nan)
                    for col in fmat.columns:
                        if fmat[col].isna().all():
                            fmat[col] = 0
                        else:
                            fmat[col] = fmat[col].fillna(fmat[col].median())
                    valid = fmat.notna().all(axis=1)
                    fmat = fmat.loc[valid]
                    merged = merged.loc[fmat.index]
                    if not fmat.empty:
                        y = merged["future_ret_h"].astype(float).values
                        current_feat_names = list(fmat.columns)
                        _t = time.time()
                        current_model = _train_model(fmat.values, y, fast_mode=fast_mode)
                        last_train_date = today
                        if day_idx % 250 == 0 or day_idx < 3:
                            print(f"  [{today}] Model trained ({len(y):,} rows, {time.time()-_t:.1f}s)", flush=True)

        if current_model is None:
            continue

        # ── Today's prices ──
        tp = price_df[price_df["trading_date"] == today]
        if tp.empty:
            continue
        price_map = {str(r["stock_id"]): float(r["close"]) for _, r in tp.iterrows() if float(r["close"]) > 0}

        # ── Score all stocks ──
        tf_today = feat_df[feat_df["trading_date"] == today].copy()
        tf_today = tf_today[tf_today["stock_id"].str.fullmatch(r"\d{4}")]
        if tf_today.empty:
            equity_curve.append({"date": str(today), "equity": equity})
            continue

        fmat = tf_today.drop(columns=["stock_id", "trading_date"], errors="ignore")
        for col in current_feat_names:
            if col not in fmat.columns:
                fmat[col] = 0
        fmat = fmat[current_feat_names].replace([np.inf, -np.inf], np.nan)
        for col in fmat.columns:
            if fmat[col].isna().all():
                fmat[col] = 0
            else:
                fmat[col] = fmat[col].fillna(fmat[col].median())

        scores = current_model.predict(fmat.values)
        tf_today = tf_today.reset_index(drop=True)
        tf_today["score"] = scores

        # Only keep stocks with prices
        tf_today = tf_today[tf_today["stock_id"].isin(price_map.keys())]
        if tf_today.empty:
            equity_curve.append({"date": str(today), "equity": equity})
            continue

        # Rank threshold
        score_cutoff = tf_today["score"].quantile(1.0 - rank_threshold)
        force_cutoff = tf_today["score"].quantile(1.0 - force_exit_threshold)
        top_n_sids = set(tf_today.nlargest(top_entry_n, "score")["stock_id"].tolist())
        above_threshold = set(tf_today[tf_today["score"] >= score_cutoff]["stock_id"].tolist())
        above_force_threshold = set(tf_today[tf_today["score"] >= force_cutoff]["stock_id"].tolist())

        # ── Market filter: adjust max positions monthly ──
        _ym = f"{today.year}-{today.month:02d}"
        if _ym not in _monthly_bm_cache:
            _prev_m = (today.replace(day=1) - timedelta(days=1))
            _pm_ym = f"{_prev_m.year}-{_prev_m.month:02d}"
            if _pm_ym not in _monthly_bm_cache:
                _pm_start = _prev_m.replace(day=1)
                _pm_p = price_df[(price_df["trading_date"] >= _pm_start) & (price_df["trading_date"] <= _prev_m)]
                if not _pm_p.empty:
                    _f = _pm_p.groupby("stock_id")["close"].first()
                    _l = _pm_p.groupby("stock_id")["close"].last()
                    _v = _f.index.intersection(_l.index)
                    _v = _v[(_f[_v] > 0) & (_l[_v] > 0)]
                    _monthly_bm_cache[_pm_ym] = float((_l[_v] / _f[_v] - 1).mean()) if len(_v) > 0 else 0.0
                else:
                    _monthly_bm_cache[_pm_ym] = 0.0
            _prev_bm = _monthly_bm_cache.get(_pm_ym, 0.0)
            _current_max_pos = max_positions
            if market_filter_tiers:
                for _thr, _mult in reversed(market_filter_tiers):
                    if _prev_bm < _thr:
                        _current_max_pos = max(1, int(max_positions * _mult))
                        break
            _monthly_bm_cache[_ym] = 0.0  # placeholder

        # ── Check exits ──
        for sid in list(positions.keys()):
            pos = positions[sid]
            pos.days_held += 1
            exit_reason = None

            if pos.days_held < min_hold_days:
                # 最小持倉保護期：只有排名掉出 force_exit_threshold 才出場
                if sid not in above_force_threshold:
                    exit_reason = "Force Exit"
            else:
                # 正常出場邏輯
                if sid not in above_threshold:
                    exit_reason = "Rank Drop"
                elif pos.days_held >= max_hold_days:
                    exit_reason = "Max Hold Days"

            # Force exit if over current max positions
            if not exit_reason and len(positions) > _current_max_pos:
                # Exit the lowest-scoring held position
                held_scores = {s: tf_today[tf_today["stock_id"] == s]["score"].values for s in positions}
                held_scores = {s: float(v[0]) if len(v) > 0 else -999 for s, v in held_scores.items()}
                worst = min(held_scores, key=held_scores.get)
                if worst == sid:
                    exit_reason = "Market Filter Reduce"

            if exit_reason:
                exit_px = price_map.get(sid, pos.entry_price)
                ret = exit_px / pos.entry_price - 1 - transaction_cost_pct
                ret = max(ret, -0.50)
                trades_log.append({
                    "stock_id": sid, "entry_date": str(pos.entry_date),
                    "exit_date": str(today), "entry_price": pos.entry_price,
                    "exit_price": exit_px, "realized_pnl_pct": ret,
                    "stoploss_triggered": False, "exit_reason": exit_reason,
                    "score": pos.score, "days_held": pos.days_held,
                })
                exit_reasons[exit_reason] += 1
                weight = 1.0 / max_positions
                equity *= (1 + ret * weight)
                del positions[sid]

        # ── Check entries ──
        if len(positions) < _current_max_pos:
            candidates = tf_today[tf_today["stock_id"].isin(top_n_sids)]
            candidates = candidates[~candidates["stock_id"].isin(positions.keys())]
            candidates = candidates.sort_values("score", ascending=False)
            slots = _current_max_pos - len(positions)
            for _, cand in candidates.head(slots).iterrows():
                _sid = str(cand["stock_id"])
                if _sid in price_map:
                    positions[_sid] = Position(
                        stock_id=_sid, entry_date=today,
                        entry_price=price_map[_sid], score=float(cand["score"]),
                    )

        equity_curve.append({"date": str(today), "equity": equity})

        if day_idx % 250 == 0 and day_idx > 0:
            _el = time.time() - _t_start
            print(f"  [{today}] {day_idx/len(bt_dates)*100:.0f}% | eq: {equity:,.0f} | "
                  f"pos: {len(positions)} | trades: {len(trades_log)} | {_el:.0f}s", flush=True)

    # ── Close remaining ──
    last_date = bt_dates[-1] if bt_dates else data_end
    lp = {str(r["stock_id"]): float(r["close"]) for _, r in price_df[price_df["trading_date"] == last_date].iterrows()}
    for sid in list(positions.keys()):
        pos = positions[sid]
        exit_px = lp.get(sid, pos.entry_price)
        ret = exit_px / pos.entry_price - 1 - transaction_cost_pct
        ret = max(ret, -0.50)
        trades_log.append({
            "stock_id": sid, "entry_date": str(pos.entry_date),
            "exit_date": str(last_date), "entry_price": pos.entry_price,
            "exit_price": exit_px, "realized_pnl_pct": ret,
            "stoploss_triggered": False, "exit_reason": "End of Backtest",
            "score": pos.score, "days_held": pos.days_held,
        })
        exit_reasons["End of Backtest"] += 1
        equity *= (1 + ret * (1.0 / max_positions))

    _total = time.time() - _t_start

    # ── Stats ──
    total_return = equity / 10000.0 - 1
    n_years = len(bt_dates) / 252.0
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    eq_vals = [e["equity"] for e in equity_curve]
    peak, max_dd = eq_vals[0] if eq_vals else 10000, 0.0
    for v in eq_vals:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / peak)

    wins = sum(1 for t in trades_log if t["realized_pnl_pct"] > 0)
    win_rate = wins / max(len(trades_log), 1)
    avg_hold = np.mean([t["days_held"] for t in trades_log]) if trades_log else 0

    monthly_eq = {}
    for e in equity_curve:
        monthly_eq[e["date"][:7]] = e["equity"]
    mk = sorted(monthly_eq.keys())
    monthly_rets = []
    for i in range(1, len(mk)):
        p, c = monthly_eq[mk[i-1]], monthly_eq[mk[i]]
        if p > 0:
            monthly_rets.append(c / p - 1)
    rf_m = 0.015 / 12
    sharpe = (np.mean(monthly_rets) - rf_m) / np.std(monthly_rets) * np.sqrt(12) if monthly_rets and np.std(monthly_rets) > 0 else 0
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    gp = sum(t["realized_pnl_pct"] for t in trades_log if t["realized_pnl_pct"] > 0)
    gl = abs(sum(t["realized_pnl_pct"] for t in trades_log if t["realized_pnl_pct"] < 0))
    pf = gp / gl if gl > 0 else float("inf")

    yearly_eq = {}
    for e in equity_curve:
        yearly_eq[e["date"][:4]] = e["equity"]
    yk = sorted(yearly_eq.keys())
    yearly_rets = {}
    _pe = 10000.0
    for yr in yk:
        yearly_rets[yr] = yearly_eq[yr] / _pe - 1
        _pe = yearly_eq[yr]

    # Periods (monthly for review_pack compatibility)
    periods = []
    mt = defaultdict(list)
    for t in trades_log:
        mt[t["exit_date"][:7]].append(t)
    _pe2 = 10000.0
    for ym in sorted(monthly_eq.keys()):
        _eq = monthly_eq[ym]
        _r = _eq / _pe2 - 1 if _pe2 > 0 else 0
        _tl = mt.get(ym, [])
        periods.append({
            "rebalance_date": f"{ym}-01", "exit_date": f"{ym}-28",
            "return": _r, "benchmark_return": 0.0,
            "trades": len(_tl),
            "stock_returns": {t["stock_id"]: t["realized_pnl_pct"] for t in _tl},
            "stoploss_triggered": 0,
        })
        _pe2 = _eq

    total_cost = len(trades_log) * transaction_cost_pct
    cost_drag = total_cost / max(n_years, 1)

    print(f"\n{'='*60}")
    print(f"Strategy C Results (rank>{rank_threshold:.0%}, hold<={max_hold_days}d)")
    print(f"{'='*60}")
    print(f"  Period: {bt_dates[0]} ~ {bt_dates[-1]} ({_total:.0f}s)")
    print(f"  Total Return:  {total_return:+.2%}")
    print(f"  Annualized:    {ann_return:+.2%}")
    print(f"  MDD:           {max_dd:.2%}")
    print(f"  Sharpe:        {sharpe:.4f}")
    print(f"  Calmar:        {calmar:.4f}")
    print(f"  Win Rate:      {win_rate:.2%}")
    print(f"  Profit Factor: {pf:.3f}")
    print(f"  Total Trades:  {len(trades_log)}")
    print(f"  Avg Hold Days: {avg_hold:.1f}")
    print(f"  Cost Drag/yr:  {cost_drag:.2%}")
    print(f"\n  Exit Reasons:")
    for r, c in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        print(f"    {r}: {c} ({c/max(len(trades_log),1)*100:.1f}%)")
    print(f"\n  Yearly Returns:")
    for yr, r in sorted(yearly_rets.items()):
        print(f"    {yr}: {r:+.2%}")

    summary = {
        "total_return": total_return, "annualized_return": ann_return,
        "benchmark_total_return": 0.0, "max_drawdown": max_dd,
        "sharpe_ratio": sharpe, "calmar_ratio": calmar,
        "win_rate": win_rate, "profit_factor": pf,
        "total_trades": len(trades_log), "total_periods": len(periods),
        "stoploss_triggered": 0,
        "backtest_start": str(bt_dates[0]) if bt_dates else "",
        "backtest_end": str(bt_dates[-1]) if bt_dates else "",
        "avg_hold_days": avg_hold, "exit_reasons": dict(exit_reasons),
        "yearly_returns": yearly_rets, "cost_drag_annual": cost_drag,
        "config": {
            "strategy": "C_rotation", "max_positions": max_positions,
            "rank_threshold": rank_threshold, "max_hold_days": max_hold_days,
            "min_hold_days": min_hold_days, "force_exit_threshold": force_exit_threshold,
            "top_entry_n": top_entry_n, "transaction_cost_pct": transaction_cost_pct,
        },
    }
    return {"summary": summary, "periods": periods, "equity_curve": equity_curve, "trades_log": trades_log}


def main():
    parser = argparse.ArgumentParser(description="Strategy C: Dynamic Rotation")
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--rank-threshold", type=float, default=None)
    parser.add_argument("--max-hold", type=int, default=None)
    parser.add_argument("--min-hold", type=int, default=0,
                        help="最小持倉天數保護（預設 0=停用）")
    parser.add_argument("--force-exit-threshold", type=float, default=None,
                        help="保護期間強制出場門檻，如 0.30=top30%（預設=rank_threshold×1.5）")
    parser.add_argument("--config", type=int, default=None, choices=[1, 2, 3],
                        help="Preset: 1=loose, 2=moderate, 3=aggressive")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    presets = {
        1: (0.30, 45, "loose"),
        2: (0.20, 30, "moderate"),
        3: (0.15, 20, "aggressive"),
    }

    if args.config:
        rank_thr, max_hold, label = presets[args.config]
    else:
        rank_thr = args.rank_threshold or 0.20
        max_hold = args.max_hold or 30

    config = load_config()
    mf_tiers = [(-0.05, 0.5), (-0.10, 0.33), (-0.15, 0.17)]

    with get_session() as session:
        result = run_rotation(
            config=config, db_session=session,
            backtest_months=args.months,
            rank_threshold=rank_thr, max_hold_days=max_hold,
            fast_mode=args.fast, market_filter_tiers=mf_tiers,
            min_hold_days=args.min_hold,
            force_exit_threshold=args.force_exit_threshold,
        )

    output_path = args.output
    if output_path is None:
        from datetime import datetime
        d = ROOT / "artifacts" / "backtest"
        d.mkdir(parents=True, exist_ok=True)
        output_path = str(d / f"rotation_c{args.config or 2}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
