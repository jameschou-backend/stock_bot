#!/usr/bin/env python3
"""Strategy B: Daily-frequency backtest with individual position management.

Unlike Strategy A (monthly rebalance), Strategy B scans daily for entry signals
and manages each position independently with per-stock exit rules.

Usage:
    python scripts/backtest_daily.py
    python scripts/backtest_daily.py --fast
    python scripts/backtest_daily.py --output artifacts/backtest/strategy_b.json
"""
from __future__ import annotations

import argparse
import gc
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


# ── Model training (reuse same logic as backtest.py) ──

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


# ── Position class ──

class Position:
    __slots__ = ("stock_id", "entry_date", "entry_price", "score",
                 "days_held", "peak_price", "foreign_sell_streak")

    def __init__(self, stock_id: str, entry_date: date, entry_price: float, score: float):
        self.stock_id = stock_id
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.score = score
        self.days_held = 0
        self.peak_price = entry_price
        self.foreign_sell_streak = 0


# ── Main backtest ──

def run_strategy_b(
    config,
    db_session,
    backtest_months: int = 120,
    max_positions: int = 6,
    position_pct: float = 0.15,
    max_hold_days: int = 20,
    stoploss_pct: float = -0.10,
    rsi_exit: float = 80.0,
    foreign_sell_exit_days: int = 3,
    rsi_entry_min: float = 45.0,
    rsi_entry_max: float = 70.0,
    volume_surge_mult: float = 1.2,
    score_percentile: float = 0.80,
    retrain_freq_months: int = 3,
    label_horizon_buffer: int = 20,
    transaction_cost_pct: float = 0.00584,
    fast_mode: bool = False,
    market_filter_tiers: Optional[List[tuple]] = None,
) -> Dict:

    print("\n" + "=" * 60)
    print("Strategy B: Daily Frequency Backtest")
    print("=" * 60)

    # ── 1. Load data ──
    from sqlalchemy import func
    from app.models import Feature, Label

    max_feat_date = db_session.query(func.max(Feature.trading_date)).scalar()
    min_feat_date = db_session.query(func.min(Feature.trading_date)).scalar()
    max_label_date = db_session.query(func.max(Label.trading_date)).scalar()

    data_end = min(max_feat_date, max_label_date)
    backtest_start = data_end - timedelta(days=30 * backtest_months)
    data_start = min_feat_date

    print(f"  Data range: {data_start} ~ {data_end}")
    print(f"  Backtest: {backtest_start} ~ {data_end}")
    print(f"  Max positions: {max_positions}, hold limit: {max_hold_days}d")
    print(f"  Entry: score top {score_percentile:.0%}, RSI {rsi_entry_min}-{rsi_entry_max}, vol surge {volume_surge_mult}x")
    print(f"  Exit: stoploss {stoploss_pct:.0%}, RSI>{rsi_exit}, foreign sell {foreign_sell_exit_days}d, max {max_hold_days}d")

    t0 = time.time()
    price_df = data_store.get_prices(db_session, data_start, data_end)
    price_df["trading_date"] = pd.to_datetime(price_df["trading_date"]).dt.date
    price_df["stock_id"] = price_df["stock_id"].astype(str)
    for col in ["open", "high", "low", "close", "volume"]:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
    print(f"  Prices loaded: {len(price_df):,} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    feat_df = data_store.get_features(db_session, data_start, data_end)
    feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
    feat_df["stock_id"] = feat_df["stock_id"].astype(str)
    print(f"  Features loaded: {len(feat_df):,} rows ({time.time()-t0:.1f}s)")

    t0 = time.time()
    label_df = data_store.get_labels(db_session, data_start, data_end)
    label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
    label_df["stock_id"] = label_df["stock_id"].astype(str)
    print(f"  Labels loaded: {len(label_df):,} rows ({time.time()-t0:.1f}s)")

    # ── 2. Precompute daily foreign net buy ──
    # Use foreign_net_5 from features as proxy; also need raw institutional data
    # For simplicity, derive foreign_buy from features: foreign_net_5 > 0 as daily proxy
    # (In production, would use raw institutional data for exact daily values)

    # ── 3. Precompute volume moving average ──
    price_df = price_df.sort_values(["stock_id", "trading_date"])
    price_df["vol_ma20"] = price_df.groupby("stock_id")["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )

    # ── 4. Get trading dates for backtest period ──
    all_dates = sorted(price_df["trading_date"].unique())
    bt_dates = [d for d in all_dates if d >= backtest_start and d <= data_end]
    print(f"  Trading days in backtest: {len(bt_dates)}")

    # ── 5. Walk-forward training setup ──
    _meta_cols = {"stock_id", "trading_date", "future_ret_h"}
    feat_cols = [c for c in feat_df.columns if c not in _meta_cols]

    current_model = None
    current_feat_names = feat_cols
    last_train_date = None
    retrain_interval = timedelta(days=30 * retrain_freq_months)

    # ── 6. Simulation state ──
    equity = 10000.0
    positions: Dict[str, Position] = {}  # stock_id -> Position
    trades_log: List[Dict] = []
    equity_curve: List[Dict] = []
    daily_returns: List[float] = []

    # Monthly benchmark tracking
    monthly_bm: Dict[str, float] = {}  # "YYYY-MM" -> return
    _prev_month_bm = 0.0
    _market_pause = False  # pause new entries
    _market_exit_all = False  # force exit all

    # Exit reason counters
    exit_reasons = defaultdict(int)

    print(f"\n  Starting simulation...", flush=True)
    _t_start = time.time()

    for day_idx, today in enumerate(bt_dates):
        # ── 6a. Retrain model if needed ──
        need_retrain = (current_model is None or
                        (last_train_date is not None and today - last_train_date >= retrain_interval))

        if need_retrain:
            train_cutoff = today - timedelta(days=label_horizon_buffer)
            train_feat = feat_df[feat_df["trading_date"] < today]
            train_label = label_df[label_df["trading_date"] < train_cutoff]

            if not train_feat.empty and not train_label.empty:
                merged = train_feat.merge(train_label, on=["stock_id", "trading_date"], how="inner")
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
                        if day_idx % 60 == 0 or day_idx < 5:
                            print(f"  [{today}] Model trained ({len(y):,} rows, {time.time()-_t:.1f}s)", flush=True)

        if current_model is None:
            continue

        # ── 6b. Get today's prices ──
        today_prices = price_df[price_df["trading_date"] == today]
        if today_prices.empty:
            continue

        today_price_map = {str(r["stock_id"]): r for _, r in today_prices.iterrows()}

        # ── 6c. Monthly benchmark calculation (for market filter) ──
        _ym = f"{today.year}-{today.month:02d}"
        if _ym not in monthly_bm:
            # Calculate previous month's benchmark return
            _prev_month = (today.replace(day=1) - timedelta(days=1))
            _pm_ym = f"{_prev_month.year}-{_prev_month.month:02d}"
            if _pm_ym not in monthly_bm:
                # Compute: all-stock equal-weight return for prev month
                _pm_start = _prev_month.replace(day=1)
                _pm_prices = price_df[
                    (price_df["trading_date"] >= _pm_start) &
                    (price_df["trading_date"] <= _prev_month)
                ]
                if not _pm_prices.empty:
                    _first = _pm_prices.groupby("stock_id")["close"].first()
                    _last = _pm_prices.groupby("stock_id")["close"].last()
                    _valid = _first.index.intersection(_last.index)
                    _valid = _valid[(_first[_valid] > 0) & (_last[_valid] > 0)]
                    if len(_valid) > 0:
                        _bm_rets = _last[_valid] / _first[_valid] - 1
                        monthly_bm[_pm_ym] = float(_bm_rets.mean())
                    else:
                        monthly_bm[_pm_ym] = 0.0
                else:
                    monthly_bm[_pm_ym] = 0.0

            _prev_month_bm = monthly_bm.get(_pm_ym, 0.0)

            # Apply market filter
            _market_pause = False
            _market_exit_all = False
            if market_filter_tiers:
                for _thr, _mult in reversed(market_filter_tiers):
                    if _prev_month_bm < _thr:
                        if _mult <= 0.1:
                            _market_exit_all = True
                            _market_pause = True
                        else:
                            _market_pause = True
                        break
            monthly_bm[_ym] = 0.0  # placeholder

        # ── 6d. Force exit all if market crash ──
        if _market_exit_all and positions:
            for sid in list(positions.keys()):
                pos = positions[sid]
                if sid in today_price_map:
                    exit_px = float(today_price_map[sid]["close"])
                    ret = exit_px / pos.entry_price - 1 - transaction_cost_pct
                    ret = max(ret, -0.50)
                    trades_log.append({
                        "stock_id": sid, "entry_date": str(pos.entry_date),
                        "exit_date": str(today), "entry_price": pos.entry_price,
                        "exit_price": exit_px, "realized_pnl_pct": ret,
                        "stoploss_triggered": False, "exit_reason": "Market Exit All",
                        "score": pos.score, "days_held": pos.days_held,
                    })
                    exit_reasons["market_exit"] += 1
                    equity *= (1 + ret * position_pct)
                    del positions[sid]
            _market_exit_all = False

        # ── 6e. Check exit conditions for existing positions ──
        for sid in list(positions.keys()):
            pos = positions[sid]
            if sid not in today_price_map:
                continue

            row = today_price_map[sid]
            close = float(row["close"])
            pos.days_held += 1
            pos.peak_price = max(pos.peak_price, close)

            # Get today's features for this stock
            _sf = feat_df[(feat_df["stock_id"] == sid) & (feat_df["trading_date"] == today)]
            _rsi = float(_sf["rsi_14"].iloc[0]) if not _sf.empty and "rsi_14" in _sf.columns else 50.0
            _fnet = float(_sf["foreign_net_5"].iloc[0]) if not _sf.empty and "foreign_net_5" in _sf.columns else 0.0

            # Track foreign sell streak
            if _fnet < 0:
                pos.foreign_sell_streak += 1
            else:
                pos.foreign_sell_streak = 0

            # Check exit conditions
            ret_unrealized = close / pos.entry_price - 1
            exit_reason = None

            if pos.days_held >= max_hold_days:
                exit_reason = "Max Hold Days"
            elif pos.foreign_sell_streak >= foreign_sell_exit_days:
                exit_reason = "Foreign Sell Streak"
            elif _rsi > rsi_exit:
                exit_reason = "RSI Overbought"
            elif ret_unrealized <= stoploss_pct:
                exit_reason = "Stop Loss"

            if exit_reason:
                ret = close / pos.entry_price - 1 - transaction_cost_pct
                ret = max(ret, -0.50)
                trades_log.append({
                    "stock_id": sid, "entry_date": str(pos.entry_date),
                    "exit_date": str(today), "entry_price": pos.entry_price,
                    "exit_price": close, "realized_pnl_pct": ret,
                    "stoploss_triggered": exit_reason == "Stop Loss",
                    "exit_reason": exit_reason, "score": pos.score,
                    "days_held": pos.days_held,
                })
                exit_reasons[exit_reason] += 1
                equity *= (1 + ret * position_pct)
                del positions[sid]

        # ── 6f. Check entry conditions (if not paused and have slots) ──
        if not _market_pause and len(positions) < max_positions:
            today_feat = feat_df[feat_df["trading_date"] == today].copy()
            today_feat = today_feat[today_feat["stock_id"].str.fullmatch(r"\d{4}")]

            if not today_feat.empty and current_model is not None:
                # Score all stocks
                fmat = today_feat.drop(columns=["stock_id", "trading_date"], errors="ignore")
                for col in current_feat_names:
                    if col not in fmat.columns:
                        fmat[col] = 0
                fmat = fmat[current_feat_names]
                fmat = fmat.replace([np.inf, -np.inf], np.nan)
                for col in fmat.columns:
                    if fmat[col].isna().all():
                        fmat[col] = 0
                    else:
                        fmat[col] = fmat[col].fillna(fmat[col].median())

                scores = current_model.predict(fmat.values)
                today_feat = today_feat.reset_index(drop=True)
                today_feat["score"] = scores

                # Filter: top percentile by score
                _threshold = today_feat["score"].quantile(score_percentile)
                candidates = today_feat[today_feat["score"] >= _threshold].copy()

                # Filter: RSI in range
                if "rsi_14" in candidates.columns:
                    candidates = candidates[
                        (candidates["rsi_14"] >= rsi_entry_min) &
                        (candidates["rsi_14"] <= rsi_entry_max)
                    ]

                # Filter: volume surge
                if not candidates.empty:
                    _vol_candidates = []
                    for _, cand in candidates.iterrows():
                        _sid = str(cand["stock_id"])
                        if _sid in today_price_map:
                            _vol = float(today_price_map[_sid]["volume"])
                            _vol_ma = float(today_price_map[_sid].get("vol_ma20", 0))
                            if _vol_ma > 0 and _vol >= _vol_ma * volume_surge_mult:
                                _vol_candidates.append(cand)
                    if _vol_candidates:
                        candidates = pd.DataFrame(_vol_candidates)

                # Filter: foreign buy (use foreign_net_5 > 0 as proxy)
                if not candidates.empty and "foreign_net_5" in candidates.columns:
                    candidates = candidates[candidates["foreign_net_5"] > 0]

                # Exclude already-held stocks
                if not candidates.empty:
                    candidates = candidates[~candidates["stock_id"].astype(str).isin(positions.keys())]

                # Sort by score, take available slots
                if not candidates.empty:
                    candidates = candidates.sort_values("score", ascending=False)
                    slots = max_positions - len(positions)
                    for _, cand in candidates.head(slots).iterrows():
                        _sid = str(cand["stock_id"])
                        if _sid in today_price_map:
                            _entry_px = float(today_price_map[_sid]["close"])
                            if _entry_px > 0:
                                positions[_sid] = Position(
                                    stock_id=_sid, entry_date=today,
                                    entry_price=_entry_px, score=float(cand["score"]),
                                )

        # ── 6g. Track equity ──
        # Mark-to-market: update equity for unrealized P&L
        _daily_pnl = 0.0
        for sid, pos in positions.items():
            if sid in today_price_map:
                _close = float(today_price_map[sid]["close"])
                if pos.days_held > 0:
                    _prev_close = pos.peak_price  # approximate
                    # Simple: just track realized P&L through trades_log
                    pass

        equity_curve.append({"date": str(today), "equity": equity})

        # Progress
        if day_idx % 250 == 0 and day_idx > 0:
            _elapsed = time.time() - _t_start
            _pct = day_idx / len(bt_dates) * 100
            print(f"  [{today}] {_pct:.0f}% done | equity: {equity:,.0f} | "
                  f"positions: {len(positions)} | trades: {len(trades_log)} | "
                  f"{_elapsed:.0f}s elapsed", flush=True)

    # ── 7. Close remaining positions at end ──
    last_date = bt_dates[-1] if bt_dates else data_end
    last_prices = price_df[price_df["trading_date"] == last_date]
    last_price_map = {str(r["stock_id"]): float(r["close"]) for _, r in last_prices.iterrows()}
    for sid in list(positions.keys()):
        pos = positions[sid]
        exit_px = last_price_map.get(sid, pos.entry_price)
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
        equity *= (1 + ret * position_pct)

    _total_time = time.time() - _t_start
    print(f"\n  Simulation complete: {_total_time:.1f}s")

    # ── 8. Compute summary statistics ──
    total_return = equity / 10000.0 - 1
    n_years = len(bt_dates) / 252.0
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # MDD from equity curve
    eq_vals = [e["equity"] for e in equity_curve]
    peak = eq_vals[0] if eq_vals else 10000
    max_dd = 0.0
    for v in eq_vals:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)

    # Trade statistics
    wins = sum(1 for t in trades_log if t["realized_pnl_pct"] > 0)
    losses = sum(1 for t in trades_log if t["realized_pnl_pct"] <= 0)
    win_rate = wins / max(len(trades_log), 1)
    avg_hold = np.mean([t["days_held"] for t in trades_log]) if trades_log else 0

    # Monthly returns for Sharpe
    monthly_eq = {}
    for e in equity_curve:
        _ym = e["date"][:7]
        monthly_eq[_ym] = e["equity"]
    _monthly_keys = sorted(monthly_eq.keys())
    monthly_rets = []
    for i in range(1, len(_monthly_keys)):
        _prev = monthly_eq[_monthly_keys[i-1]]
        _curr = monthly_eq[_monthly_keys[i]]
        if _prev > 0:
            monthly_rets.append(_curr / _prev - 1)

    risk_free_monthly = 0.015 / 12
    if monthly_rets and np.std(monthly_rets) > 0:
        sharpe = (np.mean(monthly_rets) - risk_free_monthly) / np.std(monthly_rets) * np.sqrt(12)
    else:
        sharpe = 0.0
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Profit factor
    gross_profit = sum(t["realized_pnl_pct"] for t in trades_log if t["realized_pnl_pct"] > 0)
    gross_loss = abs(sum(t["realized_pnl_pct"] for t in trades_log if t["realized_pnl_pct"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Yearly returns
    yearly_equity = {}
    for e in equity_curve:
        yr = e["date"][:4]
        yearly_equity[yr] = e["equity"]
    _yr_keys = sorted(yearly_equity.keys())
    yearly_rets = {}
    _prev_eq = 10000.0
    for yr in _yr_keys:
        _yr_eq = yearly_equity[yr]
        yearly_rets[yr] = _yr_eq / _prev_eq - 1
        _prev_eq = _yr_eq

    # Build periods (monthly aggregation for compatibility with generate_review_pack)
    periods = []
    _monthly_trades = defaultdict(list)
    for t in trades_log:
        _ym = t["exit_date"][:7]
        _monthly_trades[_ym].append(t)

    _prev_eq_p = 10000.0
    for ym in sorted(monthly_eq.keys()):
        _eq = monthly_eq[ym]
        _ret = _eq / _prev_eq_p - 1 if _prev_eq_p > 0 else 0
        _trades_in_month = _monthly_trades.get(ym, [])
        _sr = {t["stock_id"]: t["realized_pnl_pct"] for t in _trades_in_month}
        periods.append({
            "rebalance_date": f"{ym}-01",
            "exit_date": f"{ym}-28",
            "return": _ret,
            "benchmark_return": 0.0,
            "trades": len(_trades_in_month),
            "stock_returns": _sr,
            "stoploss_triggered": sum(1 for t in _trades_in_month if t.get("stoploss_triggered")),
        })
        _prev_eq_p = _eq

    # ── 9. Print summary ──
    print(f"\n{'='*60}")
    print(f"Strategy B Results")
    print(f"{'='*60}")
    print(f"  Period: {bt_dates[0]} ~ {bt_dates[-1]}")
    print(f"  Total Return: {total_return:+.2%}")
    print(f"  Annualized:   {ann_return:+.2%}")
    print(f"  MDD:          {max_dd:.2%}")
    print(f"  Sharpe:       {sharpe:.4f}")
    print(f"  Calmar:       {calmar:.4f}")
    print(f"  Win Rate:     {win_rate:.2%}")
    print(f"  Profit Factor:{profit_factor:.3f}")
    print(f"  Total Trades: {len(trades_log)}")
    print(f"  Avg Hold Days:{avg_hold:.1f}")
    print(f"\n  Exit Reasons:")
    for reason, cnt in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        pct = cnt / max(len(trades_log), 1) * 100
        print(f"    {reason}: {cnt} ({pct:.1f}%)")
    print(f"\n  Yearly Returns:")
    for yr, ret in sorted(yearly_rets.items()):
        print(f"    {yr}: {ret:+.2%}")

    # ── 10. Build output ──
    summary = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "benchmark_total_return": 0.0,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_trades": len(trades_log),
        "total_periods": len(periods),
        "stoploss_triggered": exit_reasons.get("Stop Loss", 0),
        "backtest_start": str(bt_dates[0]) if bt_dates else "",
        "backtest_end": str(bt_dates[-1]) if bt_dates else "",
        "avg_hold_days": avg_hold,
        "exit_reasons": dict(exit_reasons),
        "yearly_returns": yearly_rets,
        "config": {
            "strategy": "B_daily",
            "max_positions": max_positions,
            "position_pct": position_pct,
            "max_hold_days": max_hold_days,
            "stoploss_pct": stoploss_pct,
            "rsi_exit": rsi_exit,
            "foreign_sell_exit_days": foreign_sell_exit_days,
            "rsi_entry_min": rsi_entry_min,
            "rsi_entry_max": rsi_entry_max,
            "volume_surge_mult": volume_surge_mult,
            "score_percentile": score_percentile,
            "transaction_cost_pct": transaction_cost_pct,
        },
    }

    return {
        "summary": summary,
        "periods": periods,
        "equity_curve": equity_curve,
        "trades_log": trades_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Strategy B: Daily-frequency backtest")
    parser.add_argument("--months", type=int, default=120, help="Backtest months")
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--max-hold", type=int, default=20, help="Max hold days")
    parser.add_argument("--stoploss", type=float, default=-0.10)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config()

    market_filter_tiers = [(-0.05, 0.5), (-0.10, 0.25), (-0.15, 0.0)]

    with get_session() as session:
        result = run_strategy_b(
            config=config,
            db_session=session,
            backtest_months=args.months,
            max_positions=args.max_positions,
            max_hold_days=args.max_hold,
            stoploss_pct=args.stoploss,
            fast_mode=args.fast,
            market_filter_tiers=market_filter_tiers,
        )

    output_path = args.output
    if output_path is None:
        from datetime import datetime
        artifacts_dir = ROOT / "artifacts" / "backtest"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(artifacts_dir / "strategy_b.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
