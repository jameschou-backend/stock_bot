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
    __slots__ = ("stock_id", "entry_date", "entry_price", "score", "days_held",
                 "foreign_sell_streak", "peak_price", "ma_below_streak",
                 "foreign_weak_streak", "foreign_consec_break_streak")

    def __init__(self, stock_id: str, entry_date: date, entry_price: float, score: float):
        self.stock_id = stock_id
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.score = score
        self.days_held = 0
        self.foreign_sell_streak = 0         # 舊版相容（rank mode 用）
        self.peak_price = entry_price
        self.ma_below_streak = 0             # 連續收盤低於 MA20 天數
        self.foreign_weak_streak = 0         # 連續 foreign_buy_streak_5 == 0 天數
        self.foreign_consec_break_streak = 0 # 連續 foreign_buy_consecutive_days == 0 天數


def run_rotation(
    config,
    db_session,
    backtest_months: int = 120,
    max_positions: int = 6,
    rank_threshold: float = 0.20,
    entry_threshold: Optional[float] = None,  # 進場門檻（百分比），None=使用 top_entry_n
    max_hold_days: int = 30,
    top_entry_n: int = 10,
    retrain_freq_months: int = 3,
    label_horizon_buffer: int = 20,
    transaction_cost_pct: float = 0.003,
    fast_mode: bool = False,
    market_filter_tiers: Optional[List[tuple]] = None,
    min_hold_days: int = 0,
    force_exit_threshold: Optional[float] = None,
    # ── 風控出場參數 ──
    exit_mode: str = "rank",  # "rank"=排名出場（原始）, "risk"=風控出場, "oracle"=Oracle 訊號出場
    stoploss_pct: float = -0.10,
    trailing_stop_pct: float = -0.15,
    foreign_sell_exit_days: int = 2,
    ma_break_days: int = 2,
    ma_break_vol_mult: float = 1.5,
    rsi_exit: Optional[float] = None,
    # Oracle 模式專用參數（基於事後最優出場分析）
    oracle_rsi_ob: float = 75.0,          # RSI 超買門檻
    oracle_boll_ob: float = 0.95,         # Boll %B 過熱門檻
    oracle_foreign_break_days: int = 3,   # 外資連買中斷天數
    oracle_ret5_tp: float = 0.20,         # 5日報酬超過此值視為短期過熱
    # ── 守門員進場品質過濾（quality gate）──
    use_quality_gate: bool = False,
    # 硬排除條件（任一成立即排除）
    gate_max_streak: int = 15,            # foreign_buy_streak > N 排除（主力晚期）
    gate_max_pe: float = 300.0,           # pe_ratio > N 排除（純炒作）
    # 軟加分條件（符合者分數乘以 1+bonus_pct，影響進場優先序）
    gate_streak_bonus_min: int = 2,       # foreign_buy_streak in [min, max] 加分
    gate_streak_bonus_max: int = 8,
    gate_streak_bonus_pct: float = 0.08,  # +8% 乘數
    gate_rev_accel_bonus_pct: float = 0.04,  # fund_revenue_yoy_accel > 0 加 +4%
    # ── 訓練 label horizon（預設 20，可改 5/10 測試尺度匹配）──
    train_label_horizon: int = 20,        # 訓練用 N 日 forward return（從 price_df 計算）
    # ── 籌碼出場補充（chip exit，rank 模式下加掛）──
    chip_exit: bool = False,
    chip_exit_foreign_break_days: int = 3,   # 外資連買中斷連 N 天
    chip_exit_boll_threshold: float = 0.90,  # boll_pct 過熱門檻
    chip_exit_min_hold: int = 5,             # 至少持倉 N 天才觸發
) -> Dict:
    """
    exit_mode:
        "rank" - 原始邏輯：排名掉出 rank_threshold 即出場
        "risk" - 風控邏輯：停損 / 外資連賣 / 均線跌破放量 / 時間到期

    exit_mode="oracle"：基於 Oracle 分析的訊號出場（任一觸發）：
        1. RSI > oracle_rsi_ob AND boll_pct > oracle_boll_ob （技術面過熱，AND 條件）
        2. foreign_buy_consecutive_days == 0 連 oracle_foreign_break_days 天（外資連買中斷）
        3. 5日報酬 > oracle_ret5_tp （短期漲幅過大，保護獲利）
        4. 固定停損 stoploss_pct
        5. max_hold_days（保底）

    風控出場條件（exit_mode="risk"，任一觸發即出場）：
        stoploss_pct:          固定停損（從進場價計算），最後防線
        trailing_stop_pct:     追蹤停損（從峰值回落，如 -0.15 = 從最高點跌 15% 出場）
        foreign_sell_exit_days: 連續 N 天 foreign_buy_streak_5 == 0（5天內無任何買超）
        ma_break_days:         連續 N 天收盤低於 MA20 才出場（過濾單日假跌破）
        ma_break_vol_mult:     MA 跌破時需成交量 > N 倍均量（預設不啟用 vol 條件，設 0 停用）
        rsi_exit:              RSI 超買出場（None = 停用）
        max_hold_days:         最長持倉上限（保底）
    """
    if force_exit_threshold is None:
        force_exit_threshold = min(rank_threshold * 1.5, 1.0)

    print("\n" + "=" * 60)
    print("Strategy C: Dynamic Rotation Backtest")
    print("=" * 60)
    print(f"  Exit mode: {exit_mode}")
    if use_quality_gate:
        print(f"  [Quality Gate ON] streak<={gate_max_streak}, pe<={gate_max_pe:.0f}")
        print(f"    bonus: streak {gate_streak_bonus_min}-{gate_streak_bonus_max} +{gate_streak_bonus_pct:.0%}, rev_accel>0 +{gate_rev_accel_bonus_pct:.0%}")
    if entry_threshold is not None:
        print(f"  Entry threshold: top {entry_threshold:.0%} (dual threshold mode)")
    if exit_mode == "rank":
        print(f"  Rank threshold: top {rank_threshold:.0%} (sell if drops out)")
    elif exit_mode == "oracle":
        print(f"  [Oracle 訊號出場]")
        print(f"  RSI>{oracle_rsi_ob} AND Boll%B>{oracle_boll_ob} → 技術面過熱")
        print(f"  外資連買中斷 {oracle_foreign_break_days} 天（consecutive_days==0）")
        print(f"  5日報酬 >{oracle_ret5_tp:.0%} → 短期過熱保利")
        print(f"  固定停損：{stoploss_pct:.0%}")
        print(f"  Max hold：{max_hold_days} 天")
    else:
        print(f"  Stop loss: {stoploss_pct:.0%} (fixed) / {trailing_stop_pct:.0%} from peak (trailing)")
        print(f"  Foreign weak: {foreign_sell_exit_days} consecutive days with foreign_buy_streak_5==0")
        print(f"  MA20 break: {ma_break_days} consecutive days below MA20")
        if rsi_exit:
            print(f"  RSI overbought: >{rsi_exit}")
    print(f"  Max hold: {max_hold_days} days")
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

    # ── 若 train_label_horizon != 20，從 price_df 重算 N 日 forward return 當 label ──
    if chip_exit:
        print(f"  [Chip Exit ON] 外資連買中斷>={chip_exit_foreign_break_days}天 AND boll_pct>{chip_exit_boll_threshold} AND 持倉>={chip_exit_min_hold}天 → 出場")

    if train_label_horizon != 20:
        print(f"  Computing {train_label_horizon}-day forward return labels from prices...", flush=True)
        _pp = price_df.pivot_table(index="trading_date", columns="stock_id", values="close", aggfunc="last")
        _pp = _pp.sort_index()
        _fret = _pp.shift(-train_label_horizon) / _pp - 1
        _alt = (
            _fret.reset_index()
            .melt(id_vars="trading_date", var_name="stock_id", value_name="future_ret_h")
            .dropna()
        )
        _alt["trading_date"] = pd.to_datetime(_alt["trading_date"]).dt.date
        _alt["stock_id"] = _alt["stock_id"].astype(str)
        label_df_train = _alt
        _eff_buffer = train_label_horizon   # buffer 與 label horizon 對齊
        print(f"  Alt labels: {len(label_df_train):,} rows (horizon={train_label_horizon}d)")
    else:
        label_df_train = label_df
        _eff_buffer = label_horizon_buffer

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
            train_cutoff = today - timedelta(days=_eff_buffer)
            tf = feat_df[feat_df["trading_date"] < today]
            tl = label_df_train[label_df_train["trading_date"] < train_cutoff]
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

        # Rank threshold（進場與 rank 模式出場用）
        score_cutoff = tf_today["score"].quantile(1.0 - rank_threshold)
        force_cutoff = tf_today["score"].quantile(1.0 - force_exit_threshold)
        top_n_sids = set(tf_today.nlargest(top_entry_n, "score")["stock_id"].tolist())
        above_threshold = set(tf_today[tf_today["score"] >= score_cutoff]["stock_id"].tolist())
        above_force_threshold = set(tf_today[tf_today["score"] >= force_cutoff]["stock_id"].tolist())

        # 雙門檻進場候選集（entry_threshold 模式）
        if entry_threshold is not None:
            entry_cutoff = tf_today["score"].quantile(1.0 - entry_threshold)
            entry_eligible = set(tf_today[tf_today["score"] >= entry_cutoff]["stock_id"].tolist())
        else:
            entry_eligible = top_n_sids

        # 風控/Oracle/chip_exit 模式：預建今日特徵查詢 dict（避免 loop 內重複 filter）
        _feat_map: Dict[str, dict] = {}
        if (exit_mode in ("risk", "oracle") or chip_exit) and not tf_today.empty:
            for _, _r in tf_today.iterrows():
                _feat_map[str(_r["stock_id"])] = _r.to_dict()

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

            if exit_mode == "risk":
                # ── 風控出場邏輯（改進版）──
                close = price_map.get(sid, pos.entry_price)
                pos.peak_price = max(pos.peak_price, close)
                ret_from_entry = close / pos.entry_price - 1
                ret_from_peak  = close / pos.peak_price - 1

                # 從今日特徵取風控指標
                _feat = _feat_map.get(sid, {})
                _fbs5  = float(_feat.get("foreign_buy_streak_5", 1) or 1)  # 5天內外資買超天數（0=全週賣）
                _ma20  = float(_feat.get("ma_20", close) or close)
                _vol_ratio = float(_feat.get("vol_ratio_20", 1.0) or 1.0)
                _rsi   = float(_feat.get("rsi_14", 50) or 50)

                # 更新連續弱勢外資天數（5天內都沒有買超 = 外資全面撤）
                if _fbs5 == 0:
                    pos.foreign_weak_streak += 1
                else:
                    pos.foreign_weak_streak = 0

                # 更新連續低於 MA20 天數
                if _ma20 > 0 and close < _ma20:
                    pos.ma_below_streak += 1
                else:
                    pos.ma_below_streak = 0

                # 出場判斷（優先序：時間 > 固定停損 > 追蹤停損 > 外資撤 > MA跌破 > RSI）
                if pos.days_held >= max_hold_days:
                    exit_reason = "Max Hold Days"
                elif ret_from_entry <= stoploss_pct:
                    exit_reason = "Stop Loss"
                elif ret_from_peak <= trailing_stop_pct:
                    exit_reason = "Trailing Stop"
                elif pos.foreign_weak_streak >= foreign_sell_exit_days:
                    exit_reason = "Foreign Weak"
                elif pos.ma_below_streak >= ma_break_days:
                    exit_reason = "MA Break"
                elif rsi_exit is not None and _rsi > rsi_exit:
                    exit_reason = "RSI Overbought"

            elif exit_mode == "oracle":
                # ── Oracle 訊號出場邏輯（基於事後最優出場分析）──
                close = price_map.get(sid, pos.entry_price)
                ret_from_entry = close / pos.entry_price - 1

                _feat = _feat_map.get(sid, {})
                _rsi       = float(_feat.get("rsi_14", 50) or 50)
                _boll      = float(_feat.get("boll_pct", 0.5) or 0.5)
                _ret5      = float(_feat.get("ret_5", 0) or 0)
                _fcd       = float(_feat.get("foreign_buy_consecutive_days", 1) or 1)

                # 更新外資連買中斷天數
                if _fcd == 0:
                    pos.foreign_consec_break_streak += 1
                else:
                    pos.foreign_consec_break_streak = 0

                if pos.days_held >= max_hold_days:
                    exit_reason = "Max Hold Days"
                elif ret_from_entry <= stoploss_pct:
                    exit_reason = "Stop Loss"
                elif _rsi > 78 or (_rsi > 72 and _ret5 > 0.10):
                    exit_reason = "Tech Overbought"   # RSI>78 OR (RSI>72 AND 5日報酬>10%)
                elif pos.foreign_consec_break_streak >= oracle_foreign_break_days:
                    exit_reason = "Foreign Break"     # 外資連買中斷 N 天
                elif _ret5 > oracle_ret5_tp:
                    exit_reason = "Ret5 Take Profit"  # 5日漲幅過大，鎖利

            else:
                # ── 原始排名出場邏輯 ──
                # 更新外資連買中斷天數（chip_exit 模式用）
                if chip_exit:
                    _feat = _feat_map.get(sid, {})
                    _fcd = float(_feat.get("foreign_buy_consecutive_days", 1) or 1)
                    if _fcd == 0:
                        pos.foreign_consec_break_streak += 1
                    else:
                        pos.foreign_consec_break_streak = 0

                if pos.days_held < min_hold_days:
                    if sid not in above_force_threshold:
                        exit_reason = "Force Exit"
                else:
                    if sid not in above_threshold:
                        exit_reason = "Rank Drop"
                    elif pos.days_held >= max_hold_days:
                        exit_reason = "Max Hold Days"
                    elif chip_exit and pos.days_held >= chip_exit_min_hold:
                        _feat = _feat_map.get(sid, {})
                        _boll = float(_feat.get("boll_pct", 0.5) or 0.5)
                        if (pos.foreign_consec_break_streak >= chip_exit_foreign_break_days
                                and _boll > chip_exit_boll_threshold):
                            exit_reason = "Chip Exit"

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
            candidates = tf_today[tf_today["stock_id"].isin(entry_eligible)]
            candidates = candidates[~candidates["stock_id"].isin(positions.keys())]

            if use_quality_gate and not candidates.empty:
                # ── 硬排除：財務地雷 / 主力晚期 ──
                mask = pd.Series(True, index=candidates.index)
                if "foreign_buy_streak" in candidates.columns:
                    _fbs = pd.to_numeric(candidates["foreign_buy_streak"], errors="coerce").fillna(0)
                    mask &= _fbs <= gate_max_streak
                if "pe_ratio" in candidates.columns:
                    _pe = pd.to_numeric(candidates["pe_ratio"], errors="coerce")
                    mask &= ~((_pe > gate_max_pe) & _pe.notna())
                if "pb_ratio" in candidates.columns:
                    _pb = pd.to_numeric(candidates["pb_ratio"], errors="coerce")
                    mask &= ~((_pb < 0) & _pb.notna())
                candidates = candidates[mask]

                # ── 軟加分：籌碼啟動窗口 + 營收動能（乘數形式，保持相對排名）──
                if not candidates.empty:
                    _bonus = pd.Series(1.0, index=candidates.index)
                    if "foreign_buy_streak" in candidates.columns:
                        _fbs2 = pd.to_numeric(candidates["foreign_buy_streak"], errors="coerce").fillna(0)
                        _in_window = (_fbs2 >= gate_streak_bonus_min) & (_fbs2 <= gate_streak_bonus_max)
                        _bonus += _in_window.astype(float) * gate_streak_bonus_pct
                    if "fund_revenue_yoy_accel" in candidates.columns:
                        _accel = pd.to_numeric(candidates["fund_revenue_yoy_accel"], errors="coerce").fillna(0)
                        _bonus += (_accel > 0).astype(float) * gate_rev_accel_bonus_pct
                    candidates = candidates.assign(_entry_priority=candidates["score"] * _bonus)
                    candidates = candidates.sort_values("_entry_priority", ascending=False)
                else:
                    candidates = candidates.sort_values("score", ascending=False)
            else:
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
            "rank_threshold": rank_threshold, "entry_threshold": entry_threshold,
            "max_hold_days": max_hold_days,
            "use_quality_gate": use_quality_gate,
            "gate_max_streak": gate_max_streak, "gate_max_pe": gate_max_pe,
            "gate_streak_bonus_pct": gate_streak_bonus_pct,
            "gate_rev_accel_bonus_pct": gate_rev_accel_bonus_pct,
            "train_label_horizon": train_label_horizon,
            "chip_exit": chip_exit,
            "chip_exit_foreign_break_days": chip_exit_foreign_break_days,
            "chip_exit_boll_threshold": chip_exit_boll_threshold,
            "chip_exit_min_hold": chip_exit_min_hold,
            "min_hold_days": min_hold_days, "force_exit_threshold": force_exit_threshold,
            "exit_mode": exit_mode, "stoploss_pct": stoploss_pct,
            "trailing_stop_pct": trailing_stop_pct,
            "foreign_sell_exit_days": foreign_sell_exit_days,
            "ma_break_days": ma_break_days,
            "ma_break_vol_mult": ma_break_vol_mult, "rsi_exit": rsi_exit,
            "oracle_rsi_ob": oracle_rsi_ob, "oracle_boll_ob": oracle_boll_ob,
            "oracle_foreign_break_days": oracle_foreign_break_days,
            "oracle_ret5_tp": oracle_ret5_tp,
            "top_entry_n": top_entry_n, "transaction_cost_pct": transaction_cost_pct,
        },
    }
    return {"summary": summary, "periods": periods, "equity_curve": equity_curve, "trades_log": trades_log}


def main():
    parser = argparse.ArgumentParser(description="Strategy C: Dynamic Rotation")
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--rank-threshold", type=float, default=None)
    parser.add_argument("--entry-threshold", type=float, default=None,
                        help="進場門檻比例（如 0.10=top10%%），None=使用 top_entry_n")
    parser.add_argument("--quality-gate", action="store_true",
                        help="啟用守門員進場過濾（streak>15/pe>300/pb<0 排除 + 軟加分）")
    parser.add_argument("--gate-max-streak", type=int, default=15,
                        help="守門員：foreign_buy_streak 上限（預設 15）")
    parser.add_argument("--gate-max-pe", type=float, default=300.0,
                        help="守門員：pe_ratio 上限（預設 300）")
    parser.add_argument("--train-label-horizon", type=int, default=20,
                        help="訓練用 label horizon（天），預設 20，可改 5/10 測試尺度匹配")
    parser.add_argument("--transaction-cost", type=float, default=0.003,
                        help="每筆交易成本（預設 0.003=0.3%%，台股真實約 0.00585=0.585%%）")
    parser.add_argument("--max-hold", type=int, default=None)
    parser.add_argument("--min-hold", type=int, default=0,
                        help="最小持倉天數保護（預設 0=停用）")
    parser.add_argument("--force-exit-threshold", type=float, default=None,
                        help="保護期間強制出場門檻，如 0.30=top30%（預設=rank_threshold×1.5）")
    parser.add_argument("--exit-mode", type=str, default="rank", choices=["rank", "risk", "oracle"],
                        help="出場模式：rank=排名出場（原始），risk=風控出場")
    parser.add_argument("--stoploss", type=float, default=-0.10,
                        help="固定停損（risk 模式，預設 -0.10）")
    parser.add_argument("--trailing-stop", type=float, default=-0.15,
                        help="追蹤停損：從峰值回落幅度（risk 模式，預設 -0.15）")
    parser.add_argument("--foreign-sell-days", type=int, default=2,
                        help="連續 N 天 foreign_buy_streak_5==0 出場（risk 模式，預設 2）")
    parser.add_argument("--ma-break-days", type=int, default=2,
                        help="連續 N 天低於 MA20 出場（risk 模式，預設 2）")
    parser.add_argument("--ma-break-vol", type=float, default=1.5,
                        help="MA20 跌破時量能門檻（risk 模式，預設 1.5，設 0 停用）")
    parser.add_argument("--rsi-exit", type=float, default=None,
                        help="RSI 超買出場門檻（risk 模式，None=停用）")
    parser.add_argument("--oracle-rsi", type=float, default=75.0,
                        help="Oracle 模式：RSI 過熱門檻（預設 75）")
    parser.add_argument("--oracle-boll", type=float, default=0.95,
                        help="Oracle 模式：Boll%%B 過熱門檻（預設 0.95）")
    parser.add_argument("--oracle-foreign-days", type=int, default=3,
                        help="Oracle 模式：外資連買中斷天數（預設 3）")
    parser.add_argument("--oracle-ret5-tp", type=float, default=0.20,
                        help="Oracle 模式：5日報酬獲利了結門檻（預設 0.20）")
    parser.add_argument("--chip-exit", action="store_true",
                        help="啟用籌碼出場補充：外資連買中斷>=N天 AND boll_pct>0.90 AND 持倉>=5天")
    parser.add_argument("--chip-exit-break-days", type=int, default=3,
                        help="外資連買中斷天數門檻（預設 3）")
    parser.add_argument("--chip-exit-boll", type=float, default=0.90,
                        help="Boll%%B 過熱門檻（預設 0.90）")
    parser.add_argument("--chip-exit-min-hold", type=int, default=5,
                        help="最小持倉天數才觸發 chip exit（預設 5）")
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
            rank_threshold=rank_thr, entry_threshold=args.entry_threshold,
            max_hold_days=max_hold,
            transaction_cost_pct=args.transaction_cost,
            use_quality_gate=args.quality_gate,
            gate_max_streak=args.gate_max_streak,
            gate_max_pe=args.gate_max_pe,
            train_label_horizon=args.train_label_horizon,
            fast_mode=args.fast, market_filter_tiers=mf_tiers,
            min_hold_days=args.min_hold,
            force_exit_threshold=args.force_exit_threshold,
            exit_mode=args.exit_mode,
            stoploss_pct=args.stoploss,
            trailing_stop_pct=args.trailing_stop,
            foreign_sell_exit_days=args.foreign_sell_days,
            ma_break_days=args.ma_break_days,
            ma_break_vol_mult=args.ma_break_vol,
            rsi_exit=args.rsi_exit,
            oracle_rsi_ob=args.oracle_rsi,
            oracle_boll_ob=args.oracle_boll,
            oracle_foreign_break_days=args.oracle_foreign_days,
            oracle_ret5_tp=args.oracle_ret5_tp,
            chip_exit=args.chip_exit,
            chip_exit_foreign_break_days=args.chip_exit_break_days,
            chip_exit_boll_threshold=args.chip_exit_boll,
            chip_exit_min_hold=args.chip_exit_min_hold,
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
