"""Strategy C 每日選股腳本。

使用 LightGBM 訓練 10 日 forward return 模型，對今日股票打分，
依 Rank Drop 邏輯決定持倉進出場。

狀態管理：artifacts/daily_signal/strategy_c_state.json
輸出：    artifacts/daily_signal/strategy_c_YYYY-MM-DD.json

用法：
    python scripts/strategy_c_pick.py                    # 最新有效日期
    python scripts/strategy_c_pick.py --date 2026-03-13  # 指定日期
    python scripts/strategy_c_pick.py --capital 2000000
    python scripts/strategy_c_pick.py --reset-state      # 清空持倉重新開始
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.breakthrough import check_today as _check_breakthrough_today

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_LGBM = False

from app.db import get_session
from app.models import Stock, StrategyCTrade
from skills import data_store
# cross_section_normalize 已移除：backtest 未使用此正規化，保留會造成 production ≠ backtest

# ─────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────
OUTPUT_DIR  = ROOT / "artifacts" / "daily_signal"
STATE_FILE  = OUTPUT_DIR / "strategy_c_state.json"

TRAIN_LABEL_HORIZON    = 10    # 10 日 forward return
LABEL_BUFFER_DAYS      = 20    # 訓練截止緩衝（避免標籤前向洩漏）
RANK_THRESHOLD         = 0.20  # 維持持倉的最低排名門檻（top 20%）
TOP_ENTRY_N            = 10    # 每日候補選股池
MAX_POSITIONS          = 4     # 最大同時持倉（2026-04-16 實驗：pos=4 報酬 3.3x，Calmar 2.649）
MAX_HOLD_DAYS          = 30    # 強制出場天數
MAX_BREAKTHROUGH_DIST  = 0.30  # 新買進距突破上限（>30% 的股不進場，等更近的候補）
RETRAIN_MONTHS         = 36    # 訓練資料回看月數（3 年）
RETRAIN_FREQ_DAYS      = 30    # 每隔幾天重訓一次
MIN_AVG_TURNOVER_BILL  = 1.0   # 流動性門檻：20日平均日成交金額（億元），0=不過濾
USE_EXCESS_LABEL       = True  # 超額報酬 label：future_ret_h -= 同日截面均值（與 backtest --excess-label 一致）
USE_RANK_LABEL         = True  # 截面排名 label：excess return → 當日百分位排名 [0,1]→[-1,1]
                                #   → MSE on rank ≈ 直接優化 Spearman IC，消除極端值主導 loss
                                #   36m 回測：Sharpe 1.47→1.73（+17%）、Calmar +31%、MDD -9.6pp、勝率 49.5%→53.3%
USE_RANKING_OBJ        = False # LGBMRanker(LambdaRank)：直接優化 NDCG 排序目標
                                #   ⚠️ 36m 回測：cumulative 572%→97%（int 0-4 離散化太粗，退化）
                                #   維持 False，使用 Rank Label + MSE 即可達到 IC 優化效果

_META_COLS = {"stock_id", "trading_date", "future_ret_h"}


# ─────────────────────────────────────────────
# 訓練 / 打分
# ─────────────────────────────────────────────
def _train_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[List[int]] = None,
) -> object:
    """
    groups != None  → LGBMRanker(LambdaRank)，直接優化 NDCG 排序目標
                      y 必須為 int 相關度標籤（0-4），且 X/y 依 trading_date 排序。
    groups is None  → LGBMRegressor(MSE)，搭配 rank label 或 excess label 使用。
    """
    if _HAS_LGBM and groups is not None:
        # ── LambdaRank：直接優化排序 ──
        m = lgb.LGBMRanker(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        m.fit(X, y, group=groups)
    elif _HAS_LGBM:
        # ── MSE 回歸（搭配 rank label 使用）──
        m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        m.fit(X, y)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42,
        )
        m.fit(X, y)
    return m


def _prep_fmat(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """填 NaN、inf，對齊欄位順序。"""
    fmat = df.reindex(columns=feat_cols).replace([np.inf, -np.inf], np.nan)
    for col in fmat.columns:
        if fmat[col].isna().all():
            fmat[col] = 0.0
        else:
            fmat[col] = fmat[col].fillna(float(fmat[col].median()))
    return fmat


# ─────────────────────────────────────────────
# 狀態檔
# ─────────────────────────────────────────────
def _load_state() -> Dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"positions": {}, "last_run_date": None}


def _save_state(state: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────
# 股票名稱
# ─────────────────────────────────────────────
def _load_names(session, stock_ids: List[str]) -> Dict[str, str]:
    rows = session.query(Stock.stock_id, Stock.name).filter(Stock.stock_id.in_(stock_ids)).all()
    result = {str(r.stock_id): str(r.name or r.stock_id) for r in rows}
    for sid in stock_ids:
        result.setdefault(sid, sid)
    return result


# ─────────────────────────────────────────────
# DB 稽核 log
# ─────────────────────────────────────────────
def _write_trades_to_db(
    run_date: date,
    buy_list: List[Dict],
    sell_list: List[Dict],
    hold_list: List[Dict],
    score_map: Dict[str, float],
    bt_status: Dict,
    amount_per_pos: int = 0,
) -> None:
    """將今日 buy/sell/hold 動作寫入 strategy_c_trades（append-only）。"""
    records = []

    def _pct_dist(sid: str) -> Optional[float]:
        v = bt_status.get(sid, {}).get("pct_to_price_bt")
        return float(v) if v is not None else None

    for p in buy_list:
        sid = p["stock_id"]
        records.append(StrategyCTrade(
            run_date=run_date,
            stock_id=sid,
            action="buy",
            entry_date=run_date,
            entry_score=p.get("entry_score"),
            days_held=0,
            exit_reason=None,
            amount=amount_per_pos or None,
            score_today=score_map.get(sid),
            pct_to_breakthrough=_pct_dist(sid),
        ))

    for p in sell_list:
        sid = p["stock_id"]
        records.append(StrategyCTrade(
            run_date=run_date,
            stock_id=sid,
            action="sell",
            entry_date=date.fromisoformat(p["entry_date"]) if p.get("entry_date") else None,
            entry_score=p.get("entry_score"),
            days_held=p.get("days_held"),
            exit_reason=p.get("exit_reason"),
            amount=None,
            score_today=score_map.get(sid),
            pct_to_breakthrough=_pct_dist(sid),
        ))

    for p in hold_list:
        sid = p["stock_id"]
        records.append(StrategyCTrade(
            run_date=run_date,
            stock_id=sid,
            action="hold",
            entry_date=date.fromisoformat(p["entry_date"]) if p.get("entry_date") else None,
            entry_score=p.get("entry_score"),
            days_held=p.get("days_held"),
            exit_reason=None,
            amount=amount_per_pos or None,
            score_today=score_map.get(sid),
            pct_to_breakthrough=_pct_dist(sid),
        ))

    if not records:
        return

    try:
        with get_session() as session:
            session.add_all(records)
            session.commit()
        print(f"  DB 稽核 log：{len(records)} 筆寫入 strategy_c_trades ✅")
    except Exception as e:
        print(f"  ⚠️  DB 稽核 log 寫入失敗（不影響主流程）：{e}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def run_pick(
    target_date: Optional[date] = None,
    capital: int = 1_000_000,
    reset_state: bool = False,
) -> Dict:
    t_start = time.time()
    today = target_date or date.today()

    # ── 載入資料 ──
    data_end   = today
    data_start = today - timedelta(days=RETRAIN_MONTHS * 31 + 60)

    print(f"[Strategy C Pick]  日期：{today}  資金：{capital:,}")
    print(f"  資料範圍：{data_start} ~ {data_end}")

    with get_session() as session:
        t0 = time.time()
        price_df = data_store.get_prices(session, data_start, data_end)
        price_df["trading_date"] = pd.to_datetime(price_df["trading_date"]).dt.date
        price_df["stock_id"]     = price_df["stock_id"].astype(str)
        for col in ("close", "volume"):
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
        print(f"  Prices: {len(price_df):,} rows ({time.time()-t0:.1f}s)")

        t0 = time.time()
        feat_df = data_store.get_features(session, data_start, data_end)
        feat_df["trading_date"] = pd.to_datetime(feat_df["trading_date"]).dt.date
        feat_df["stock_id"]     = feat_df["stock_id"].astype(str)
        print(f"  Features: {len(feat_df):,} rows ({time.time()-t0:.1f}s)")

        # 股票名稱（稍後查）
        all_sids = feat_df["stock_id"].unique().tolist()
        names = _load_names(session, all_sids)

        # 興櫃清單：議價交易、外資無法參與，排除以免 foreign_buy_* 特徵永遠為 0
        _emerging_ids = {
            str(r.stock_id)
            for r in session.query(Stock.stock_id).filter(Stock.market == "EMERGING").all()
        }

    # ── 確認今日有特徵資料 ──
    available_feat_dates = sorted(feat_df["trading_date"].unique(), reverse=True)
    if not available_feat_dates:
        print("❌ 無特徵資料，請先執行 make pipeline")
        sys.exit(1)

    if today not in available_feat_dates:
        fallback = available_feat_dates[0]
        print(f"⚠️  {today} 無特徵資料，使用最近有效日期：{fallback}")
        today = fallback

    # ── 計算 10 日 forward return label ──
    t0 = time.time()
    _pp   = price_df.pivot_table(index="trading_date", columns="stock_id", values="close", aggfunc="last").sort_index()
    _fret = _pp.shift(-TRAIN_LABEL_HORIZON) / _pp - 1
    label_df = (
        _fret.reset_index()
        .melt(id_vars="trading_date", var_name="stock_id", value_name="future_ret_h")
        .dropna()
    )
    label_df["trading_date"] = pd.to_datetime(label_df["trading_date"]).dt.date
    label_df["stock_id"]     = label_df["stock_id"].astype(str)
    print(f"  Labels ({TRAIN_LABEL_HORIZON}d): {len(label_df):,} rows ({time.time()-t0:.1f}s)")

    # ── Step 1：超額報酬 label（market-neutral，與 backtest --excess-label 一致）──
    if USE_EXCESS_LABEL:
        label_df["future_ret_h"] = label_df["future_ret_h"].replace([np.inf, -np.inf], np.nan)
        _mkt_ret = label_df.groupby("trading_date")["future_ret_h"].mean().rename("mkt_ret_h")
        label_df = label_df.merge(_mkt_ret.reset_index(), on="trading_date", how="left")
        label_df["future_ret_h"] = label_df["future_ret_h"] - label_df["mkt_ret_h"].fillna(0)
        label_df = label_df.drop(columns=["mkt_ret_h"])
        label_df["future_ret_h"] = label_df["future_ret_h"].clip(-1.5, 1.5)
        label_df = label_df.dropna(subset=["future_ret_h"])
        print(f"  Excess label applied: {len(label_df):,} rows")

    # ── Step 2：截面排名 label（IC 優化核心）──
    # 把 excess return 轉為當日截面百分位排名 [0,1]，再 scale 到 [-1,1]
    # MSE on rank ≈ 優化 Spearman IC；消除極端值（暴漲/暴跌股）主導 loss 的問題
    if USE_RANK_LABEL:
        label_df["future_ret_h"] = (
            label_df.groupby("trading_date")["future_ret_h"]
            .rank(pct=True)          # [0, 1] 截面百分位
            .mul(2).sub(1)           # → [-1, 1]，0 = 中位數
        )
        print(f"  Rank label applied: range [{label_df['future_ret_h'].min():.2f}, "
              f"{label_df['future_ret_h'].max():.2f}]")

    # ── 特徵欄位 ──
    feat_cols = [c for c in feat_df.columns if c not in _META_COLS]

    # ── 訓練模型（截止到 today - LABEL_BUFFER_DAYS）──
    train_cutoff = today - timedelta(days=LABEL_BUFFER_DAYS)
    tf = feat_df[feat_df["trading_date"] < today].copy()
    tl = label_df[label_df["trading_date"] < train_cutoff].copy()

    merged = tf.merge(tl, on=["stock_id", "trading_date"], how="inner")
    if len(merged) < 1000:
        print(f"❌ 訓練樣本不足（{len(merged)} 筆），請回填更多歷史資料")
        sys.exit(1)

    # ── 準備訓練矩陣 ──
    # LambdaRank 需要依 trading_date 排序，一般回歸則不需要
    if USE_RANKING_OBJ and _HAS_LGBM:
        merged = merged.sort_values(["trading_date", "stock_id"]).reset_index(drop=True)

    fmat_train = _prep_fmat(
        merged.drop(columns=[c for c in _META_COLS if c in merged.columns]),
        feat_cols,
    )
    valid = fmat_train.notna().all(axis=1)
    fmat_train = fmat_train.loc[valid]
    merged_valid = merged.loc[fmat_train.index]

    if USE_RANKING_OBJ and _HAS_LGBM:
        # LambdaRank：rank label [-1,1] → int 相關度 0-4
        _pct = (merged_valid["future_ret_h"].values + 1) / 2  # [-1,1] → [0,1]
        y_train = np.clip((_pct * 5).astype(int), 0, 4)       # [0,1] → int 0-4
        # group = 每個 trading_date 有幾支股票（必須與排序對齊）
        _groups = merged_valid.groupby("trading_date", sort=False).size().tolist()
    else:
        y_train = merged_valid["future_ret_h"].astype(float).values
        _groups = None

    t0 = time.time()
    obj_label = "LambdaRank" if (_groups is not None) else ("RankLabel+MSE" if USE_RANK_LABEL else "MSE")
    print(f"  訓練模型（{len(y_train):,} 樣本，{len(feat_cols)} 特徵，{obj_label}）...", end=" ", flush=True)
    model = _train_model(fmat_train.values, y_train, groups=_groups)
    print(f"{time.time()-t0:.1f}s")

    # ── 對今日打分 ──
    tf_today = feat_df[feat_df["trading_date"] == today].copy()
    tf_today  = tf_today[tf_today["stock_id"].str.fullmatch(r"\d{4}")].reset_index(drop=True)
    tf_today  = tf_today[~tf_today["stock_id"].isin(_emerging_ids)].reset_index(drop=True)

    # 今日有收盤價的股票
    tp = price_df[price_df["trading_date"] == today]
    price_map = {str(r["stock_id"]): float(r["close"]) for _, r in tp.iterrows() if float(r["close"] or 0) > 0}
    tf_today  = tf_today[tf_today["stock_id"].isin(price_map)].reset_index(drop=True)

    # 流動性過濾：保留 20 日平均日成交金額 >= MIN_AVG_TURNOVER_BILL 億的股票
    if MIN_AVG_TURNOVER_BILL > 0 and not price_df.empty:
        _pv = price_df[["stock_id", "trading_date", "close", "volume"]].copy()
        _pv = _pv.sort_values(["stock_id", "trading_date"])
        _pv["_tv"] = _pv["close"] * _pv["volume"]
        _pv["_avg_tv20"] = _pv.groupby("stock_id")["_tv"].transform(
            lambda s: s.rolling(20, min_periods=1).mean()
        )
        _threshold = MIN_AVG_TURNOVER_BILL * 1e8
        _today_liq = _pv[_pv["trading_date"] == today]
        _liquid_sids = set(_today_liq[_today_liq["_avg_tv20"] >= _threshold]["stock_id"].astype(str).tolist())
        _before_liq = len(tf_today)
        tf_today = tf_today[tf_today["stock_id"].isin(_liquid_sids)].reset_index(drop=True)
        print(f"  流動性過濾 (>={MIN_AVG_TURNOVER_BILL:.0f}億): {_before_liq} → {len(tf_today)} 支")

    if tf_today.empty:
        print(f"❌ {today} 無有效股票特徵")
        sys.exit(1)

    fmat_today = _prep_fmat(
        tf_today.drop(columns=["stock_id", "trading_date"], errors="ignore"),
        feat_cols,
    )
    tf_today["score"] = model.predict(fmat_today.values)

    # Rank threshold：top 20%
    score_cutoff = float(tf_today["score"].quantile(1.0 - RANK_THRESHOLD))
    above_threshold = set(tf_today[tf_today["score"] >= score_cutoff]["stock_id"].tolist())
    top_n_pool      = set(tf_today.nlargest(TOP_ENTRY_N, "score")["stock_id"].tolist())

    score_map = {str(r["stock_id"]): float(r["score"]) for _, r in tf_today.iterrows()}

    # ── 載入持倉狀態 ──
    state = {"positions": {}, "last_run_date": None} if reset_state else _load_state()
    positions: Dict[str, Dict] = state.get("positions", {})

    # 更新持倉天數（根據上次跑的日期到今日的交易日數）
    last_run = state.get("last_run_date")
    if last_run:
        last_run_dt = date.fromisoformat(last_run)
        trading_days_since = len([
            d for d in available_feat_dates
            if last_run_dt < d <= today
        ])
        for sid in positions:
            positions[sid]["days_held"] = positions[sid].get("days_held", 0) + trading_days_since

    # ── 出場判斷 ──
    sell_list  = []
    hold_list  = []
    exits_done = set()

    for sid, pos in list(positions.items()):
        days = pos.get("days_held", 0)
        if days >= MAX_HOLD_DAYS:
            sell_list.append({**pos, "stock_id": sid, "exit_reason": "Max Hold Days"})
            exits_done.add(sid)
        elif sid not in above_threshold:
            sell_list.append({**pos, "stock_id": sid, "exit_reason": "Rank Drop"})
            exits_done.add(sid)
        else:
            hold_list.append({**pos, "stock_id": sid})

    remaining_slots = MAX_POSITIONS - len(hold_list)

    # ── 突破確認偵測（先算，供進場過濾 + /signal 顯示）──
    # 對「top 20 候選 + 維持持倉」計算今日突破狀態
    _top20_sids = set(tf_today.nlargest(20, "score")["stock_id"].astype(str).tolist())
    _bt_sids = list(_top20_sids | {p["stock_id"] for p in hold_list})
    _bt_feat_today = feat_df[feat_df["trading_date"] == today].copy() if "trading_date" in feat_df.columns else pd.DataFrame()
    try:
        bt_status = _check_breakthrough_today(
            price_df=price_df,
            feature_df=_bt_feat_today,
            stock_ids=_bt_sids,
            target_date=today,
        )
    except Exception:
        bt_status = {}

    # ── 進場判斷 ──
    held_sids = {p["stock_id"] for p in hold_list}
    buy_list  = []
    candidates = (
        tf_today[tf_today["stock_id"].isin(top_n_pool - held_sids - exits_done)]
        .sort_values("score", ascending=False)
    )
    for _, row in candidates.iterrows():
        if len(buy_list) >= remaining_slots:
            break
        sid = str(row["stock_id"])
        # 距突破上限過濾：距條件一 >30% 且尚未突破 → 跳過，用後排候補遞補
        _bt_info = bt_status.get(sid, {})
        _pct_dist = _bt_info.get("pct_to_price_bt", 0.0) or 0.0
        if not _bt_info.get("ready", False) and _pct_dist > MAX_BREAKTHROUGH_DIST:
            continue
        buy_list.append({
            "stock_id": sid,
            "entry_date": today.isoformat(),
            "entry_score": round(float(row["score"]), 6),
            "days_held": 0,
        })

    # ── 建立今日持倉清單 ──
    total_positions = len(hold_list) + len(buy_list)
    amount_per_pos  = capital // total_positions if total_positions > 0 else 0

    def _enrich(pos_list: List[Dict], action: str) -> List[Dict]:
        out = []
        for p in pos_list:
            sid = p["stock_id"]
            score = score_map.get(sid, p.get("entry_score", 0.0))
            entry = {
                "stock_id": sid,
                "name": names.get(sid, sid),
                "action": action,
                "score_today": round(score, 6),
                "entry_date": p.get("entry_date"),
                "entry_score": round(p.get("entry_score", score), 6),
                "days_held": p.get("days_held", 0),
                "amount": amount_per_pos,
            }
            if action == "sell":
                entry["exit_reason"] = p.get("exit_reason", "Rank Drop")
                entry.pop("amount", None)
            # 突破狀態（buy / hold 才附加）
            if action in ("buy", "hold") and sid in bt_status:
                bt = bt_status[sid]
                entry["breakthrough_ready"]    = bt["ready"]
                entry["breakthrough_type"]     = bt.get("type")
                entry["close_max_20"]          = round(bt.get("close_max_20", 0), 2)
                entry["vol_ratio"]             = round(bt.get("vol_ratio", 0), 2) if bt.get("vol_ratio") else None
                entry["pct_to_price_bt"]       = round(bt.get("pct_to_price_bt", 0), 4) if bt.get("pct_to_price_bt") is not None else None
                entry["ma_20"]                 = round(bt.get("ma_20", 0), 2)
            out.append(entry)
        return out

    holdings_today = _enrich(hold_list, "hold") + _enrich(buy_list, "buy")
    sell_enriched  = _enrich(sell_list, "sell")

    # ── 更新狀態 ──
    new_positions: Dict[str, Dict] = {}
    for p in hold_list:
        sid = p["stock_id"]
        new_positions[sid] = {
            "entry_date": p.get("entry_date"),
            "entry_score": p.get("entry_score", score_map.get(sid, 0.0)),
            "days_held": p.get("days_held", 0),
        }
    for p in buy_list:
        new_positions[p["stock_id"]] = {
            "entry_date": today.isoformat(),
            "entry_score": p["entry_score"],
            "days_held": 0,
        }
    _save_state({"positions": new_positions, "last_run_date": today.isoformat()})

    # ── 建立輸出 ──
    elapsed = time.time() - t_start
    result = {
        "strategy": "C",
        "date": today.isoformat(),
        "capital": capital,
        "num_positions": total_positions,
        "amount_per_position": amount_per_pos,
        "holdings": sorted(holdings_today, key=lambda x: -x["score_today"]),
        "changes": {
            "buy":  sorted(_enrich(buy_list,  "buy"),  key=lambda x: -x["score_today"]),
            "sell": sorted(sell_enriched,              key=lambda x: -x["score_today"]),
            "hold": sorted(_enrich(hold_list, "hold"), key=lambda x: -x["score_today"]),
        },
        "summary": {
            "buy_count": len(buy_list),
            "sell_count": len(sell_list),
            "hold_count": len(hold_list),
            "total_positions": total_positions,
        },
        # 供 telegram_bot 判斷使用者實際持倉是否應賣出
        "above_threshold_stocks": sorted(above_threshold),
        # 今日模型評分前 20 名（含名稱、分數、突破狀態），供買進建議用
        "top_candidates": [
            {
                "stock_id": str(r["stock_id"]),
                "name": names.get(str(r["stock_id"]), str(r["stock_id"])),
                "score_today": round(float(r["score"]), 6),
                **({
                    "breakthrough_ready": bt_status[str(r["stock_id"])]["ready"],
                    "breakthrough_type":  bt_status[str(r["stock_id"])].get("type"),
                    "close_max_20":       round(bt_status[str(r["stock_id"])].get("close_max_20", 0), 2),
                    "vol_ratio":          round(bt_status[str(r["stock_id"])].get("vol_ratio", 0), 2)
                                          if bt_status[str(r["stock_id"])].get("vol_ratio") is not None else None,
                    "pct_to_price_bt":    round(bt_status[str(r["stock_id"])].get("pct_to_price_bt", 0), 4)
                                          if bt_status[str(r["stock_id"])].get("pct_to_price_bt") is not None else None,
                    "ma_20":              round(bt_status[str(r["stock_id"])].get("ma_20", 0), 2),
                } if str(r["stock_id"]) in bt_status else {}),
            }
            for _, r in tf_today.nlargest(20, "score").iterrows()
        ],
        "meta": {
            "train_samples": int(len(y_train)),
            "feat_count": len(feat_cols),
            "scored_stocks": len(tf_today),
            "rank_threshold": RANK_THRESHOLD,
            "above_threshold_count": len(above_threshold),
            "score_cutoff": round(score_cutoff, 6),
            "elapsed_sec": round(elapsed, 1),
            "train_label_horizon": TRAIN_LABEL_HORIZON,
            "use_excess_label": USE_EXCESS_LABEL,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # ── 寫入 DB 稽核 log ──
    _write_trades_to_db(
        run_date=today,
        buy_list=buy_list,
        sell_list=sell_list,
        hold_list=hold_list,
        score_map=score_map,
        bt_status=bt_status,
        amount_per_pos=amount_per_pos,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"strategy_c_{today}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


# ─────────────────────────────────────────────
# 輸出
# ─────────────────────────────────────────────
def _print_result(r: Dict) -> None:
    d   = r["date"]
    cap = r["capital"]
    n   = r["num_positions"]
    amt = r["amount_per_position"]
    s   = r["summary"]
    m   = r["meta"]

    print()
    print("=" * 60)
    print(f"  Strategy C 每日選股  {d}")
    print("=" * 60)
    print(f"  資金：{cap:,}  持倉：{n} 檔  每檔：{amt:,}")
    print(f"  打分股票：{m['scored_stocks']}  top20%門檻：{m['score_cutoff']:.4f}")
    print(f"  買進 +{s['buy_count']}  賣出 -{s['sell_count']}  維持 {s['hold_count']}  耗時 {m['elapsed_sec']}s")

    if r["changes"]["buy"]:
        print("\n  ▶ 買進（新進場）")
        print(f"    {'代號':6}  {'名稱':<10}  {'今日分數':>9}  {'金額':>10}")
        for h in r["changes"]["buy"]:
            print(f"    {h['stock_id']:6}  {h['name']:<10}  {h['score_today']:>9.4f}  {h['amount']:>10,}")

    if r["changes"]["sell"]:
        print("\n  ◀ 賣出（出場）")
        print(f"    {'代號':6}  {'名稱':<10}  {'今日分數':>9}  {'原因'}")
        for h in r["changes"]["sell"]:
            print(f"    {h['stock_id']:6}  {h['name']:<10}  {h['score_today']:>9.4f}  {h.get('exit_reason','')}")

    if r["changes"]["hold"]:
        print("\n  ─ 維持持有")
        print(f"    {'代號':6}  {'名稱':<10}  {'今日分數':>9}  {'持倉天':>6}  {'金額':>10}")
        for h in r["changes"]["hold"]:
            print(f"    {h['stock_id']:6}  {h['name']:<10}  {h['score_today']:>9.4f}  {h['days_held']:>6}天  {h['amount']:>10,}")

    out_path = ROOT / "artifacts" / "daily_signal" / f"strategy_c_{r['date']}.json"
    print(f"\n  已儲存：{out_path.relative_to(ROOT)}")
    print()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main() -> None:
    global MIN_AVG_TURNOVER_BILL
    parser = argparse.ArgumentParser(description="Strategy C 每日選股")
    parser.add_argument("--date", type=lambda s: date.fromisoformat(s), default=date.today())
    parser.add_argument("--capital", type=int, default=1_000_000)
    parser.add_argument("--reset-state", action="store_true", help="清空持倉狀態重新開始")
    parser.add_argument("--min-avg-turnover", type=float, default=MIN_AVG_TURNOVER_BILL,
                        dest="min_avg_turnover",
                        help=f"流動性門檻（億元），0=不過濾（預設 {MIN_AVG_TURNOVER_BILL}）")
    args = parser.parse_args()
    MIN_AVG_TURNOVER_BILL = args.min_avg_turnover

    result = run_pick(args.date, args.capital, args.reset_state)
    _print_result(result)


if __name__ == "__main__":
    main()
