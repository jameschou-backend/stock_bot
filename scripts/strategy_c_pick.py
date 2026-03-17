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

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_LGBM = False

from app.db import get_session
from app.models import Stock
from skills import data_store

# ─────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────
OUTPUT_DIR  = ROOT / "artifacts" / "daily_signal"
STATE_FILE  = OUTPUT_DIR / "strategy_c_state.json"

TRAIN_LABEL_HORIZON = 10    # 10 日 forward return
LABEL_BUFFER_DAYS   = 20    # 訓練截止緩衝（避免標籤前向洩漏）
RANK_THRESHOLD      = 0.20  # 維持持倉的最低排名門檻（top 20%）
TOP_ENTRY_N         = 10    # 每日候補選股池
MAX_POSITIONS       = 6     # 最大同時持倉
MAX_HOLD_DAYS       = 30    # 強制出場天數
RETRAIN_MONTHS      = 36    # 訓練資料回看月數（3 年）
RETRAIN_FREQ_DAYS   = 30    # 每隔幾天重訓一次

_META_COLS = {"stock_id", "trading_date", "future_ret_h"}


# ─────────────────────────────────────────────
# 訓練 / 打分
# ─────────────────────────────────────────────
def _train_model(X: np.ndarray, y: np.ndarray):
    if _HAS_LGBM:
        m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50,
            random_state=42, n_jobs=-1, verbose=-1,
        )
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

    fmat_train = _prep_fmat(
        merged.drop(columns=[c for c in _META_COLS if c in merged.columns]),
        feat_cols,
    )
    valid = fmat_train.notna().all(axis=1)
    fmat_train = fmat_train.loc[valid]
    y_train    = merged.loc[fmat_train.index]["future_ret_h"].astype(float).values

    t0 = time.time()
    print(f"  訓練模型（{len(y_train):,} 樣本，{len(feat_cols)} 特徵）...", end=" ", flush=True)
    model = _train_model(fmat_train.values, y_train)
    print(f"{time.time()-t0:.1f}s")

    # ── 對今日打分 ──
    tf_today = feat_df[feat_df["trading_date"] == today].copy()
    tf_today  = tf_today[tf_today["stock_id"].str.fullmatch(r"\d{4}")].reset_index(drop=True)

    # 今日有收盤價的股票
    tp = price_df[price_df["trading_date"] == today]
    price_map = {str(r["stock_id"]): float(r["close"]) for _, r in tp.iterrows() if float(r["close"] or 0) > 0}
    tf_today  = tf_today[tf_today["stock_id"].isin(price_map)].reset_index(drop=True)

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
        # 今日模型評分前 20 名（含名稱與分數），供買進建議用
        "top_candidates": [
            {
                "stock_id": str(r["stock_id"]),
                "name": names.get(str(r["stock_id"]), str(r["stock_id"])),
                "score_today": round(float(r["score"]), 6),
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
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

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
    parser = argparse.ArgumentParser(description="Strategy C 每日選股")
    parser.add_argument("--date", type=lambda s: date.fromisoformat(s), default=date.today())
    parser.add_argument("--capital", type=int, default=1_000_000)
    parser.add_argument("--reset-state", action="store_true", help="清空持倉狀態重新開始")
    args = parser.parse_args()

    result = run_pick(args.date, args.capital, args.reset_state)
    _print_result(result)


if __name__ == "__main__":
    main()
