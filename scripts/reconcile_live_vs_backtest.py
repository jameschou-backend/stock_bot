#!/usr/bin/env python
"""Live vs Backtest 對帳工具（2026-07-04 新增，健檢後續制度化項目）。

動機：daily_pick 的 index 錯位 bug（P0-1）潛伏近五個月才被健檢抓到——若有例行對帳，
2 月就會發現 live picks 與模型行為脫鉤。本工具提供兩層防線：

1. **sanity**：抽驗最近 picks 的 reason_json 特徵值 == features 表同 (stock, date) 的值
   （P0-1 哨兵：任何 MISMATCH 代表打分特徵again錯位，立即報警）。
2. **reconcile**：以每個「月度再平衡日」的實際 picks 計算 20 交易日等權還原報酬，
   與基準回測 JSON 同期 period return 比較，輸出偏差表。
   偏差大 ≠ 一定有 bug（live=每日 pick、回測=月頻模擬，universe 快照也不同），
   但持續、單向的大偏差是 live 與回測脫鉤的訊號。

用法：
    python scripts/reconcile_live_vs_backtest.py --sanity          # 只跑哨兵（快，可每日跑）
    python scripts/reconcile_live_vs_backtest.py \
        --baseline artifacts/backtest/backtest_20260703_204141.json  # 完整對帳（每月跑）

注意：2026-02-13 ~ 2026-07-03 的 picks 受 P0-1 影響（錯位特徵打分），對帳無意義；
有效 live track record 從 2026-07-06（修復後第一個交易日）起算。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import get_session

P0_FIX_DATE = date(2026, 7, 4)  # P0-1 修復後第一天（之前的 picks 特徵錯位，不對帳）
HORIZON_TDAYS = 20              # 與 label horizon 一致


# ── 哨兵：picks 特徵值 == features 表 ────────────────────────────────────────

def run_sanity(session, n_dates: int = 3, sample_per_date: int = 10) -> int:
    """抽驗最近 n_dates 個 pick 日的特徵一致性。回傳 MISMATCH 數（0=通過）。"""
    dates = [r[0] for r in session.execute(text(
        "SELECT DISTINCT pick_date FROM picks ORDER BY pick_date DESC LIMIT :n"
    ), {"n": n_dates}).fetchall()]
    if not dates:
        print("[sanity] picks 表為空，跳過")
        return 0

    mismatches = 0
    checked = 0
    for d in dates:
        rows = session.execute(text(
            "SELECT stock_id, reason_json FROM picks WHERE pick_date = :d LIMIT :k"
        ), {"d": d, "k": sample_per_date}).fetchall()
        for sid, reason in rows:
            reasons = json.loads(reason) if isinstance(reason, str) else reason
            feat_vals = {
                k: v.get("value")
                for k, v in reasons.items()
                if isinstance(v, dict) and "value" in v
            }
            if not feat_vals:
                continue
            frow = session.execute(text(
                "SELECT features_json FROM features "
                "WHERE stock_id = :s AND trading_date = :d"
            ), {"s": sid, "d": d}).fetchone()
            if frow is None:
                # pick 日可能是 fallback 舊特徵日：改比 pick 當日之前最近的特徵日
                frow = session.execute(text(
                    "SELECT features_json FROM features "
                    "WHERE stock_id = :s AND trading_date <= :d "
                    "ORDER BY trading_date DESC LIMIT 1"
                ), {"s": sid, "d": d}).fetchone()
            if frow is None:
                print(f"  [sanity] {d} {sid}: features 表無資料（無法驗證）")
                continue
            fjson = json.loads(frow[0]) if isinstance(frow[0], str) else frow[0]
            checked += 1
            for feat, val in feat_vals.items():
                truth = fjson.get(feat)
                if truth is None or val is None:
                    continue
                if not np.isclose(float(val), float(truth), rtol=1e-6, atol=1e-9):
                    mismatches += 1
                    print(f"  ❌ MISMATCH {d} {sid} {feat}: pick={val} features={truth}")
                    break

    status = "✅ 通過" if mismatches == 0 else f"❌ {mismatches} 檔 MISMATCH（P0-1 型錯位！）"
    print(f"[sanity] 抽驗 {checked} 檔（{len(dates)} 個 pick 日）：{status}")
    return mismatches


# ── 對帳：月度 picks 實際報酬 vs 回測 period return ──────────────────────────

def _adj_close_frame(session, start: date, end: date) -> pd.DataFrame:
    """讀取區間內還原收盤價（close × adj_factor，factor 缺日已由 populate ffill）。"""
    df = pd.read_sql(text(
        "SELECT p.stock_id, p.trading_date, p.close, COALESCE(f.adj_factor, 1.0) AS adj_factor "
        "FROM raw_prices p LEFT JOIN price_adjust_factors f "
        "ON p.stock_id = f.stock_id AND p.trading_date = f.trading_date "
        "WHERE p.trading_date BETWEEN :s AND :e"
    ), session.connection(), params={"s": start, "e": end})
    df["adj_close"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(
        df["adj_factor"], errors="coerce"
    )
    return df[["stock_id", "trading_date", "adj_close"]]


def run_reconcile(session, baseline_path: str, since: date) -> pd.DataFrame:
    baseline = json.loads(Path(baseline_path).read_text())
    bt_periods = {
        str(p["rebalance_date"]): p for p in baseline.get("periods", [])
    }

    # 月度再平衡日 = 每月第一個 pick 日（與生產月頻邏輯一致）
    pick_dates = [r[0] for r in session.execute(text(
        "SELECT DISTINCT pick_date FROM picks WHERE pick_date >= :d ORDER BY pick_date"
    ), {"d": since}).fetchall()]
    if not pick_dates:
        print(f"[reconcile] {since} 之後無 picks（有效 track record 從 2026-07-06 起算）")
        return pd.DataFrame()

    monthly_first: dict[str, date] = {}
    for d in pick_dates:
        key = f"{d:%Y-%m}"
        monthly_first.setdefault(key, d)

    all_dates = [r[0] for r in session.execute(text(
        "SELECT DISTINCT trading_date FROM raw_prices WHERE trading_date >= :d ORDER BY trading_date"
    ), {"d": since}).fetchall()]

    rows = []
    for month, rb in sorted(monthly_first.items()):
        picks = [r[0] for r in session.execute(text(
            "SELECT stock_id FROM picks WHERE pick_date = :d"
        ), {"d": rb}).fetchall()]
        if not picks:
            continue
        try:
            i = all_dates.index(rb)
        except ValueError:
            continue
        if i + HORIZON_TDAYS >= len(all_dates):
            rows.append({"month": month, "rb_date": rb, "n_picks": len(picks),
                         "live_ret": None, "bt_ret": None, "note": "horizon 未到期"})
            continue
        exit_d = all_dates[i + HORIZON_TDAYS]
        px = _adj_close_frame(session, rb, exit_d)
        rets = []
        for sid in picks:
            sub = px[px["stock_id"] == str(sid)]
            p0 = sub[sub["trading_date"] == rb]["adj_close"]
            p1 = sub[sub["trading_date"] == exit_d]["adj_close"]
            if len(p0) and len(p1) and float(p0.iloc[0]) > 0:
                rets.append(max(float(p1.iloc[0]) / float(p0.iloc[0]) - 1, -0.50))
        live_ret = float(np.mean(rets)) if rets else None
        bt = bt_periods.get(str(rb))
        bt_ret = float(bt["return"]) if bt else None
        rows.append({
            "month": month, "rb_date": rb, "n_picks": len(picks),
            "live_ret": live_ret, "bt_ret": bt_ret,
            "diff": (live_ret - bt_ret) if (live_ret is not None and bt_ret is not None) else None,
            "note": "" if bt else "基準 JSON 無同日 period（回測未涵蓋）",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        print(out.to_string(index=False))
        diffs = out["diff"].dropna() if "diff" in out.columns else pd.Series(dtype=float)
        if len(diffs) >= 3 and abs(diffs.mean()) > 0.03:
            print(f"⚠️ 持續偏差：live-bt 平均 {diffs.mean():+.2%}（>3% 門檻）——檢查 universe/特徵/模型是否脫鉤")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Live vs Backtest 對帳")
    ap.add_argument("--sanity", action="store_true", help="只跑特徵一致性哨兵（可每日跑）")
    ap.add_argument("--baseline", default=None, help="基準回測 JSON 路徑（完整對帳用）")
    ap.add_argument("--since", default=str(P0_FIX_DATE), help="對帳起始日（預設 P0-1 修復日）")
    args = ap.parse_args()

    with get_session() as s:
        bad = run_sanity(s)
        if args.sanity:
            return 1 if bad else 0
        if args.baseline:
            run_reconcile(s, args.baseline, date.fromisoformat(args.since))
        else:
            print("（未指定 --baseline，只跑了 sanity；完整對帳請帶基準 JSON）")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
