#!/usr/bin/env python
"""官方 adj factor 切換腳本（Phase 2）：官方自算 factor → price_adjust_factors。

**預設 --dry-run**：只讀 DB、印 diff 統計 + 寫報告 JSON，不寫任何生產表；
`--apply` 才 TRUNCATE price_adjust_factors + 重灌（何時 apply 由人工決定，
apply 後需全量重建 features/labels + 重訓）。

資料來源：scripts/build_official_adj_factors.py 產出的
artifacts/adj_official/factors.parquet（官方事件累積 factor，僅含有事件的股票）
與 events.parquet（事件明細，供颱風停市修復清單與凍結後事件統計）。

展開語義（比照 scripts/populate_adj_factors.py）：
- per-stock 把 factor ffill/bfill 展開到該股「全部」raw 交易日
  （事件步階間缺日沿用前值；leading gap 用最早已知 factor；
  build 端之後的新交易日由最後值 ffill = 1.0）
- clip 到 [0.1, 10]，觸 clip 股票輸出 official_factor_clip_touched.json
  （深度分割股如 5314 世紀一拆二十，累積 factor by construction < 0.1，
  會落入清單——與 FinMind 快照口徑一致，消費端據清單評估排除）
- 無事件股票不寫 → build 端自動 1.0（與 populate「整檔無 adj 不寫」語義一致）

dry-run diff 統計：
- 新舊逐 (stock, date) 比對：會變動列數、變幅分佈、新增/移除股票
- 移除股票中 factor 非平坦（≠1.0）者 = FinMind 資訊損失清單（人工審閱）
- 颱風停市修復清單：事件日為平日但全市場停市（2024 凱米/山陀兒/康芮），
  FinMind 快照已證實整批漏調整，官方 factor 修復這批股票
- 凍結日（2026-06-23）後新事件將被調整的股票數（切換的主要動機）

用法：
    python scripts/cutover_official_adj_factors.py            # dry-run（預設）
    python scripts/cutover_official_adj_factors.py --apply    # 真正 TRUNCATE+重灌
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from skills.official_adj_factors import SNAPSHOT_FREEZE_DATE

DEFAULT_DIR = ROOT / "artifacts" / "adj_official"
FACTORS_PARQUET = DEFAULT_DIR / "factors.parquet"
EVENTS_PARQUET = DEFAULT_DIR / "events.parquet"
CLIP_TOUCHED_JSON = DEFAULT_DIR / "official_factor_clip_touched.json"
DRY_RUN_REPORT_JSON = DEFAULT_DIR / "cutover_dry_run_report.json"

#: clip 邊界（比照 scripts/populate_adj_factors.py）
LO, HI = 0.1, 10.0

#: 變幅分佈 bucket（相對變動 |new/old - 1|）
DIFF_BUCKETS = [(1e-6, 0.001, "<0.1%"), (0.001, 0.01, "0.1%~1%"),
                (0.01, 0.05, "1%~5%"), (0.05, 0.20, "5%~20%"),
                (0.20, float("inf"), ">20%")]

#: --apply 安全底線：官方 factor 覆蓋股票數低於此值視為資料異常，拒絕重灌
MIN_APPLY_STOCKS = 1000


# ──────────────────────────────────────────────
# 純函數（單元測試對象）
# ──────────────────────────────────────────────

def expand_official_factors(factors_df: pd.DataFrame,
                            raw_days_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """官方 factor（僅事件股、build 窗口交易日）→ 該股全部 raw 交易日。

    比照 populate_adj_factors：per-stock ffill（內部缺日/窗口後新交易日沿用前值）
    + bfill（窗口前 leading gap 用最早已知 factor）。

    Returns:
        (expanded_df[stock_id, trading_date, adj_factor], n_gap_filled)
    """
    f = factors_df.copy()
    f["stock_id"] = f["stock_id"].astype(str)
    f["trading_date"] = pd.to_datetime(f["trading_date"]).dt.normalize()
    f["adj_factor"] = pd.to_numeric(f["adj_factor"], errors="coerce")

    days = raw_days_df.copy()
    days["stock_id"] = days["stock_id"].astype(str)
    days["trading_date"] = pd.to_datetime(days["trading_date"]).dt.normalize()

    covered = days[days["stock_id"].isin(set(f["stock_id"].unique()))]
    full = covered.merge(f, on=["stock_id", "trading_date"], how="left")
    full = full.sort_values(["stock_id", "trading_date"])
    n_gap = int(full["adj_factor"].isna().sum())
    full["adj_factor"] = full.groupby("stock_id", sort=False)["adj_factor"].ffill()
    full["adj_factor"] = full.groupby("stock_id", sort=False)["adj_factor"].bfill()
    out = full.dropna(subset=["adj_factor"]).reset_index(drop=True)
    return out[["stock_id", "trading_date", "adj_factor"]], n_gap


def clip_and_record(df: pd.DataFrame, lo: float = LO,
                    hi: float = HI) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """clip factor 到 [lo, hi] 並記錄觸 clip 股票（比照 populate_adj_factors P2-9）。

    Returns:
        (clipped_df, clip_payload)——payload schema 與
        artifacts/adj_prices/adj_factor_clip_touched.json 一致
        （stock_id → {clipped_days, min_ratio, max_ratio}）。
    """
    out = df.copy()
    raw = out["adj_factor"]
    clipped_mask = (raw < lo) | (raw > hi)
    payload: Dict[str, Any] = {}
    if clipped_mask.any():
        stats = (
            out[clipped_mask]
            .assign(_ratio=raw[clipped_mask])
            .groupby("stock_id")["_ratio"]
            .agg(["min", "max", "size"])
        )
        payload = {
            str(sid): {
                "clipped_days": int(row["size"]),
                "min_ratio": float(row["min"]),
                "max_ratio": float(row["max"]),
            }
            for sid, row in stats.iterrows()
        }
    out["adj_factor"] = raw.clip(lo, hi)
    return out, payload


def find_market_closure_fix_events(events_df: pd.DataFrame,
                                   market_days: set) -> pd.DataFrame:
    """事件日為平日但全市場停市（颱風假）→ FinMind 快照漏調整、官方修復的事件。

    market_days：raw_prices 全市場交易日集合（datetime.date）。
    """
    if events_df.empty:
        return events_df.copy()
    ev = events_df.copy()
    ev["event_date"] = pd.to_datetime(ev["event_date"]).dt.date
    mask = ev["event_date"].map(
        lambda d: d.weekday() < 5 and d not in market_days
    )
    return ev[mask].reset_index(drop=True)


def compute_cutover_diff(new_df: pd.DataFrame, old_df: pd.DataFrame,
                         tol: float = 1e-6,
                         freeze_date: Optional[date] = None) -> Dict[str, Any]:
    """新（官方）vs 舊（DB 現況）factor 逐 (stock, date) 比對 → diff 統計。

    - n_changed / 變幅分佈：兩邊都有的列中 |new/old - 1| > tol 者
    - dropped：舊有新無（TRUNCATE 後消失）——其中 factor 非平坦（≠1.0）的股票
      = FinMind 資訊損失（官方窗口外事件 / FinMind 特有調整）
    - added：新有舊無（官方覆蓋 DB 缺口）
    """
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["stock_id"] = d["stock_id"].astype(str)
        d["trading_date"] = pd.to_datetime(d["trading_date"]).dt.normalize()
        d["adj_factor"] = pd.to_numeric(d["adj_factor"], errors="coerce")
        return d.dropna(subset=["adj_factor"])

    new = _norm(new_df)
    old = _norm(old_df)
    merged = old.merge(new, on=["stock_id", "trading_date"], how="outer",
                       suffixes=("_old", "_new"), indicator=True)

    both = merged[merged["_merge"] == "both"]
    rel = (both["adj_factor_new"] / both["adj_factor_old"] - 1.0).abs()
    changed = rel[rel > tol]

    buckets = {label: int(((changed > lo_) & (changed <= hi_)).sum())
               for lo_, hi_, label in DIFF_BUCKETS}
    quantiles = ({"p50": float(changed.quantile(0.5)),
                  "p90": float(changed.quantile(0.9)),
                  "p99": float(changed.quantile(0.99)),
                  "max": float(changed.max())}
                 if len(changed) else {})

    only_old = merged[merged["_merge"] == "left_only"]
    only_new = merged[merged["_merge"] == "right_only"]

    old_stocks = set(old["stock_id"].unique())
    new_stocks = set(new["stock_id"].unique())
    dropped_stocks = sorted(old_stocks - new_stocks)
    added_stocks = sorted(new_stocks - old_stocks)

    # 被移除股票中 factor 非平坦者 = FinMind 資訊損失清單
    dropped_nonflat = []
    if dropped_stocks:
        drop_rows = old[old["stock_id"].isin(set(dropped_stocks))]
        dev = (drop_rows["adj_factor"] - 1.0).abs()
        nonflat_ids = drop_rows.loc[dev > tol, "stock_id"].unique()
        dropped_nonflat = sorted(map(str, nonflat_ids))

    # 每檔最大變幅（人工審閱 top 清單用）
    per_stock_max = (
        both.assign(_rel=rel)
        .groupby("stock_id")["_rel"].max()
        .sort_values(ascending=False)
    )
    top_changed = [
        {"stock_id": str(sid), "max_rel_diff": float(v)}
        for sid, v in per_stock_max.head(20).items() if v > tol
    ]

    stats: Dict[str, Any] = {
        "n_old_rows": int(len(old)),
        "n_new_rows": int(len(new)),
        "n_common_rows": int(len(both)),
        "n_changed_rows": int(len(changed)),
        "changed_pct_of_common": (float(len(changed)) / len(both) if len(both) else None),
        "change_magnitude_buckets": buckets,
        "change_magnitude_quantiles": quantiles,
        "n_rows_only_in_old": int(len(only_old)),
        "n_rows_only_in_new": int(len(only_new)),
        "n_stocks_old": len(old_stocks),
        "n_stocks_new": len(new_stocks),
        "n_stocks_dropped": len(dropped_stocks),
        "n_stocks_dropped_nonflat": len(dropped_nonflat),
        "stocks_dropped_nonflat": dropped_nonflat,
        "n_stocks_added": len(added_stocks),
        "stocks_added": added_stocks,
        "top_changed_stocks": top_changed,
        "tolerance": tol,
    }
    if freeze_date is not None:
        freeze_ts = pd.Timestamp(freeze_date)
        post = both[both["trading_date"] > freeze_ts]
        post_rel = (post["adj_factor_new"] / post["adj_factor_old"] - 1.0).abs()
        stats["n_changed_rows_post_freeze"] = int((post_rel > tol).sum())
        stats["n_stocks_changed_post_freeze"] = int(
            post.loc[post_rel > tol, "stock_id"].nunique())
    return stats


# ──────────────────────────────────────────────
# DB I/O（main 專用；dry-run 只 SELECT）
# ──────────────────────────────────────────────

def _load_raw_days() -> pd.DataFrame:
    from sqlalchemy import text
    from app.db import get_session
    with get_session() as s:
        return pd.read_sql(
            text(r"SELECT stock_id, trading_date FROM raw_prices "
                 r"WHERE stock_id REGEXP '^[0-9]{4}$'"),
            s.connection(),
        )


def _load_db_factors() -> pd.DataFrame:
    from sqlalchemy import text
    from app.db import get_session
    with get_session() as s:
        return pd.read_sql(
            text("SELECT stock_id, trading_date, adj_factor FROM price_adjust_factors"),
            s.connection(),
        )


def _apply_to_db(out: pd.DataFrame) -> None:
    """TRUNCATE + 重灌（比照 populate_adj_factors 寫入路徑）。僅 --apply 走到這裡。"""
    from sqlalchemy import text
    from app.db import get_session
    with get_session() as s:
        s.execute(text("TRUNCATE TABLE price_adjust_factors"))
        s.commit()
        eng = s.get_bind()
        ch = 50000
        for i in range(0, len(out), ch):
            out.iloc[i:i + ch].to_sql("price_adjust_factors", eng, if_exists="append",
                                      index=False, method="multi")
            if (i // ch) % 10 == 0:
                print(f"  寫入 {min(i + ch, len(out)):,}/{len(out):,}", flush=True)
        r = s.execute(text(
            "SELECT COUNT(*), AVG(adj_factor), MIN(adj_factor), MAX(adj_factor) "
            "FROM price_adjust_factors")).fetchone()
    print(f"完成：price_adjust_factors {r[0]:,} 列  avg={r[1]:.4f} "
          f"min={r[2]:.4f} max={r[3]:.4f}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factors", type=Path, default=FACTORS_PARQUET)
    ap.add_argument("--events", type=Path, default=EVENTS_PARQUET)
    ap.add_argument("--apply", action="store_true",
                    help="真正 TRUNCATE+重灌 price_adjust_factors（預設 dry-run 不寫 DB）")
    args = ap.parse_args()

    mode = "APPLY" if args.apply else "DRY-RUN（不寫 DB）"
    print(f"=== 官方 adj factor 切換：{mode} ===")

    factors = pd.read_parquet(args.factors)
    events = pd.read_parquet(args.events)
    if factors.empty or events.empty:
        print("factors/events parquet 為空——先跑 scripts/build_official_adj_factors.py")
        return 1
    print(f"官方 factor：{len(factors):,} 列 / {factors['stock_id'].nunique()} 檔；"
          f"事件 {len(events):,} 筆")

    raw_days = _load_raw_days()
    market_days = set(pd.to_datetime(raw_days["trading_date"]).dt.date.unique())

    expanded, n_gap = expand_official_factors(factors, raw_days)
    new_df, clip_payload = clip_and_record(expanded)
    print(f"展開後：{len(new_df):,} 列（缺日補 {n_gap:,} 列）；"
          f"觸 clip 股票 {len(clip_payload)} 檔")

    CLIP_TOUCHED_JSON.parent.mkdir(parents=True, exist_ok=True)
    CLIP_TOUCHED_JSON.write_text(
        json.dumps(clip_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {CLIP_TOUCHED_JSON}")

    old_df = _load_db_factors()
    stats = compute_cutover_diff(new_df, old_df, freeze_date=SNAPSHOT_FREEZE_DATE)

    # 颱風停市修復清單（FinMind 快照已證實漏調整、官方 factor 修復）
    closure_events = find_market_closure_fix_events(events, market_days)
    closure_stocks = sorted(closure_events["stock_id"].unique().tolist())
    ev_dates = pd.to_datetime(events["event_date"]).dt.date
    post_freeze = events[ev_dates > SNAPSHOT_FREEZE_DATE]

    report: Dict[str, Any] = {
        "mode": "apply" if args.apply else "dry_run",
        "generated_at": pd.Timestamp.now().isoformat(),
        "factors_parquet": str(args.factors),
        "snapshot_freeze_date": SNAPSHOT_FREEZE_DATE.isoformat(),
        **stats,
        "n_clip_touched_stocks": len(clip_payload),
        "clip_touched_stocks": sorted(clip_payload),
        "n_market_closure_fix_events": int(len(closure_events)),
        "n_market_closure_fix_stocks": len(closure_stocks),
        "market_closure_fix_stocks": closure_stocks,
        "n_post_freeze_events": int(len(post_freeze)),
        "n_post_freeze_stocks": int(post_freeze["stock_id"].nunique()),
    }
    DRY_RUN_REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[write] {DRY_RUN_REPORT_JSON}")

    # ── 人類可讀摘要 ──
    print(f"\n--- diff 統計（官方 vs DB 現況）---")
    print(f"舊列數 {stats['n_old_rows']:,} / 新列數 {stats['n_new_rows']:,} "
          f"/ 交集 {stats['n_common_rows']:,}")
    pct = stats["changed_pct_of_common"]
    print(f"會變動列數：{stats['n_changed_rows']:,}"
          f"（{pct:.2%}）" if pct is not None else "會變動列數：0")
    print(f"變幅分佈：{stats['change_magnitude_buckets']}")
    if stats["change_magnitude_quantiles"]:
        q = stats["change_magnitude_quantiles"]
        print(f"變幅分位：p50={q['p50']:.4%} p90={q['p90']:.4%} "
              f"p99={q['p99']:.4%} max={q['max']:.2%}")
    print(f"凍結日後會變動：{stats.get('n_changed_rows_post_freeze', 0):,} 列 / "
          f"{stats.get('n_stocks_changed_post_freeze', 0)} 檔（切換主要動機）")
    print(f"移除股票：{stats['n_stocks_dropped']}"
          f"（其中 factor 非平坦 = FinMind 資訊損失：{stats['n_stocks_dropped_nonflat']} 檔）")
    print(f"新增股票：{stats['n_stocks_added']}")
    print(f"颱風停市修復：{report['n_market_closure_fix_events']} 事件 / "
          f"{report['n_market_closure_fix_stocks']} 檔")
    print(f"凍結後新事件將被調整：{report['n_post_freeze_events']} 事件 / "
          f"{report['n_post_freeze_stocks']} 檔")

    if not args.apply:
        print("\nDRY-RUN 結束（未寫 DB）。確認後用 --apply 執行切換，"
              "apply 後需全量重建 features/labels + 重訓。")
        return 0

    # ── --apply 安全底線 ──
    if stats["n_stocks_new"] < MIN_APPLY_STOCKS:
        print(f"拒絕 apply：官方 factor 僅覆蓋 {stats['n_stocks_new']} 檔 "
              f"(< {MIN_APPLY_STOCKS})，資料疑似不完整")
        return 1
    if new_df["adj_factor"].isna().any() or (new_df["adj_factor"] <= 0).any():
        print("拒絕 apply：factor 含 NaN 或非正值")
        return 1
    new_max_date = new_df["trading_date"].max().date()
    if new_max_date < SNAPSHOT_FREEZE_DATE:
        print(f"拒絕 apply：官方 factor 最新日 {new_max_date} 未覆蓋凍結日"
              f"（{SNAPSHOT_FREEZE_DATE}），先重跑 build 到最新")
        return 1

    out = new_df.copy()
    out["trading_date"] = out["trading_date"].dt.date
    print(f"\nAPPLY：TRUNCATE price_adjust_factors + 重灌 {len(out):,} 列 ...")
    _apply_to_db(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
