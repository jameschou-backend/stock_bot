#!/usr/bin/env python
"""抓 TWSE/TPEx 官方除權息 + 減資公告 → 自算 adj factor（Phase 1：引擎 + 對帳，不切生產）。

輸出（--out-dir，預設 artifacts/adj_official/）：
- events.parquet   事件明細（stock_id/event_date/market/source/event_type/比率/現金增資旗標）
- factors.parquet  per-stock 全交易日累積 factor（僅含有事件的股票；缺股 = 1.0）
- checkpoints/     每 chunk 原始 JSON（續跑直接讀檔，不重打 API）

--reconcile：與凍結快照（DB price_adjust_factors，populate_adj_factors 由
FinMind adj parquet 反推）在重疊窗逐 (stock, date) 對帳，輸出
reconcile_report.json + reconcile_mismatches.parquet + extra_snapshot_jumps.parquet。
match 判準：|r_official / r_snap - 1| < 0.5%。

用法：
    python scripts/build_official_adj_factors.py --start 2024-01-01 --end 2026-07-09
    python scripts/build_official_adj_factors.py --start 2024-01-01 --reconcile

僅讀 DB（SELECT raw_prices / price_adjust_factors），不寫任何生產表。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import text

from skills.official_adj_factors import (
    FETCH_SPECS,
    SNAPSHOT_FREEZE_DATE,
    SOURCE_CAPITAL_REDUCTION,
    OfficialAdjClient,
    build_factor_frame,
    events_to_dataframe,
    load_clip_touched,
    reconcile_events_vs_snapshot,
)

DEFAULT_OUT_DIR = ROOT / "artifacts" / "adj_official"
CLIP_TOUCHED_JSON = ROOT / "artifacts" / "adj_prices" / "adj_factor_clip_touched.json"


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """[start, end] 切成日曆月 chunk（checkpoint 粒度）。"""
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunks.append((cur, min(nxt - timedelta(days=1), end)))
        cur = nxt
    return chunks


def year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """減資事件稀少，用年 chunk 省 request。"""
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        year_end = date(cur.year, 12, 31)
        chunks.append((cur, min(year_end, end)))
        cur = year_end + timedelta(days=1)
    return chunks


def _fetch_chunk_validated(client: OfficialAdjClient, fetcher_attr: str, parser,
                           c_start: date, c_end: date, retries: int = 3) -> tuple:
    """抓一個 chunk 並「先 parse 驗證再回傳」。

    TWSE 偶發回無意義錯誤 stat（如「查詢開始日期小於92年5月5日」），且 CDN 會把
    錯誤以 query string 為 key 快取住——parser 對這類 stat 會 raise TWSEError，
    此處帶 cache_bust（隨機 `_` 參數）重試繞開毒快取；靜默當空結果會漏抓整月事件。
    """
    import time as _time
    from app.twse_client import TWSEError
    last_exc = None
    for attempt in range(retries + 1):
        payload = getattr(client, fetcher_attr)(c_start, c_end, cache_bust=attempt > 0)
        try:
            return payload, parser(payload)
        except TWSEError as exc:
            last_exc = exc
            wait_s = 5.0 * (attempt + 1)
            print(f"  ⚠ {fetcher_attr} {c_start}~{c_end} 錯誤 stat（{exc}），"
                  f"{wait_s:.0f}s 後帶 cache-bust 重試 {attempt + 1}/{retries}", flush=True)
            _time.sleep(wait_s)
    raise SystemExit(f"chunk 抓取失敗（{fetcher_attr} {c_start}~{c_end}）：{last_exc}")


def _write_checkpoint_atomic(ckpt: Path, payload) -> None:
    """checkpoint 原子寫入（temp + os.replace）。

    直接 write_text 在程序中斷/磁碟滿時會留下截斷 JSON，之後每次 resume 都
    parse 失敗；tmp 檔帶 pid 後綴避免並行執行互踩。
    """
    tmp = ckpt.with_name(f"{ckpt.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, ckpt)
    finally:
        tmp.unlink(missing_ok=True)


def fetch_all_events(client: OfficialAdjClient, start: date, end: date,
                     ckpt_dir: Path, resume: bool) -> pd.DataFrame:
    """四個來源 × chunk 抓取（含 checkpoint 續跑）→ 事件 DataFrame。

    checkpoint 只在 parse 驗證通過後「原子」寫入（temp + os.replace）；
    resume 時若舊 checkpoint 損壞（截斷 JSON → JSONDecodeError/ValueError）
    或 parse 失敗（先前版本存了暫時性錯誤 payload → TWSEError）會自動重抓，
    不需人工刪檔。
    """
    from app.twse_client import TWSEError
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    all_events = []
    for kind, parser, fetcher_attr in FETCH_SPECS:
        chunker = year_chunks if kind.endswith("capital_reduction") else month_chunks
        for c_start, c_end in chunker(start, end):
            ckpt = ckpt_dir / f"{kind}_{c_start:%Y%m%d}_{c_end:%Y%m%d}.json"
            events, src = None, "checkpoint"
            if resume and ckpt.exists():
                try:
                    events = parser(json.loads(ckpt.read_text(encoding="utf-8")))
                except (TWSEError, json.JSONDecodeError, ValueError) as exc:
                    # json.JSONDecodeError 是 ValueError 子類；一併捕 ValueError
                    # 涵蓋其他損壞形態（如非 UTF-8 殘骸）——損壞即重抓，不 crash
                    print(f"  ⚠ checkpoint {ckpt.name} 內容無效（{type(exc).__name__}: {exc}），重抓",
                          flush=True)
            if events is None:
                payload, events = _fetch_chunk_validated(client, fetcher_attr, parser,
                                                         c_start, c_end)
                _write_checkpoint_atomic(ckpt, payload)
                src = "http"
            all_events.extend(events)
            print(f"  {kind} {c_start} ~ {c_end}: {len(events)} events ({src})", flush=True)
    return events_to_dataframe(all_events)


def load_trading_days(start: date, end: date, stock_ids: list[str]) -> pd.DataFrame:
    """讀有事件股票在窗口內的實際交易日（只 SELECT，不寫）。"""
    from app.db import get_session
    with get_session() as s:
        df = pd.read_sql(
            text("SELECT stock_id, trading_date FROM raw_prices "
                 "WHERE trading_date BETWEEN :a AND :b "
                 "AND stock_id REGEXP '^[0-9]{4}$'"),
            s.connection(), params={"a": start.isoformat(), "b": end.isoformat()},
        )
    df["stock_id"] = df["stock_id"].astype(str)
    return df[df["stock_id"].isin(set(stock_ids))]


def load_snapshot_factors(start: date, end: date) -> pd.DataFrame:
    """讀凍結快照 factor（DB price_adjust_factors，只 SELECT）。"""
    from app.db import get_session
    with get_session() as s:
        return pd.read_sql(
            text("SELECT stock_id, trading_date, adj_factor FROM price_adjust_factors "
                 "WHERE trading_date BETWEEN :a AND :b"),
            s.connection(), params={"a": start.isoformat(), "b": end.isoformat()},
        )


def run_reconcile(events_df: pd.DataFrame, start: date, snapshot_end: date,
                  out_dir: Path, tolerance: float) -> dict:
    overlap_events = events_df[
        (pd.to_datetime(events_df["event_date"]).dt.date >= start)
        & (pd.to_datetime(events_df["event_date"]).dt.date <= snapshot_end)
    ]
    post_freeze = events_df[pd.to_datetime(events_df["event_date"]).dt.date > snapshot_end]

    print(f"\n[reconcile] 重疊窗 {start} ~ {snapshot_end}："
          f"官方事件 {len(overlap_events)}，凍結後未調整事件 {len(post_freeze)}")
    snapshot = load_snapshot_factors(start - timedelta(days=14), snapshot_end)
    print(f"[reconcile] 快照 factor {len(snapshot):,} 列 / {snapshot['stock_id'].nunique()} 檔")

    clip = load_clip_touched(str(CLIP_TOUCHED_JSON))
    result = reconcile_events_vs_snapshot(
        overlap_events, snapshot, tolerance=tolerance, clip_touched=clip,
        extra_jump_min_date=start,
    )

    mismatches = result.event_results[~result.event_results["matched"]]
    mismatches = mismatches.sort_values(["reason", "stock_id", "event_date"])
    _to_parquet(mismatches, out_dir / "reconcile_mismatches.parquet")
    _to_parquet(result.extra_jumps, out_dir / "extra_snapshot_jumps.parquet")

    report = {
        "overlap_window": [start.isoformat(), snapshot_end.isoformat()],
        **result.summary,
        "post_freeze_unadjusted_events": int(len(post_freeze)),
        "post_freeze_stocks": sorted(post_freeze["stock_id"].unique().tolist()),
    }
    (out_dir / "reconcile_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n=== 對帳結果（tolerance {tolerance:.2%}）===")
    print(f"官方事件（重疊窗）: {result.summary['n_official_events']}")
    print(f"可對帳事件        : {result.summary['n_eligible']}")
    mr = result.summary["match_rate"]
    print(f"match rate        : {mr:.4%}" if mr is not None else "match rate        : n/a")
    print(f"  其中經開盤競價基準才 match: {result.summary['n_matched_via_opening_ref']}")
    print("mismatch 分類:")
    for reason, cnt in sorted(result.summary["reason_counts"].items(), key=lambda kv: -kv[1]):
        if not reason.startswith("matched"):
            print(f"  {reason}: {cnt}")
    print(f"快照多出跳動（官方無事件）: {result.summary['n_extra_snapshot_jumps']}")
    print(f"凍結後（>{snapshot_end}）未調整官方事件: {len(post_freeze)} "
          f"檔數 {post_freeze['stock_id'].nunique()}")
    if not mismatches.empty:
        print("\nmismatch top 15（依 rel_diff）：")
        cols = ["stock_id", "event_date", "market", "event_type", "reason",
                "r_official", "r_snap", "rel_diff"]
        top = mismatches.dropna(subset=["rel_diff"]).nlargest(15, "rel_diff")
        print(top[cols].to_string(index=False))
    return report


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, date)).any():
            out[c] = pd.to_datetime(out[c], errors="coerce")
    out.to_parquet(path, index=False)
    print(f"[write] {path}（{len(out):,} 列）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--delay", type=float, default=None, help="request 間隔秒數（預設 env 或 2.0）")
    ap.add_argument("--reconcile", action="store_true", help="與凍結快照對帳（讀 DB）")
    ap.add_argument("--tolerance", type=float, default=0.005, help="match 容忍度（預設 0.5%%）")
    ap.add_argument("--snapshot-end", type=date.fromisoformat, default=SNAPSHOT_FREEZE_DATE,
                    help="快照凍結日（預設 2026-06-23；之後的 DB factor 是 pipeline 補 1.0）")
    ap.add_argument("--no-resume", action="store_true", help="忽略 checkpoint 重抓")
    ap.add_argument("--skip-factors", action="store_true",
                    help="只抓事件不展開 factor（不需 DB）")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OfficialAdjClient(**({"delay": args.delay} if args.delay is not None else {}))
    print(f"抓取官方事件 {args.start} ~ {args.end}（checkpoint: {out_dir/'checkpoints'}）")
    events_df = fetch_all_events(client, args.start, args.end,
                                 out_dir / "checkpoints", resume=not args.no_resume)
    print(f"\n事件合計 {len(events_df)} 筆 / {events_df['stock_id'].nunique() if not events_df.empty else 0} 檔 "
          f"（除權息 {(events_df['source'] != SOURCE_CAPITAL_REDUCTION).sum() if not events_df.empty else 0}、"
          f"減資 {(events_df['source'] == SOURCE_CAPITAL_REDUCTION).sum() if not events_df.empty else 0}）")
    _to_parquet(events_df, out_dir / "events.parquet")

    if not args.skip_factors and not events_df.empty:
        td = load_trading_days(args.start, args.end, events_df["stock_id"].unique().tolist())
        factors_df = build_factor_frame(events_df, td)
        _to_parquet(factors_df, out_dir / "factors.parquet")

    if args.reconcile:
        run_reconcile(events_df, args.start, min(args.snapshot_end, args.end),
                      out_dir, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
