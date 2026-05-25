#!/usr/bin/env python
"""Stage 11.1 一次性：補 5 年 news（從 2021-05-25 到 2026-02-23）。

bypass ingest_stock_news 的增量模式，直接呼叫 fetch_dataset per-day。
"""
from __future__ import annotations

import sys
import time as _time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hashlib

import pandas as pd
from sqlalchemy.dialects.mysql import insert

from app.config import load_config
from app.db import get_session
from app.finmind import fetch_dataset
from app.models import RawStockNews

DATASET = "TaiwanStockNews"


def md5_title(t):
    return hashlib.md5((t or "").encode("utf-8")).hexdigest()


def normalize(df):
    df = df.copy()
    df["news_datetime"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    df["title"] = df.get("title", "").fillna("").astype(str).str[:500]
    df["source"] = df.get("source", "").fillna("").astype(str).str[:64]
    df["link"] = df.get("link", "").fillna("").astype(str).str[:500]
    df["title_hash"] = df["title"].apply(md5_title)
    out = df[["stock_id", "news_datetime", "source", "title", "link", "title_hash"]]
    out = out.drop_duplicates(subset=["stock_id", "news_datetime", "title_hash"])
    return out


def upsert(session, df):
    if df.empty:
        return 0
    records = df.to_dict("records")
    n = 0
    for i in range(0, len(records), 5000):
        batch = records[i:i+5000]
        stmt = insert(RawStockNews).values(batch).prefix_with("IGNORE")
        r = session.execute(stmt)
        session.commit()
        n += r.rowcount or 0
    return n


def main():
    cfg = load_config()
    start = date(2021, 5, 25)
    end = date(2026, 2, 23)
    print(f"\n=== 5 year news backfill: {start} → {end} ===")
    print(f"Total days: {(end - start).days}")

    cur = start
    total = 0
    n_days = 0
    n_skipped = 0
    consec_fail = 0
    t0 = _time.time()
    with get_session() as s:
        while cur <= end:
            for attempt in range(2):
                try:
                    df = fetch_dataset(DATASET, start_date=cur, token=cfg.finmind_token)
                    if df.empty:
                        n_skipped += 1
                    else:
                        norm = normalize(df)
                        n = upsert(s, norm)
                        total += n
                        n_days += 1
                    consec_fail = 0
                    break
                except Exception as exc:
                    msg = str(exc)
                    if "your level is free" in msg.lower() and attempt == 0:
                        _time.sleep(60)
                        continue
                    print(f"  [fail] {cur}: {msg[:80]}")
                    consec_fail += 1
                    if consec_fail >= 10:
                        print(f"  連續 10 天失敗，abort. 已寫 {total:,} 筆")
                        return
                    break

            if n_days % 50 == 0 and n_days > 0:
                elapsed = _time.time() - t0
                done_pct = ((cur - start).days / (end - start).days) * 100
                est_total = elapsed / max(done_pct/100, 0.01)
                est_remain = est_total - elapsed
                print(f"  [{cur}] {n_days} days, {total:,} rows, {done_pct:.1f}%, "
                      f"est remain {est_remain/60:.0f}min")

            cur = cur + timedelta(days=1)
            _time.sleep(2.0)

    elapsed = _time.time() - t0
    print(f"\n=== DONE ===")
    print(f"  rows: {total:,}, days w/ data: {n_days}, days skipped: {n_skipped}")
    print(f"  elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
