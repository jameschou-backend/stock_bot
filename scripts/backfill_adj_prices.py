#!/usr/bin/env python
"""趁 FinMind sponsor 到期前（2026-06-24），搶抓全市場 10 年『還原股價』。

TaiwanStockPriceAdj 為 sponsor 專屬（TWSE/TPEx 免費版沒有）。這是修正
adj_close=1.0 偏差所缺的唯一資料源。本腳本一檔一 call（支援 data_id+日期範圍），
分批 checkpoint 寫 parquet，可續跑（重跑會跳過已完成的 stock_id）。

用法：
    python scripts/backfill_adj_prices.py                  # 全 universe，2016-01-01 起
    python scripts/backfill_adj_prices.py --start 2016-01-01

輸出：artifacts/adj_prices/batch_*.parquet（每批一檔）+ 完成後可 merge。
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import glob
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(str(ROOT / ".env"))

import pandas as pd
from sqlalchemy import text

from app.db import get_session
from app.finmind import fetch_dataset, FinMindError

OUT_DIR = ROOT / "artifacts" / "adj_prices"
DATASET = "TaiwanStockPriceAdj"
BATCH = 100  # 每 100 檔 flush 一次 parquet
RENAME = {
    "date": "trading_date", "open": "open", "max": "high", "min": "low",
    "close": "close", "Trading_Volume": "volume", "Trading_money": "amount",
}


def _done_stock_ids() -> set:
    done = set()
    for f in glob.glob(str(OUT_DIR / "batch_*.parquet")):
        try:
            done |= set(pd.read_parquet(f, columns=["stock_id"])["stock_id"].astype(str).unique())
        except Exception:
            pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = os.getenv("FINMIND_TOKEN")
    if not tok:
        print("FINMIND_TOKEN 不存在，中止"); return 1

    with get_session() as s:
        # 只抓四碼普通股（排除權證/ETF 等；strategy universe 也只用四碼）
        all_ids = [str(r[0]) for r in s.execute(
            text(r"SELECT DISTINCT stock_id FROM raw_prices "
                 r"WHERE stock_id REGEXP '^[0-9]{4}$' ORDER BY stock_id")).fetchall()]
    done = _done_stock_ids()
    todo = [sid for sid in all_ids if sid not in done]
    print(f"universe={len(all_ids)}  已完成={len(done)}  待抓={len(todo)}  區間 {start}~{end}", flush=True)

    batch_rows = []
    batch_idx = len(glob.glob(str(OUT_DIR / "batch_*.parquet")))
    ok = err = empty = 0
    t0 = time.time()
    for i, sid in enumerate(todo, 1):
        try:
            df = fetch_dataset(DATASET, start, end, token=tok, data_id=sid, requests_per_hour=6000)
            if df is not None and len(df):
                keep = [c for c in RENAME if c in df.columns]
                df = df[["stock_id"] + keep].rename(columns=RENAME)
                df["stock_id"] = df["stock_id"].astype(str)
                batch_rows.append(df)
                ok += 1
            else:
                empty += 1
        except FinMindError as e:
            print(f"[{i}/{len(todo)}] {sid} FinMindError（疑配額）：{str(e)[:80]} → 中止保存進度", flush=True)
            break
        except Exception as e:
            err += 1
            print(f"[{i}/{len(todo)}] {sid} 失敗：{type(e).__name__} {str(e)[:60]}", flush=True)
        # flush
        if len(batch_rows) >= BATCH:
            out = OUT_DIR / f"batch_{batch_idx:04d}.parquet"
            pd.concat(batch_rows, ignore_index=True).to_parquet(out, index=False)
            batch_idx += 1; batch_rows = []
            rate = i / max(1e-9, (time.time() - t0))
            print(f"[{i}/{len(todo)}] flush {out.name}  ok={ok} empty={empty} err={err}  {rate*3600:.0f}/hr", flush=True)
    # 收尾
    if batch_rows:
        out = OUT_DIR / f"batch_{batch_idx:04d}.parquet"
        pd.concat(batch_rows, ignore_index=True).to_parquet(out, index=False)
        print(f"final flush {out.name}", flush=True)
    print(f"完成：ok={ok} empty={empty} err={err}  耗時 {(time.time()-t0)/60:.1f} 分", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
