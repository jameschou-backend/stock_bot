"""One-shot prices backfill via FinMind single-stock query.

繞過 fetch_dataset_by_stocks 的 batch_query bug（"parameter data_id is illegal"）。
每股 1 個 API call 涵蓋整個日期區間。

用法：
    # 預設：補從 DB latest+1 到今天
    python scripts/backfill_prices_single_stock.py

    # 指定起始日期（覆寫 DB latest）
    python scripts/backfill_prices_single_stock.py --start-date 2026-05-08

    # 限制股票數（測試用）
    python scripts/backfill_prices_single_stock.py --limit 50

    # 從特定 stock_id 之後開始（中斷後續跑用）
    python scripts/backfill_prices_single_stock.py --resume-from 2330
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert

from app.config import load_config
from app.db import get_session
from app.finmind import FinMindError, fetch_dataset, fetch_stock_list
from app.models import RawPrice
from skills.ingest_prices import _normalize_prices


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD；不給則用 DB latest+1")
    p.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD；不給則用今天")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 支股票（測試用）")
    p.add_argument("--resume-from", type=str, default=None, help="跳過直到看到此 stock_id（中斷恢復用）")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config()
    today = datetime.now(ZoneInfo(config.tz)).date()

    # 解析日期區間
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        with get_session() as s:
            max_date = s.query(func.max(RawPrice.trading_date)).scalar()
        start_date = (max_date + timedelta(days=1)) if max_date else (today - timedelta(days=30))

    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else today

    if start_date > end_date:
        print(f"start_date {start_date} > end_date {end_date}，無事可做")
        return 0

    print(f"=== 單股 backfill prices ===")
    print(f"  期間：{start_date} ~ {end_date} ({(end_date - start_date).days + 1} 天)")

    # 取得股票清單
    print("  取得 stock list ...", end="", flush=True)
    stock_ids = fetch_stock_list(
        config.finmind_token,
        requests_per_hour=config.finmind_requests_per_hour,
        max_retries=config.finmind_retry_max,
        backoff_seconds=config.finmind_retry_backoff,
    )
    print(f" {len(stock_ids)} 支")

    if args.resume_from:
        if args.resume_from in stock_ids:
            idx = stock_ids.index(args.resume_from)
            stock_ids = stock_ids[idx:]
            print(f"  resume from {args.resume_from}（剩 {len(stock_ids)} 支）")
        else:
            print(f"  ⚠ resume-from {args.resume_from} 不在清單中，忽略")

    if args.limit:
        stock_ids = stock_ids[: args.limit]
        print(f"  --limit {args.limit}：只跑前 {len(stock_ids)} 支")

    print()
    ok_cnt = 0
    empty_cnt = 0
    err_cnt = 0
    total_rows = 0
    t0 = time.monotonic()

    for i, sid in enumerate(stock_ids, 1):
        try:
            df = fetch_dataset(
                "TaiwanStockPrice",
                start_date=start_date,
                end_date=end_date,
                token=config.finmind_token,
                data_id=sid,
                requests_per_hour=config.finmind_requests_per_hour,
                max_retries=config.finmind_retry_max,
                backoff_seconds=config.finmind_retry_backoff,
            )
            if df.empty:
                empty_cnt += 1
            else:
                df = _normalize_prices(df)
                if df.empty:
                    empty_cnt += 1
                else:
                    records = df.to_dict("records")
                    with get_session() as s:
                        stmt = insert(RawPrice).values(records)
                        update_cols = {c: stmt.inserted[c] for c in ["open", "high", "low", "close", "volume"]}
                        stmt = stmt.on_duplicate_key_update(**update_cols)
                        s.execute(stmt)
                        s.commit()
                    total_rows += len(records)
                    ok_cnt += 1
        except FinMindError as exc:
            err_cnt += 1
            msg = str(exc)
            if "402" in msg or "429" in msg or "quota" in msg.lower():
                print(f"\n❌ [{i}/{len(stock_ids)}] {sid} quota error，中止：{exc}")
                print(f"   已成功：ok={ok_cnt} empty={empty_cnt} err={err_cnt} rows={total_rows}")
                print(f"   resume 指令：python scripts/backfill_prices_single_stock.py --resume-from {sid} --start-date {start_date}")
                return 1
            if i <= 10 or err_cnt <= 5:
                print(f"  [{i}/{len(stock_ids)}] {sid} ERROR: {exc}")
        except Exception as exc:
            err_cnt += 1
            if i <= 10 or err_cnt <= 5:
                print(f"  [{i}/{len(stock_ids)}] {sid} EXC: {exc}")

        if i % 100 == 0 or i == len(stock_ids):
            elapsed = time.monotonic() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(stock_ids) - i) / rate if rate > 0 else 0
            print(
                f"  [{i:>4}/{len(stock_ids)}] "
                f"ok={ok_cnt} empty={empty_cnt} err={err_cnt} rows={total_rows} | "
                f"{rate:.1f} stocks/s, ETA {eta/60:.1f} 分"
            )

    elapsed = time.monotonic() - t0
    print()
    print(f"=== 完成 ===")
    print(f"  ok={ok_cnt} empty={empty_cnt} err={err_cnt} total_rows={total_rows}")
    print(f"  耗時 {elapsed/60:.1f} 分鐘")

    # 確認 DB latest
    with get_session() as s:
        latest = s.query(func.max(RawPrice.trading_date)).scalar()
        cnt = s.query(func.count()).select_from(RawPrice).filter(RawPrice.trading_date == latest).scalar()
    print(f"  DB latest: {latest} ({cnt} 筆)")
    return 0 if err_cnt < len(stock_ids) // 10 else 1


if __name__ == "__main__":
    sys.exit(main())
