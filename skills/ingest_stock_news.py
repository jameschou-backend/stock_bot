"""Stage 11.1：Ingest 個股新聞到 raw_stock_news（FinMind TaiwanStockNews）。

FinMind sponsor dataset，每日 ~3900 筆跨 1100+ stocks。
chunked by date 避免單次 response 太大。

用法：
  python -m skills.ingest_stock_news
  python -m skills.ingest_stock_news --days 30   # 回補 30 天
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import fetch_dataset
from app.job_utils import finish_job, start_job
from app.models import RawStockNews

logger = logging.getLogger(__name__)

DATASET = "TaiwanStockNews"


def _md5_title(title: str) -> str:
    if not title:
        return ""
    return hashlib.md5(title.encode("utf-8")).hexdigest()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["news_datetime"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    df["title"] = df.get("title", "").fillna("").astype(str).str[:500]
    df["source"] = df.get("source", "").fillna("").astype(str).str[:64]
    df["link"] = df.get("link", "").fillna("").astype(str).str[:500]
    df["title_hash"] = df["title"].apply(_md5_title)
    keep = ["stock_id", "news_datetime", "source", "title", "link", "title_hash"]
    out = df[keep]
    # 去重（同一篇可能多 source 重 push）
    out = out.drop_duplicates(subset=["stock_id", "news_datetime", "title_hash"])
    return out


def _upsert(session: Session, df: pd.DataFrame, chunk: int = 5000) -> int:
    if df.empty:
        return 0
    total = 0
    records = df.to_dict(orient="records")
    for i in range(0, len(records), chunk):
        batch = records[i:i + chunk]
        stmt = insert(RawStockNews).values(batch)
        # 用 dedup index (stock_id, news_datetime, title_hash)：duplicate 的就 skip
        # MySQL ON DUPLICATE KEY 預設不 raise，但需 unique index，這裡用 INSERT IGNORE 邏輯
        # SQLAlchemy 用 prefix_with("IGNORE") 替代
        stmt = stmt.prefix_with("IGNORE")
        result = session.execute(stmt)
        session.commit()
        total += result.rowcount or 0
    return total


def run(config, session: Session, days_back: Optional[int] = None) -> dict:
    """從 FinMind 取得 TaiwanStockNews。

    Args:
        days_back: 回補天數（None=自動：若 DB 空 90 天，否則從 DB max 增量）
    """
    job_id = start_job(session, "ingest_stock_news")
    try:
        last_dt = session.execute(
            select(func.max(RawStockNews.news_datetime))
        ).scalar_one_or_none()

        if last_dt is None:
            start_date = date.today() - timedelta(days=days_back or 90)
            logger.info("[ingest_stock_news] DB 空，從 %s backfill", start_date)
        else:
            start_date = (last_dt - timedelta(hours=12)).date()  # 多取 12h 避免漏單
            logger.info("[ingest_stock_news] 增量從 %s", start_date)

        end_date = date.today()
        if start_date > end_date:
            finish_job(session, job_id, "success", logs={"rows": 0, "skipped": "up-to-date"})
            return {"rows": 0, "status": "up-to-date"}

        # FinMind 限制：TaiwanStockNews 每次只能 1 天（end_date 不能傳）
        # → per-day query，每 call 間 sleep 2s 避免 rate limit 降級
        # 若遇 "Your level is free" → sleep 60s 重試 1 次
        import time as _time
        total = 0
        cur = start_date
        n_days = 0
        n_skipped = 0
        consecutive_fails = 0
        while cur <= end_date:
            retry = False
            for attempt in range(2):
                try:
                    df = fetch_dataset(DATASET, start_date=cur,
                                        token=config.finmind_token)
                    if df.empty:
                        n_skipped += 1
                    else:
                        norm = _normalize(df)
                        n = _upsert(session, norm)
                        total += n
                        n_days += 1
                        if n_days % 10 == 0:
                            logger.info("[ingest_stock_news] %s: cumulative %d days, %d rows",
                                        cur, n_days, total)
                    consecutive_fails = 0
                    break
                except Exception as exc:
                    exc_msg = str(exc)
                    if "your level is free" in exc_msg.lower() and attempt == 0:
                        logger.warning("[ingest_stock_news] day %s rate-limited, sleep 60s & retry", cur)
                        _time.sleep(60)
                        retry = True
                        continue
                    logger.warning("[ingest_stock_news] day %s 失敗: %s", cur, exc_msg[:120])
                    consecutive_fails += 1
                    if consecutive_fails >= 5:
                        logger.error("[ingest_stock_news] 連續 5 天失敗，abort")
                        finish_job(session, job_id, "failed",
                                   error_text="connect/quota 連續失敗",
                                   logs={"rows": int(total), "days_done": n_days})
                        return {"rows": int(total)}
                    break
            cur = cur + timedelta(days=1)
            _time.sleep(2.0)  # 較長 sleep 避免 rate limit 降級

        finish_job(session, job_id, "success", logs={"rows": int(total)})
        return {"rows": int(total)}

    except Exception as exc:
        logger.error("[ingest_stock_news] 失敗: %s", exc)
        finish_job(session, job_id, "failed", error_text=str(exc))
        raise


if __name__ == "__main__":
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None,
                        help="回補天數（None=自動偵測 DB 最新或預設 90 天）")
    args = parser.parse_args()
    from app.config import load_config
    from app.db import get_session
    cfg = load_config()
    with get_session() as s:
        result = run(cfg, s, days_back=args.days)
        print(f"done: {result}")
