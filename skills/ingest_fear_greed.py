"""Priority 5：CNN 恐懼貪婪指數（CnnFearGreedIndex）

從 FinMind 抓取 CNN Fear & Greed Index，寫入 raw_fear_greed 表。
這是全球市場情緒指標（非個股），用於 market-level 特徵。

Dataset: CnnFearGreedIndex
Fields:  date, score (0-100), rating (Extreme Fear/Fear/Neutral/Greed/Extreme Greed)
頻率：   每日
限制：   Sponsor 計劃；資料從 2011 年起
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import FinMindError, fetch_dataset
from app.job_utils import finish_job, start_job
from app.models import RawFearGreed

DATASET = "CnnFearGreedIndex"

UPDATE_COLS = ["score", "rating"]


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawFearGreed.date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_fear_greed", commit=True)
    logs: Dict = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = date(2020, 1, 1)  # 2011 年起有資料，取近幾年即可
        start_date = _resolve_start_date(db_session, default_start)
        end_date   = today

        logs["start_date"] = start_date.isoformat()
        logs["end_date"]   = end_date.isoformat()

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "already_up_to_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        print(f"[ingest_fear_greed] {start_date} ~ {end_date}", flush=True)

        # CnnFearGreedIndex 不需要 data_id（非個股資料）
        df = fetch_dataset(
            dataset=DATASET,
            start_date=start_date,
            end_date=end_date,
            token=config.finmind_token,
            requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
            max_retries=getattr(config, "finmind_retry_max", 3),
            backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
        )

        if df is None or df.empty:
            logs["rows"] = 0
            logs["skip_reason"] = "no_data_returned"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        # 正規化
        df.columns = [c.lower() for c in df.columns]  # 統一小寫欄位名
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})
        df["date"]   = pd.to_datetime(df["date"]).dt.date
        score_col = next((c for c in ["score", "value", "close"] if c in df.columns), None)
        df["score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0).astype(int) \
            if score_col else 0
        rating_col = next((c for c in ["rating", "name"] if c in df.columns), None)
        df["rating"] = df[rating_col].astype(str).str.strip() if rating_col else "Unknown"
        df = df[["date", "score", "rating"]].drop_duplicates(subset=["date"])

        # Upsert
        rows = df.to_dict("records")
        if rows:
            stmt = insert(RawFearGreed).values(rows)
            stmt = stmt.on_duplicate_key_update(
                score=stmt.inserted.score,
                rating=stmt.inserted.rating,
            )
            db_session.execute(stmt)
            db_session.commit()

        logs["rows"] = len(rows)
        print(f"  ✅ Fear & Greed: {len(rows)} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": len(rows)}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
