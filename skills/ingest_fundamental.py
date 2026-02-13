"""Ingest Fundamental 模組

從 FinMind 抓取月營收資料寫入 raw_fundamentals 表。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import (
    FinMindError,
    date_chunks,
    fetch_dataset_by_stocks,
    fetch_stock_list,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawFundamental


DATASET = "TaiwanStockMonthRevenue"


def _normalize_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """正規化月營收欄位。

    FinMind 月營收欄位在不同版本命名可能略有差異，因此採用候選欄位映射。
    """
    if df.empty:
        return pd.DataFrame()

    rename_map = {"date": "trading_date"}
    df = df.rename(columns=rename_map)

    date_col = "trading_date" if "trading_date" in df.columns else None
    if date_col is None:
        raise FinMindError("Fundamental dataset missing date/trading_date column")

    def pick_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    rev_col = pick_col(["revenue", "monthly_revenue", "Revenue"])
    rev_last_month_col = pick_col(["revenue_last_month", "RevenueLastMonth"])
    rev_last_year_col = pick_col(["revenue_last_year", "RevenueLastYear"])
    mom_col = pick_col(["revenue_mom", "YoM", "revenue_growth_month"])
    yoy_col = pick_col(["revenue_yoy", "YoY", "revenue_growth_year"])

    if rev_col is None:
        raise FinMindError("Fundamental dataset missing revenue column")

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.date
    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]
    df = df.dropna(subset=["stock_id", "trading_date"])

    df["revenue_current_month"] = pd.to_numeric(df[rev_col], errors="coerce")
    df["revenue_last_month"] = pd.to_numeric(df[rev_last_month_col], errors="coerce") if rev_last_month_col else None
    df["revenue_last_year"] = pd.to_numeric(df[rev_last_year_col], errors="coerce") if rev_last_year_col else None

    if mom_col:
        df["revenue_mom"] = pd.to_numeric(df[mom_col], errors="coerce")
    else:
        denom = df["revenue_last_month"].replace(0, pd.NA)
        df["revenue_mom"] = df["revenue_current_month"] / denom - 1

    if yoy_col:
        df["revenue_yoy"] = pd.to_numeric(df[yoy_col], errors="coerce")
    else:
        denom = df["revenue_last_year"].replace(0, pd.NA)
        df["revenue_yoy"] = df["revenue_current_month"] / denom - 1

    return df[
        [
            "stock_id",
            "trading_date",
            "revenue_current_month",
            "revenue_last_month",
            "revenue_last_year",
            "revenue_mom",
            "revenue_yoy",
        ]
    ].drop_duplicates(subset=["stock_id", "trading_date"])


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawFundamental.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_fundamental", commit=True)
    logs: Dict[str, object] = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        # 月營收不需要太細粒度，抓長歷史做研究
        default_start = today - timedelta(days=365 * max(config.train_lookback_years, 10))
        start_date = _resolve_start_date(db_session, default_start)
        end_date = today
        logs.update({"start_date": start_date.isoformat(), "end_date": end_date.isoformat()})

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "start_date > end_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0, "start_date": start_date, "end_date": end_date}

        stock_ids = fetch_stock_list(
            config.finmind_token,
            requests_per_hour=config.finmind_requests_per_hour,
            max_retries=config.finmind_retry_max,
            backoff_seconds=config.finmind_retry_backoff,
        )
        logs["stock_count"] = len(stock_ids)
        if not stock_ids:
            logs["rows"] = 0
            logs["warning"] = "無法取得股票清單"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        total_rows = 0
        chunk_ranges = list(date_chunks(start_date, end_date, chunk_days=365))
        total_chunks = len(chunk_ranges)
        logs["chunks_total"] = total_chunks

        for chunk_count, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            logs["progress"] = {
                "current_chunk": chunk_count,
                "total_chunks": total_chunks,
                "chunk_start": chunk_start.isoformat(),
                "chunk_end": chunk_end.isoformat(),
                "rows": total_rows,
            }
            update_job(db_session, job_id, logs=logs, commit=True)

            df = fetch_dataset_by_stocks(
                DATASET,
                chunk_start,
                chunk_end,
                stock_ids,
                token=config.finmind_token,
                batch_size=500,
                use_batch_query=True,
                requests_per_hour=config.finmind_requests_per_hour,
                max_retries=config.finmind_retry_max,
                backoff_seconds=config.finmind_retry_backoff,
                timeout=120,
            )
            if df.empty:
                continue

            norm = _normalize_fundamentals(df)
            records: List[Dict] = norm.to_dict("records")
            if not records:
                continue

            stmt = insert(RawFundamental).values(records)
            stmt = stmt.on_duplicate_key_update(
                revenue_current_month=stmt.inserted.revenue_current_month,
                revenue_last_month=stmt.inserted.revenue_last_month,
                revenue_last_year=stmt.inserted.revenue_last_year,
                revenue_mom=stmt.inserted.revenue_mom,
                revenue_yoy=stmt.inserted.revenue_yoy,
            )
            db_session.execute(stmt)
            total_rows += len(records)

        logs.update({"rows": total_rows, "chunks": total_chunks})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:  # pragma: no cover - pipeline runtime
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
