"""Ingest Fundamental 模組

從 FinMind 抓取月營收資料寫入 raw_fundamentals 表。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import (
    FinMindError,
    fetch_dataset,
    fetch_stock_list,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawFundamental, Stock


DATASET = "TaiwanStockMonthRevenue"
DEFAULT_COMPLETENESS_THRESHOLD = 0.98


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


def _first_day_next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _normalize_stock_ids(stock_ids: List[str]) -> List[str]:
    return sorted({str(s) for s in stock_ids if str(s).isdigit() and len(str(s)) == 4})


def _get_listed_stock_ids_from_db(session: Session) -> List[str]:
    stmt = (
        select(Stock.stock_id)
        .where(Stock.is_listed == True)
        .where(Stock.security_type == "stock")
    )
    rows = session.execute(stmt).scalars().all()
    return _normalize_stock_ids(list(rows))


def _get_existing_stock_ids_for_date(session: Session, trading_date: date) -> List[str]:
    stmt = select(RawFundamental.stock_id).where(RawFundamental.trading_date == trading_date)
    rows = session.execute(stmt).scalars().all()
    return _normalize_stock_ids(list(rows))


def _to_mysql_safe_records(df: pd.DataFrame) -> List[Dict]:
    """將 DataFrame 轉為 MySQL 可接受 records（NaN/NaT -> None）。"""
    if df.empty:
        return []
    safe_df = df.astype(object).copy()
    # 避免 pandas replace downcasting FutureWarning，直接以 mask 處理 inf/-inf。
    safe_df = safe_df.mask(safe_df.isin([float("inf"), float("-inf")]), pd.NA)
    safe_df = safe_df.where(pd.notna(safe_df), None)
    return safe_df.to_dict("records")


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

        latest_date = db_session.query(func.max(RawFundamental.trading_date)).scalar()
        completeness_threshold = float(kwargs.get("completeness_threshold", DEFAULT_COMPLETENESS_THRESHOLD))

        # 優先用 DB 的上市普通股池，避免每次都先打 FinMind 取得 stock list。
        stock_ids = _get_listed_stock_ids_from_db(db_session)
        stock_source = "db"
        if not stock_ids:
            stock_ids = fetch_stock_list(
                config.finmind_token,
                requests_per_hour=config.finmind_requests_per_hour,
                max_retries=config.finmind_retry_max,
                backoff_seconds=config.finmind_retry_backoff,
            )
            stock_ids = _normalize_stock_ids(stock_ids)
            stock_source = "finmind"

        logs["stock_source"] = stock_source
        logs["stock_count"] = len(stock_ids)
        if not stock_ids:
            logs["rows"] = 0
            logs["warning"] = "無法取得股票清單"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        if latest_date is not None:
            next_month_start = _first_day_next_month(latest_date)
            covered_ids = set(_get_existing_stock_ids_for_date(db_session, latest_date))
            missing_ids = sorted(set(stock_ids) - covered_ids)
            coverage_ratio = len(covered_ids) / len(stock_ids) if stock_ids else 0.0
            logs.update(
                {
                    "latest_date": latest_date.isoformat(),
                    "latest_date_coverage": round(coverage_ratio, 6),
                    "latest_date_missing_stocks": len(missing_ids),
                }
            )

            # 月頻資料：未進入下個月份時，若最新月份已完整，不需要再打 API。
            if today < next_month_start and coverage_ratio >= completeness_threshold:
                logs["rows"] = 0
                logs["skip_reason"] = "latest_month_complete"
                finish_job(db_session, job_id, "success", logs=logs)
                return {"rows": 0, "start_date": start_date, "end_date": end_date}

            # 尚未跨月但資料不完整，只補缺漏股票，避免全量重抓。
            if today < next_month_start and missing_ids:
                stock_ids = missing_ids
                start_date = latest_date
            else:
                # 已跨月時，從下個月份起抓即可，避免在同月內無效查詢。
                start_date = max(start_date, next_month_start)
                logs["start_date"] = start_date.isoformat()

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "start_date > end_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0, "start_date": start_date, "end_date": end_date}

        total_rows = 0
        total_stocks = len(stock_ids)
        error_stocks = 0
        logs["stocks_total"] = total_stocks

        for stock_idx, stock_id in enumerate(stock_ids, start=1):
            logs["progress"] = {
                "current_stock": stock_idx,
                "total_stocks": total_stocks,
                "current_chunk": stock_idx,
                "total_chunks": total_stocks,
                "chunk_start": start_date.isoformat(),
                "chunk_end": end_date.isoformat(),
                "rows": total_rows,
                "stock_id": stock_id,
            }
            update_job(db_session, job_id, logs=logs, commit=True)

            try:
                df = fetch_dataset(
                    DATASET,
                    start_date,
                    end_date,
                    token=config.finmind_token,
                    data_id=stock_id,
                    requests_per_hour=config.finmind_requests_per_hour,
                    max_retries=config.finmind_retry_max,
                    backoff_seconds=config.finmind_retry_backoff,
                    timeout=120,
                )
            except FinMindError:
                error_stocks += 1
                continue

            if df.empty:
                continue

            norm = _normalize_fundamentals(df)
            records: List[Dict] = _to_mysql_safe_records(norm)
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
            db_session.commit()
            total_rows += len(records)

        logs.update({"rows": total_rows, "stocks_with_error": error_stocks})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:  # pragma: no cover - pipeline runtime
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
