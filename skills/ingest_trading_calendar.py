from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Set
from zoneinfo import ZoneInfo

from sqlalchemy import distinct, func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import RawPrice, TradingCalendar
from app.finmind import fetch_dataset, FinMindError

logger = logging.getLogger(__name__)


def _seed_rows(start_date: date, end_date: date) -> List[Dict]:
    """初始預設用 weekday heuristic。"""
    rows: List[Dict] = []
    cursor = start_date
    while cursor <= end_date:
        is_open = cursor.weekday() < 5
        rows.append(
            {
                "trading_date": cursor,
                "is_open": is_open,
                "session_type": "FULL" if is_open else "CLOSED",
                "note": "weekday-heuristic",
            }
        )
        cursor += timedelta(days=1)
    return rows


def _actual_trading_dates(session: Session, start_date: date, end_date: date) -> Set[date]:
    """從 raw_prices 取得有價格記錄的日期，僅供資料稽核。

    沒有價格記錄可能是 ingest 缺漏，不能據此推定休市。
    """
    rows = (
        session.query(distinct(RawPrice.trading_date))
        .filter(RawPrice.trading_date >= start_date)
        .filter(RawPrice.trading_date <= end_date)
        .all()
    )
    return {r[0] for r in rows}


def seed_calendar(session: Session, start_date: date, end_date: date) -> int:
    records = _seed_rows(start_date, end_date)
    if not records:
        return 0
    stmt = insert(TradingCalendar).values(records)
    stmt = stmt.on_duplicate_key_update(
        is_open=stmt.inserted.is_open,
        session_type=stmt.inserted.session_type,
        note=stmt.inserted.note,
    )
    session.execute(stmt)
    return len(records)


def next_trading_day(session: Session, from_date: date) -> date | None:
    stmt = (
        select(TradingCalendar.trading_date)
        .where(TradingCalendar.trading_date > from_date)
        .where(TradingCalendar.is_open == True)
        .order_by(TradingCalendar.trading_date.asc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def prev_trading_day(session: Session, from_date: date) -> date | None:
    stmt = (
        select(TradingCalendar.trading_date)
        .where(TradingCalendar.trading_date < from_date)
        .where(TradingCalendar.is_open == True)
        .order_by(TradingCalendar.trading_date.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def run(config, db_session: Session, **kwargs) -> Dict:
    """Seed trading_calendar:

    1) 全範圍用 weekday heuristic seed（未來日的最佳猜測）
    2) 對 <=today 範圍以 FinMind 交易日期校準，缺行情不代表休市
    """
    job_id = start_job(db_session, "ingest_trading_calendar")
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        years = int(kwargs.get("years", 10))
        start_date = today - timedelta(days=365 * years)
        end_date = today + timedelta(days=365)
        rows = seed_calendar(db_session, start_date, end_date)
        # Use provider trading dates. Missing price rows indicate an ingest gap, not a holiday.
        calendar = fetch_dataset("TaiwanStockTradingDate", start_date, today,
                                 token=config.finmind_token,
                                 requests_per_hour=config.finmind_requests_per_hour,
                                 cache_ttl=86400)
        if calendar.empty or "date" not in calendar:
            raise FinMindError("交易日曆來源未回傳有效日期，停止更新")
        import pandas as pd
        open_dates = set(pd.to_datetime(calendar["date"]).dt.date)
        historical = []
        cursor = start_date
        while cursor <= today:
            is_open = cursor in open_dates
            historical.append({"trading_date": cursor, "is_open": is_open,
                               "session_type": "FULL" if is_open else "HOLIDAY",
                               "note": "FinMind TaiwanStockTradingDate"})
            cursor += timedelta(days=1)
        stmt = insert(TradingCalendar).values(historical)
        db_session.execute(stmt.on_duplicate_key_update(
            is_open=stmt.inserted.is_open, session_type=stmt.inserted.session_type,
            note=stmt.inserted.note))
        fixed = len(historical)
        db_session.commit()
        logs = {
            "rows_upserted": rows,
            "rows_calibrated": fixed,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "note": "FinMind trading dates; future dates use weekday heuristic",
        }
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover
        logger.error("[ingest_trading_calendar] 失敗: %s", exc, exc_info=True)
        try:
            finish_job(
                db_session, job_id, "failed",
                error_text=str(exc), logs={"error": str(exc)},
            )
        except Exception as finish_exc:
            logger.warning(
                "[ingest_trading_calendar] finish_job 寫入失敗（保留原始例外）: %s",
                finish_exc,
            )
        raise
