from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import TradingCalendar


def _seed_rows(start_date: date, end_date: date) -> List[Dict]:
    rows: List[Dict] = []
    cursor = start_date
    while cursor <= end_date:
        is_open = cursor.weekday() < 5
        rows.append(
            {
                "trading_date": cursor,
                "is_open": is_open,
                "session_type": "FULL" if is_open else "CLOSED",
                "note": "TODO: replace with TWSE official calendar (holiday/half-day aware)",
            }
        )
        cursor += timedelta(days=1)
    return rows


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
    job_id = start_job(db_session, "ingest_trading_calendar")
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        years = int(kwargs.get("years", 10))
        start_date = today - timedelta(days=365 * years)
        end_date = today + timedelta(days=365)
        rows = seed_calendar(db_session, start_date, end_date)
        logs = {
            "rows_upserted": rows,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "warning": "Calendar currently seeded by weekday heuristic; integrate official TWSE calendar source.",
        }
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
