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
    """從 raw_prices 取得實際有交易記錄的日期（ground truth）。

    若該日全市場無任何 prices row，視為非交易日（國定假日 / 颱風假）。
    需要 raw_prices 已 backfill 到 end_date 才能準確校準。
    """
    rows = (
        session.query(distinct(RawPrice.trading_date))
        .filter(RawPrice.trading_date >= start_date)
        .filter(RawPrice.trading_date <= end_date)
        .all()
    )
    return {r[0] for r in rows}


def _calibrate_from_prices(
    session: Session, start_date: date, end_date: date, today: date
) -> int:
    """用 raw_prices 校準歷史日期：weekday=True 但無交易記錄 → 國定假日。

    僅校準 trading_date <= today（未來日無法判斷）。回傳被修正的天數。
    """
    cal_end = min(end_date, today)
    if start_date > cal_end:
        return 0

    actual = _actual_trading_dates(session, start_date, cal_end)
    fixed = 0
    cursor = start_date
    while cursor <= cal_end:
        if cursor.weekday() < 5 and cursor not in actual:
            stmt = (
                insert(TradingCalendar)
                .values(
                    trading_date=cursor,
                    is_open=False,
                    session_type="HOLIDAY",
                    note="auto-calibrated: no prices recorded",
                )
                .on_duplicate_key_update(
                    is_open=False,
                    session_type="HOLIDAY",
                    note="auto-calibrated: no prices recorded",
                )
            )
            session.execute(stmt)
            fixed += 1
        # 額外：weekend 但有交易 → 補班日（罕見但 2019/2020 颱風補班發生過）
        elif cursor.weekday() >= 5 and cursor in actual:
            stmt = (
                insert(TradingCalendar)
                .values(
                    trading_date=cursor,
                    is_open=True,
                    session_type="MAKEUP",
                    note="auto-calibrated: prices recorded on weekend",
                )
                .on_duplicate_key_update(
                    is_open=True,
                    session_type="MAKEUP",
                    note="auto-calibrated: prices recorded on weekend",
                )
            )
            session.execute(stmt)
            fixed += 1
        cursor += timedelta(days=1)
    return fixed


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
    2) 對 <=today 範圍用 raw_prices 校準（國定假日 + 補班日）
    """
    job_id = start_job(db_session, "ingest_trading_calendar")
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        years = int(kwargs.get("years", 10))
        start_date = today - timedelta(days=365 * years)
        end_date = today + timedelta(days=365)
        rows = seed_calendar(db_session, start_date, end_date)
        # 校準歷史日（raw_prices 是 ground truth）
        fixed = _calibrate_from_prices(db_session, start_date, end_date, today)
        db_session.commit()
        logs = {
            "rows_upserted": rows,
            "rows_calibrated": fixed,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "note": "weekday seed + raw_prices calibration（holidays inferred from missing prices）",
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
