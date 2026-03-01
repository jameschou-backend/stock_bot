from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import CorporateAction, PriceAdjustFactor, RawPrice


TODO_SOURCE_HINTS = [
    "TWSE 除權除息公告",
    "公開資訊觀測站（MOPS）公司行為事件",
    "FinMind 若提供對應 corporate action dataset",
]


def _date_range_from_input(date_range: Tuple[date, date] | None, tz: str) -> Tuple[date, date]:
    if date_range is not None:
        return date_range
    today = datetime.now(ZoneInfo(tz)).date()
    return today - timedelta(days=365), today


def _fetch_external_actions(
    start_date: date,
    end_date: date,
    config,
) -> pd.DataFrame:
    # TODO: 接上正式來源（TWSE / MOPS / FinMind）
    _ = start_date, end_date, config
    return pd.DataFrame()


def _upsert_corporate_actions(session: Session, records: List[Dict]) -> int:
    if not records:
        return 0
    stmt = insert(CorporateAction).values(records)
    stmt = stmt.on_duplicate_key_update(
        action_type=stmt.inserted.action_type,
        adj_factor=stmt.inserted.adj_factor,
        payload_json=stmt.inserted.payload_json,
    )
    session.execute(stmt)
    return len(records)


def _build_default_adjust_factors(
    session: Session,
    start_date: date,
    end_date: date,
) -> Iterable[Dict]:
    stmt = (
        select(RawPrice.stock_id, RawPrice.trading_date)
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.stock_id, RawPrice.trading_date)
    )
    rows = session.execute(stmt).fetchall()
    for stock_id, trading_date in rows:
        yield {
            "stock_id": str(stock_id),
            "trading_date": trading_date,
            "adj_factor": 1.0,
        }


def _upsert_adjust_factors(session: Session, records: List[Dict]) -> int:
    if not records:
        return 0
    upserted = 0
    batch_size = 1000
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        stmt = insert(PriceAdjustFactor).values(batch)
        stmt = stmt.on_duplicate_key_update(
            adj_factor=stmt.inserted.adj_factor,
        )
        session.execute(stmt)
        upserted += len(batch)
    return upserted


def run_date_range(config, db_session: Session, date_range: Tuple[date, date] | None = None) -> Dict:
    start_date, end_date = _date_range_from_input(date_range, config.tz)
    logs: Dict[str, object] = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "todo_sources": TODO_SOURCE_HINTS,
    }

    action_df = _fetch_external_actions(start_date, end_date, config)
    logs["external_action_rows"] = int(len(action_df))

    action_records: List[Dict] = []
    if not action_df.empty:
        action_df = action_df.copy()
        action_df["stock_id"] = action_df["stock_id"].astype(str)
        action_df["action_date"] = pd.to_datetime(action_df["action_date"], errors="coerce").dt.date
        action_df["adj_factor"] = pd.to_numeric(action_df.get("adj_factor"), errors="coerce")
        for _, row in action_df.iterrows():
            action_records.append(
                {
                    "stock_id": row["stock_id"],
                    "action_date": row["action_date"],
                    "action_type": str(row.get("action_type", "OTHER")),
                    "adj_factor": None if pd.isna(row["adj_factor"]) else float(row["adj_factor"]),
                    "payload_json": row.get("payload_json", {}),
                }
            )

    actions_upserted = _upsert_corporate_actions(db_session, action_records)

    factor_records = list(_build_default_adjust_factors(db_session, start_date, end_date))
    factors_upserted = _upsert_adjust_factors(db_session, factor_records)

    logs.update(
        {
            "actions_upserted": actions_upserted,
            "factors_upserted": factors_upserted,
            "warning": (
                "No external corporate action source connected yet; "
                "default adj_factor=1.0 was written for observed raw_prices dates."
            ),
        }
    )
    return logs


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_corporate_actions")
    try:
        logs = run_date_range(config, db_session, kwargs.get("date_range"))
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
