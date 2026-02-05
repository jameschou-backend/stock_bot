from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import date_chunks, fetch_dataset
from app.job_utils import finish_job, start_job
from app.models import RawInstitutional, RawPrice
from skills.ingest_institutional import _normalize_institutional
from skills.ingest_prices import _normalize_prices


PRICE_DATASET = "TaiwanStockPrice"
INST_DATASET = "TaiwanStockInstitutionalInvestorsBuySell"


@dataclass(frozen=True)
class HistoryStatus:
    needs_backfill: bool
    reason: str
    price_min: date | None
    price_max: date | None
    inst_min: date | None
    inst_max: date | None


def _span_days(min_date: date | None, max_date: date | None) -> int | None:
    if min_date is None or max_date is None:
        return None
    return (max_date - min_date).days


def _should_backfill(
    price_min: date | None,
    price_max: date | None,
    inst_min: date | None,
    inst_max: date | None,
    required_days: int,
) -> HistoryStatus:
    price_span = _span_days(price_min, price_max)
    inst_span = _span_days(inst_min, inst_max)
    if price_span is None:
        return HistoryStatus(True, "raw_prices empty", price_min, price_max, inst_min, inst_max)
    if price_span < required_days:
        return HistoryStatus(True, f"raw_prices span {price_span}d < {required_days}d", price_min, price_max, inst_min, inst_max)
    if inst_span is None:
        return HistoryStatus(True, "raw_institutional empty", price_min, price_max, inst_min, inst_max)
    if inst_span < required_days:
        return HistoryStatus(True, f"raw_institutional span {inst_span}d < {required_days}d", price_min, price_max, inst_min, inst_max)
    return HistoryStatus(False, "ok", price_min, price_max, inst_min, inst_max)


def _backfill_range(config) -> Tuple[date, date]:
    today = datetime.now(ZoneInfo(config.tz)).date()
    start_date = today - timedelta(days=config.bootstrap_days)
    return start_date, today


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "bootstrap_history")
    logs: Dict[str, object] = {}
    try:
        price_min = db_session.query(func.min(RawPrice.trading_date)).scalar()
        price_max = db_session.query(func.max(RawPrice.trading_date)).scalar()
        inst_min = db_session.query(func.min(RawInstitutional.trading_date)).scalar()
        inst_max = db_session.query(func.max(RawInstitutional.trading_date)).scalar()

        status = _should_backfill(price_min, price_max, inst_min, inst_max, config.bootstrap_days)
        logs.update(
            {
                "price_min": price_min.isoformat() if price_min else None,
                "price_max": price_max.isoformat() if price_max else None,
                "inst_min": inst_min.isoformat() if inst_min else None,
                "inst_max": inst_max.isoformat() if inst_max else None,
                "reason": status.reason,
            }
        )

        if not status.needs_backfill:
            logs["mode"] = "skip"
            finish_job(db_session, job_id, "success", logs=logs)
            return logs

        start_date, end_date = _backfill_range(config)
        logs.update({"mode": "backfill", "start_date": start_date.isoformat(), "end_date": end_date.isoformat()})

        price_rows = 0
        inst_rows = 0
        for chunk_start, chunk_end in date_chunks(start_date, end_date, chunk_days=30):
            price_df = fetch_dataset(PRICE_DATASET, chunk_start, chunk_end, token=config.finmind_token)
            if not price_df.empty:
                price_df = _normalize_prices(price_df)
                records: List[Dict] = price_df.to_dict("records")
                if records:
                    stmt = insert(RawPrice).values(records)
                    update_cols = {col: stmt.inserted[col] for col in ["open", "high", "low", "close", "volume"]}
                    stmt = stmt.on_duplicate_key_update(**update_cols)
                    db_session.execute(stmt)
                    price_rows += len(records)

            inst_df = fetch_dataset(INST_DATASET, chunk_start, chunk_end, token=config.finmind_token)
            if not inst_df.empty:
                inst_df = _normalize_institutional(inst_df)
                records = inst_df.to_dict("records")
                if records:
                    stmt = insert(RawInstitutional).values(records)
                    update_cols = {
                        col: stmt.inserted[col]
                        for col in [
                            "foreign_buy",
                            "foreign_sell",
                            "foreign_net",
                            "trust_buy",
                            "trust_sell",
                            "trust_net",
                            "dealer_buy",
                            "dealer_sell",
                            "dealer_net",
                        ]
                    }
                    stmt = stmt.on_duplicate_key_update(**update_cols)
                    db_session.execute(stmt)
                    inst_rows += len(records)

        price_min_after = db_session.query(func.min(RawPrice.trading_date)).scalar()
        price_max_after = db_session.query(func.max(RawPrice.trading_date)).scalar()
        inst_min_after = db_session.query(func.min(RawInstitutional.trading_date)).scalar()
        inst_max_after = db_session.query(func.max(RawInstitutional.trading_date)).scalar()

        status_after = _should_backfill(
            price_min_after, price_max_after, inst_min_after, inst_max_after, config.bootstrap_days
        )

        logs.update(
            {
                "rows_prices": price_rows,
                "rows_institutional": inst_rows,
                "price_min_after": price_min_after.isoformat() if price_min_after else None,
                "price_max_after": price_max_after.isoformat() if price_max_after else None,
                "inst_min_after": inst_min_after.isoformat() if inst_min_after else None,
                "inst_max_after": inst_max_after.isoformat() if inst_max_after else None,
            }
        )

        if status_after.needs_backfill:
            logs["reason_after"] = status_after.reason
            finish_job(db_session, job_id, "failed", error_text=status_after.reason, logs=logs)
            raise ValueError(status_after.reason)

        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc), **logs})
        raise
