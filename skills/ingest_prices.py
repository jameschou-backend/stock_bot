from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import FinMindError, date_chunks, fetch_dataset
from app.job_utils import finish_job, start_job
from app.models import RawPrice


DATASET = "TaiwanStockPrice"


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date": "trading_date",
        "Trading_Volume": "volume",
        "trading_volume": "volume",
    }
    df = df.rename(columns=rename_map)
    required = {"stock_id", "trading_date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise FinMindError(f"Price dataset missing columns: {sorted(missing)}")

    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = None

    df = df.dropna(subset=["stock_id", "trading_date"])
    return df[["stock_id", "trading_date", "open", "high", "low", "close", "volume"]].drop_duplicates(
        subset=["stock_id", "trading_date"]
    )


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawPrice.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_prices")
    logs: Dict[str, object] = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = today - timedelta(days=365 * config.train_lookback_years)
        start_date = _resolve_start_date(db_session, default_start)
        end_date = today
        logs.update({"start_date": start_date.isoformat(), "end_date": end_date.isoformat()})

        if start_date > end_date:
            logs["rows"] = 0
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0, "start_date": start_date, "end_date": end_date}

        total_rows = 0
        chunk_count = 0
        for chunk_start, chunk_end in date_chunks(start_date, end_date, chunk_days=30):
            chunk_count += 1
            df = fetch_dataset(DATASET, chunk_start, chunk_end, token=config.finmind_token)
            if df.empty:
                continue
            df = _normalize_prices(df)
            records: List[Dict] = df.to_dict("records")
            if not records:
                continue

            stmt = insert(RawPrice).values(records)
            update_cols = {col: stmt.inserted[col] for col in ["open", "high", "low", "close", "volume"]}
            stmt = stmt.on_duplicate_key_update(**update_cols)
            db_session.execute(stmt)
            total_rows += len(records)

        logs.update({"rows": total_rows, "chunks": chunk_count})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
