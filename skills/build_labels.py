from __future__ import annotations

from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import Label, RawPrice


def _compute_labels(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values(["stock_id", "trading_date"])

    def apply_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trading_date").copy()
        group["future_ret_h"] = group["close"].shift(-horizon) / group["close"] - 1
        return group

    return df.groupby("stock_id", group_keys=False).apply(apply_group)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "build_labels")
    try:
        max_price_date = db_session.query(func.max(RawPrice.trading_date)).scalar()
        if max_price_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        horizon = config.label_horizon_days
        last_label_date = db_session.query(func.max(Label.trading_date)).scalar()
        if last_label_date is None:
            target_start = db_session.query(func.min(RawPrice.trading_date)).scalar()
        else:
            target_start = last_label_date + timedelta(days=1)

        if target_start is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        max_label_date = max_price_date - timedelta(days=horizon)
        if target_start > max_label_date:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        stmt = (
            select(RawPrice.stock_id, RawPrice.trading_date, RawPrice.close)
            .where(RawPrice.trading_date.between(target_start, max_price_date))
            .order_by(RawPrice.stock_id, RawPrice.trading_date)
        )
        df = pd.read_sql(stmt, db_session.get_bind())
        if df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        df = _compute_labels(df, horizon)
        df = df[df["trading_date"] <= max_label_date]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["future_ret_h"])
        if df.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        records: List[Dict] = df[["stock_id", "trading_date", "future_ret_h"]].to_dict("records")
        insert_stmt = insert(Label).values(records)
        insert_stmt = insert_stmt.on_duplicate_key_update(future_ret_h=insert_stmt.inserted.future_ret_h)
        db_session.execute(insert_stmt)

        logs = {
            "rows": len(records),
            "start_date": target_start.isoformat(),
            "end_date": max_label_date.isoformat(),
        }
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
