from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.constants import NON_TRADABLE_STATUSES, map_external_status
from app.models import StockStatusHistory


def _latest_status_map(session: Session, stock_ids: List[str], asof_date: date) -> Dict[str, str]:
    if not stock_ids:
        return {}
    stmt = (
        select(
            StockStatusHistory.stock_id,
            StockStatusHistory.effective_date,
            StockStatusHistory.status_type,
        )
        .where(StockStatusHistory.stock_id.in_(stock_ids))
        .where(StockStatusHistory.effective_date <= asof_date)
        .order_by(StockStatusHistory.stock_id, StockStatusHistory.effective_date.desc())
    )
    rows = session.execute(stmt).fetchall()
    status_map: Dict[str, str] = {}
    for stock_id, _, status_type in rows:
        sid = str(stock_id)
        if sid in status_map:
            continue
        status_map[sid] = map_external_status(status_type)
    return status_map


def is_tradable(session: Session, stock_id: str, asof_date: date) -> Tuple[bool, List[str]]:
    status_map = _latest_status_map(session, [str(stock_id)], asof_date)
    status = status_map.get(str(stock_id))
    if status is None:
        return True, ["missing_status"]
    if status in NON_TRADABLE_STATUSES:
        return False, [status]
    return True, []


def filter_universe(
    session: Session,
    df: pd.DataFrame,
    asof_date: date,
    return_stats: bool = False,
):
    if df.empty:
        empty = df.copy()
        stats = {
            "input_count": 0,
            "filtered_count": 0,
            "excluded_count": 0,
            "excluded_ratio": 0.0,
            "missing_status_count": 0,
            "missing_status_ratio": 0.0,
            "excluded_reason_counts": {},
        }
        return (empty, stats) if return_stats else empty

    out = df.copy()
    out["stock_id"] = out["stock_id"].astype(str)
    stock_ids = out["stock_id"].dropna().unique().tolist()
    status_map = _latest_status_map(session, stock_ids, asof_date)

    mapped_status = out["stock_id"].map(status_map)
    out["tradability_status"] = mapped_status.fillna("MISSING")
    out["is_tradable"] = ~out["tradability_status"].isin(NON_TRADABLE_STATUSES)
    filtered = out[out["is_tradable"]].drop(columns=["is_tradable"])

    excluded = out[~out["is_tradable"]]
    reason_counts = (
        excluded["tradability_status"].value_counts(dropna=False).sort_index().to_dict()
        if not excluded.empty
        else {}
    )
    missing_status_count = int((out["tradability_status"] == "MISSING").sum())
    total = len(out)
    stats = {
        "input_count": int(total),
        "filtered_count": int(len(filtered)),
        "excluded_count": int(len(excluded)),
        "excluded_ratio": float(len(excluded) / total) if total else 0.0,
        "missing_status_count": missing_status_count,
        "missing_status_ratio": float(missing_status_count / total) if total else 0.0,
        "excluded_reason_counts": reason_counts,
    }
    return (filtered.drop(columns=["tradability_status"]), stats) if return_stats else filtered.drop(
        columns=["tradability_status"]
    )
