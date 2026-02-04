from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import Feature, RawInstitutional, RawPrice


FEATURE_COLUMNS = [
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ma_5",
    "ma_20",
    "ma_60",
    "bias_20",
    "vol_20",
    "vol_ratio_20",
    "foreign_net_5",
    "foreign_net_20",
    "trust_net_5",
    "trust_net_20",
    "dealer_net_5",
    "dealer_net_20",
]


def _fetch_data(session: Session, start_date: date, end_date: date) -> pd.DataFrame:
    price_stmt = (
        select(
            RawPrice.stock_id,
            RawPrice.trading_date,
            RawPrice.close,
            RawPrice.volume,
        )
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.stock_id, RawPrice.trading_date)
    )
    price_df = pd.read_sql(price_stmt, session.get_bind())
    if price_df.empty:
        return price_df

    inst_stmt = (
        select(
            RawInstitutional.stock_id,
            RawInstitutional.trading_date,
            RawInstitutional.foreign_net,
            RawInstitutional.trust_net,
            RawInstitutional.dealer_net,
        )
        .where(RawInstitutional.trading_date.between(start_date, end_date))
        .order_by(RawInstitutional.stock_id, RawInstitutional.trading_date)
    )
    inst_df = pd.read_sql(inst_stmt, session.get_bind())

    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    price_df["volume"] = pd.to_numeric(price_df["volume"], errors="coerce")

    if inst_df.empty:
        price_df["foreign_net"] = 0
        price_df["trust_net"] = 0
        price_df["dealer_net"] = 0
        return price_df

    inst_df["foreign_net"] = pd.to_numeric(inst_df["foreign_net"], errors="coerce").fillna(0)
    inst_df["trust_net"] = pd.to_numeric(inst_df["trust_net"], errors="coerce").fillna(0)
    inst_df["dealer_net"] = pd.to_numeric(inst_df["dealer_net"], errors="coerce").fillna(0)

    merged = price_df.merge(inst_df, on=["stock_id", "trading_date"], how="left")
    for col in ["foreign_net", "trust_net", "dealer_net"]:
        merged[col] = merged[col].fillna(0)
    return merged


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values(["stock_id", "trading_date"]).copy()

    def apply_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trading_date").copy()
        close = group["close"]
        volume = group["volume"]

        group["ret_5"] = close.pct_change(5)
        group["ret_10"] = close.pct_change(10)
        group["ret_20"] = close.pct_change(20)
        group["ret_60"] = close.pct_change(60)

        group["ma_5"] = close.rolling(5).mean()
        group["ma_20"] = close.rolling(20).mean()
        group["ma_60"] = close.rolling(60).mean()

        group["bias_20"] = close / group["ma_20"] - 1

        daily_ret = close.pct_change(1)
        group["vol_20"] = daily_ret.rolling(20).std()
        group["vol_ratio_20"] = volume / volume.rolling(20).mean()

        group["foreign_net_5"] = group["foreign_net"].rolling(5).sum()
        group["foreign_net_20"] = group["foreign_net"].rolling(20).sum()
        group["trust_net_5"] = group["trust_net"].rolling(5).sum()
        group["trust_net_20"] = group["trust_net"].rolling(20).sum()
        group["dealer_net_5"] = group["dealer_net"].rolling(5).sum()
        group["dealer_net_20"] = group["dealer_net"].rolling(20).sum()

        return group

    return df.groupby("stock_id", group_keys=False).apply(apply_group)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "build_features")
    logs: Dict[str, object] = {}
    try:
        max_price_date = db_session.query(func.max(RawPrice.trading_date)).scalar()
        if max_price_date is None:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        max_feature_date = db_session.query(func.max(Feature.trading_date)).scalar()
        if max_feature_date is None:
            target_start = db_session.query(func.min(RawPrice.trading_date)).scalar()
        else:
            target_start = max_feature_date + timedelta(days=1)

        if target_start is None or target_start > max_price_date:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        calc_start = target_start - timedelta(days=120)
        merged = _fetch_data(db_session, calc_start, max_price_date)
        if merged.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        featured = _compute_features(merged)
        featured = featured[featured["trading_date"] >= target_start]
        featured = featured.dropna(subset=FEATURE_COLUMNS)
        if featured.empty:
            finish_job(db_session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        featured = featured.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
        records: List[Dict] = []
        for _, row in featured.iterrows():
            features = {col: float(row[col]) for col in FEATURE_COLUMNS}
            records.append(
                {
                    "stock_id": row["stock_id"],
                    "trading_date": row["trading_date"],
                    "features_json": features,
                }
            )

        stmt = insert(Feature).values(records)
        stmt = stmt.on_duplicate_key_update(features_json=stmt.inserted.features_json)
        db_session.execute(stmt)

        logs = {
            "rows": len(records),
            "start_date": target_start.isoformat(),
            "end_date": max_price_date.isoformat(),
        }
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
