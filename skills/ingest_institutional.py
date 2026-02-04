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
from app.models import RawInstitutional


DATASET = "TaiwanStockInstitutionalInvestorsBuySell"

CATEGORY_MAP = {
    "foreign_investor": "foreign",
    "foreign_dealer_self": "foreign",
    "investment_trust": "trust",
    "dealer_self": "dealer",
    "dealer_hedging": "dealer",
}


def _normalize_institutional(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"date": "trading_date"}
    df = df.rename(columns=rename_map)

    required = {"stock_id", "trading_date", "name"}
    missing = required - set(df.columns)
    if missing:
        raise FinMindError(f"Institutional dataset missing columns: {sorted(missing)}")

    buy_col = next((c for c in ["buy", "buy_amount", "buy_volume"] if c in df.columns), None)
    sell_col = next((c for c in ["sell", "sell_amount", "sell_volume"] if c in df.columns), None)
    if buy_col is None or sell_col is None:
        raise FinMindError("Institutional dataset missing buy/sell columns")

    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["name"] = df["name"].astype(str).str.strip()
    df["category"] = df["name"].str.lower().map(CATEGORY_MAP)
    df = df[df["category"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "trading_date"])

    df[buy_col] = pd.to_numeric(df[buy_col], errors="coerce").fillna(0)
    df[sell_col] = pd.to_numeric(df[sell_col], errors="coerce").fillna(0)

    agg = (
        df.groupby(["stock_id", "trading_date", "category"], as_index=False)[[buy_col, sell_col]]
        .sum()
        .rename(columns={buy_col: "buy", sell_col: "sell"})
    )
    agg["net"] = agg["buy"] - agg["sell"]

    base = agg[["stock_id", "trading_date"]].drop_duplicates()
    result = base.copy()

    for category in ["foreign", "trust", "dealer"]:
        sub = agg[agg["category"] == category].copy()
        if sub.empty:
            result[f"{category}_buy"] = 0
            result[f"{category}_sell"] = 0
            result[f"{category}_net"] = 0
            continue
        sub = sub[["stock_id", "trading_date", "buy", "sell", "net"]].rename(
            columns={
                "buy": f"{category}_buy",
                "sell": f"{category}_sell",
                "net": f"{category}_net",
            }
        )
        result = result.merge(sub, on=["stock_id", "trading_date"], how="left")

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
    ]:
        if col not in result.columns:
            result[col] = 0
        result[col] = result[col].fillna(0).astype(int)

    return result[
        [
            "stock_id",
            "trading_date",
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
    ].drop_duplicates(subset=["stock_id", "trading_date"])


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawInstitutional.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_institutional")
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
            df = _normalize_institutional(df)
            records: List[Dict] = df.to_dict("records")
            if not records:
                continue

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
            total_rows += len(records)

        logs.update({"rows": total_rows, "chunks": chunk_count})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
