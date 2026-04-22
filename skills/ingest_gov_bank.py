"""Priority 4：官股銀行買賣超（TaiwanstockGovernmentBankBuySell）

從 FinMind 抓取官股銀行每日買賣超明細，彙整成聚合指標，
寫入 raw_gov_bank 表。

Dataset: TaiwanstockGovernmentBankBuySell
Fields:  date, stock_id, securities_trader_id, securities_trader, buy, sell
頻率：   每日
限制：   Sponsor 計劃；資料從 2010 年起

官股銀行共 8 家：台灣銀行、土地銀行、合作金庫銀行、第一銀行、
                  華南銀行、彰化銀行、臺灣企銀、兆豐銀行

聚合指標：
- gov_net:          8 行庫合計淨買超（張）
- bank_count_buy:   當日淨買超的行庫數
- bank_count_sell:  當日淨賣超的行庫數
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import (
    FinMindError,
    date_chunks,
    fetch_dataset_by_stocks,
    fetch_stock_list,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawGovBank, Stock

DATASET = "TaiwanstockGovernmentBankBuySell"
UPDATE_COLS = ["gov_net", "bank_count_buy", "bank_count_sell"]


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawGovBank.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def _load_allowed_stock_ids(session: Session) -> Set[str]:
    rows = (
        session.query(Stock.stock_id)
        .filter(Stock.is_listed == True)
        .filter(Stock.security_type == "stock")
        .all()
    )
    return {row[0] for row in rows}


def _aggregate_gov_bank(df: pd.DataFrame, allowed_stock_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """將官股銀行原始明細彙整成每日每股聚合指標。"""
    if df.empty:
        return pd.DataFrame()

    rename = {"date": "trading_date"}
    df = df.rename(columns=rename)
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["stock_id"] = df["stock_id"].astype(str)

    if allowed_stock_ids:
        df = df[df["stock_id"].isin(allowed_stock_ids)]
        if df.empty:
            return pd.DataFrame()

    # 欄位容錯
    buy_col  = next((c for c in ["buy", "Buy", "buy_volume"] if c in df.columns), None)
    sell_col = next((c for c in ["sell", "Sell", "sell_volume"] if c in df.columns), None)
    if buy_col is None or sell_col is None:
        raise FinMindError(
            f"TaiwanstockGovernmentBankBuySell missing buy/sell columns; got: {df.columns.tolist()}"
        )

    df["buy"]  = pd.to_numeric(df[buy_col],  errors="coerce").fillna(0)
    df["sell"] = pd.to_numeric(df[sell_col], errors="coerce").fillna(0)
    df["net"]  = df["buy"] - df["sell"]

    results = []
    for (stock_id, trading_date), grp in df.groupby(["stock_id", "trading_date"]):
        gov_net        = int(grp["net"].sum())
        bank_count_buy  = int((grp["net"] > 0).sum())
        bank_count_sell = int((grp["net"] < 0).sum())

        results.append({
            "stock_id": stock_id,
            "trading_date": trading_date,
            "gov_net": gov_net,
            "bank_count_buy": bank_count_buy,
            "bank_count_sell": bank_count_sell,
        })

    return pd.DataFrame(results)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_gov_bank", commit=True)
    logs: Dict = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = today - timedelta(days=365 * config.train_lookback_years)
        start_date = _resolve_start_date(db_session, default_start)
        end_date   = today

        logs["start_date"] = start_date.isoformat()
        logs["end_date"]   = end_date.isoformat()

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "already_up_to_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        print(f"[ingest_gov_bank] {start_date} ~ {end_date}", flush=True)

        allowed_stock_ids = _load_allowed_stock_ids(db_session)
        logs["allowed_stock_ids"] = len(allowed_stock_ids)

        stock_ids = fetch_stock_list(
            config.finmind_token,
            requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
            max_retries=getattr(config, "finmind_retry_max", 3),
            backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
        )
        if not stock_ids:
            logs["warning"] = "無法取得股票清單，跳過抓取"
            logs["rows"] = 0
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        chunk_days = getattr(config, "chunk_days", 180)
        chunk_ranges = list(date_chunks(start_date, end_date, chunk_days=chunk_days))
        logs["chunks_total"] = len(chunk_ranges)
        total_rows = 0

        for i, (chunk_start, chunk_end) in enumerate(chunk_ranges, 1):
            update_job(db_session, job_id, logs={**logs, "progress": f"{i}/{len(chunk_ranges)}"}, commit=True)

            df = fetch_dataset_by_stocks(
                DATASET,
                chunk_start,
                chunk_end,
                stock_ids,
                token=config.finmind_token,
                batch_size=500,
                use_batch_query=True,
                requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
                max_retries=getattr(config, "finmind_retry_max", 3),
                backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
            )
            if df is None or df.empty:
                continue

            agg_df = _aggregate_gov_bank(df, allowed_stock_ids=allowed_stock_ids or None)
            if agg_df.empty:
                continue

            records: List[Dict] = agg_df.to_dict("records")
            stmt = insert(RawGovBank).values(records)
            stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
            db_session.execute(stmt)
            db_session.commit()
            total_rows += len(records)
            print(f"  chunk {i}/{len(chunk_ranges)}: {len(records)} 筆", flush=True)

        logs["rows"] = total_rows
        print(f"  ✅ gov_bank: {total_rows} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
