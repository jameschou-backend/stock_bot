"""Priority 6：本益比/殖利率/本淨比（TaiwanStockPER）

從 FinMind 抓取每日 PER/PBR/殖利率，寫入 raw_per 表。

Dataset: TaiwanStockPER
Fields:  date, stock_id, dividend_yield, PER, PBR
頻率：   每日（交易日）
限制：   Sponsor 計劃；支援 data_id + 日期區間查詢（每檔股票一次 API call）

回補策略：
- 增量：從 DB 最大日期 +1 開始抓取今日資料（快速）
- 全量回補：FORCE_RECOMPUTE_PER=1 env var 或 backfill_per.py 腳本
- 每次按股票批次查詢，每批 ~500 檔（FinMind data_id 支援多股）
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
    fetch_dataset,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawPER, Stock

DATASET = "TaiwanStockPER"
UPDATE_COLS = ["per", "pbr", "dividend_yield"]


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawPER.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def _load_stock_ids(session: Session) -> List[str]:
    rows = (
        session.query(Stock.stock_id)
        .filter(Stock.is_listed == True)
        .filter(Stock.security_type == "stock")
        .order_by(Stock.stock_id)
        .all()
    )
    return [row[0] for row in rows]


def _normalize_per(df: pd.DataFrame) -> pd.DataFrame:
    """正規化 TaiwanStockPER 欄位"""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    rename = {"date": "trading_date"}
    df = df.rename(columns=rename)

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.date
    df["stock_id"] = df["stock_id"].astype(str)
    df = df.dropna(subset=["stock_id", "trading_date"])
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]

    # PER 欄位容錯（大小寫、不同命名）
    per_col = next((c for c in ["PER", "per", "PE"] if c in df.columns), None)
    pbr_col = next((c for c in ["PBR", "pbr", "PB"] if c in df.columns), None)
    div_col = next((c for c in ["dividend_yield", "DividendYield", "yield"] if c in df.columns), None)

    df["per"] = pd.to_numeric(df[per_col], errors="coerce") if per_col else None
    df["pbr"] = pd.to_numeric(df[pbr_col], errors="coerce") if pbr_col else None
    df["dividend_yield"] = pd.to_numeric(df[div_col], errors="coerce") if div_col else None

    return df[["stock_id", "trading_date", "per", "pbr", "dividend_yield"]]


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_per", commit=True)
    logs: Dict = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        # PER 資料從 2010 年起有效
        default_start = max(
            today - timedelta(days=365 * config.train_lookback_years),
            date(2010, 1, 1),
        )
        start_date = _resolve_start_date(db_session, default_start)
        end_date = today

        logs["start_date"] = start_date.isoformat()
        logs["end_date"] = end_date.isoformat()

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "already_up_to_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        stock_ids = _load_stock_ids(db_session)
        if not stock_ids:
            logs["rows"] = 0
            logs["skip_reason"] = "no_stocks"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        logs["stocks"] = len(stock_ids)
        print(f"[ingest_per] {start_date} ~ {end_date}，{len(stock_ids)} 檔", flush=True)

        # TaiwanStockPER 支援 data_id + 日期區間 → 每檔股票一次 API call
        # 為避免 10 年資料量過大單次超時，若超過 365 天改為 chunk
        CHUNK_DAYS = 365
        total_rows = 0
        commit_buffer: List[Dict] = []

        date_ranges: List[tuple] = []
        cur = start_date
        while cur <= end_date:
            chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end_date)
            date_ranges.append((cur, chunk_end))
            cur = chunk_end + timedelta(days=1)

        for i, stock_id in enumerate(stock_ids, 1):
            if i % 200 == 0:
                update_job(
                    db_session, job_id,
                    logs={**logs, "progress": f"{i}/{len(stock_ids)}", "rows": total_rows},
                    commit=True,
                )
                print(f"  [{i}/{len(stock_ids)}] rows={total_rows}", flush=True)

            for range_start, range_end in date_ranges:
                df = fetch_dataset(
                    dataset=DATASET,
                    start_date=range_start,
                    end_date=range_end,
                    token=config.finmind_token,
                    data_id=stock_id,
                    requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
                    max_retries=getattr(config, "finmind_retry_max", 3),
                    backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
                )
                if df is None or df.empty:
                    continue

                normalized = _normalize_per(df)
                if normalized.empty:
                    continue

                commit_buffer.extend(normalized.to_dict("records"))

                if len(commit_buffer) >= 5000:
                    _flush(db_session, commit_buffer)
                    total_rows += len(commit_buffer)
                    commit_buffer.clear()

        if commit_buffer:
            _flush(db_session, commit_buffer)
            total_rows += len(commit_buffer)

        logs["rows"] = total_rows
        print(f"  ✅ ingest_per: {total_rows} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise


def _flush(session: Session, buffer: List[Dict]) -> None:
    stmt = insert(RawPER).values(buffer)
    stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
    session.execute(stmt)
    session.commit()
