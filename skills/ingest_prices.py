"""Ingest Prices 模組

從 FinMind 抓取股價資料寫入 raw_prices 表。

支援兩種模式：
1. 全市場抓取（需較高權限）
2. 逐檔抓取（適用免費/低階會員）

會自動偵測權限並切換模式。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import (
    FinMindError,
    date_chunks,
    fetch_dataset,
    fetch_dataset_by_stocks,
    fetch_stock_list,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawPrice


DATASET = "TaiwanStockPrice"

# 全市場抓取失敗時，改用逐檔抓取的股票清單快取
_stock_list_cache: Optional[List[str]] = None


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date": "trading_date",
        "Trading_Volume": "volume",
        "trading_volume": "volume",
        "max": "high",
        "min": "low",
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
    df["stock_id"] = df["stock_id"].astype(str)
    # FinMind 可能包含指數/產業字串，僅保留台股四碼代碼。
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]
    return df[["stock_id", "trading_date", "open", "high", "low", "close", "volume"]].drop_duplicates(
        subset=["stock_id", "trading_date"]
    )


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawPrice.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def _get_stock_list(token: str | None, requests_per_hour: int, max_retries: int, backoff_seconds: float) -> List[str]:
    """取得股票清單（有快取）"""
    global _stock_list_cache
    if _stock_list_cache is None:
        _stock_list_cache = fetch_stock_list(
            token,
            requests_per_hour=requests_per_hour,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
    return _stock_list_cache


def _fetch_prices_smart(
    start_date: date,
    end_date: date,
    token: str | None,
    requests_per_hour: int,
    max_retries: int,
    backoff_seconds: float,
    logs: Dict[str, object],
) -> pd.DataFrame:
    """智慧抓取：先嘗試全市場，失敗則改逐檔
    
    Returns:
        合併後的 DataFrame
    """
    # 先嘗試全市場抓取
    df = fetch_dataset(
        DATASET,
        start_date,
        end_date,
        token=token,
        requests_per_hour=requests_per_hour,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    if not df.empty:
        logs["fetch_mode"] = "bulk"
        return df
    
    # 全市場抓取回傳空值，改用逐檔抓取
    logs["fetch_mode"] = "by_stock"
    stock_ids = _get_stock_list(token, requests_per_hour, max_retries, backoff_seconds)
    logs["stock_list_count"] = len(stock_ids)
    
    if not stock_ids:
        logs["warning"] = "無法取得股票清單，跳過抓取"
        return pd.DataFrame()
    
    def progress_callback(current: int, total: int) -> None:
        # 可在此加入進度記錄
        pass
    
    return fetch_dataset_by_stocks(
        DATASET,
        start_date,
        end_date,
        stock_ids,
        token=token,
        requests_per_hour=requests_per_hour,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        progress_callback=progress_callback,
    )


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_prices", commit=True)
    logs: Dict[str, object] = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = today - timedelta(days=365 * config.train_lookback_years)
        start_date = _resolve_start_date(db_session, default_start)
        end_date = today
        logs.update({"start_date": start_date.isoformat(), "end_date": end_date.isoformat()})

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "start_date > end_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0, "start_date": start_date, "end_date": end_date}

        total_rows = 0
        chunk_ranges = list(date_chunks(start_date, end_date, chunk_days=config.chunk_days))
        total_chunks = len(chunk_ranges)
        logs["chunks_total"] = total_chunks

        for chunk_count, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            chunk_logs: Dict[str, object] = {}
            logs["progress"] = {
                "current_chunk": chunk_count,
                "total_chunks": total_chunks,
                "chunk_start": chunk_start.isoformat(),
                "chunk_end": chunk_end.isoformat(),
                "rows": total_rows,
            }
            update_job(db_session, job_id, logs=logs, commit=True)

            df = _fetch_prices_smart(
                chunk_start,
                chunk_end,
                config.finmind_token,
                config.finmind_requests_per_hour,
                config.finmind_retry_max,
                config.finmind_retry_backoff,
                chunk_logs,
            )
            logs.update({f"chunk_{chunk_count}": chunk_logs})
            
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

        logs.update({"rows": total_rows, "chunks": total_chunks})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
