"""Ingest Prices 模組

從 FinMind 或 TWSE/TPEx 官方 API 抓取股價資料寫入 raw_prices 表。

來源切換：env var INGEST_PRICES_SOURCE
  - finmind（預設，向後相容）：批次查詢 + 逗號分隔 data_id，每批 500 檔
  - twse：TWSE/TPEx Legacy endpoints 逐日抓全市場（一日一 call，無需 token）

切換後 daily pipeline 與既有 backfill 機制不變；DB schema 不變。
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import Dict, List
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
    probe_dataset_has_data,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawPrice
from app.twse_client import TWSEClient, TWSEError

logger = logging.getLogger(__name__)


DATASET = "TaiwanStockPrice"
SOURCE_ENV = "INGEST_PRICES_SOURCE"


def _resolve_source() -> str:
    """讀 INGEST_PRICES_SOURCE，預設 'finmind'。未知值 fall-back finmind（typo 防呆）。"""
    val = os.environ.get(SOURCE_ENV, "finmind").lower().strip()
    if val not in ("finmind", "twse"):
        logger.warning(
            "[ingest_prices] unknown %s=%r, falling back to 'finmind'", SOURCE_ENV, val
        )
        return "finmind"
    return val


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
    df = df[df["stock_id"].str.fullmatch(r"\d{4,6}")]
    return df[["stock_id", "trading_date", "open", "high", "low", "close", "volume"]].drop_duplicates(
        subset=["stock_id", "trading_date"]
    )


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawPrice.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def run(config, db_session: Session, **kwargs) -> Dict:
    """Dispatch 到 FinMind 或 TWSE 後端，依 INGEST_PRICES_SOURCE 決定。"""
    source = _resolve_source()
    if source == "twse":
        return _run_twse(config, db_session)
    return _run_finmind(config, db_session)


def _normalize_twse_prices(rows: List[Dict], allowed_pattern: str = r"\d{4,6}") -> pd.DataFrame:
    """TWSEClient.fetch_prices_history 回傳的 row 已含必要欄位，僅做型別 + 過濾。

    過濾條件：
    - stock_id 必為 4-6 碼數字（含 ETF），與 _normalize_prices() 行為一致
    - 必要欄位（open/high/low/close）至少有一個非 NaN（全空則整列丟棄）
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["stock_id"].str.fullmatch(allowed_pattern)]
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = None
    df = df.dropna(subset=["stock_id", "trading_date"], how="any")
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")
    df = df.drop_duplicates(subset=["stock_id", "trading_date"])
    return df[["stock_id", "trading_date", "open", "high", "low", "close", "volume"]]


def _run_twse(config, db_session: Session) -> Dict:
    """TWSE/TPEx Legacy 後端：逐日抓全市場 OHLCV。

    對 daily 增量（start==end==今日）只 2 個 API call（TWSE + TPEx）。
    對 backfill 需逐日 loop，rate-limit 預設 1.5s/req → 約 2500 日 × 2 call × 1.5s ≈ 125 分鐘。
    """
    job_id = start_job(db_session, "ingest_prices", commit=True)
    logs: Dict[str, object] = {"source": "twse"}
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

        client = TWSEClient()
        total_rows = 0
        skipped_days = 0
        current = start_date
        days_total = (end_date - start_date).days + 1
        days_done = 0

        while current <= end_date:
            # 週末快速 skip（TWSE/TPEx legacy 對假日回空 data，但仍會打網路；本地跳過更省）
            if current.weekday() >= 5:
                current += timedelta(days=1)
                days_done += 1
                continue
            try:
                rows = client.fetch_prices_history(current)
            except TWSEError as exc:
                logger.warning("[ingest_prices/twse] %s fetch failed: %s", current, exc)
                skipped_days += 1
                current += timedelta(days=1)
                days_done += 1
                continue

            df = _normalize_twse_prices(rows)
            if not df.empty:
                records: List[Dict] = df.to_dict("records")
                BATCH_SIZE = 5000
                for i in range(0, len(records), BATCH_SIZE):
                    batch = records[i : i + BATCH_SIZE]
                    stmt = insert(RawPrice).values(batch)
                    update_cols = {
                        col: stmt.inserted[col] for col in ["open", "high", "low", "close", "volume"]
                    }
                    stmt = stmt.on_duplicate_key_update(**update_cols)
                    db_session.execute(stmt)
                    db_session.commit()
                total_rows += len(records)

            days_done += 1
            if days_done % 20 == 0:
                logs["progress"] = {"days_done": days_done, "days_total": days_total, "rows": total_rows}
                update_job(db_session, job_id, logs=logs, commit=True)
            current += timedelta(days=1)

        logs.update({"rows": total_rows, "days": days_total, "skipped_days": skipped_days})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:
        logger.error("[ingest_prices/twse] 失敗: %s", exc, exc_info=True)
        try:
            db_session.rollback()
        except Exception as rb_exc:
            logger.warning("[ingest_prices/twse] rollback 失敗: %s", rb_exc)
        try:
            logs["error"] = str(exc)
            finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        except Exception as finish_exc:
            logger.warning("[ingest_prices/twse] finish_job 寫入失敗: %s", finish_exc)
        raise


def _run_finmind(config, db_session: Session) -> Dict:
    job_id = start_job(db_session, "ingest_prices", commit=True)
    logs: Dict[str, object] = {"source": "finmind"}
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

        # 先用探針股票檢查這段期間是否有資料，避免假日/空窗期逐檔 call API。
        probe = probe_dataset_has_data(
            dataset=DATASET,
            start_date=start_date,
            end_date=end_date,
            token=config.finmind_token,
            requests_per_hour=config.finmind_requests_per_hour,
            max_retries=config.finmind_retry_max,
            backoff_seconds=config.finmind_retry_backoff,
        )
        logs["probe"] = probe
        if not probe.get("has_data", False):
            logs["rows"] = 0
            logs["skip_reason"] = "no_data_in_range"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0, "start_date": start_date, "end_date": end_date}

        # 取得股票清單
        stock_ids = fetch_stock_list(
            config.finmind_token,
            requests_per_hour=config.finmind_requests_per_hour,
            max_retries=config.finmind_retry_max,
            backoff_seconds=config.finmind_retry_backoff,
        )
        logs["stock_count"] = len(stock_ids)
        logs["fetch_mode"] = "by_stock_batch"

        if not stock_ids:
            logs["warning"] = "無法取得股票清單，跳過抓取"
            logs["rows"] = 0
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0, "start_date": start_date, "end_date": end_date}

        total_rows = 0
        chunk_ranges = list(date_chunks(start_date, end_date, chunk_days=config.chunk_days))
        total_chunks = len(chunk_ranges)
        logs["chunks_total"] = total_chunks

        for chunk_count, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            logs["progress"] = {
                "current_chunk": chunk_count,
                "total_chunks": total_chunks,
                "chunk_start": chunk_start.isoformat(),
                "chunk_end": chunk_end.isoformat(),
                "rows": total_rows,
            }
            update_job(db_session, job_id, logs=logs, commit=True)

            # 使用批次查詢（每 500 檔一次 API call，逗號分隔 data_id）
            df = fetch_dataset_by_stocks(
                DATASET,
                chunk_start,
                chunk_end,
                stock_ids,
                token=config.finmind_token,
                batch_size=500,
                use_batch_query=True,
                requests_per_hour=config.finmind_requests_per_hour,
                max_retries=config.finmind_retry_max,
                backoff_seconds=config.finmind_retry_backoff,
            )

            if df.empty:
                continue
            df = _normalize_prices(df)
            records: List[Dict] = df.to_dict("records")
            if not records:
                continue

            BATCH_SIZE = 5000
            for i in range(0, len(records), BATCH_SIZE):
                batch = records[i : i + BATCH_SIZE]
                stmt = insert(RawPrice).values(batch)
                update_cols = {col: stmt.inserted[col] for col in ["open", "high", "low", "close", "volume"]}
                stmt = stmt.on_duplicate_key_update(**update_cols)
                db_session.execute(stmt)
                db_session.commit()
            total_rows += len(records)

        logs.update({"rows": total_rows, "chunks": total_chunks})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:
        logger.error("[ingest_prices/finmind] 失敗: %s", exc, exc_info=True)
        try:
            db_session.rollback()
        except Exception as rb_exc:
            logger.warning("[ingest_prices/finmind] rollback 失敗: %s", rb_exc)
        try:
            logs["error"] = str(exc)
            finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        except Exception as finish_exc:
            logger.warning(
                "[ingest_prices/finmind] finish_job 寫入失敗（保留原始例外）: %s", finish_exc
            )
        raise
