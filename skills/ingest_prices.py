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
    df = df[["stock_id", "trading_date", "open", "high", "low", "close", "volume"]]
    # 個別欄位 NaN -> None（MySQL 不接受 NaN）
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


def _run_twse(config, db_session: Session) -> Dict:
    """TWSE/TPEx 後端 (daily 增量限定)。

    ⚠️ 重要限制：TWSE STOCK_DAY_ALL endpoint **不接受 date 參數**，永遠回最新一天
    （實測 2026-05-19 確認，與 deep-research 結論不同）。所以本後端：
    - 只能處理「DB 最新一日 + 1 ~ today」≤ 1 天的場景（=正常 daily 增量）
    - 若 DB 落後超過 1 天（連假/休假/中斷），本後端會 raise，要求暫時切回
      INGEST_PRICES_SOURCE=finmind 把缺失補完，再切回 twse
    - row 內的 trading_date 用 TWSE/TPEx server 回傳的最新交易日（不一定是今天，
      週末/假日會是上一個工作日）

    institutional / margin / per 的 legacy endpoint 真的接受 date，不受此限制。
    """
    job_id = start_job(db_session, "ingest_prices", commit=True)
    logs: Dict[str, object] = {"source": "twse"}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = today - timedelta(days=365 * config.train_lookback_years)
        start_date = _resolve_start_date(db_session, default_start)
        end_date = today
        days_gap = (end_date - start_date).days
        logs.update({
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days_gap": days_gap,
        })

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "start_date > end_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0, "start_date": start_date, "end_date": end_date}

        # TWSE STOCK_DAY_ALL 只回最新一天，無法 backfill 多天。
        # 容忍 1 天 gap（涵蓋假日/週末延遲），更長則拒絕並提示 fallback。
        # 注意：1 個自然週末 = 2 天 gap，因此 7 天才算「真的長時間中斷」。
        MAX_GAP_DAYS = 7
        if days_gap > MAX_GAP_DAYS:
            msg = (
                f"TWSE prices backend 無法 backfill 超過 {MAX_GAP_DAYS} 天的歷史："
                f"DB 最新 {start_date - timedelta(days=1)}, 今天 {end_date}, 缺 {days_gap} 天。"
                f"請暫時設 INGEST_PRICES_SOURCE=finmind 把缺失歷史補完，再切回 twse。"
                f"原因：TWSE STOCK_DAY_ALL endpoint 不接受 date 參數，無法歷史查詢。"
            )
            logger.error("[ingest_prices/twse] %s", msg)
            logs["error"] = msg
            logs["fallback_hint"] = "set INGEST_PRICES_SOURCE=finmind temporarily"
            finish_job(db_session, job_id, "failed", error_text=msg, logs=logs)
            raise RuntimeError(msg)

        client = TWSEClient()
        # 抓最新一天（TWSE OpenAPI 與 legacy 結果相同，這裡用 OpenAPI 較單純）
        rows = client.fetch_prices_latest()
        df = _normalize_twse_prices(rows)
        total_rows = 0

        if not df.empty:
            # 用 row 內的 trading_date（TWSE 自帶的最新交易日，可能是今天或上一個工作日）
            unique_dates = sorted(df["trading_date"].unique())
            logs["fetched_trading_dates"] = [str(d) for d in unique_dates]
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
            total_rows = len(records)

        logs["rows"] = total_rows
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
