"""Ingest Institutional 模組

從 FinMind 或 TWSE/TPEx 官方 API 抓取三大法人買賣超寫入 raw_institutional 表。

來源切換：env var INGEST_INSTITUTIONAL_SOURCE
  - finmind（預設）：批次查詢 + 逗號分隔 data_id
  - twse：TWSE T86 Legacy + TPEx Legacy 3insti 逐日抓全市場
    （TWSE OpenAPI 沒有 T86 對應 endpoint，**必走 legacy**）

DB schema 不變。
"""

from __future__ import annotations

import logging
import os
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
    probe_dataset_has_data,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawInstitutional, Stock
from app.twse_client import TWSEClient, TWSEError

logger = logging.getLogger(__name__)


DATASET = "TaiwanStockInstitutionalInvestorsBuySell"
SOURCE_ENV = "INGEST_INSTITUTIONAL_SOURCE"


def _resolve_source() -> str:
    val = os.environ.get(SOURCE_ENV, "finmind").lower().strip()
    if val not in ("finmind", "twse"):
        logger.warning(
            "[ingest_institutional] unknown %s=%r, falling back to 'finmind'", SOURCE_ENV, val
        )
        return "finmind"
    return val

CATEGORY_MAP = {
    "foreign_investor": "foreign",
    "foreign_dealer_self": "foreign",
    "investment_trust": "trust",
    "dealer_self": "dealer",
    "dealer_hedging": "dealer",
}


def _normalize_institutional(
    df: pd.DataFrame,
    allowed_stock_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
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

    df["stock_id"] = df["stock_id"].astype(str)
    if allowed_stock_ids:
        df = df[df["stock_id"].isin(allowed_stock_ids)]
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


def _load_allowed_stock_ids(session: Session) -> Set[str]:
    rows = (
        session.query(Stock.stock_id)
        .filter(Stock.is_listed == True)
        .filter(Stock.security_type == "stock")
        .all()
    )
    return {row[0] for row in rows}


INST_UPDATE_COLS = [
    "foreign_buy", "foreign_sell", "foreign_net",
    "trust_buy", "trust_sell", "trust_net",
    "dealer_buy", "dealer_sell", "dealer_net",
]


def run(config, db_session: Session, **kwargs) -> Dict:
    """Dispatch FinMind vs TWSE 後端，依 INGEST_INSTITUTIONAL_SOURCE 決定。"""
    source = _resolve_source()
    if source == "twse":
        return _run_twse(config, db_session)
    return _run_finmind(config, db_session)


def _normalize_twse_institutional(rows: List[Dict]) -> pd.DataFrame:
    """TWSEClient.fetch_institutional_history 回傳含上市+上櫃。對應 DB schema：
    foreign_buy/sell/net, trust_buy/sell/net, dealer_buy/sell/net
    （TWSE 沒有獨立 dealer_buy/sell 欄位，用 dealer_self + dealer_hedging 合計。）
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]  # 與生產 universe 一致（排除 ETF/權證）
    df = df.dropna(subset=["stock_id", "trading_date"], how="any")

    # 補 dealer_buy / dealer_sell（TWSE T86 row 沒拆細，只給 self + hedging 的 net）
    # 對於 daily ingest，net 已足夠（前端 features 主要用 net）。
    # 若需要 buy/sell，需另外從 r[12..17] 抓細項；此處保守填 0 避免 NULL constraint。
    for col in ("dealer_buy", "dealer_sell"):
        if col not in df.columns:
            df[col] = 0

    keep_cols = [
        "stock_id", "trading_date",
        "foreign_buy", "foreign_sell", "foreign_net",
        "trust_buy", "trust_sell", "trust_net",
        "dealer_buy", "dealer_sell", "dealer_net",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    df = df.drop_duplicates(subset=["stock_id", "trading_date"])
    df = df[keep_cols]
    # 數值欄位 NaN -> None（MySQL 不接受 NaN）
    for col in keep_cols:
        if col in ("stock_id", "trading_date"):
            continue
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


def _run_twse(config, db_session: Session) -> Dict:
    job_id = start_job(db_session, "ingest_institutional", commit=True)
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
            if current.weekday() >= 5:
                current += timedelta(days=1)
                days_done += 1
                continue
            try:
                rows = client.fetch_institutional_history(current)
            except TWSEError as exc:
                logger.warning("[ingest_institutional/twse] %s fetch failed: %s", current, exc)
                skipped_days += 1
                current += timedelta(days=1)
                days_done += 1
                continue

            df = _normalize_twse_institutional(rows)
            if not df.empty:
                records = df.to_dict("records")
                BATCH_SIZE = 5000
                for i in range(0, len(records), BATCH_SIZE):
                    batch = records[i : i + BATCH_SIZE]
                    stmt = insert(RawInstitutional).values(batch)
                    update_cols = {col: stmt.inserted[col] for col in INST_UPDATE_COLS}
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
        logger.error("[ingest_institutional/twse] 失敗: %s", exc, exc_info=True)
        try:
            db_session.rollback()
        except Exception as rb_exc:
            logger.warning("[ingest_institutional/twse] rollback 失敗: %s", rb_exc)
        try:
            logs["error"] = str(exc)
            finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        except Exception as finish_exc:
            logger.warning("[ingest_institutional/twse] finish_job 寫入失敗: %s", finish_exc)
        raise


def _run_finmind(config, db_session: Session) -> Dict:
    job_id = start_job(db_session, "ingest_institutional", commit=True)
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

        # 先做探針檢查，若區間無資料就不進入逐檔抓取。
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

        total_rows = 0
        allowed_stock_ids = _load_allowed_stock_ids(db_session)
        logs["allowed_stock_ids"] = len(allowed_stock_ids)

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
            df = _normalize_institutional(df, allowed_stock_ids=allowed_stock_ids or None)
            records: List[Dict] = df.to_dict("records")
            if not records:
                continue

            stmt = insert(RawInstitutional).values(records)
            update_cols = {col: stmt.inserted[col] for col in INST_UPDATE_COLS}
            stmt = stmt.on_duplicate_key_update(**update_cols)
            db_session.execute(stmt)
            total_rows += len(records)

        logs.update({"rows": total_rows, "chunks": total_chunks})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        logger.error("[ingest_institutional] 失敗: %s", exc, exc_info=True)
        logs["error"] = str(exc)
        try:
            finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        except Exception as finish_exc:
            logger.warning(
                "[ingest_institutional] finish_job 寫入失敗（保留原始例外）: %s",
                finish_exc,
            )
        raise
