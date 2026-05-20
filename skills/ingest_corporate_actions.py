from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import CorporateAction, PriceAdjustFactor, RawPrice
from app.twse_client import TWSEClient, TWSEError

logger = logging.getLogger(__name__)


TODO_SOURCE_HINTS = [
    "TWSE 除權除息公告（已實作，設 INGEST_CORPORATE_ACTIONS_SOURCE=twse 啟用）",
    "TPEx 除權除息（未實作，TODO）",
    "公開資訊觀測站（MOPS）—— 用於配股配息細節（未實作）",
]

SOURCE_ENV = "INGEST_CORPORATE_ACTIONS_SOURCE"


def _resolve_source() -> str:
    """讀 INGEST_CORPORATE_ACTIONS_SOURCE。

    - 'none'（預設）：不抓外部資料，price_adjust_factors 全填 1.0 保留現行行為
    - 'twse'：呼叫 TWSE TWT49U 抓除權除息事件，寫入 corporate_actions 表
      （price_adjust_factors 仍維持 1.0，cumulative adj 重算屬 Stage 後續工作）
    """
    val = os.environ.get(SOURCE_ENV, "none").lower().strip()
    if val not in ("none", "twse"):
        logger.warning("[ingest_corporate_actions] unknown %s=%r, falling back to 'none'", SOURCE_ENV, val)
        return "none"
    return val


def _date_range_from_input(date_range: Tuple[date, date] | None, tz: str) -> Tuple[date, date]:
    if date_range is not None:
        return date_range
    today = datetime.now(ZoneInfo(tz)).date()
    return today - timedelta(days=365), today


def _fetch_external_actions(
    start_date: date,
    end_date: date,
    config,
) -> pd.DataFrame:
    """根據 INGEST_CORPORATE_ACTIONS_SOURCE 抓除權除息事件。"""
    source = _resolve_source()
    if source == "none":
        _ = start_date, end_date, config
        return pd.DataFrame()
    if source == "twse":
        try:
            client = TWSEClient()
            rows = client.fetch_ex_rights_history(start_date, end_date)
        except TWSEError as exc:
            logger.warning("[ingest_corporate_actions] TWSE fetch failed: %s", exc)
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # 對齊舊 schema：CorporateAction 表用 action_date / action_type / adj_factor / payload_json
        # 已在 client 內就是這個欄位名
        df = df[df["stock_id"].astype(str).str.fullmatch(r"\d{4}")]
        return df
    return pd.DataFrame()


def _upsert_corporate_actions(session: Session, records: List[Dict]) -> int:
    if not records:
        return 0
    stmt = insert(CorporateAction).values(records)
    stmt = stmt.on_duplicate_key_update(
        action_type=stmt.inserted.action_type,
        adj_factor=stmt.inserted.adj_factor,
        payload_json=stmt.inserted.payload_json,
    )
    session.execute(stmt)
    return len(records)


def _build_default_adjust_factors(
    session: Session,
    start_date: date,
    end_date: date,
) -> Iterable[Dict]:
    # 增量模式：只補算 price_adjust_factors 尚未覆蓋的日期，避免每次重新 upsert 全部 365 天
    max_existing = session.execute(
        select(func.max(PriceAdjustFactor.trading_date))
    ).scalar()
    if max_existing is not None:
        incremental_start = max_existing + timedelta(days=1)
        if incremental_start > end_date:
            return  # 已是最新，無需補算
        start_date = max(start_date, incremental_start)

    stmt = (
        select(RawPrice.stock_id, RawPrice.trading_date)
        .where(RawPrice.trading_date.between(start_date, end_date))
        .order_by(RawPrice.stock_id, RawPrice.trading_date)
    )
    rows = session.execute(stmt).fetchall()
    for stock_id, trading_date in rows:
        yield {
            "stock_id": str(stock_id),
            "trading_date": trading_date,
            "adj_factor": 1.0,
        }


def _upsert_adjust_factors(session: Session, records: List[Dict]) -> int:
    if not records:
        return 0
    upserted = 0
    batch_size = 1000
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        stmt = insert(PriceAdjustFactor).values(batch)
        stmt = stmt.on_duplicate_key_update(
            adj_factor=stmt.inserted.adj_factor,
        )
        session.execute(stmt)
        upserted += len(batch)
    return upserted


def run_date_range(config, db_session: Session, date_range: Tuple[date, date] | None = None) -> Dict:
    start_date, end_date = _date_range_from_input(date_range, config.tz)
    logs: Dict[str, object] = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "todo_sources": TODO_SOURCE_HINTS,
    }

    action_df = _fetch_external_actions(start_date, end_date, config)
    logs["external_action_rows"] = int(len(action_df))

    action_records: List[Dict] = []
    if not action_df.empty:
        action_df = action_df.copy()
        action_df["stock_id"] = action_df["stock_id"].astype(str)
        action_df["action_date"] = pd.to_datetime(action_df["action_date"], errors="coerce").dt.date
        action_df["adj_factor"] = pd.to_numeric(action_df.get("adj_factor"), errors="coerce")
        for _, row in action_df.iterrows():
            # payload_json：保留來源原始細節（pre_close / ref_price / value_amount）方便未來 audit
            payload = row.get("payload_json")
            if not isinstance(payload, dict):
                payload = {}
            for k in ("pre_close", "ref_price", "value_amount", "market"):
                if k in row and pd.notna(row[k]):
                    payload[k] = row[k] if not isinstance(row[k], float) else float(row[k])
            action_records.append(
                {
                    "stock_id": row["stock_id"],
                    "action_date": row["action_date"],
                    "action_type": str(row.get("action_type", "OTHER")),
                    "adj_factor": None if pd.isna(row["adj_factor"]) else float(row["adj_factor"]),
                    "payload_json": payload,
                }
            )

    actions_upserted = _upsert_corporate_actions(db_session, action_records)

    factor_records = list(_build_default_adjust_factors(db_session, start_date, end_date))
    factors_upserted = _upsert_adjust_factors(db_session, factor_records)

    source = _resolve_source()
    logs.update(
        {
            "source": source,
            "actions_upserted": actions_upserted,
            "factors_upserted": factors_upserted,
        }
    )
    if source == "none":
        logs["warning"] = (
            "INGEST_CORPORATE_ACTIONS_SOURCE=none：未抓除權除息事件，"
            "price_adjust_factors 全填 1.0（保留舊行為）。"
            "切到 'twse' 啟用後 corporate_actions 表會填上事件，"
            "price_adjust_factors cumulative 重算屬後續 Stage 工作。"
        )
    return logs


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_corporate_actions")
    try:
        logs = run_date_range(config, db_session, kwargs.get("date_range"))
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover
        logger.error("[ingest_corporate_actions] 失敗: %s", exc, exc_info=True)
        try:
            db_session.rollback()
        except Exception as rb_exc:
            logger.warning("[ingest_corporate_actions] rollback 失敗: %s", rb_exc)
        try:
            finish_job(
                db_session, job_id, "failed",
                error_text=str(exc), logs={"error": str(exc)},
            )
        except Exception as finish_exc:
            logger.warning(
                "[ingest_corporate_actions] finish_job 寫入失敗（保留原始例外）: %s",
                finish_exc,
            )
        raise
