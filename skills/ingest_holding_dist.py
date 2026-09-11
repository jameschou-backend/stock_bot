"""Priority 2：持股分級週報（TaiwanStockHoldingSharesPer）

從 FinMind 抓取每週持股分級資料，彙整成大/小戶持股比例，
寫入 validated_holding_dist 表（舊表隔離，不再供特徵使用）。

Dataset: TaiwanStockHoldingSharesPer
Fields:  date, stock_id, HoldingSharesLevel, people, unit
頻率：   每週（週五更新）
限制：   Sponsor 計劃；資料從 2010 年起

聚合邏輯：
- large_holder_pct:  unit 合計中，持有 > 1,000,000 股（1000 張）的比例
- small_holder_pct:  unit 合計中，持有 <= 1,000,000 股的比例
- top_level_pct:     unit 合計中，最高級別持股人的比例（含 HoldingSharesLevel 最大類別）
- holder_count:      所有 HoldingSharesLevel 的 people 合計
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import (
    FinMindError,
    fetch_dataset_by_stocks,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import ValidatedHoldingDist as RawHoldingDist, Stock

logger = logging.getLogger(__name__)

DATASET = "TaiwanStockHoldingSharesPer"
UPDATE_COLS = ["large_holder_pct", "small_holder_pct", "top_level_pct", "holder_count", "available_date"]

def _load_allowed_stock_ids(session: Session) -> Set[str]:
    rows = (
        session.query(Stock.stock_id)
        .filter(Stock.is_listed == True)
        .filter(Stock.security_type == "stock")
        .all()
    )
    return {row[0] for row in rows}


def _aggregate_holding(df: pd.DataFrame, allowed_stock_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    from skills.holding_validation import aggregate
    return aggregate(df, allowed_stock_ids)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_holding_dist", commit=True)
    logs: Dict = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = today - timedelta(days=365 * config.train_lookback_years)
        allowed_stock_ids = _load_allowed_stock_ids(db_session)
        if not allowed_stock_ids:
            raise FinMindError('No eligible stock metadata; refresh stock list before holder ingest')
        latest_by_stock = dict(db_session.query(RawHoldingDist.stock_id,
            func.max(RawHoldingDist.trading_date)).group_by(RawHoldingDist.stock_id).all())
        starts = {sid:max(default_start, latest_by_stock[sid]+timedelta(days=1))
                  if sid in latest_by_stock else default_start for sid in allowed_stock_ids}
        start_date = min(starts.values())
        end_date   = today

        logs["start_date"] = start_date.isoformat()
        logs["end_date"]   = end_date.isoformat()

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "already_up_to_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        logger.info("[ingest_holding_dist] %s ~ %s", start_date, end_date)

        logs["allowed_stock_ids"] = len(allowed_stock_ids)
        stock_ids = sorted(sid for sid in allowed_stock_ids if starts[sid] <= end_date)

        # TaiwanStockHoldingSharesPer 不支援 batch data_id（comma-separated）
        # 須逐股查詢（use_batch_query=False）。週資料量小，一次抓全區間（一個 chunk）
        logs["fetch_mode"] = "per_stock_no_batch"
        total_rows = 0
        commit_buffer: List[Dict] = []
        stock_ids = sorted(set(stock_ids) & allowed_stock_ids)
        total_stocks = len(stock_ids)

        for i, sid in enumerate(stock_ids, 1):
            if i % 200 == 0:
                update_job(db_session, job_id, logs={**logs, "progress": f"{i}/{total_stocks}", "rows": total_rows}, commit=True)
                logger.info("[%d/%d] 已寫 %d 筆...", i, total_stocks, total_rows)

            df = fetch_dataset_by_stocks(
                DATASET,
                starts[sid],
                end_date,
                [sid],                   # 一次一股
                token=config.finmind_token,
                batch_size=1,
                use_batch_query=False,   # 跳過無效的 batch attempt
                requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
                max_retries=getattr(config, "finmind_retry_max", 3),
                backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
            )
            if df is None or df.empty:
                continue

            agg_df = _aggregate_holding(df, allowed_stock_ids=allowed_stock_ids)
            if agg_df.empty:
                continue

            commit_buffer.extend(agg_df.to_dict("records"))
            if len(commit_buffer) >= 2000:
                stmt = insert(RawHoldingDist).values(commit_buffer)
                stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
                db_session.execute(stmt)
                db_session.commit()
                total_rows += len(commit_buffer)
                commit_buffer.clear()

        if commit_buffer:
            stmt = insert(RawHoldingDist).values(commit_buffer)
            stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
            db_session.execute(stmt)
            db_session.commit()
            total_rows += len(commit_buffer)

        logs["rows"] = total_rows
        logger.info("holding_dist: %d 筆", total_rows)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        logger.error("[ingest_holding_dist] 失敗: %s", exc, exc_info=True)
        try:
            db_session.rollback()
        except Exception as rb_exc:
            logger.warning("[ingest_holding_dist] rollback 失敗: %s", rb_exc)
        try:
            finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        except Exception as finish_exc:
            logger.warning(
                "[ingest_holding_dist] finish_job 寫入失敗（保留原始例外）: %s",
                finish_exc,
            )
        raise
