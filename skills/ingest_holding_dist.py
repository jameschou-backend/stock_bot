"""Priority 2：持股分級週報（TaiwanStockHoldingSharesPer）

從 FinMind 抓取每週持股分級資料，彙整成大/小戶持股比例，
寫入 raw_holding_dist 表。

Dataset: TaiwanStockHoldingSharesPer
Fields:  date, stock_id, HoldingSharesLevel, people, unit
頻率：   每週（週五更新）
限制：   Sponsor 計劃；資料從 2010 年起

聚合邏輯：
- large_holder_pct:  unit 合計中，持有 >= 1,000,000 股（1000 張）的比例
- small_holder_pct:  unit 合計中，持有 < 1,000,000 股的比例
- top_level_pct:     unit 合計中，最高級別持股人的比例（含 HoldingSharesLevel 最大類別）
- holder_count:      所有 HoldingSharesLevel 的 people 合計
"""
from __future__ import annotations

import re
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
from app.models import RawHoldingDist, Stock

DATASET = "TaiwanStockHoldingSharesPer"
UPDATE_COLS = ["large_holder_pct", "small_holder_pct", "top_level_pct", "holder_count"]

# 大戶門檻：1,000,000 股（= 1000 張）
LARGE_HOLDER_MIN_SHARES = 1_000_000


def _parse_level_min(level_str: str) -> int:
    """將 HoldingSharesLevel 字串解析為最小股數（整數）。

    範例：'1-999' → 1, '1,000-5,000' → 1000, 'over 400,000' → 400000
    """
    s = str(level_str).replace(",", "").strip().lower()
    if s.startswith("over"):
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 0
    m = re.match(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawHoldingDist.trading_date)).scalar()
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


def _aggregate_holding(df: pd.DataFrame, allowed_stock_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """將 TaiwanStockHoldingSharesPer 原始資料彙整成每週每股聚合指標。"""
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
    level_col = next((c for c in ["HoldingSharesLevel", "holdingSharesLevel", "level"] if c in df.columns), None)
    unit_col = next((c for c in ["unit", "Unit", "shares"] if c in df.columns), None)
    people_col = next((c for c in ["people", "People", "holders"] if c in df.columns), None)

    if level_col is None or unit_col is None:
        raise FinMindError(
            f"TaiwanStockHoldingSharesPer missing required columns; got: {df.columns.tolist()}"
        )

    df["unit"] = pd.to_numeric(df[unit_col], errors="coerce").fillna(0)
    df["people"] = pd.to_numeric(df[people_col], errors="coerce").fillna(0) if people_col else 0
    df["level_min"] = df[level_col].apply(_parse_level_min)

    results = []
    for (stock_id, trading_date), grp in df.groupby(["stock_id", "trading_date"]):
        total_unit = grp["unit"].sum()
        total_people = int(grp["people"].sum())

        if total_unit <= 0:
            continue

        large = grp[grp["level_min"] >= LARGE_HOLDER_MIN_SHARES]["unit"].sum()
        small = grp[grp["level_min"] < LARGE_HOLDER_MIN_SHARES]["unit"].sum()

        # 最高級別（level_min 最大的那一組）
        top_row = grp.loc[grp["level_min"].idxmax()]
        top_pct = float(top_row["unit"]) / float(total_unit)

        results.append({
            "stock_id": stock_id,
            "trading_date": trading_date,
            "large_holder_pct": round(float(large) / float(total_unit), 4),
            "small_holder_pct": round(float(small) / float(total_unit), 4),
            "top_level_pct": round(top_pct, 4),
            "holder_count": total_people,
        })

    return pd.DataFrame(results)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_holding_dist", commit=True)
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

        print(f"[ingest_holding_dist] {start_date} ~ {end_date}", flush=True)

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

        # 持股分級為週資料，用較大 chunk 減少 API 次數
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

            agg_df = _aggregate_holding(df, allowed_stock_ids=allowed_stock_ids or None)
            if agg_df.empty:
                continue

            records: List[Dict] = agg_df.to_dict("records")
            stmt = insert(RawHoldingDist).values(records)
            stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
            db_session.execute(stmt)
            db_session.commit()
            total_rows += len(records)
            print(f"  chunk {i}/{len(chunk_ranges)}: {len(records)} 筆", flush=True)

        logs["rows"] = total_rows
        print(f"  ✅ holding_dist: {total_rows} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
