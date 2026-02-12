"""Ingest Margin Short 模組

從 FinMind 抓取融資融券資料寫入 raw_margin_short 表。

FinMind Dataset: TaiwanStockMarginPurchaseShortSale
- 資料區間：2001-01-01 至今
- 更新時間：星期一至五 21:00

支援兩種模式：
1. 全市場抓取（需較高權限）
2. 逐檔抓取（適用免費/低階會員）
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
    fetch_dataset_by_stocks,
    fetch_stock_list,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawMarginShort


DATASET = "TaiwanStockMarginPurchaseShortSale"

# 股票清單快取
_stock_list_cache: Optional[List[str]] = None


# FinMind 欄位映射（根據實際 API 回傳調整）
COLUMN_MAP = {
    "date": "trading_date",
    "stock_id": "stock_id",
    # 融資相關
    "MarginPurchaseBuy": "margin_purchase_buy",
    "MarginPurchaseSell": "margin_purchase_sell",
    "MarginPurchaseCashRepayment": "margin_purchase_cash_repay",
    "MarginPurchaseLimit": "margin_purchase_limit",
    "MarginPurchaseTodayBalance": "margin_purchase_balance",
    # 融券相關
    "ShortSaleBuy": "short_sale_buy",
    "ShortSaleSell": "short_sale_sell",
    "ShortSaleCashRepayment": "short_sale_cash_repay",
    "ShortSaleLimit": "short_sale_limit",
    "ShortSaleTodayBalance": "short_sale_balance",
    # 資券互抵
    "OffsetLoanAndShort": "offset_loan_and_short",
    "Note": "note",
}


def _normalize_margin_short(df: pd.DataFrame) -> pd.DataFrame:
    """正規化融資融券資料
    
    Args:
        df: 原始 DataFrame
    
    Returns:
        正規化後的 DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    # 欄位重命名
    rename_map = {}
    for orig, target in COLUMN_MAP.items():
        if orig in df.columns:
            rename_map[orig] = target
    
    df = df.rename(columns=rename_map)
    
    # 檢查必要欄位
    required = {"stock_id", "trading_date"}
    missing = required - set(df.columns)
    if missing:
        raise FinMindError(f"Margin dataset missing columns: {sorted(missing)}")
    
    # 日期處理
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    
    # 數值欄位處理
    numeric_cols = [
        "margin_purchase_buy", "margin_purchase_sell", "margin_purchase_cash_repay",
        "margin_purchase_limit", "margin_purchase_balance",
        "short_sale_buy", "short_sale_sell", "short_sale_cash_repay",
        "short_sale_limit", "short_sale_balance",
        "offset_loan_and_short",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = None
    
    # note 欄位
    if "note" not in df.columns:
        df["note"] = None
    
    # 清理
    df = df.dropna(subset=["stock_id", "trading_date"])
    df["stock_id"] = df["stock_id"].astype(str)
    
    # 只保留台股四碼代碼
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]
    
    # 選擇需要的欄位
    output_cols = ["stock_id", "trading_date"] + numeric_cols + ["note"]
    output_cols = [c for c in output_cols if c in df.columns]
    
    return df[output_cols].drop_duplicates(subset=["stock_id", "trading_date"])


def _resolve_start_date(session: Session, default_start: date) -> date:
    """決定抓取起始日期"""
    max_date = session.query(func.max(RawMarginShort.trading_date)).scalar()
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


def _fetch_margin_smart(
    start_date: date,
    end_date: date,
    token: str | None,
    requests_per_hour: int,
    max_retries: int,
    backoff_seconds: float,
    bulk_chunk_days: int,
    logs: Dict[str, object],
) -> pd.DataFrame:
    """逐檔抓取（融資融券建議直接逐檔）
    
    Returns:
        合併後的 DataFrame
    """
    logs["fetch_mode"] = "by_stock"
    stock_ids = _get_stock_list(token, requests_per_hour, max_retries, backoff_seconds)
    logs["stock_list_count"] = len(stock_ids)
    
    if not stock_ids:
        logs["warning"] = "無法取得股票清單，跳過抓取"
        return pd.DataFrame()
    
    return fetch_dataset_by_stocks(
        DATASET,
        start_date,
        end_date,
        stock_ids,
        token=token,
        use_batch_query=False,
        requests_per_hour=requests_per_hour,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        debug=True,
    )


def run(config, db_session: Session, **kwargs) -> Dict:
    """執行融資融券資料抓取
    
    Returns:
        包含抓取統計的 dict
    """
    job_id = start_job(db_session, "ingest_margin_short", commit=True)
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
        chunk_days = config.chunk_days
        if getattr(config, "margin_bulk_chunk_days", 0):
            chunk_days = min(chunk_days, config.margin_bulk_chunk_days)
        chunk_ranges = list(date_chunks(start_date, end_date, chunk_days=chunk_days))
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
            
            df = _fetch_margin_smart(
                chunk_start,
                chunk_end,
                config.finmind_token,
                config.finmind_requests_per_hour,
                config.finmind_retry_max,
                config.finmind_retry_backoff,
                config.margin_bulk_chunk_days,
                chunk_logs,
            )
            logs.update({f"chunk_{chunk_count}": chunk_logs})
            
            if df.empty:
                continue
            
            df = _normalize_margin_short(df)
            records: List[Dict] = df.to_dict("records")
            if not records:
                continue
            
            # Upsert
            stmt = insert(RawMarginShort).values(records)
            update_cols = {
                col: stmt.inserted[col]
                for col in [
                    "margin_purchase_buy", "margin_purchase_sell", "margin_purchase_cash_repay",
                    "margin_purchase_limit", "margin_purchase_balance",
                    "short_sale_buy", "short_sale_sell", "short_sale_cash_repay",
                    "short_sale_limit", "short_sale_balance",
                    "offset_loan_and_short", "note",
                ]
                if col in df.columns
            }
            stmt = stmt.on_duplicate_key_update(**update_cols)
            db_session.execute(stmt)
            total_rows += len(records)
        
        logs.update({"rows": total_rows, "chunks": total_chunks})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
        
    except Exception as exc:  # pragma: no cover
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
