"""Ingest Stock Master 模組

從 FinMind 抓取股票基本資料與下市櫃資訊，寫入 stocks 與 stock_status_history 表。

FinMind Datasets:
- TaiwanStockInfo: 台股總覽（股票代碼、名稱、產業類別、市場別）
- TaiwanStockDelisting: 台灣股票下市櫃表
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Dict, List, Set
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import FinMindError, fetch_dataset
from app.job_utils import finish_job, start_job
from app.models import Stock, StockStatusHistory


STOCK_INFO_DATASET = "TaiwanStockInfo"
DELISTING_DATASET = "TaiwanStockDelisting"

# 根據 FinMind type 欄位判斷 security_type
# type 可能的值：twse(上市), tpex(上櫃), etc.
# stock_id 格式判斷：
#   - 4碼數字：一般股票
#   - 00xx, 006xxx：ETF
#   - 其他：權證、特別股等
ETF_PATTERNS = [
    r"^00\d{2}$",      # 0050, 0056 等
    r"^006\d{3}$",     # 006XXX 系列
    r"^00\d{3}[A-Z]?$",  # 00XXX 或 00XXXA
]

WARRANT_PATTERNS = [
    r"^\d{5,6}$",  # 5-6 碼數字（權證）
]


def _detect_security_type(stock_id: str, stock_name: str) -> str:
    """判斷證券類型
    
    Args:
        stock_id: 股票代碼
        stock_name: 股票名稱
    
    Returns:
        security_type: stock/etf/warrant/other
    """
    # ETF 判斷
    for pattern in ETF_PATTERNS:
        if re.match(pattern, stock_id):
            return "etf"
    
    # 名稱包含 ETF 相關字眼
    if stock_name and any(kw in stock_name for kw in ["ETF", "元大", "富邦", "國泰", "群益"]):
        # 需進一步確認是 ETF 還是一般股票
        if re.match(r"^00\d+", stock_id):
            return "etf"
    
    # 權證判斷
    for pattern in WARRANT_PATTERNS:
        if re.match(pattern, stock_id):
            return "warrant"
    
    # 4 碼數字視為一般股票
    if re.fullmatch(r"\d{4}", stock_id):
        return "stock"
    
    return "other"


def _normalize_market(type_value: str) -> str:
    """標準化市場代碼
    
    Args:
        type_value: FinMind 的 type 欄位值
    
    Returns:
        標準化後的市場代碼
    """
    if not type_value:
        return "unknown"
    
    type_lower = type_value.lower().strip()
    
    mapping = {
        "twse": "TWSE",
        "tpex": "TPEX",
        "otc": "TPEX",
        "emerging": "EMERGING",
        "rotc": "ROTC",
    }
    
    return mapping.get(type_lower, type_value.upper())


def _fetch_stock_info(
    token: str | None,
    requests_per_hour: int,
    max_retries: int,
    backoff_seconds: float,
) -> pd.DataFrame:
    """從 TaiwanStockInfo 抓取股票基本資料"""
    df = fetch_dataset(
        STOCK_INFO_DATASET,
        date(2020, 1, 1),
        date(2030, 12, 31),
        token=token,
        requests_per_hour=requests_per_hour,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    return df


def _fetch_delisting_info(
    token: str | None,
    requests_per_hour: int,
    max_retries: int,
    backoff_seconds: float,
) -> pd.DataFrame:
    """從 TaiwanStockDelisting 抓取下市櫃資料"""
    try:
        df = fetch_dataset(
            DELISTING_DATASET,
            date(1990, 1, 1),
            date(2030, 12, 31),
            token=token,
            requests_per_hour=requests_per_hour,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        return df
    except FinMindError:
        # 下市櫃表可能因權限問題無法取得
        return pd.DataFrame()


def _get_existing_stocks(session: Session) -> Dict[str, Stock]:
    """取得現有的 stocks 資料"""
    stocks = session.query(Stock).all()
    return {s.stock_id: s for s in stocks}


def run(config, db_session: Session, **kwargs) -> Dict:
    """執行股票主檔更新
    
    Returns:
        包含更新統計的 dict
    """
    job_id = start_job(db_session, "ingest_stock_master")
    logs: Dict[str, object] = {}
    
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        
        # 抓取股票基本資料
        info_df = _fetch_stock_info(
            config.finmind_token,
            config.finmind_requests_per_hour,
            config.finmind_retry_max,
            config.finmind_retry_backoff,
        )
        logs["stock_info_rows"] = len(info_df)
        
        if info_df.empty:
            logs["warning"] = "TaiwanStockInfo 回傳空資料"
            finish_job(db_session, job_id, "success", logs=logs)
            return logs
        
        # 抓取下市櫃資料
        delisting_df = _fetch_delisting_info(
            config.finmind_token,
            config.finmind_requests_per_hour,
            config.finmind_retry_max,
            config.finmind_retry_backoff,
        )
        logs["delisting_rows"] = len(delisting_df)
        
        # 建立下市股票集合
        delisted_stocks: Set[str] = set()
        delisting_dates: Dict[str, date] = {}
        if not delisting_df.empty:
            if "stock_id" in delisting_df.columns:
                delisted_stocks = set(delisting_df["stock_id"].astype(str).unique())
            if "date" in delisting_df.columns:
                for _, row in delisting_df.iterrows():
                    sid = str(row.get("stock_id", ""))
                    d = row.get("date")
                    if sid and d:
                        try:
                            delisting_dates[sid] = pd.to_datetime(d).date()
                        except Exception:
                            pass
        
        # 取得現有 stocks
        existing = _get_existing_stocks(db_session)
        
        # 處理股票資料
        records: List[Dict] = []
        status_records: List[Dict] = []
        
        upserted_count = 0
        new_count = 0
        delisted_count = 0
        
        for _, row in info_df.iterrows():
            stock_id = str(row.get("stock_id", "")).strip()
            if not stock_id:
                continue
            
            stock_name = str(row.get("stock_name", "") or "").strip()
            industry = str(row.get("industry_category", "") or "").strip()
            market_type = str(row.get("type", "") or "").strip()
            
            market = _normalize_market(market_type)
            security_type = _detect_security_type(stock_id, stock_name)
            is_listed = stock_id not in delisted_stocks
            delisted_date = delisting_dates.get(stock_id)
            
            record = {
                "stock_id": stock_id,
                "name": stock_name,
                "market": market,
                "is_listed": is_listed,
                "industry_category": industry,
                "security_type": security_type,
                "delisted_date": delisted_date,
            }
            records.append(record)
            
            # 檢查是否有變更需要記錄到 history
            old = existing.get(stock_id)
            if old is None:
                new_count += 1
                status_records.append({
                    "stock_id": stock_id,
                    "effective_date": today,
                    "status_type": "listed",
                    "payload_json": {"name": stock_name, "market": market},
                })
            elif old.is_listed and not is_listed:
                delisted_count += 1
                status_records.append({
                    "stock_id": stock_id,
                    "effective_date": delisted_date or today,
                    "status_type": "delisted",
                    "payload_json": {"delisted_date": delisted_date.isoformat() if delisted_date else None},
                })
            
            upserted_count += 1
        
        # 批次 upsert stocks
        if records:
            stmt = insert(Stock).values(records)
            update_cols = {
                col: stmt.inserted[col]
                for col in ["name", "market", "is_listed", "industry_category", "security_type", "delisted_date"]
            }
            stmt = stmt.on_duplicate_key_update(**update_cols)
            db_session.execute(stmt)
        
        # 記錄狀態變更
        if status_records:
            for sr in status_records:
                hist = StockStatusHistory(
                    stock_id=sr["stock_id"],
                    effective_date=sr["effective_date"],
                    status_type=sr["status_type"],
                    payload_json=sr["payload_json"],
                )
                db_session.add(hist)
        
        logs.update({
            "upserted_count": upserted_count,
            "new_count": new_count,
            "delisted_count": delisted_count,
            "status_history_records": len(status_records),
            "source_datasets": [STOCK_INFO_DATASET, DELISTING_DATASET],
        })
        
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
        
    except Exception as exc:  # pragma: no cover
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
