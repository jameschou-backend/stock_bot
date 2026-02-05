"""FinMind API 封裝模組

提供 FinMind 資料存取功能，支援全市場抓取與逐檔抓取兩種模式。

注意：免費/低階會員可能無法使用全市場抓取，需改用逐檔抓取模式。
"""

from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"

# 逐檔抓取時的批次大小與延遲（避免被限流）
BATCH_SIZE = 50  # 每批抓取股票數
BATCH_DELAY = 0.5  # 批次間延遲（秒）


class FinMindError(RuntimeError):
    pass


def _build_headers(token: str | None) -> Dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def fetch_dataset(
    dataset: str,
    start_date: date,
    end_date: date,
    token: str | None = None,
    data_id: str | None = None,
) -> pd.DataFrame:
    """抓取單一 dataset 資料
    
    Args:
        dataset: FinMind dataset 名稱
        start_date: 開始日期
        end_date: 結束日期
        token: FinMind API token
        data_id: 股票代碼（若為 None 則全市場抓取）
    
    Returns:
        DataFrame 包含抓取的資料
    """
    params: Dict[str, Any] = {
        "dataset": dataset,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    if data_id:
        params["data_id"] = data_id

    resp = requests.get(FINMIND_DATA_URL, params=params, headers=_build_headers(token), timeout=30)
    if resp.status_code != 200:
        raise FinMindError(f"FinMind HTTP {resp.status_code}: {resp.text[:200]}")

    payload = resp.json()
    status = payload.get("status")
    if status not in (200, "200", None):
        raise FinMindError(f"FinMind status={status}, msg={payload.get('msg')}")

    data = payload.get("data")
    if data is None:
        raise FinMindError(f"FinMind missing data field: {payload}")
    return pd.DataFrame(data)


def fetch_stock_list(token: str | None = None) -> List[str]:
    """取得台股股票代碼清單（僅四碼數字）
    
    從 TaiwanStockInfo 取得所有股票，過濾出四碼數字代碼。
    
    Returns:
        四碼股票代碼清單
    """
    df = fetch_dataset(
        "TaiwanStockInfo",
        date(2020, 1, 1),
        date(2030, 12, 31),
        token=token,
    )
    if df.empty:
        return []
    
    # 只保留四碼數字股票代碼
    stock_ids = df["stock_id"].unique().tolist()
    four_digit = [s for s in stock_ids if re.fullmatch(r"\d{4}", str(s))]
    return sorted(four_digit)


def fetch_dataset_by_stocks(
    dataset: str,
    start_date: date,
    end_date: date,
    stock_ids: List[str],
    token: str | None = None,
    batch_size: int = BATCH_SIZE,
    batch_delay: float = BATCH_DELAY,
    progress_callback: Optional[callable] = None,
) -> pd.DataFrame:
    """逐檔抓取資料（當全市場抓取不可用時）
    
    將股票清單分批，每批抓取後合併。
    
    Args:
        dataset: FinMind dataset 名稱
        start_date: 開始日期
        end_date: 結束日期
        stock_ids: 要抓取的股票代碼清單
        token: FinMind API token
        batch_size: 每批抓取的股票數
        batch_delay: 批次間延遲（秒）
        progress_callback: 進度回報函數 callback(current, total)
    
    Returns:
        合併後的 DataFrame
    """
    if not stock_ids:
        return pd.DataFrame()
    
    all_dfs = []
    total = len(stock_ids)
    
    for i in range(0, total, batch_size):
        batch = stock_ids[i:i + batch_size]
        
        for stock_id in batch:
            try:
                df = fetch_dataset(dataset, start_date, end_date, token=token, data_id=stock_id)
                if not df.empty:
                    all_dfs.append(df)
            except FinMindError:
                # 單檔失敗不中斷整體流程
                pass
        
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)
        
        # 批次間延遲
        if i + batch_size < total:
            time.sleep(batch_delay)
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def date_chunks(start_date: date, end_date: date, chunk_days: int = 30) -> Iterable[tuple[date, date]]:
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + timedelta(days=chunk_days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)
