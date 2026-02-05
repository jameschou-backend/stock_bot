"""FinMind API 封裝模組

提供 FinMind 資料存取功能，支援全市場抓取與逐檔抓取兩種模式。

注意：免費/低階會員可能無法使用全市場抓取，需改用逐檔抓取模式。

優化特點：
1. 支援真正的批次查詢（一次傳多個 stock_id）
2. 整合 Rate Limiter 控制每小時 API 請求數
3. 可配置的 chunk_days（建議 180 天減少 API 次數）
"""

from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import requests

from app.rate_limiter import get_rate_limiter

FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"

# 批次抓取設定（優化後）
BATCH_SIZE = 100  # 每批抓取股票數（FinMind 支援多個 data_id）
BATCH_DELAY = 0.5  # 批次間最小延遲（秒）
DEFAULT_CHUNK_DAYS = 180  # 預設 chunk 天數（從 30 改為 180）


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
    rate_limit: bool = True,
    requests_per_hour: int = 6000,
) -> pd.DataFrame:
    """抓取單一 dataset 資料
    
    Args:
        dataset: FinMind dataset 名稱
        start_date: 開始日期
        end_date: 結束日期
        token: FinMind API token
        data_id: 股票代碼（若為 None 則全市場抓取）
        rate_limit: 是否啟用速率限制
        requests_per_hour: 每小時最大請求數
    
    Returns:
        DataFrame 包含抓取的資料
    """
    # Rate limiting
    if rate_limit:
        limiter = get_rate_limiter(requests_per_hour)
        limiter.acquire()
    
    params: Dict[str, Any] = {
        "dataset": dataset,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    if data_id:
        params["data_id"] = data_id

    resp = requests.get(FINMIND_DATA_URL, params=params, headers=_build_headers(token), timeout=60)
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


def fetch_stock_list(
    token: str | None = None,
    requests_per_hour: int = 6000,
) -> List[str]:
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
        requests_per_hour=requests_per_hour,
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
    progress_callback: Optional[Callable[[int, int], None]] = None,
    requests_per_hour: int = 6000,
    use_batch_query: bool = True,
) -> pd.DataFrame:
    """批次抓取資料（當全市場抓取不可用時）
    
    將股票清單分批，每批抓取後合併。
    
    優化：支援真正的批次查詢（一次 API call 抓多檔股票），
    大幅減少 API 次數。
    
    Args:
        dataset: FinMind dataset 名稱
        start_date: 開始日期
        end_date: 結束日期
        stock_ids: 要抓取的股票代碼清單
        token: FinMind API token
        batch_size: 每批抓取的股票數
        batch_delay: 批次間延遲（秒）
        progress_callback: 進度回報函數 callback(current, total)
        requests_per_hour: 每小時最大請求數
        use_batch_query: 是否使用批次查詢（一次傳多個 data_id）
    
    Returns:
        合併後的 DataFrame
    """
    if not stock_ids:
        return pd.DataFrame()
    
    all_dfs = []
    total = len(stock_ids)
    api_calls = 0
    
    for i in range(0, total, batch_size):
        batch = stock_ids[i:i + batch_size]
        
        if use_batch_query:
            # 優化：一次 API call 抓取整批股票
            # FinMind 支援 data_id 用逗號分隔多個股票代碼
            batch_data_id = ",".join(batch)
            try:
                df = fetch_dataset(
                    dataset,
                    start_date,
                    end_date,
                    token=token,
                    data_id=batch_data_id,
                    requests_per_hour=requests_per_hour,
                )
                api_calls += 1
                if not df.empty:
                    all_dfs.append(df)
            except FinMindError:
                # 批次失敗時，降級為逐檔抓取
                for stock_id in batch:
                    try:
                        df = fetch_dataset(
                            dataset,
                            start_date,
                            end_date,
                            token=token,
                            data_id=stock_id,
                            requests_per_hour=requests_per_hour,
                        )
                        api_calls += 1
                        if not df.empty:
                            all_dfs.append(df)
                    except FinMindError:
                        # 單檔失敗不中斷整體流程
                        pass
        else:
            # 傳統模式：逐檔抓取
            for stock_id in batch:
                try:
                    df = fetch_dataset(
                        dataset,
                        start_date,
                        end_date,
                        token=token,
                        data_id=stock_id,
                        requests_per_hour=requests_per_hour,
                    )
                    api_calls += 1
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


def date_chunks(
    start_date: date,
    end_date: date,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
) -> Iterable[tuple[date, date]]:
    """將日期範圍切分成多個 chunk
    
    Args:
        start_date: 開始日期
        end_date: 結束日期
        chunk_days: 每個 chunk 的天數（預設 180 天）
    
    Yields:
        (chunk_start, chunk_end) 日期對
    """
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + timedelta(days=chunk_days - 1))
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)
