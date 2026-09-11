"""FinMind API 封裝模組

提供 FinMind 資料存取功能，支援全市場抓取與逐檔抓取兩種模式。

注意：免費/低階會員可能無法使用全市場抓取，需改用逐檔抓取模式。

優化特點：
1. 已驗證資料集按日期抓全市場，其餘按單股日期區間抓取
2. 整合 Rate Limiter 控制每小時 API 請求數
3. 可配置的 chunk_days（建議 180 天減少 API 次數）
"""

from __future__ import annotations

import logging
import math
import threading
import random
import re
import time
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

import pandas as pd
import requests

from app.rate_limiter import get_rate_limiter
from app.file_lock import file_lock
from app.finmind_cache import cache_path, read_cache, write_cache

FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"

# 批次抓取設定（優化後）
BATCH_SIZE = 500  # 每批寫入資料前累積的查詢數
BATCH_DELAY = 0.1  # 批次間最小延遲（秒）
DEFAULT_CHUNK_DAYS = 180  # 預設 chunk 天數（從 30 改為 180）


class FinMindError(RuntimeError):
    pass


def _build_headers(token: str | None) -> Dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


class FinMindQuotaError(FinMindError):
    """Paused work can resume after retry_after_seconds; do not retry immediately."""
    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = max(0, retry_after_seconds)
        super().__init__(f"FinMind quota 暫停，約 {math.ceil(self.retry_after_seconds)} 秒後可重試；已完成資料保留")


_http = threading.local()


def _http_session() -> requests.Session:
    if not hasattr(_http, "session"):
        _http.session = requests.Session()
    return _http.session


def _sleep_backoff(attempt: int, base_seconds: float) -> None:
    backoff = min(30.0, base_seconds * (2 ** attempt))
    time.sleep(backoff + random.uniform(0, backoff))


def _retry_after(value: str | None) -> float:
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return 3600.0
    return max(1.0, seconds) if math.isfinite(seconds) else 3600.0


def fetch_dataset(
    dataset: str,
    start_date: date,
    end_date: Optional[date] = None,
    token: str | None = None,
    data_id: str | None = None,
    rate_limit: bool = True,
    requests_per_hour: int = 6000,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    timeout: int = 60,
    *,
    force_refresh: bool = False,
    cache_ttl: float = 300,
) -> pd.DataFrame:
    """Fetch with pooled HTTP, shared quota and 5-minute duplicate-request reuse.

    Each attempt is charged. Empty/error responses are never cached. An exhausted
    quota fails fast so jobs can pause; a process never sleeps for an hour.
    DataFrame.attrs contains retrieved_at, cache_hit and source provenance.
    """
    if not rate_limit:
        raise ValueError("FinMind quota protection cannot be disabled")
    if max_retries < 0 or timeout <= 0 or cache_ttl < 0 or not math.isfinite(cache_ttl):
        raise ValueError("Invalid FinMind retry/timeout/cache settings")
    if data_id and "," in data_id:
        raise ValueError("FinMind data_id 必須為單一代碼；全市場請按日期查詢")
    if end_date is not None and end_date < start_date:
        raise ValueError("end_date must not precede start_date")
    params: Dict[str, Any] = {"dataset": dataset, "start_date": start_date.isoformat()}
    if end_date is not None:
        params["end_date"] = end_date.isoformat()
    if data_id:
        params["data_id"] = data_id
    # Sponsor snapshot has a dedicated endpoint; reuse the same quota/cache route.
    snapshot = dataset == "TaiwanStockTickSnapshot"
    url = FINMIND_DATA_URL
    if snapshot:
        if cache_ttl > 10:
            cache_ttl = 10
        url = "https://api.finmindtrade.com/api/v4/taiwan_stock_tick_snapshot"
        params = {"data_id": data_id or ""}
    path = cache_path({**params, "_endpoint": url}, token) if snapshot else cache_path(params, token)
    # Identical simultaneous requests use the first worker's result, even across processes.
    with file_lock(path.with_suffix(".lock"), timeout=timeout):
        cached = None if force_refresh else read_cache(path, cache_ttl)
        if cached is not None:
            df = pd.DataFrame(cached["data"])
            df.attrs.update(retrieved_at=cached["retrieved_at"], cache_hit=True, source="finmind")
            return df
        limiter = get_rate_limiter(requests_per_hour)
        for attempt in range(max_retries + 1):
            if not limiter.acquire(timeout=0):
                raise FinMindQuotaError(limiter.get_stats().retry_after_seconds)
            try:
                resp = _http_session().get(
                    url, params=params, headers=_build_headers(token), timeout=timeout,
                )
            except requests.RequestException as exc:
                if attempt < max_retries:
                    _sleep_backoff(attempt, backoff_seconds)
                    continue
                # Never reflect URLs/provider bodies/credentials into the UI or job logs.
                raise FinMindError(f"FinMind network error ({type(exc).__name__})") from None
            if resp.status_code in (402, 429):
                limiter.defer(_retry_after(resp.headers.get("Retry-After")))
                raise FinMindQuotaError(limiter.get_stats().retry_after_seconds)
            if resp.status_code != 200:
                if resp.status_code in {500, 502, 503, 504} and attempt < max_retries:
                    _sleep_backoff(attempt, backoff_seconds)
                    continue
                raise FinMindError(f"FinMind HTTP {resp.status_code}, dataset={dataset}")
            try:
                payload = resp.json()
            except ValueError:
                raise FinMindError("FinMind returned invalid JSON") from None
            if not isinstance(payload, dict):
                raise FinMindError("FinMind returned invalid payload shape")
            status = payload.get("status")
            if str(status) in {"402", "429"}:
                limiter.defer(_retry_after(resp.headers.get("Retry-After")))
                raise FinMindQuotaError(limiter.get_stats().retry_after_seconds)
            if status not in (200, "200", None):
                raise FinMindError(f"FinMind status={status}, dataset={dataset}")
            data = payload.get("data")
            if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
                raise FinMindError("FinMind missing or invalid data field")
            retrieved_at = time.time()
            if data and cache_ttl > 0:
                write_cache(path, data, retrieved_at)
            df = pd.DataFrame(data)
            df.attrs.update(retrieved_at=retrieved_at, cache_hit=False, source="finmind")
            return df
    raise FinMindError("FinMind request failed after retries")


def fetch_dataset_bulk_subchunks(
    dataset: str,
    start_date: date,
    end_date: date,
    chunk_days: int,
    token: str | None = None,
    requests_per_hour: int = 6000,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> tuple[pd.DataFrame, int]:
    """用較小 chunk 嘗試全市場抓取，避免一次回傳過大而空回。

    Returns:
        (DataFrame, api_calls)
    """
    if chunk_days <= 0:
        return pd.DataFrame(), 0

    dfs: list[pd.DataFrame] = []
    api_calls = 0
    for sub_start, sub_end in date_chunks(start_date, end_date, chunk_days=chunk_days):
        df = fetch_dataset(
            dataset,
            sub_start,
            sub_end,
            token=token,
            requests_per_hour=requests_per_hour,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        api_calls += 1
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(), api_calls
    return pd.concat(dfs, ignore_index=True), api_calls


def fetch_stock_list(
    token: str | None = None,
    requests_per_hour: int = 6000,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
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
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
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
    batch_write_callback: Optional[Callable[[pd.DataFrame], int]] = None,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    debug: bool = False,
    timeout: int = 60,
    error_rate_threshold: float = 0.5,
) -> pd.DataFrame:
    """Choose documented market-per-date queries when cheaper, otherwise single-stock ranges.

    Successful pages can be written incrementally. Any missing/error page raises,
    so partial coverage is never reported as a successful complete ingestion.
    """
    if not stock_ids:
        return pd.DataFrame()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    stock_ids = list(dict.fromkeys(stock_ids))
    # Documented Sponsor all-market daily endpoints; never guess comma-separated IDs.
    bulk_datasets = {"TaiwanStockPrice", "TaiwanStockPriceAdj", "TaiwanStockPER", "TaiwanStockMonthRevenue"}
    days = (end_date - start_date).days + 1
    if days <= 0:
        raise ValueError("end_date must not precede start_date")
    dates = [start_date + timedelta(days=i) for i in range(days)]
    if dataset == "TaiwanStockMonthRevenue":
        # FinMind's date is the following month's first day, not announcement time.
        dates = [day for day in dates if day.day == 1]
    bulk = use_batch_query and dataset in bulk_datasets and len(dates) <= len(stock_ids)
    queries = ([(day, day, None) for day in dates] if bulk
               else [(start_date, end_date, sid) for sid in stock_ids])
    logger.info("[finmind] %s plan=%s requests<=%d (before cache/retries)",
                dataset, "market_by_date" if bulk else "stock_by_range", len(queries))
    all_dfs, pending = [], []
    errors = 0
    first_error = None

    def flush():
        if pending:
            frame = pd.concat(pending, ignore_index=True)
            if batch_write_callback:
                batch_write_callback(frame)
            else:
                all_dfs.append(frame)
            pending.clear()

    for i, (start, end, sid) in enumerate(queries, 1):
        try:
            df = fetch_dataset(dataset, start, end, token=token, data_id=sid,
                               requests_per_hour=requests_per_hour, max_retries=max_retries,
                               backoff_seconds=backoff_seconds, timeout=timeout)
            if bulk and not df.empty:
                if "stock_id" not in df:
                    raise FinMindError(f"{dataset}: missing stock_id")
                df = df[df["stock_id"].astype(str).isin(stock_ids)]
            if not df.empty:
                pending.append(df)
        except FinMindQuotaError:
            flush()  # Preserve completed pages before pausing a resumable writer.
            raise
        except FinMindError as exc:
            errors += 1
            first_error = first_error or str(exc)
            logger.warning("[finmind] %s query %d/%d failed: %s", dataset, i, len(queries), exc)
        if i % batch_size == 0 or i == len(queries):
            flush()
        if progress_callback:
            progress_callback(i, len(queries))
        if errors and error_rate_threshold > 0 and (i >= 10 or i == len(queries)):
            if errors / i > error_rate_threshold:
                flush()
                raise FinMindError(f"FinMind {dataset}: {errors}/{i} queries failed; {first_error}")
        if i % batch_size == 0 and i < len(queries) and batch_delay > 0:
            time.sleep(batch_delay)
    # Partial output must not be recorded as a successful complete ingest.
    if errors:
        raise FinMindError(f"FinMind {dataset}: incomplete ({errors} failed queries); {first_error}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


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


def probe_dataset_has_data(
    dataset: str,
    start_date: date,
    end_date: date,
    token: str | None = None,
    probe_stock_ids: Optional[List[str]] = None,
    requests_per_hour: int = 6000,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    timeout: int = 30,
) -> Dict[str, Any]:
    """用少量探針股票先確認區間是否有資料，避免空窗期逐檔呼叫 API。

    回傳:
      {
        "has_data": bool,
        "probe_stock_id": str | None,
        "rows": int
      }
    """
    probe_ids = probe_stock_ids or ["2330", "2317"]
    for sid in probe_ids:
        df = fetch_dataset(
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            token=token,
            data_id=sid,
            requests_per_hour=requests_per_hour,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            timeout=timeout,
        )
        if not df.empty:
            return {"has_data": True, "probe_stock_id": sid, "rows": int(len(df))}
    return {"has_data": False, "probe_stock_id": None, "rows": 0}
