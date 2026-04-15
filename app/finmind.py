"""FinMind API 封裝模組

提供 FinMind 資料存取功能，支援全市場抓取與逐檔抓取兩種模式。

注意：免費/低階會員可能無法使用全市場抓取，需改用逐檔抓取模式。

優化特點：
1. 支援真正的批次查詢（一次傳多個 stock_id）
2. 整合 Rate Limiter 控制每小時 API 請求數
3. 可配置的 chunk_days（建議 180 天減少 API 次數）
"""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

import pandas as pd
import requests

from app.rate_limiter import get_rate_limiter

FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"

# 批次抓取設定（優化後）
BATCH_SIZE = 500  # 每批抓取股票數（FinMind 支援多個 data_id）
BATCH_DELAY = 0.1  # 批次間最小延遲（秒）
DEFAULT_CHUNK_DAYS = 180  # 預設 chunk 天數（從 30 改為 180）


class FinMindError(RuntimeError):
    pass


def _build_headers(token: str | None) -> Dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _is_retryable_payload(status: object, msg: str) -> bool:
    if status in (429, "429", 402, "402"):
        return True
    msg_lower = (msg or "").lower()
    return any(term in msg_lower for term in ["rate", "limit", "頻率", "超過", "exceed"])


def _sleep_backoff(attempt: int, base_seconds: float, retry_after: float | None = None) -> None:
    backoff = base_seconds * (2 ** attempt)
    jitter = random.uniform(0, backoff)
    wait_time = max(retry_after or 0.0, backoff + jitter)
    time.sleep(wait_time)


def fetch_dataset(
    dataset: str,
    start_date: date,
    end_date: date,
    token: str | None = None,
    data_id: str | None = None,
    rate_limit: bool = True,
    requests_per_hour: int = 6000,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    timeout: int = 60,
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
        timeout: HTTP 請求超時秒數（預設 60，長區間建議 120）
    
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

    retryable_http = {429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        # Rate limiting
        if rate_limit:
            limiter = get_rate_limiter(requests_per_hour)
            limiter.acquire()

        try:
            resp = requests.get(
                FINMIND_DATA_URL,
                params=params,
                headers=_build_headers(token),
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt < max_retries:
                _sleep_backoff(attempt, backoff_seconds)
                continue
            raise FinMindError(f"FinMind request failed: {exc}") from exc

        if resp.status_code != 200:
            if resp.status_code in retryable_http and attempt < max_retries:
                retry_after = resp.headers.get("Retry-After")
                retry_after_sec = float(retry_after) if retry_after and retry_after.isdigit() else None
                _sleep_backoff(attempt, backoff_seconds, retry_after_sec)
                continue
            raise FinMindError(f"FinMind HTTP {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        status = payload.get("status")
        if status not in (200, "200", None):
            msg = payload.get("msg") or ""
            if _is_retryable_payload(status, msg) and attempt < max_retries:
                _sleep_backoff(attempt, backoff_seconds)
                continue
            raise FinMindError(f"FinMind status={status}, msg={msg}")

        data = payload.get("data")
        if data is None:
            raise FinMindError(f"FinMind missing data field: {payload}")
        return pd.DataFrame(data)

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
        batch_write_callback: 每批寫入回調函數 callback(df) -> rows_written
                              如果提供，每批抓完後立即寫入 DB，中斷也不會丟失已寫資料
        error_rate_threshold: 錯誤率閾值（0.0~1.0），超過此比例則拋出 FinMindError
                              預設 0.5（50% 的 batch 失敗即視為配額耗盡，不繼續）

    Returns:
        合併後的 DataFrame（如果有 batch_write_callback 則回傳空 DataFrame）

    Raises:
        FinMindError: 配額錯誤（402/429）或錯誤率超過 error_rate_threshold
    """
    if not stock_ids:
        return pd.DataFrame()
    
    all_dfs = []
    total = len(stock_ids)
    api_calls = 0
    total_written = 0
    error_count = 0
    quota_error_count = 0  # 402/429 配額錯誤（所有後續 call 都會失敗，應立即停止）
    empty_count = 0
    first_error: Optional[str] = None
    _first_data_logged = False

    def _is_quota_error(exc: FinMindError) -> bool:
        """判斷是否為配額耗盡錯誤（402/429）"""
        msg = str(exc)
        return "402" in msg or "429" in msg or "quota" in msg.lower() or "超過" in msg

    total_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = stock_ids[i:i + batch_size]
        batch_dfs = []

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
                    max_retries=max_retries,
                    backoff_seconds=backoff_seconds,
                    timeout=timeout,
                )
                api_calls += 1
                if not df.empty:
                    batch_dfs.append(df)
            except FinMindError as exc:
                error_count += 1
                if first_error is None:
                    first_error = f"batch data_id={batch_data_id}: {exc}"
                if _is_quota_error(exc):
                    quota_error_count += 1
                    logger.error(
                        "[finmind] 配額錯誤（%s），已完成 %d/%d batches，停止抓取 %s",
                        exc, i // batch_size + 1, total_batches, dataset,
                    )
                    raise FinMindError(
                        f"FinMind 配額耗盡 ({exc})，dataset={dataset} 已中止，"
                        f"請稍後重試或升級方案"
                    ) from exc
                # 非配額錯誤：降級為逐檔抓取
                for stock_id in batch:
                    try:
                        df = fetch_dataset(
                            dataset,
                            start_date,
                            end_date,
                            token=token,
                            data_id=stock_id,
                            requests_per_hour=requests_per_hour,
                            max_retries=max_retries,
                            backoff_seconds=backoff_seconds,
                            timeout=timeout,
                        )
                        api_calls += 1
                        if not df.empty:
                            batch_dfs.append(df)
                    except FinMindError as exc2:
                        error_count += 1
                        if first_error is None:
                            first_error = f"stock_id={stock_id}: {exc2}"
                        if _is_quota_error(exc2):
                            quota_error_count += 1
                            logger.error(
                                "[finmind] 配額錯誤（%s），停止抓取 %s", exc2, dataset
                            )
                            raise FinMindError(
                                f"FinMind 配額耗盡 ({exc2})，dataset={dataset} 已中止"
                            ) from exc2
        else:
            # 傳統模式：逐檔抓取
            for j, stock_id in enumerate(batch):
                try:
                    df = fetch_dataset(
                        dataset,
                        start_date,
                        end_date,
                        token=token,
                        data_id=stock_id,
                        requests_per_hour=requests_per_hour,
                        max_retries=max_retries,
                        backoff_seconds=backoff_seconds,
                        timeout=timeout,
                    )
                    api_calls += 1
                    if not df.empty:
                        batch_dfs.append(df)
                        if debug and not _first_data_logged:
                            _first_data_logged = True
                            print(f"\n  [debug] 首筆回傳: stock_id={stock_id}, rows={len(df)}, cols={list(df.columns[:6])}", flush=True)
                    else:
                        empty_count += 1
                except FinMindError as exc:
                    error_count += 1
                    if first_error is None:
                        first_error = f"stock_id={stock_id}: {exc}"
                    if _is_quota_error(exc):
                        quota_error_count += 1
                        logger.error(
                            "[finmind] 配額錯誤（%s），停止抓取 %s", exc, dataset
                        )
                        raise FinMindError(
                            f"FinMind 配額耗盡 ({exc})，dataset={dataset} 已中止"
                        ) from exc

                # 每檔都更新進度（逐檔模式下即時顯示）
                if progress_callback:
                    progress_callback(i + j + 1, total)

        # 處理這批資料
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            if batch_write_callback:
                # 立即寫入這批資料
                rows = batch_write_callback(batch_df)
                total_written += rows
            else:
                # 累積到最後合併
                all_dfs.append(batch_df)

        # batch_query 模式在 batch 結束時更新進度
        if use_batch_query and progress_callback:
            progress_callback(min(i + batch_size, total), total)

        # 批次間延遲
        if i + batch_size < total:
            time.sleep(batch_delay)

        # 錯誤率檢查：每 10 批次或處理完畢時檢查
        batches_done = i // batch_size + 1
        if error_rate_threshold > 0 and batches_done % 10 == 0:
            err_rate = error_count / max(api_calls, 1)
            if err_rate > error_rate_threshold:
                logger.error(
                    "[finmind] %s 錯誤率過高 %.1f%% (%d/%d calls)，first_error=%s，停止抓取",
                    dataset, err_rate * 100, error_count, api_calls, first_error,
                )
                raise FinMindError(
                    f"FinMind {dataset} 錯誤率 {err_rate:.0%} 超過閾值 {error_rate_threshold:.0%}，"
                    f"共 {error_count} 次失敗（{api_calls} calls），first_error={first_error}"
                )

    # 最終錯誤率檢查
    if error_count > 0:
        err_rate = error_count / max(api_calls, 1)
        log_fn = logger.warning if err_rate < error_rate_threshold else logger.error
        log_fn(
            "[finmind] %s 完成：api_calls=%d, errors=%d (%.1f%%), empty=%d, written=%d, first_error=%s",
            dataset, api_calls, error_count, err_rate * 100, empty_count, total_written, first_error,
        )
        if error_rate_threshold > 0 and err_rate > error_rate_threshold:
            raise FinMindError(
                f"FinMind {dataset} 最終錯誤率 {err_rate:.0%} 超過閾值 {error_rate_threshold:.0%}，"
                f"共 {error_count} 次失敗（{api_calls} calls），first_error={first_error}"
            )

    # 如果有 batch_write_callback，資料已經寫入，回傳空 DataFrame
    if batch_write_callback:
        if debug and (error_count or empty_count):
            print(f"\n[finmind] {dataset} api_calls={api_calls} empty={empty_count} errors={error_count}", flush=True)
        return pd.DataFrame()

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
