"""Bootstrap History 模組

負責檢查 DB 資料狀態，並執行小規模增量回補（bootstrap_days）。

重要：
- 此模組只負責 bootstrap_days 範圍的增量回補（預設 365 天）
- 10 年完整 backfill 需要使用 `make backfill-10y` 顯式觸發
- 若 DB 完全為空，會提示使用者先執行 backfill-10y

行為邏輯：
1. 若 DB 為空 → 提示需要先執行 make backfill-10y，不自動回補
2. 若 DB 有資料但跨度不足 bootstrap_days → 執行增量回補補足
3. 若 DB 資料充足 → 跳過
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import (
    FinMindError,
    date_chunks,
    fetch_dataset,
    fetch_dataset_by_stocks,
    fetch_stock_list,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawInstitutional, RawPrice
from skills.ingest_institutional import _normalize_institutional
from skills.ingest_prices import _normalize_prices


PRICE_DATASET = "TaiwanStockPrice"
INST_DATASET = "TaiwanStockInstitutionalInvestorsBuySell"

# 最低資料品質門檻
MIN_STOCKS_THRESHOLD = 500  # 每日至少要有這麼多股票
BENCHMARK_STOCK_ID = "0050"  # 用於驗證日頻資料


@dataclass(frozen=True)
class HistoryStatus:
    needs_backfill: bool
    reason: str
    reason_category: str  # 'ok', 'empty', 'insufficient', 'wrong_frequency', 'api_issue'
    price_min: date | None
    price_max: date | None
    inst_min: date | None
    inst_max: date | None
    details: Dict[str, object] | None = None


def _span_days(min_date: date | None, max_date: date | None) -> int | None:
    if min_date is None or max_date is None:
        return None
    return (max_date - min_date).days


def _should_backfill(
    price_min: date | None,
    price_max: date | None,
    inst_min: date | None,
    inst_max: date | None,
    required_days: int,
) -> HistoryStatus:
    """檢查是否需要 backfill
    
    Returns:
        HistoryStatus，其中 reason_category 可能為：
        - 'empty': DB 完全為空，需要先執行 make backfill-10y
        - 'insufficient': 資料不足，可執行增量回補
        - 'ok': 資料充足
    """
    price_span = _span_days(price_min, price_max)
    inst_span = _span_days(inst_min, inst_max)
    
    # DB 完全為空 - 需要先執行 backfill-10y
    if price_span is None:
        return HistoryStatus(
            True, "raw_prices empty - run 'make backfill-10y' first", "empty",
            price_min, price_max, inst_min, inst_max
        )
    if inst_span is None:
        return HistoryStatus(
            True, "raw_institutional empty - run 'make backfill-10y' first", "empty",
            price_min, price_max, inst_min, inst_max
        )
    
    # 資料不足 - 可執行增量回補
    if price_span < required_days:
        return HistoryStatus(
            True, f"raw_prices span {price_span}d < {required_days}d", "insufficient",
            price_min, price_max, inst_min, inst_max
        )
    if inst_span < required_days:
        return HistoryStatus(
            True, f"raw_institutional span {inst_span}d < {required_days}d", "insufficient",
            price_min, price_max, inst_min, inst_max
        )
    
    return HistoryStatus(False, "ok", "ok", price_min, price_max, inst_min, inst_max)


def _backfill_range(config) -> Tuple[date, date]:
    today = datetime.now(ZoneInfo(config.tz)).date()
    start_date = today - timedelta(days=config.bootstrap_days)
    return start_date, today


def _diagnose_failure(
    session: Session,
    price_rows: int,
    inst_rows: int,
    start_date: date,
    end_date: date,
    config,
) -> Tuple[str, str]:
    """診斷 backfill 失敗原因並產生明確錯誤訊息
    
    Returns:
        (error_category, error_message)
    """
    # 計算預期交易日數（約略）
    expected_trading_days = ((end_date - start_date).days * 5) // 7
    
    # 檢查是否完全沒有資料（可能是 token/API 問題）
    if price_rows == 0 and inst_rows == 0:
        return (
            "api_issue",
            f"FinMind API 回傳 0 筆資料，可能原因：(1) FINMIND_TOKEN 無效或過期 "
            f"(2) API 限流 (3) 網路問題。請確認 token 並重試。"
        )
    
    if price_rows == 0:
        return (
            "prices_insufficient",
            f"raw_prices 抓取 0 筆，但 raw_institutional 有 {inst_rows} 筆。"
            f"可能 FINMIND_TOKEN 無權存取 TaiwanStockPrice dataset。"
        )
    
    if inst_rows == 0:
        return (
            "inst_insufficient",
            f"raw_institutional 抓取 0 筆，但 raw_prices 有 {price_rows} 筆。"
            f"可能 FINMIND_TOKEN 無權存取 TaiwanStockInstitutionalInvestorsBuySell dataset。"
        )
    
    # 檢查法人資料是否日頻（用 0050 驗證）
    benchmark_count = (
        session.query(func.count())
        .select_from(RawInstitutional)
        .where(RawInstitutional.stock_id == BENCHMARK_STOCK_ID)
        .scalar()
    ) or 0
    
    # 預期 0050 應有約 expected_trading_days 筆
    if benchmark_count < expected_trading_days * 0.5:
        return (
            "wrong_frequency",
            f"法人資料可能非日頻：{BENCHMARK_STOCK_ID} 僅 {benchmark_count} 筆，"
            f"預期約 {expected_trading_days} 筆。請確認 FinMind dataset 為 "
            f"TaiwanStockInstitutionalInvestorsBuySell（日頻），而非月/週頻。"
        )
    
    # 檢查每日股票數量（universe 大小）
    daily_counts = (
        session.query(
            RawPrice.trading_date,
            func.count(func.distinct(RawPrice.stock_id))
        )
        .group_by(RawPrice.trading_date)
        .order_by(RawPrice.trading_date.desc())
        .limit(10)
        .all()
    )
    
    if daily_counts:
        avg_daily = sum(c for _, c in daily_counts) / len(daily_counts)
        if avg_daily < MIN_STOCKS_THRESHOLD:
            return (
                "universe_too_small",
                f"股票 universe 太小：最近 {len(daily_counts)} 交易日平均每日僅 {avg_daily:.0f} 檔，"
                f"低於門檻 {MIN_STOCKS_THRESHOLD}。可能是 token 權限或 API 問題。"
            )
    
    # 一般性資料不足
    return (
        "insufficient",
        f"資料跨度不足：prices={price_rows} rows, institutional={inst_rows} rows，"
        f"期間 {start_date.isoformat()} ~ {end_date.isoformat()}。"
        f"請確認 FINMIND_TOKEN 權限與 API 狀態。"
    )


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "bootstrap_history", commit=True)
    logs: Dict[str, object] = {}
    api_errors: List[str] = []
    
    try:
        price_min = db_session.query(func.min(RawPrice.trading_date)).scalar()
        price_max = db_session.query(func.max(RawPrice.trading_date)).scalar()
        inst_min = db_session.query(func.min(RawInstitutional.trading_date)).scalar()
        inst_max = db_session.query(func.max(RawInstitutional.trading_date)).scalar()
        
        # 記錄 backfill 前狀態
        rows_before_prices = db_session.query(func.count()).select_from(RawPrice).scalar() or 0
        rows_before_inst = db_session.query(func.count()).select_from(RawInstitutional).scalar() or 0

        status = _should_backfill(price_min, price_max, inst_min, inst_max, config.bootstrap_days)
        logs.update(
            {
                "price_min": price_min.isoformat() if price_min else None,
                "price_max": price_max.isoformat() if price_max else None,
                "inst_min": inst_min.isoformat() if inst_min else None,
                "inst_max": inst_max.isoformat() if inst_max else None,
                "rows_before_prices": rows_before_prices,
                "rows_before_inst": rows_before_inst,
                "reason": status.reason,
                "reason_category": status.reason_category,
            }
        )

        if not status.needs_backfill:
            logs["mode"] = "skip"
            finish_job(db_session, job_id, "success", logs=logs)
            return logs

        # DB 完全為空 - 需要先執行 make backfill-10y
        if status.reason_category == "empty":
            error_msg = (
                "資料庫為空，無法進行增量回補。\n"
                "請先執行 'make backfill-10y' 進行完整歷史資料回補。\n"
                f"原因：{status.reason}"
            )
            logs["mode"] = "error"
            logs["error"] = error_msg
            finish_job(db_session, job_id, "failed", error_text=error_msg, logs=logs)
            raise RuntimeError(error_msg)

        start_date, end_date = _backfill_range(config)
        logs.update({
            "mode": "backfill_incremental",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        })

        price_rows = 0
        inst_rows = 0
        chunk_count = 0
        fetch_mode = "bulk"  # 預設全市場抓取
        stock_list: List[str] = []
        
        chunk_ranges = list(date_chunks(start_date, end_date, chunk_days=config.chunk_days))
        total_chunks = len(chunk_ranges)
        logs["chunks_total"] = total_chunks

        for chunk_count, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            logs["progress"] = {
                "current_chunk": chunk_count,
                "total_chunks": total_chunks,
                "chunk_start": chunk_start.isoformat(),
                "chunk_end": chunk_end.isoformat(),
                "prices_rows": price_rows,
                "institutional_rows": inst_rows,
            }
            update_job(db_session, job_id, logs=logs, commit=True)
            
            # 抓取價格資料
            try:
                price_df = fetch_dataset(
                    PRICE_DATASET,
                    chunk_start,
                    chunk_end,
                    token=config.finmind_token,
                    requests_per_hour=config.finmind_requests_per_hour,
                    max_retries=config.finmind_retry_max,
                    backoff_seconds=config.finmind_retry_backoff,
                )
                
                # 如果全市場抓取回傳空值，改用逐檔抓取
                if price_df.empty and fetch_mode == "bulk":
                    fetch_mode = "by_stock"
                    stock_list = fetch_stock_list(
                        config.finmind_token,
                        requests_per_hour=config.finmind_requests_per_hour,
                        max_retries=config.finmind_retry_max,
                        backoff_seconds=config.finmind_retry_backoff,
                    )
                    logs["fetch_mode"] = "by_stock"
                    logs["stock_list_count"] = len(stock_list)
                    
                    if stock_list:
                        price_df = fetch_dataset_by_stocks(
                            PRICE_DATASET,
                            chunk_start,
                            chunk_end,
                            stock_list,
                            token=config.finmind_token,
                            requests_per_hour=config.finmind_requests_per_hour,
                            max_retries=config.finmind_retry_max,
                            backoff_seconds=config.finmind_retry_backoff,
                        )
                elif price_df.empty and fetch_mode == "by_stock" and stock_list:
                    price_df = fetch_dataset_by_stocks(
                        PRICE_DATASET,
                        chunk_start,
                        chunk_end,
                        stock_list,
                        token=config.finmind_token,
                        requests_per_hour=config.finmind_requests_per_hour,
                        max_retries=config.finmind_retry_max,
                        backoff_seconds=config.finmind_retry_backoff,
                    )
                
                if not price_df.empty:
                    price_df = _normalize_prices(price_df)
                    records: List[Dict] = price_df.to_dict("records")
                    if records:
                        stmt = insert(RawPrice).values(records)
                        update_cols = {col: stmt.inserted[col] for col in ["open", "high", "low", "close", "volume"]}
                        stmt = stmt.on_duplicate_key_update(**update_cols)
                        db_session.execute(stmt)
                        price_rows += len(records)
            except FinMindError as e:
                api_errors.append(f"prices chunk {chunk_start}~{chunk_end}: {e}")

            # 抓取法人資料
            try:
                inst_df = fetch_dataset(
                    INST_DATASET,
                    chunk_start,
                    chunk_end,
                    token=config.finmind_token,
                    requests_per_hour=config.finmind_requests_per_hour,
                    max_retries=config.finmind_retry_max,
                    backoff_seconds=config.finmind_retry_backoff,
                )
                
                # 如果全市場抓取回傳空值，改用逐檔抓取
                if inst_df.empty and fetch_mode == "by_stock":
                    if not stock_list:
                        stock_list = fetch_stock_list(
                            config.finmind_token,
                            requests_per_hour=config.finmind_requests_per_hour,
                            max_retries=config.finmind_retry_max,
                            backoff_seconds=config.finmind_retry_backoff,
                        )
                    if stock_list:
                        inst_df = fetch_dataset_by_stocks(
                            INST_DATASET,
                            chunk_start,
                            chunk_end,
                            stock_list,
                            token=config.finmind_token,
                            requests_per_hour=config.finmind_requests_per_hour,
                            max_retries=config.finmind_retry_max,
                            backoff_seconds=config.finmind_retry_backoff,
                        )
                
                if not inst_df.empty:
                    inst_df = _normalize_institutional(inst_df)
                    records = inst_df.to_dict("records")
                    if records:
                        stmt = insert(RawInstitutional).values(records)
                        update_cols = {
                            col: stmt.inserted[col]
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
                            ]
                        }
                        stmt = stmt.on_duplicate_key_update(**update_cols)
                        db_session.execute(stmt)
                        inst_rows += len(records)
            except FinMindError as e:
                api_errors.append(f"institutional chunk {chunk_start}~{chunk_end}: {e}")

        # 記錄 backfill 後狀態
        price_min_after = db_session.query(func.min(RawPrice.trading_date)).scalar()
        price_max_after = db_session.query(func.max(RawPrice.trading_date)).scalar()
        inst_min_after = db_session.query(func.min(RawInstitutional.trading_date)).scalar()
        inst_max_after = db_session.query(func.max(RawInstitutional.trading_date)).scalar()
        rows_after_prices = db_session.query(func.count()).select_from(RawPrice).scalar() or 0
        rows_after_inst = db_session.query(func.count()).select_from(RawInstitutional).scalar() or 0

        status_after = _should_backfill(
            price_min_after, price_max_after, inst_min_after, inst_max_after, config.bootstrap_days
        )

        logs.update(
            {
                "chunks_processed": chunk_count,
                "rows_prices": price_rows,
                "rows_institutional": inst_rows,
                "rows_after_prices": rows_after_prices,
                "rows_after_inst": rows_after_inst,
                "price_min_after": price_min_after.isoformat() if price_min_after else None,
                "price_max_after": price_max_after.isoformat() if price_max_after else None,
                "inst_min_after": inst_min_after.isoformat() if inst_min_after else None,
                "inst_max_after": inst_max_after.isoformat() if inst_max_after else None,
            }
        )
        
        if api_errors:
            logs["api_errors"] = api_errors[:10]  # 限制錯誤數量

        if status_after.needs_backfill:
            # 診斷失敗原因
            error_category, error_message = _diagnose_failure(
                db_session, price_rows, inst_rows, start_date, end_date, config
            )
            logs["reason_after"] = status_after.reason
            logs["error_category"] = error_category
            logs["error_diagnosis"] = error_message
            
            finish_job(db_session, job_id, "failed", error_text=error_message, logs=logs)
            raise ValueError(error_message)

        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc), **logs})
        raise
