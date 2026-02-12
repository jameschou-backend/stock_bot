#!/usr/bin/env python3
"""歷史資料回補腳本

用於抓取過去 N 年的歷史資料，支援：
1. 中斷續傳（從上次進度繼續）
2. 進度顯示與預估時間
3. Rate Limiting（避免觸發 API 限流）
4. 批次查詢優化（大幅減少 API 次數）
5. 支援選擇性回補（prices/institutional/margin）

使用方式：
    # 抓取過去 10 年資料（所有 datasets）
    python scripts/backfill_history.py --years 10
    
    # 只回補特定 dataset
    python scripts/backfill_history.py --years 10 --datasets prices,institutional
    python scripts/backfill_history.py --years 10 --datasets margin
    
    # 抓取特定日期範圍
    python scripts/backfill_history.py --start 2016-01-01 --end 2026-02-05
    
    # 顯示目前進度和預估
    python scripts/backfill_history.py --status
    
    # 重置進度（重新抓取）
    python scripts/backfill_history.py --reset --years 10

API 用量估算（付費會員 6000 次/小時）：
- 全市場模式：10 年資料約需 60-120 次 API calls（3 datasets）
- 逐檔批次模式：10 年資料約需 1000-2000 次 API calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

# 加入 project root 到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert

from app.config import load_config
from app.db import get_session
from app.finmind import (
    DEFAULT_CHUNK_DAYS,
    FinMindError,
    date_chunks,
    fetch_dataset,
    fetch_dataset_by_stocks,
    fetch_dataset_bulk_subchunks,
    fetch_stock_list,
)
from app.models import RawInstitutional, RawMarginShort, RawPrice
from app.rate_limiter import get_rate_limiter
from skills.ingest_institutional import _normalize_institutional
from skills.ingest_margin_short import _normalize_margin_short
from skills.ingest_prices import _normalize_prices


# 進度檔案位置
PROGRESS_FILE = Path(__file__).resolve().parent.parent / "storage" / "backfill_progress.json"
# 融資融券逐檔進度檔（記錄已完成的 stock_id，支援中斷續傳）
MARGIN_DONE_FILE = Path(__file__).resolve().parent.parent / "storage" / "backfill_margin_done.json"
PRICES_DONE_FILE = Path(__file__).resolve().parent.parent / "storage" / "backfill_prices_done.json"
INST_DONE_FILE = Path(__file__).resolve().parent.parent / "storage" / "backfill_inst_done.json"

# Dataset 名稱
PRICE_DATASET = "TaiwanStockPrice"
INST_DATASET = "TaiwanStockInstitutionalInvestorsBuySell"
MARGIN_DATASET = "TaiwanStockMarginPurchaseShortSale"

# 支援的 dataset 類型
SUPPORTED_DATASETS = {"prices", "institutional", "margin"}


def get_listed_stocks_from_db() -> List[str]:
    """從 stocks 表取得目前上市櫃股票清單（排除下市、ETF、權證）"""
    from app.models import Stock
    
    with get_session() as session:
        stocks = (
            session.query(Stock.stock_id)
            .filter(Stock.is_listed == True)
            .filter(Stock.security_type == "stock")
            .all()
        )
        return sorted([s[0] for s in stocks])


def _bulk_insufficient(df: pd.DataFrame, start_date: date, end_date: date) -> bool:
    """判斷全市場回補是否只回單日資料（對多日區間來說不足）"""
    if df.empty:
        return True
    if start_date == end_date:
        return False
    date_col = None
    for col in ("date", "trading_date"):
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        return False
    unique_dates = pd.to_datetime(df[date_col], errors="coerce").dt.date.nunique()
    return unique_dates <= 1


@dataclass
class BackfillProgress:
    """回補進度追蹤"""
    start_date: date
    end_date: date
    current_date: date  # 目前處理到的日期
    total_chunks: int
    completed_chunks: int
    prices_rows: int
    inst_rows: int
    margin_rows: int
    api_calls: int
    fetch_mode: str  # "bulk" or "by_stock"
    datasets: List[str]  # 要回補的 datasets
    started_at: str
    last_updated: str
    
    def to_dict(self) -> Dict:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "current_date": self.current_date.isoformat(),
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "prices_rows": self.prices_rows,
            "inst_rows": self.inst_rows,
            "margin_rows": self.margin_rows,
            "api_calls": self.api_calls,
            "fetch_mode": self.fetch_mode,
            "datasets": self.datasets,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "BackfillProgress":
        return cls(
            start_date=date.fromisoformat(d["start_date"]),
            end_date=date.fromisoformat(d["end_date"]),
            current_date=date.fromisoformat(d["current_date"]),
            total_chunks=d["total_chunks"],
            completed_chunks=d["completed_chunks"],
            prices_rows=d.get("prices_rows", 0),
            inst_rows=d.get("inst_rows", 0),
            margin_rows=d.get("margin_rows", 0),
            api_calls=d["api_calls"],
            fetch_mode=d["fetch_mode"],
            datasets=d.get("datasets", ["prices", "institutional"]),
            started_at=d["started_at"],
            last_updated=d["last_updated"],
        )


def load_progress() -> Optional[BackfillProgress]:
    """載入進度檔案"""
    if not PROGRESS_FILE.exists():
        return None
    try:
        with open(PROGRESS_FILE, "r") as f:
            return BackfillProgress.from_dict(json.load(f))
    except Exception:
        return None


def save_progress(progress: BackfillProgress) -> None:
    """儲存進度檔案"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress.last_updated = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress.to_dict(), f, indent=2)


def delete_progress() -> None:
    """刪除進度檔案"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def load_margin_done() -> Set[str]:
    """載入已完成的融資融券股票清單"""
    if not MARGIN_DONE_FILE.exists():
        return set()
    try:
        with open(MARGIN_DONE_FILE, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_margin_done(done: Set[str]) -> None:
    """儲存已完成的融資融券股票清單"""
    MARGIN_DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MARGIN_DONE_FILE, "w") as f:
        json.dump(sorted(done), f)


def delete_margin_done() -> None:
    """刪除融資融券逐檔進度檔"""
    if MARGIN_DONE_FILE.exists():
        MARGIN_DONE_FILE.unlink()


def load_prices_done() -> Set[str]:
    """載入已完成的價格回補股票清單"""
    if not PRICES_DONE_FILE.exists():
        return set()
    try:
        with open(PRICES_DONE_FILE, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_prices_done(done: Set[str]) -> None:
    """儲存已完成的價格回補股票清單"""
    PRICES_DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRICES_DONE_FILE, "w") as f:
        json.dump(sorted(done), f)


def delete_prices_done() -> None:
    """刪除價格逐檔進度檔"""
    if PRICES_DONE_FILE.exists():
        PRICES_DONE_FILE.unlink()


def load_inst_done() -> Set[str]:
    """載入已完成的法人回補股票清單"""
    if not INST_DONE_FILE.exists():
        return set()
    try:
        with open(INST_DONE_FILE, "r") as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_inst_done(done: Set[str]) -> None:
    """儲存已完成的法人回補股票清單"""
    INST_DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INST_DONE_FILE, "w") as f:
        json.dump(sorted(done), f)


def delete_inst_done() -> None:
    """刪除法人逐檔進度檔"""
    if INST_DONE_FILE.exists():
        INST_DONE_FILE.unlink()


def estimate_api_calls(
    start_date: date,
    end_date: date,
    chunk_days: int,
    fetch_mode: str,
    datasets: List[str],
    stock_count: int = 1800,
    batch_size: int = 100,
) -> Tuple[int, int]:
    """估算 API 次數和所需時間
    
    Returns:
        (estimated_calls, estimated_minutes)
    """
    total_days = (end_date - start_date).days
    chunks = (total_days + chunk_days - 1) // chunk_days
    num_datasets = len(datasets)
    
    if fetch_mode == "bulk":
        # 全市場模式：每個 chunk 抓 N 次（每個 dataset 一次）+ 1 次 stock list
        calls = chunks * num_datasets + 1
    else:
        # 逐檔批次模式：每個 chunk 抓 (stock_count / batch_size) * N 次
        batches_per_chunk = (stock_count + batch_size - 1) // batch_size
        calls = chunks * batches_per_chunk * num_datasets + 1
    
    # 以每小時 5400 次（90% buffer）計算所需時間
    minutes = calls / 90  # 5400 / 60 = 90 calls per minute
    
    return calls, int(minutes) + 1


def print_status(progress: Optional[BackfillProgress], config) -> None:
    """印出目前狀態"""
    print("\n" + "=" * 60)
    print("歷史資料回補狀態")
    print("=" * 60)
    
    if progress is None:
        print("沒有進行中的回補任務")
        return
    
    pct = (progress.completed_chunks / progress.total_chunks * 100) if progress.total_chunks > 0 else 0
    
    # 估算剩餘時間
    est_calls, est_minutes = estimate_api_calls(
        progress.current_date,
        progress.end_date,
        config.chunk_days,
        progress.fetch_mode,
        progress.datasets,
    )
    
    print(f"日期範圍: {progress.start_date} ~ {progress.end_date}")
    print(f"目前進度: {progress.current_date}")
    print(f"Chunks:   {progress.completed_chunks}/{progress.total_chunks} ({pct:.1f}%)")
    print(f"抓取模式: {progress.fetch_mode}")
    print(f"Datasets: {', '.join(progress.datasets)}")
    print(f"已抓取:   prices={progress.prices_rows:,}, institutional={progress.inst_rows:,}, margin={progress.margin_rows:,}")
    print(f"API 呼叫: {progress.api_calls:,} 次")
    print(f"預估剩餘: ~{est_calls:,} 次 API calls, ~{est_minutes} 分鐘")
    print(f"開始時間: {progress.started_at}")
    print(f"更新時間: {progress.last_updated}")
    
    # Rate limiter 狀態
    limiter = get_rate_limiter(config.finmind_requests_per_hour)
    remaining = limiter.remaining_requests()
    print(f"Rate Limit: {remaining}/{limiter.effective_limit} 剩餘（本小時）")
    print("=" * 60)


INST_UPDATE_COLS = [
    "foreign_buy", "foreign_sell", "foreign_net",
    "trust_buy", "trust_sell", "trust_net",
    "dealer_buy", "dealer_sell", "dealer_net",
]


def _backfill_institutional(
    start_date: date,
    end_date: date,
    config,
    stock_list: List[str],
    db_session,
    progress: BackfillProgress,
    allowed_stock_ids: Optional[Set[str]] = None,
) -> None:
    """逐檔抓取法人資料，每檔一次 API call 抓完整區間，每檔立即 upsert"""
    inst_done = load_inst_done()
    remaining = [s for s in stock_list if s not in inst_done]
    total = len(stock_list)
    done_count = total - len(remaining)

    print(f"\n{'='*50}")
    print(f"抓取法人資料 (逐檔模式)")
    print(f"期間: {start_date} ~ {end_date}")
    print(f"股票: {total} 檔, 已完成: {done_count}, 待抓取: {len(remaining)}")
    print(f"{'='*50}")

    if not remaining:
        print("所有股票已完成，跳過")
        return

    for idx, stock_id in enumerate(remaining, start=1):
        current = done_count + idx
        try:
            df = fetch_dataset(
                INST_DATASET,
                start_date,
                end_date,
                token=config.finmind_token,
                data_id=stock_id,
                requests_per_hour=config.finmind_requests_per_hour,
                max_retries=config.finmind_retry_max,
                backoff_seconds=config.finmind_retry_backoff,
                timeout=120,
            )
            progress.api_calls += 1

            if not df.empty:
                df = _normalize_institutional(df, allowed_stock_ids=allowed_stock_ids)
                records = df.to_dict("records")
                if records:
                    for k in range(0, len(records), 5000):
                        batch = records[k:k + 5000]
                        stmt = insert(RawInstitutional).values(batch)
                        update_cols = {
                            col: stmt.inserted[col]
                            for col in INST_UPDATE_COLS
                        }
                        stmt = stmt.on_duplicate_key_update(**update_cols)
                        db_session.execute(stmt)
                        db_session.commit()
                    rows = len(records)
                    progress.inst_rows += rows
                    print(f"  [{current}/{total}] {stock_id} +{rows:,} 筆", flush=True)
                else:
                    print(f"  [{current}/{total}] {stock_id} 空 (normalize 後)", flush=True)
            else:
                print(f"  [{current}/{total}] {stock_id} 無資料", flush=True)

            inst_done.add(stock_id)
            save_inst_done(inst_done)

        except FinMindError as e:
            print(f"  [{current}/{total}] {stock_id} ERR: {e}", flush=True)
        except Exception as e:
            db_session.rollback()
            print(f"  [{current}/{total}] {stock_id} DB ERR: {e}", flush=True)

    print(f"\n法人資料完成: 共 {progress.inst_rows:,} 筆")


def _backfill_prices(
    start_date: date,
    end_date: date,
    config,
    stock_list: List[str],
    db_session,
    progress: BackfillProgress,
) -> None:
    """逐檔抓取價格資料，每檔一次 API call 抓完整區間，每檔立即 upsert"""
    prices_done = load_prices_done()
    remaining = [s for s in stock_list if s not in prices_done]
    total = len(stock_list)
    done_count = total - len(remaining)

    print(f"\n{'='*50}")
    print(f"抓取價格資料 (逐檔模式)")
    print(f"期間: {start_date} ~ {end_date}")
    print(f"股票: {total} 檔, 已完成: {done_count}, 待抓取: {len(remaining)}")
    print(f"{'='*50}")

    if not remaining:
        print("所有股票已完成，跳過")
        return

    for idx, stock_id in enumerate(remaining, start=1):
        current = done_count + idx
        try:
            df = fetch_dataset(
                PRICE_DATASET,
                start_date,
                end_date,
                token=config.finmind_token,
                data_id=stock_id,
                requests_per_hour=config.finmind_requests_per_hour,
                max_retries=config.finmind_retry_max,
                backoff_seconds=config.finmind_retry_backoff,
                timeout=120,
            )
            progress.api_calls += 1

            if not df.empty:
                df = _normalize_prices(df)
                records = df.to_dict("records")
                if records:
                    for k in range(0, len(records), 5000):
                        batch = records[k:k + 5000]
                        stmt = insert(RawPrice).values(batch)
                        update_cols = {
                            col: stmt.inserted[col]
                            for col in ["open", "high", "low", "close", "volume"]
                        }
                        stmt = stmt.on_duplicate_key_update(**update_cols)
                        db_session.execute(stmt)
                        db_session.commit()
                    rows = len(records)
                    progress.prices_rows += rows
                    print(f"  [{current}/{total}] {stock_id} +{rows:,} 筆", flush=True)
                else:
                    print(f"  [{current}/{total}] {stock_id} 空 (normalize 後)", flush=True)
            else:
                print(f"  [{current}/{total}] {stock_id} 無資料", flush=True)

            prices_done.add(stock_id)
            save_prices_done(prices_done)

        except FinMindError as e:
            print(f"  [{current}/{total}] {stock_id} ERR: {e}", flush=True)
        except Exception as e:
            db_session.rollback()
            print(f"  [{current}/{total}] {stock_id} DB ERR: {e}", flush=True)

    print(f"\n價格資料完成: 共 {progress.prices_rows:,} 筆")


def run_backfill(
    start_date: date,
    end_date: date,
    config,
    datasets: List[str],
    resume: bool = True,
    listed_stock_list: Optional[List[str]] = None,
    chunk_days: Optional[int] = None,
) -> Dict:
    """執行歷史資料回補
    
    Args:
        start_date: 開始日期
        end_date: 結束日期
        config: AppConfig
        datasets: 要回補的 datasets 列表
        resume: 是否從上次進度繼續
        listed_stock_list: 指定股票清單（若提供則不從 API 取得）
    
    Returns:
        結果摘要 dict
    """
    # 載入或建立進度
    progress = load_progress() if resume else None
    
    if progress:
        # 檢查日期範圍和 datasets 是否一致
        if (progress.start_date != start_date or 
            progress.end_date != end_date or 
            set(progress.datasets) != set(datasets)):
            print(f"進度檔案的設定不符，將重新開始")
            progress = None
    
    # 計算 chunks
    if chunk_days is None:
        chunk_days = config.chunk_days
    margin_only = set(datasets) == {"margin"}
    if margin_only:
        # 融資融券逐檔抓取：每檔 1 次 API call 就能拉完整區間，
        # 用單一 chunk 最小化 API 次數（2339 檔 = 2339 calls）。
        # 中斷續傳由 stock-level 進度檔（backfill_margin_done.json）負責。
        chunk_days = (end_date - start_date).days + 1
    all_chunks = list(date_chunks(start_date, end_date, chunk_days))
    total_chunks = len(all_chunks)
    
    # 決定起始 chunk
    start_chunk_idx = 0
    if progress:
        # 找到應該繼續的 chunk
        for idx, (chunk_start, _) in enumerate(all_chunks):
            if chunk_start >= progress.current_date:
                start_chunk_idx = idx
                break
        print(f"從 chunk {start_chunk_idx + 1}/{total_chunks} 繼續")
    else:
        progress = BackfillProgress(
            start_date=start_date,
            end_date=end_date,
            current_date=start_date,
            total_chunks=total_chunks,
            completed_chunks=0,
            prices_rows=0,
            inst_rows=0,
            margin_rows=0,
            api_calls=0,
            fetch_mode="bulk",
            datasets=datasets,
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )
    
    # 初始化 Rate Limiter
    limiter = get_rate_limiter(config.finmind_requests_per_hour)
    
    # 先嘗試全市場抓取，失敗則獲取股票清單
    fetch_mode = "by_stock" if margin_only else "bulk"
    stock_list: List[str] = []
    
    print(f"\n開始回補歷史資料")
    print(f"日期範圍: {start_date} ~ {end_date}")
    print(f"總 chunks: {total_chunks}, chunk_days: {chunk_days}")
    print(f"Datasets: {', '.join(datasets)}")
    
    # 估算
    est_calls_bulk, est_minutes_bulk = estimate_api_calls(start_date, end_date, chunk_days, "bulk", datasets)
    est_calls_stock, est_minutes_stock = estimate_api_calls(start_date, end_date, chunk_days, "by_stock", datasets)
    print(f"預估 API 次數:")
    print(f"   - 全市場模式: ~{est_calls_bulk} 次 (~{est_minutes_bulk} 分鐘)")
    print(f"   - 逐檔批次模式: ~{est_calls_stock} 次 (~{est_minutes_stock} 分鐘)")
    print()
    
    with get_session() as db_session:
        # === Phase 1: 逐檔抓取價格資料（完整區間）===
        if "prices" in datasets:
            if not stock_list:
                if listed_stock_list:
                    stock_list = listed_stock_list
                else:
                    stock_list = fetch_stock_list(
                        config.finmind_token,
                        requests_per_hour=config.finmind_requests_per_hour,
                        max_retries=config.finmind_retry_max,
                        backoff_seconds=config.finmind_retry_backoff,
                    )
                    progress.api_calls += 1
            _backfill_prices(start_date, end_date, config, stock_list, db_session, progress)
            save_progress(progress)

        # === Phase 1b: 逐檔抓取法人資料（完整區間）===
        if "institutional" in datasets:
            if not stock_list:
                if listed_stock_list:
                    stock_list = listed_stock_list
                else:
                    stock_list = fetch_stock_list(
                        config.finmind_token,
                        requests_per_hour=config.finmind_requests_per_hour,
                        max_retries=config.finmind_retry_max,
                        backoff_seconds=config.finmind_retry_backoff,
                    )
                    progress.api_calls += 1
            allowed_stock_ids = set(listed_stock_list) if listed_stock_list else None
            _backfill_institutional(start_date, end_date, config, stock_list, db_session, progress, allowed_stock_ids)
            save_progress(progress)

        # === Phase 2: 融資融券（按時間 chunk）===
        for chunk_idx, (chunk_start, chunk_end) in enumerate(all_chunks[start_chunk_idx:], start=start_chunk_idx):
            chunk_num = chunk_idx + 1
            
            # 進度顯示
            pct = chunk_num / total_chunks * 100
            remaining = limiter.remaining_requests()
            print(f"[{chunk_num}/{total_chunks}] ({pct:.1f}%) {chunk_start} ~ {chunk_end} | Rate: {remaining}/{limiter.effective_limit}", end="", flush=True)
            
            chunk_results = []
            
            # === 抓取融資融券資料（逐檔模式 + stock-level 續傳）===
            if "margin" in datasets:
                margin_rows_written = 0
                margin_cols = [
                    "margin_purchase_buy", "margin_purchase_sell",
                    "margin_purchase_cash_repay", "margin_purchase_limit",
                    "margin_purchase_balance",
                    "short_sale_buy", "short_sale_sell",
                    "short_sale_cash_repay", "short_sale_limit",
                    "short_sale_balance", "offset_loan_and_short", "note",
                ]

                # 載入已完成股票，過濾 stock_list
                margin_done: Set[str] = load_margin_done()

                # 取得股票清單
                if not stock_list:
                    if listed_stock_list:
                        stock_list = listed_stock_list
                    else:
                        stock_list = fetch_stock_list(
                            config.finmind_token,
                            requests_per_hour=config.finmind_requests_per_hour,
                            max_retries=config.finmind_retry_max,
                            backoff_seconds=config.finmind_retry_backoff,
                        )
                        progress.api_calls += 1

                remaining_stocks = [s for s in stock_list if s not in margin_done]
                total_stocks = len(stock_list)
                done_count = total_stocks - len(remaining_stocks)
                print(f" [待抓取: {len(remaining_stocks)} 檔, 已完成: {done_count} 檔]", flush=True)

                if remaining_stocks:
                    # 批次寫入回調函數（每批寫入 DB 並記錄已完成 stock_id）
                    _current_batch_stocks: List[str] = []

                    DB_INSERT_BATCH = 5000  # 每次 INSERT 最多筆數，避免 SQLAlchemy 編譯過慢

                    def write_margin_batch(df: pd.DataFrame) -> int:
                        nonlocal margin_rows_written
                        if df.empty:
                            return 0
                        try:
                            df = _normalize_margin_short(df)
                            records = df.to_dict("records")
                            if not records:
                                return 0
                            # 分批 INSERT（避免單次 INSERT 十萬筆導致 SQLAlchemy 編譯卡住）
                            total_written = 0
                            for k in range(0, len(records), DB_INSERT_BATCH):
                                batch_records = records[k:k + DB_INSERT_BATCH]
                                stmt = insert(RawMarginShort).values(batch_records)
                                update_cols = {
                                    col: stmt.inserted[col]
                                    for col in margin_cols
                                    if col in df.columns
                                }
                                stmt = stmt.on_duplicate_key_update(**update_cols)
                                db_session.execute(stmt)
                                db_session.commit()
                                total_written += len(batch_records)
                            margin_rows_written += total_written
                            # 記錄已完成的 stock_id 到進度檔
                            written_ids = set(df["stock_id"].unique())
                            margin_done.update(written_ids)
                            save_margin_done(margin_done)
                            return total_written
                        except Exception as exc:
                            db_session.rollback()
                            print(f"\n  [margin write err] {type(exc).__name__}: {exc}", flush=True)
                        return 0

                    def margin_progress(current, total):
                        done_now = done_count + current
                        print(f"\r[{chunk_num}/{total_chunks}] margin: {done_now}/{total_stocks} 股票 (已寫入 {margin_rows_written:,} 筆)", end="", flush=True)

                    try:
                        fetch_dataset_by_stocks(
                            MARGIN_DATASET,
                            chunk_start,
                            chunk_end,
                            remaining_stocks,
                            token=config.finmind_token,
                            batch_size=10,  # 每 10 檔寫入一次 DB（~24K 筆），避免 INSERT 過大
                            use_batch_query=False,
                            requests_per_hour=config.finmind_requests_per_hour,
                            max_retries=config.finmind_retry_max,
                            backoff_seconds=config.finmind_retry_backoff,
                            progress_callback=margin_progress,
                            batch_write_callback=write_margin_batch,
                            debug=True,
                            timeout=120,
                        )
                        print()  # 換行
                        progress.api_calls += len(remaining_stocks)
                    except FinMindError as e:
                        print(f" [margin err: {e}]", end="", flush=True)

                if margin_rows_written > 0:
                    progress.margin_rows += margin_rows_written
                    chunk_results.append(f"M+{margin_rows_written:,}")
            
            # 更新進度
            progress.completed_chunks = chunk_num
            progress.current_date = chunk_end + timedelta(days=1)
            save_progress(progress)
            
            result_str = ", ".join(chunk_results) if chunk_results else "no data"
            print(f" OK [{result_str}]")
    
    # 完成
    print(f"\n回補完成!")
    print(f"總計: prices={progress.prices_rows:,}, institutional={progress.inst_rows:,}, margin={progress.margin_rows:,}")
    print(f"總 API 呼叫: {progress.api_calls:,} 次")
    
    # 刪除進度檔案（完成後）
    delete_progress()
    delete_margin_done()
    delete_prices_done()
    delete_inst_done()
    
    return {
        "prices_rows": progress.prices_rows,
        "inst_rows": progress.inst_rows,
        "margin_rows": progress.margin_rows,
        "api_calls": progress.api_calls,
        "fetch_mode": fetch_mode,
    }


def main():
    parser = argparse.ArgumentParser(
        description="歷史資料回補腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--years", type=int, default=10, help="抓取過去幾年資料 (預設: 10)")
    parser.add_argument("--start", type=str, help="開始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="結束日期 (YYYY-MM-DD)")
    parser.add_argument("--datasets", type=str, default="prices,institutional,margin",
                       help="要回補的 datasets，逗號分隔 (預設: prices,institutional,margin)")
    parser.add_argument("--listed-only", action="store_true", 
                       help="只抓目前上市櫃股票（排除下市、ETF、權證）")
    parser.add_argument("--status", action="store_true", help="顯示目前進度")
    parser.add_argument("--reset", action="store_true", help="重置進度")
    parser.add_argument("--estimate", action="store_true", help="只顯示 API 用量估算")
    
    args = parser.parse_args()
    
    config = load_config()
    
    # 解析 datasets
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    invalid = set(datasets) - SUPPORTED_DATASETS
    if invalid:
        print(f"錯誤: 不支援的 dataset: {invalid}")
        print(f"支援的 datasets: {SUPPORTED_DATASETS}")
        sys.exit(1)
    
    # 顯示狀態
    if args.status:
        progress = load_progress()
        print_status(progress, config)
        return
    
    # 重置進度
    if args.reset:
        delete_progress()
        delete_margin_done()
        delete_prices_done()
        delete_inst_done()
        print("進度已重置")
        if not args.years and not args.start:
            return
    
    # 決定日期範圍
    today = datetime.now(ZoneInfo(config.tz)).date()
    
    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        end_date = today
        start_date = today - timedelta(days=365 * args.years)
    
    # 只顯示估算
    if args.estimate:
        print(f"\nAPI 用量估算 ({start_date} ~ {end_date})")
        print(f"   chunk_days: {config.chunk_days}")
        print(f"   datasets: {datasets}")
        
        if args.listed_only:
            listed_count = len(get_listed_stocks_from_db())
            print(f"   --listed-only: {listed_count} 檔上市櫃股票")
        
        est_bulk, min_bulk = estimate_api_calls(start_date, end_date, config.chunk_days, "bulk", datasets)
        est_stock, min_stock = estimate_api_calls(start_date, end_date, config.chunk_days, "by_stock", datasets)
        
        print(f"\n   全市場模式:")
        print(f"     - API 次數: ~{est_bulk} 次")
        print(f"     - 預估時間: ~{min_bulk} 分鐘")
        
        print(f"\n   逐檔批次模式:")
        print(f"     - API 次數: ~{est_stock} 次")
        print(f"     - 預估時間: ~{min_stock} 分鐘")
        
        print(f"\n   每小時限制: {config.finmind_requests_per_hour} 次")
        return
    
    # 檢查 token
    if not config.finmind_token:
        print("錯誤: FINMIND_TOKEN 未設定")
        print("請在 .env 檔案中設定 FINMIND_TOKEN")
        sys.exit(1)
    
    # 取得上市櫃股票清單（如果指定 --listed-only）
    listed_stock_list = None
    if args.listed_only:
        listed_stock_list = get_listed_stocks_from_db()
        if not listed_stock_list:
            print("錯誤: stocks 表為空，請先執行 make pipeline 或 ingest_stock_master")
            sys.exit(1)
        print(f"使用上市櫃股票清單: {len(listed_stock_list)} 檔（排除下市、ETF、權證）")
    
    # 執行回補
    result = run_backfill(
        start_date,
        end_date,
        config,
        datasets,
        resume=not args.reset,
        listed_stock_list=listed_stock_list,
        chunk_days=config.chunk_days,
    )
    print(f"\n結果: {result}")


if __name__ == "__main__":
    main()
