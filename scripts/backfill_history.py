#!/usr/bin/env python3
"""歷史資料回補腳本

用於抓取過去 N 年的歷史資料，支援：
1. 中斷續傳（從上次進度繼續）
2. 進度顯示與預估時間
3. Rate Limiting（避免觸發 API 限流）
4. 批次查詢優化（大幅減少 API 次數）

使用方式：
    # 抓取過去 5 年資料
    python scripts/backfill_history.py --years 5
    
    # 抓取特定日期範圍
    python scripts/backfill_history.py --start 2021-01-01 --end 2026-02-05
    
    # 顯示目前進度和預估
    python scripts/backfill_history.py --status
    
    # 重置進度（重新抓取）
    python scripts/backfill_history.py --reset --years 5

API 用量估算（付費會員 6000 次/小時）：
- 全市場模式：5 年資料約需 20-40 次 API calls
- 逐檔批次模式：5 年資料約需 360-720 次 API calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    fetch_stock_list,
)
from app.models import RawInstitutional, RawPrice
from app.rate_limiter import get_rate_limiter
from skills.ingest_institutional import _normalize_institutional
from skills.ingest_prices import _normalize_prices


# 進度檔案位置
PROGRESS_FILE = Path(__file__).resolve().parent.parent / "storage" / "backfill_progress.json"

PRICE_DATASET = "TaiwanStockPrice"
INST_DATASET = "TaiwanStockInstitutionalInvestorsBuySell"


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
    api_calls: int
    fetch_mode: str  # "bulk" or "by_stock"
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
            "api_calls": self.api_calls,
            "fetch_mode": self.fetch_mode,
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
            prices_rows=d["prices_rows"],
            inst_rows=d["inst_rows"],
            api_calls=d["api_calls"],
            fetch_mode=d["fetch_mode"],
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


def estimate_api_calls(
    start_date: date,
    end_date: date,
    chunk_days: int,
    fetch_mode: str,
    stock_count: int = 1800,
    batch_size: int = 100,
) -> Tuple[int, int]:
    """估算 API 次數和所需時間
    
    Returns:
        (estimated_calls, estimated_minutes)
    """
    total_days = (end_date - start_date).days
    chunks = (total_days + chunk_days - 1) // chunk_days
    
    if fetch_mode == "bulk":
        # 全市場模式：每個 chunk 抓 2 次（價格+法人）+ 1 次 stock list
        calls = chunks * 2 + 1
    else:
        # 逐檔批次模式：每個 chunk 抓 (stock_count / batch_size) * 2 次
        batches_per_chunk = (stock_count + batch_size - 1) // batch_size
        calls = chunks * batches_per_chunk * 2 + 1
    
    # 以每小時 5400 次（90% buffer）計算所需時間
    minutes = calls / 90  # 5400 / 60 = 90 calls per minute
    
    return calls, int(minutes) + 1


def print_status(progress: Optional[BackfillProgress], config) -> None:
    """印出目前狀態"""
    print("\n" + "=" * 60)
    print("📊 歷史資料回補狀態")
    print("=" * 60)
    
    if progress is None:
        print("❌ 沒有進行中的回補任務")
        return
    
    pct = (progress.completed_chunks / progress.total_chunks * 100) if progress.total_chunks > 0 else 0
    remaining_chunks = progress.total_chunks - progress.completed_chunks
    
    # 估算剩餘時間
    est_calls, est_minutes = estimate_api_calls(
        progress.current_date,
        progress.end_date,
        config.chunk_days,
        progress.fetch_mode,
    )
    
    print(f"📅 日期範圍: {progress.start_date} ~ {progress.end_date}")
    print(f"📍 目前進度: {progress.current_date}")
    print(f"📦 Chunks:   {progress.completed_chunks}/{progress.total_chunks} ({pct:.1f}%)")
    print(f"🔄 抓取模式: {progress.fetch_mode}")
    print(f"📈 已抓取:   prices={progress.prices_rows:,} rows, institutional={progress.inst_rows:,} rows")
    print(f"🔗 API 呼叫: {progress.api_calls:,} 次")
    print(f"⏱️  預估剩餘: ~{est_calls:,} 次 API calls, ~{est_minutes} 分鐘")
    print(f"🕐 開始時間: {progress.started_at}")
    print(f"🕐 更新時間: {progress.last_updated}")
    
    # Rate limiter 狀態
    limiter = get_rate_limiter(config.finmind_requests_per_hour)
    remaining = limiter.remaining_requests()
    print(f"⚡ Rate Limit: {remaining}/{limiter.effective_limit} 剩餘（本小時）")
    print("=" * 60)


def run_backfill(
    start_date: date,
    end_date: date,
    config,
    resume: bool = True,
) -> Dict:
    """執行歷史資料回補
    
    Args:
        start_date: 開始日期
        end_date: 結束日期
        config: AppConfig
        resume: 是否從上次進度繼續
    
    Returns:
        結果摘要 dict
    """
    # 載入或建立進度
    progress = load_progress() if resume else None
    
    if progress and (progress.start_date != start_date or progress.end_date != end_date):
        print(f"⚠️  進度檔案的日期範圍不符，將重新開始")
        progress = None
    
    # 計算 chunks
    chunk_days = config.chunk_days
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
        print(f"📂 從 chunk {start_chunk_idx + 1}/{total_chunks} 繼續")
    else:
        progress = BackfillProgress(
            start_date=start_date,
            end_date=end_date,
            current_date=start_date,
            total_chunks=total_chunks,
            completed_chunks=0,
            prices_rows=0,
            inst_rows=0,
            api_calls=0,
            fetch_mode="bulk",
            started_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
        )
    
    # 初始化 Rate Limiter
    limiter = get_rate_limiter(config.finmind_requests_per_hour)
    
    # 先嘗試全市場抓取，失敗則獲取股票清單
    fetch_mode = "bulk"
    stock_list: List[str] = []
    
    print(f"\n🚀 開始回補歷史資料")
    print(f"📅 日期範圍: {start_date} ~ {end_date}")
    print(f"📦 總 chunks: {total_chunks}, chunk_days: {chunk_days}")
    
    # 估算
    est_calls_bulk, est_minutes_bulk = estimate_api_calls(start_date, end_date, chunk_days, "bulk")
    est_calls_stock, est_minutes_stock = estimate_api_calls(start_date, end_date, chunk_days, "by_stock")
    print(f"📊 預估 API 次數:")
    print(f"   - 全市場模式: ~{est_calls_bulk} 次 (~{est_minutes_bulk} 分鐘)")
    print(f"   - 逐檔批次模式: ~{est_calls_stock} 次 (~{est_minutes_stock} 分鐘)")
    print()
    
    with get_session() as db_session:
        for chunk_idx, (chunk_start, chunk_end) in enumerate(all_chunks[start_chunk_idx:], start=start_chunk_idx):
            chunk_num = chunk_idx + 1
            
            # 進度顯示
            pct = chunk_num / total_chunks * 100
            remaining = limiter.remaining_requests()
            print(f"[{chunk_num}/{total_chunks}] ({pct:.1f}%) {chunk_start} ~ {chunk_end} | Rate: {remaining}/{limiter.effective_limit}", end="", flush=True)
            
            # === 抓取價格資料 ===
            price_df = pd.DataFrame()
            try:
                price_df = fetch_dataset(
                    PRICE_DATASET,
                    chunk_start,
                    chunk_end,
                    token=config.finmind_token,
                    requests_per_hour=config.finmind_requests_per_hour,
                )
                progress.api_calls += 1
                
                # 如果全市場抓取回傳空值，切換到逐檔模式
                if price_df.empty and fetch_mode == "bulk":
                    print(" [切換到逐檔模式]", end="", flush=True)
                    fetch_mode = "by_stock"
                    progress.fetch_mode = "by_stock"
                    
                    # 取得股票清單
                    stock_list = fetch_stock_list(config.finmind_token, config.finmind_requests_per_hour)
                    progress.api_calls += 1
                    print(f" [股票數: {len(stock_list)}]", end="", flush=True)
                
                if price_df.empty and fetch_mode == "by_stock" and stock_list:
                    price_df = fetch_dataset_by_stocks(
                        PRICE_DATASET,
                        chunk_start,
                        chunk_end,
                        stock_list,
                        token=config.finmind_token,
                        requests_per_hour=config.finmind_requests_per_hour,
                    )
                    # 估算 API 次數（批次查詢）
                    progress.api_calls += (len(stock_list) + 99) // 100
                
            except FinMindError as e:
                print(f" ❌ prices: {e}")
            
            # 寫入價格資料
            if not price_df.empty:
                try:
                    price_df = _normalize_prices(price_df)
                    records = price_df.to_dict("records")
                    if records:
                        stmt = insert(RawPrice).values(records)
                        update_cols = {col: stmt.inserted[col] for col in ["open", "high", "low", "close", "volume"]}
                        stmt = stmt.on_duplicate_key_update(**update_cols)
                        db_session.execute(stmt)
                        db_session.commit()
                        progress.prices_rows += len(records)
                except Exception as e:
                    print(f" ⚠️ prices write: {e}")
            
            # === 抓取法人資料 ===
            inst_df = pd.DataFrame()
            try:
                inst_df = fetch_dataset(
                    INST_DATASET,
                    chunk_start,
                    chunk_end,
                    token=config.finmind_token,
                    requests_per_hour=config.finmind_requests_per_hour,
                )
                progress.api_calls += 1
                
                if inst_df.empty and fetch_mode == "by_stock" and stock_list:
                    inst_df = fetch_dataset_by_stocks(
                        INST_DATASET,
                        chunk_start,
                        chunk_end,
                        stock_list,
                        token=config.finmind_token,
                        requests_per_hour=config.finmind_requests_per_hour,
                    )
                    progress.api_calls += (len(stock_list) + 99) // 100
                    
            except FinMindError as e:
                print(f" ❌ inst: {e}")
            
            # 寫入法人資料
            if not inst_df.empty:
                try:
                    inst_df = _normalize_institutional(inst_df)
                    records = inst_df.to_dict("records")
                    if records:
                        stmt = insert(RawInstitutional).values(records)
                        update_cols = {
                            col: stmt.inserted[col]
                            for col in [
                                "foreign_buy", "foreign_sell", "foreign_net",
                                "trust_buy", "trust_sell", "trust_net",
                                "dealer_buy", "dealer_sell", "dealer_net",
                            ]
                        }
                        stmt = stmt.on_duplicate_key_update(**update_cols)
                        db_session.execute(stmt)
                        db_session.commit()
                        progress.inst_rows += len(records)
                except Exception as e:
                    print(f" ⚠️ inst write: {e}")
            
            # 更新進度
            progress.completed_chunks = chunk_num
            progress.current_date = chunk_end + timedelta(days=1)
            save_progress(progress)
            
            print(f" ✅ prices: +{len(price_df) if not price_df.empty else 0}, inst: +{len(inst_df) if not inst_df.empty else 0}")
    
    # 完成
    print(f"\n🎉 回補完成!")
    print(f"📈 總計: prices={progress.prices_rows:,} rows, institutional={progress.inst_rows:,} rows")
    print(f"🔗 總 API 呼叫: {progress.api_calls:,} 次")
    
    # 刪除進度檔案（完成後）
    delete_progress()
    
    return {
        "prices_rows": progress.prices_rows,
        "inst_rows": progress.inst_rows,
        "api_calls": progress.api_calls,
        "fetch_mode": fetch_mode,
    }


def main():
    parser = argparse.ArgumentParser(
        description="歷史資料回補腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--years", type=int, default=5, help="抓取過去幾年資料 (預設: 5)")
    parser.add_argument("--start", type=str, help="開始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="結束日期 (YYYY-MM-DD)")
    parser.add_argument("--status", action="store_true", help="顯示目前進度")
    parser.add_argument("--reset", action="store_true", help="重置進度")
    parser.add_argument("--estimate", action="store_true", help="只顯示 API 用量估算")
    
    args = parser.parse_args()
    
    config = load_config()
    
    # 顯示狀態
    if args.status:
        progress = load_progress()
        print_status(progress, config)
        return
    
    # 重置進度
    if args.reset:
        delete_progress()
        print("✅ 進度已重置")
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
        print(f"\n📊 API 用量估算 ({start_date} ~ {end_date})")
        print(f"   chunk_days: {config.chunk_days}")
        
        est_bulk, min_bulk = estimate_api_calls(start_date, end_date, config.chunk_days, "bulk")
        est_stock, min_stock = estimate_api_calls(start_date, end_date, config.chunk_days, "by_stock")
        
        print(f"\n   全市場模式:")
        print(f"     - API 次數: ~{est_bulk} 次")
        print(f"     - 預估時間: ~{min_bulk} 分鐘")
        
        print(f"\n   逐檔批次模式:")
        print(f"     - API 次數: ~{est_stock} 次")
        print(f"     - 預估時間: ~{min_stock} 分鐘")
        
        print(f"\n   ⚡ 每小時限制: {config.finmind_requests_per_hour} 次")
        return
    
    # 檢查 token
    if not config.finmind_token:
        print("❌ 錯誤: FINMIND_TOKEN 未設定")
        print("請在 .env 檔案中設定 FINMIND_TOKEN")
        sys.exit(1)
    
    # 執行回補
    result = run_backfill(start_date, end_date, config, resume=not args.reset)
    print(f"\n結果: {result}")


if __name__ == "__main__":
    main()
