"""Priority 4：官股銀行買賣超（TaiwanstockGovernmentBankBuySell）

從 FinMind 抓取官股銀行每日買賣超明細，彙整成聚合指標，
寫入 raw_gov_bank 表。

Dataset: TaiwanstockGovernmentBankBuySell
Fields:  date, stock_id, securities_trader_id, securities_trader, buy, sell
頻率：   每日
限制：   Sponsor 計劃；資料從 2010 年起

官股銀行共 8 家：台灣銀行、土地銀行、合作金庫銀行、第一銀行、
                  華南銀行、彰化銀行、臺灣企銀、兆豐銀行

聚合指標：
- gov_net:          8 行庫合計淨買超（張）
- bank_count_buy:   當日淨買超的行庫數
- bank_count_sell:  當日淨賣超的行庫數
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import (
    FinMindError,
    fetch_dataset,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawGovBank, Stock

DATASET = "TaiwanStockGovernmentBankBuySell"
UPDATE_COLS = ["gov_net", "bank_count_buy", "bank_count_sell"]


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawGovBank.trading_date)).scalar()
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


def _aggregate_gov_bank(df: pd.DataFrame, allowed_stock_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """將官股銀行原始明細彙整成每日每股聚合指標。"""
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
    buy_col  = next((c for c in ["buy", "Buy", "buy_volume"] if c in df.columns), None)
    sell_col = next((c for c in ["sell", "Sell", "sell_volume"] if c in df.columns), None)
    if buy_col is None or sell_col is None:
        raise FinMindError(
            f"TaiwanstockGovernmentBankBuySell missing buy/sell columns; got: {df.columns.tolist()}"
        )

    df["buy"]  = pd.to_numeric(df[buy_col],  errors="coerce").fillna(0)
    df["sell"] = pd.to_numeric(df[sell_col], errors="coerce").fillna(0)
    df["net"]  = df["buy"] - df["sell"]

    results = []
    for (stock_id, trading_date), grp in df.groupby(["stock_id", "trading_date"]):
        gov_net        = int(grp["net"].sum())
        bank_count_buy  = int((grp["net"] > 0).sum())
        bank_count_sell = int((grp["net"] < 0).sum())

        results.append({
            "stock_id": stock_id,
            "trading_date": trading_date,
            "gov_net": gov_net,
            "bank_count_buy": bank_count_buy,
            "bank_count_sell": bank_count_sell,
        })

    return pd.DataFrame(results)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_gov_bank", commit=True)
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

        print(f"[ingest_gov_bank] {start_date} ~ {end_date}", flush=True)

        # TaiwanStockGovernmentBankBuySell：
        # - 不接受 data_id（全市場資料，無法按股票篩選）
        # - 資料量過大，API 只允許單日查詢（end_date 不傳或設為 None）
        # → 逐日查詢，每日一次 API call
        allowed_stock_ids = _load_allowed_stock_ids(db_session)
        logs["allowed_stock_ids"] = len(allowed_stock_ids)
        logs["fetch_mode"] = "daily_bulk"

        # 產生所有日期（含週末，API 會自動回傳空值）
        all_dates = [
            start_date + timedelta(days=d)
            for d in range((end_date - start_date).days + 1)
        ]
        # 跳過週末（台股週一至五）
        trading_dates = [d for d in all_dates if d.weekday() < 5]
        logs["days_total"] = len(trading_dates)
        total_rows = 0
        commit_buffer: List[Dict] = []

        for i, query_date in enumerate(trading_dates, 1):
            if i % 50 == 0:
                update_job(db_session, job_id, logs={**logs, "progress": f"{i}/{len(trading_dates)}", "rows": total_rows}, commit=True)
                print(f"  [{i}/{len(trading_dates)}] 已處理 {total_rows} 筆...", flush=True)

            df = fetch_dataset(
                dataset=DATASET,
                start_date=query_date,
                end_date=None,   # 此 dataset 僅支援單日，不傳 end_date
                token=config.finmind_token,
                requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
                max_retries=getattr(config, "finmind_retry_max", 3),
                backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
            )
            if df is None or df.empty:
                continue

            agg_df = _aggregate_gov_bank(df, allowed_stock_ids=allowed_stock_ids or None)
            if agg_df.empty:
                continue

            commit_buffer.extend(agg_df.to_dict("records"))

            # 每 30 天 commit 一次，避免事務過長
            if len(commit_buffer) >= 500:
                stmt = insert(RawGovBank).values(commit_buffer)
                stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
                db_session.execute(stmt)
                db_session.commit()
                total_rows += len(commit_buffer)
                commit_buffer.clear()

        # 最後 flush
        if commit_buffer:
            stmt = insert(RawGovBank).values(commit_buffer)
            stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
            db_session.execute(stmt)
            db_session.commit()
            total_rows += len(commit_buffer)

        logs["rows"] = total_rows
        print(f"  ✅ gov_bank: {total_rows} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
