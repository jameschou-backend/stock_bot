"""Priority 7：借券餘額聚合（TaiwanStockSecuritiesLending）

從 FinMind 抓取逐筆借券資料，彙整成每日每股借券餘額/費率指標，
寫入 raw_securities_lending 表。

Dataset: TaiwanStockSecuritiesLending
Fields:  date, stock_id, transaction_type, volume, fee_rate,
         original_return_date, original_lending_period
頻率：   逐筆（每日多筆，按 transaction_type 區分借入/還券）
限制：   Sponsor 計劃

聚合邏輯：
- 只計「借出」（transaction_type 含 "lending" 或 "borrow"）累計餘額
- 以今日借出量 - 今日還券量 計算淨變動，並與前日餘額累計
- lending_fee_rate：加權平均借券費率（以 volume 加權）
- lending_transaction_count：當日借出筆數
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
from app.models import RawSecuritiesLending, Stock

DATASET = "TaiwanStockSecuritiesLending"
UPDATE_COLS = ["lending_balance", "lending_fee_rate", "lending_transaction_count"]

# 借出交易類型關鍵字（FinMind 原始資料用中文標記）
_LEND_KEYWORDS = ["借出", "lending", "borrow"]
_RETURN_KEYWORDS = ["還券", "return", "returned"]


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawSecuritiesLending.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def _load_stock_ids(session: Session) -> List[str]:
    rows = (
        session.query(Stock.stock_id)
        .filter(Stock.is_listed == True)
        .filter(Stock.security_type == "stock")
        .order_by(Stock.stock_id)
        .all()
    )
    return [row[0] for row in rows]


def _aggregate_lending(df: pd.DataFrame) -> pd.DataFrame:
    """將逐筆借券資料彙整成每日每股摘要指標。

    Returns DataFrame with: stock_id, trading_date, lending_balance,
        lending_fee_rate, lending_transaction_count
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    rename = {"date": "trading_date"}
    df = df.rename(columns=rename)

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.date
    df["stock_id"] = df["stock_id"].astype(str)
    df = df.dropna(subset=["stock_id", "trading_date"])
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]

    vol_col = next((c for c in ["volume", "Volume", "quantity"] if c in df.columns), None)
    fee_col = next((c for c in ["fee_rate", "FeeRate", "lending_fee_rate", "rate"] if c in df.columns), None)
    type_col = next((c for c in ["transaction_type", "TransactionType", "type"] if c in df.columns), None)

    if vol_col is None:
        return pd.DataFrame()

    df["volume_n"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0)
    df["fee_n"] = pd.to_numeric(df[fee_col], errors="coerce").fillna(0) if fee_col else 0

    # 區分借出 vs 還券（若 type_col 存在）
    if type_col is not None:
        type_lower = df[type_col].astype(str).str.lower()
        is_lend = type_lower.str.contains("|".join(_LEND_KEYWORDS), na=False)
        is_return = type_lower.str.contains("|".join(_RETURN_KEYWORDS), na=False)
        df["net_vol"] = df["volume_n"] * is_lend.astype(int) - df["volume_n"] * is_return.astype(int)
        df["is_lend"] = is_lend
    else:
        # 若無法區分類型，全部視為借出
        df["net_vol"] = df["volume_n"]
        df["is_lend"] = True

    results = []
    for (stock_id, trading_date), grp in df.groupby(["stock_id", "trading_date"]):
        lend_grp = grp[grp["is_lend"]]
        net_balance = int(grp["net_vol"].sum())
        lend_count = int(lend_grp["is_lend"].sum())

        # 加權平均費率（以借出量加權）
        total_vol = lend_grp["volume_n"].sum()
        if total_vol > 0:
            fee_rate = float((lend_grp["fee_n"] * lend_grp["volume_n"]).sum() / total_vol)
        else:
            fee_rate = float(grp["fee_n"].mean()) if not grp.empty else 0.0

        results.append({
            "stock_id": stock_id,
            "trading_date": trading_date,
            "lending_balance": net_balance,
            "lending_fee_rate": fee_rate,
            "lending_transaction_count": lend_count,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_securities_lending", commit=True)
    logs: Dict = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = max(
            today - timedelta(days=365 * config.train_lookback_years),
            date(2010, 1, 1),
        )
        start_date = _resolve_start_date(db_session, default_start)
        end_date = today

        logs["start_date"] = start_date.isoformat()
        logs["end_date"] = end_date.isoformat()

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "already_up_to_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        stock_ids = _load_stock_ids(db_session)
        if not stock_ids:
            logs["rows"] = 0
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        logs["stocks"] = len(stock_ids)
        print(f"[ingest_securities_lending] {start_date} ~ {end_date}，{len(stock_ids)} 檔", flush=True)

        # SecuritiesLending 資料量大（逐筆），使用 90 天 chunk 避免超時
        CHUNK_DAYS = 90
        total_rows = 0
        commit_buffer: List[Dict] = []

        date_ranges: List[tuple] = []
        cur = start_date
        while cur <= end_date:
            chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end_date)
            date_ranges.append((cur, chunk_end))
            cur = chunk_end + timedelta(days=1)

        for i, stock_id in enumerate(stock_ids, 1):
            if i % 200 == 0:
                update_job(
                    db_session, job_id,
                    logs={**logs, "progress": f"{i}/{len(stock_ids)}", "rows": total_rows},
                    commit=True,
                )
                print(f"  [{i}/{len(stock_ids)}] rows={total_rows}", flush=True)

            for range_start, range_end in date_ranges:
                try:
                    df = fetch_dataset(
                        dataset=DATASET,
                        start_date=range_start,
                        end_date=range_end,
                        token=config.finmind_token,
                        data_id=stock_id,
                        requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
                        max_retries=getattr(config, "finmind_retry_max", 3),
                        backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
                        timeout=120,
                    )
                except Exception as exc:
                    # 逐筆資料量大，可能超時；記錄但繼續下一檔
                    print(f"  ⚠️ {stock_id} {range_start}~{range_end}: {exc}", flush=True)
                    continue

                if df is None or df.empty:
                    continue

                agg = _aggregate_lending(df)
                if agg.empty:
                    continue

                commit_buffer.extend(agg.to_dict("records"))

                if len(commit_buffer) >= 5000:
                    _flush(db_session, commit_buffer)
                    total_rows += len(commit_buffer)
                    commit_buffer.clear()

        if commit_buffer:
            _flush(db_session, commit_buffer)
            total_rows += len(commit_buffer)

        logs["rows"] = total_rows
        print(f"  ✅ ingest_securities_lending: {total_rows} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise


def _flush(session: Session, buffer: List[Dict]) -> None:
    stmt = insert(RawSecuritiesLending).values(buffer)
    stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
    session.execute(stmt)
    session.commit()
