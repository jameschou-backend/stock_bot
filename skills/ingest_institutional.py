"""Ingest Institutional 模組

從 FinMind 抓取三大法人買賣超資料寫入 raw_institutional 表。

支援兩種模式：
1. 全市場抓取（需較高權限）
2. 逐檔抓取（適用免費/低階會員）

會自動偵測權限並切換模式。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
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
from app.job_utils import finish_job, start_job
from app.models import RawInstitutional


DATASET = "TaiwanStockInstitutionalInvestorsBuySell"

# 股票清單快取（與 ingest_prices 共用）
_stock_list_cache: Optional[List[str]] = None

CATEGORY_MAP = {
    "foreign_investor": "foreign",
    "foreign_dealer_self": "foreign",
    "investment_trust": "trust",
    "dealer_self": "dealer",
    "dealer_hedging": "dealer",
}


def _normalize_institutional(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"date": "trading_date"}
    df = df.rename(columns=rename_map)

    required = {"stock_id", "trading_date", "name"}
    missing = required - set(df.columns)
    if missing:
        raise FinMindError(f"Institutional dataset missing columns: {sorted(missing)}")

    buy_col = next((c for c in ["buy", "buy_amount", "buy_volume"] if c in df.columns), None)
    sell_col = next((c for c in ["sell", "sell_amount", "sell_volume"] if c in df.columns), None)
    if buy_col is None or sell_col is None:
        raise FinMindError("Institutional dataset missing buy/sell columns")

    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date
    df["name"] = df["name"].astype(str).str.strip()
    df["category"] = df["name"].str.lower().map(CATEGORY_MAP)
    df = df[df["category"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "trading_date"])

    df[buy_col] = pd.to_numeric(df[buy_col], errors="coerce").fillna(0)
    df[sell_col] = pd.to_numeric(df[sell_col], errors="coerce").fillna(0)

    agg = (
        df.groupby(["stock_id", "trading_date", "category"], as_index=False)[[buy_col, sell_col]]
        .sum()
        .rename(columns={buy_col: "buy", sell_col: "sell"})
    )
    agg["net"] = agg["buy"] - agg["sell"]

    base = agg[["stock_id", "trading_date"]].drop_duplicates()
    result = base.copy()

    for category in ["foreign", "trust", "dealer"]:
        sub = agg[agg["category"] == category].copy()
        if sub.empty:
            result[f"{category}_buy"] = 0
            result[f"{category}_sell"] = 0
            result[f"{category}_net"] = 0
            continue
        sub = sub[["stock_id", "trading_date", "buy", "sell", "net"]].rename(
            columns={
                "buy": f"{category}_buy",
                "sell": f"{category}_sell",
                "net": f"{category}_net",
            }
        )
        result = result.merge(sub, on=["stock_id", "trading_date"], how="left")

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
    ]:
        if col not in result.columns:
            result[col] = 0
        result[col] = result[col].fillna(0).astype(int)

    return result[
        [
            "stock_id",
            "trading_date",
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
    ].drop_duplicates(subset=["stock_id", "trading_date"])


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawInstitutional.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def _get_stock_list(token: str | None) -> List[str]:
    """取得股票清單（有快取）"""
    global _stock_list_cache
    if _stock_list_cache is None:
        _stock_list_cache = fetch_stock_list(token)
    return _stock_list_cache


def _fetch_institutional_smart(
    start_date: date,
    end_date: date,
    token: str | None,
    logs: Dict[str, object],
) -> pd.DataFrame:
    """智慧抓取：先嘗試全市場，失敗則改逐檔
    
    Returns:
        合併後的 DataFrame
    """
    # 先嘗試全市場抓取
    df = fetch_dataset(DATASET, start_date, end_date, token=token)
    if not df.empty:
        logs["fetch_mode"] = "bulk"
        return df
    
    # 全市場抓取回傳空值，改用逐檔抓取
    logs["fetch_mode"] = "by_stock"
    stock_ids = _get_stock_list(token)
    logs["stock_list_count"] = len(stock_ids)
    
    if not stock_ids:
        logs["warning"] = "無法取得股票清單，跳過抓取"
        return pd.DataFrame()
    
    return fetch_dataset_by_stocks(
        DATASET,
        start_date,
        end_date,
        stock_ids,
        token=token,
    )


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_institutional")
    logs: Dict[str, object] = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = today - timedelta(days=365 * config.train_lookback_years)
        start_date = _resolve_start_date(db_session, default_start)
        end_date = today
        logs.update({"start_date": start_date.isoformat(), "end_date": end_date.isoformat()})

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "start_date > end_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0, "start_date": start_date, "end_date": end_date}

        total_rows = 0
        chunk_count = 0
        for chunk_start, chunk_end in date_chunks(start_date, end_date, chunk_days=30):
            chunk_count += 1
            chunk_logs: Dict[str, object] = {}
            
            df = _fetch_institutional_smart(chunk_start, chunk_end, config.finmind_token, chunk_logs)
            logs.update({f"chunk_{chunk_count}": chunk_logs})
            
            if df.empty:
                continue
            df = _normalize_institutional(df)
            records: List[Dict] = df.to_dict("records")
            if not records:
                continue

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
            total_rows += len(records)

        logs.update({"rows": total_rows, "chunks": chunk_count})
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows, "start_date": start_date, "end_date": end_date}
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
