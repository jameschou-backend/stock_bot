"""Stage 11.0：Ingest 台指期貨日資料 (TX 大台指近月) 到 raw_futures_daily。

從 FinMind TaiwanFuturesDaily dataset 抓取。預設只 ingest 近月合約
（settlement_price 對應的最近期），避免重複資料。

設計：
  - dataset 同個 trading_date 會有多個 contract（不同月份），我們取近月
    判定方式：contract_date 最接近 trading_date 的（且不過期）
  - 寫入 raw_futures_daily，contract_id="TX" 固定，contract_month 紀錄月份字串

用法：
  from skills.ingest_futures import run
  run(config, session)

或 CLI：
  python -m skills.ingest_futures
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import FinMindError, fetch_dataset
from app.job_utils import finish_job, start_job, update_job
from app.models import RawFuturesDaily

logger = logging.getLogger(__name__)

DATASET = "TaiwanFuturesDaily"
DEFAULT_CONTRACT = "TX"  # 大台指


def _select_near_month(df: pd.DataFrame, trading_date: date) -> pd.DataFrame:
    """從同 trading_date 多 contract 中挑近月（最早未過期）。

    FinMind 同 dataset 每日多筆，columns 含 contract_date='202606' 字串。
    """
    if df.empty or "contract_date" not in df.columns:
        return df.iloc[0:0]
    td_yyyymm = trading_date.strftime("%Y%m")
    # 近月 = 不小於當月的最早合約
    sub = df[df["contract_date"] >= td_yyyymm].copy()
    if sub.empty:
        # fallback：用最晚合約（極端 case）
        sub = df.sort_values("contract_date").tail(1)
    else:
        sub = sub.sort_values("contract_date").head(1)
    return sub


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """重新命名 + 型別轉換 + NaN→None。"""
    if df.empty:
        return df
    df = df.copy()
    df["contract_id"] = DEFAULT_CONTRACT
    df["trading_date"] = pd.to_datetime(df["date"]).dt.date
    df["contract_month"] = df.get("contract_date", "")
    df["open"] = pd.to_numeric(df.get("open"), errors="coerce")
    df["high"] = pd.to_numeric(df.get("max"), errors="coerce")
    df["low"] = pd.to_numeric(df.get("min"), errors="coerce")
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0).astype("int64")
    df["open_interest"] = pd.to_numeric(df.get("open_interest"), errors="coerce").fillna(0).astype("int64")
    df["settlement_price"] = pd.to_numeric(df.get("settlement_price"), errors="coerce")
    keep = ["contract_id", "trading_date", "contract_month", "open", "high",
            "low", "close", "volume", "open_interest", "settlement_price"]
    out = df[keep]
    # NaN → None for MySQL
    out = out.astype(object).where(out.notna(), None)
    return out


def _upsert(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    stmt = insert(RawFuturesDaily).values(records)
    upd = {c.name: stmt.inserted[c.name] for c in RawFuturesDaily.__table__.columns
           if c.name not in ("contract_id", "trading_date")}
    stmt = stmt.on_duplicate_key_update(**upd)
    result = session.execute(stmt)
    session.commit()
    return result.rowcount or len(records)


def run(config, session: Session, days_back: Optional[int] = None) -> dict:
    """從 FinMind 取得 TaiwanFuturesDaily，近月合約寫入 raw_futures_daily。

    Args:
        config: AppConfig
        session: DB session
        days_back: 回補天數（None=自動偵測 DB 最新日期，缺則 730 天）
    """
    job_id = start_job(session, "ingest_futures")
    try:
        # 找 DB 內最新一筆 trading_date
        from sqlalchemy import func, select
        last_date = session.execute(
            select(func.max(RawFuturesDaily.trading_date))
            .where(RawFuturesDaily.contract_id == DEFAULT_CONTRACT)
        ).scalar_one_or_none()

        if last_date is None:
            start_date = date.today() - timedelta(days=days_back or 3650)
            logger.info("[ingest_futures] DB 無資料，從 %s 開始 backfill", start_date)
        else:
            start_date = last_date + timedelta(days=1)
            logger.info("[ingest_futures] DB 最新 %s，從 %s 增量更新", last_date, start_date)

        end_date = date.today()
        if start_date > end_date:
            logger.info("[ingest_futures] 已是最新")
            finish_job(session, job_id, "success", logs={"rows": 0, "skipped": "up-to-date"})
            return {"rows": 0, "status": "up-to-date"}

        df = fetch_dataset(
            DATASET,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        if df.empty:
            finish_job(session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        # 對每個 trading_date 挑近月
        normalized_chunks = []
        for td, group in df.groupby("date"):
            td_date = pd.to_datetime(td).date()
            near = _select_near_month(group, td_date)
            if not near.empty:
                normalized_chunks.append(_normalize(near))

        if not normalized_chunks:
            finish_job(session, job_id, "success", logs={"rows": 0})
            return {"rows": 0}

        all_df = pd.concat(normalized_chunks, ignore_index=True)
        n_rows = _upsert(session, all_df)
        logger.info("[ingest_futures] 寫入 %d 筆", n_rows)
        finish_job(session, job_id, "success", logs={"rows": int(n_rows)})
        return {"rows": int(n_rows)}

    except Exception as exc:
        logger.error("[ingest_futures] 失敗: %s", exc)
        finish_job(session, job_id, "failed", error_text=str(exc))
        raise


if __name__ == "__main__":
    import logging as _lg
    _lg.basicConfig(level=_lg.INFO)
    from app.config import load_config
    from app.db import get_session
    cfg = load_config()
    with get_session() as s:
        result = run(cfg, s)
        print(f"done: {result}")
