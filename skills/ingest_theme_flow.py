"""Ingest Theme Flow 模組

以 stocks.industry_category 作為題材代理，從 raw_prices 聚合每日金流與報酬。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Iterator, List
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import RawPrice, RawThemeFlow, Stock


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawThemeFlow.trading_date)).scalar()
    if max_date is None:
        return default_start
    return max_date + timedelta(days=1)


def _build_theme_flow(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close", "volume", "industry_category"])
    if df.empty:
        return df

    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]
    if df.empty:
        return df

    df = df.sort_values(["stock_id", "trading_date"])
    df["ret_5"] = df.groupby("stock_id")["close"].pct_change(5, fill_method=None)
    df["ret_20"] = df.groupby("stock_id")["close"].pct_change(20, fill_method=None)
    df["turnover_amount"] = df["close"] * df["volume"]

    grouped = (
        df.groupby(["industry_category", "trading_date"], as_index=False)
        .agg(
            turnover_amount=("turnover_amount", "sum"),
            theme_return_5=("ret_5", "mean"),
            theme_return_20=("ret_20", "mean"),
        )
        .rename(columns={"industry_category": "theme_id"})
    )

    total_turnover = grouped.groupby("trading_date")["turnover_amount"].transform("sum")
    grouped["turnover_ratio"] = grouped["turnover_amount"] / total_turnover.replace(0, pd.NA)

    # 熱度分數：成交占比 z-score + 20日報酬 z-score
    grouped["turnover_z"] = grouped.groupby("trading_date")["turnover_ratio"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
    )
    grouped["ret20_z"] = grouped.groupby("trading_date")["theme_return_20"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
    )
    grouped["hot_score"] = 0.6 * grouped["turnover_z"] + 0.4 * grouped["ret20_z"]

    return grouped[
        [
            "theme_id",
            "trading_date",
            "turnover_amount",
            "turnover_ratio",
            "theme_return_5",
            "theme_return_20",
            "hot_score",
        ]
    ].drop_duplicates(subset=["theme_id", "trading_date"])


def _to_mysql_safe_records(df: pd.DataFrame) -> List[Dict]:
    """將 DataFrame 轉為 MySQL 可接受 records（NaN/NaT/inf -> None）。"""
    if df.empty:
        return []
    safe_df = df.astype(object).copy()
    safe_df = safe_df.mask(safe_df.isin([float("inf"), float("-inf")]), pd.NA)
    safe_df = safe_df.where(pd.notna(safe_df), None)
    return safe_df.to_dict("records")


def _chunk_records(records: List[Dict], chunk_size: int) -> Iterator[List[Dict]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    for i in range(0, len(records), chunk_size):
        yield records[i : i + chunk_size]


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_theme_flow", commit=True)
    logs: Dict[str, object] = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = today - timedelta(days=365 * max(config.train_lookback_years, 10))
        start_date = _resolve_start_date(db_session, default_start)
        end_date = today
        logs.update({"start_date": start_date.isoformat(), "end_date": end_date.isoformat()})

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "start_date > end_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        # 為了計算 20 日報酬，向前補資料
        calc_start = start_date - timedelta(days=40)
        stmt = (
            select(
                RawPrice.stock_id,
                RawPrice.trading_date,
                RawPrice.close,
                RawPrice.volume,
                Stock.industry_category,
            )
            .join(Stock, Stock.stock_id == RawPrice.stock_id)
            .where(RawPrice.trading_date.between(calc_start, end_date))
            .where(Stock.is_listed == True)
            .where(Stock.security_type == "stock")
            .order_by(RawPrice.trading_date, RawPrice.stock_id)
        )
        df = pd.read_sql(stmt, db_session.get_bind())
        if df.empty:
            logs["rows"] = 0
            logs["warning"] = "raw_prices/stocks 無可用資料"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        theme_df = _build_theme_flow(df)
        if theme_df.empty:
            logs["rows"] = 0
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        theme_df = theme_df[theme_df["trading_date"] >= start_date]
        records: List[Dict] = _to_mysql_safe_records(theme_df)
        if not records:
            logs["rows"] = 0
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        batch_size = int(kwargs.get("batch_size", 2000))
        batch_count = 0
        for batch in _chunk_records(records, batch_size):
            stmt = insert(RawThemeFlow).values(batch)
            stmt = stmt.on_duplicate_key_update(
                turnover_amount=stmt.inserted.turnover_amount,
                turnover_ratio=stmt.inserted.turnover_ratio,
                theme_return_5=stmt.inserted.theme_return_5,
                theme_return_20=stmt.inserted.theme_return_20,
                hot_score=stmt.inserted.hot_score,
            )
            db_session.execute(stmt)
            db_session.commit()
            batch_count += 1

        logs.update({"rows": len(records), "themes": int(theme_df["theme_id"].nunique())})
        logs["batches"] = batch_count
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": len(records)}
    except Exception as exc:  # pragma: no cover
        # 若前面 SQL 失敗，先 rollback，避免後續寫 jobs 時觸發 invalid transaction。
        try:
            db_session.rollback()
        except Exception:
            pass

        logs["error"] = str(exc)
        try:
            finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        except Exception:
            # 不覆蓋原始例外，保留最初失敗原因給上層處理。
            pass
        raise
