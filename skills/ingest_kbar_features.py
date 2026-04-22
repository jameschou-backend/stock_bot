"""Priority 3：分鐘K線日內特徵（TaiwanStockKBar）

從 FinMind 抓取每日分鐘級 K 線資料，計算四個日內特徵，
寫入 raw_kbar_daily 表。原始分鐘資料不存入 DB（資料量過大）。

Dataset: TaiwanStockKBar
Fields:  date, stock_id, Time, Open, High, Low, Close, Volume
頻率：   每日
限制：   Sponsor 計劃；台股交易時間 09:00-13:30

日內特徵（4 個）：
- morning_ret:       09:01-09:30 開盤後半小時報酬（= close_0930 / open_0901 - 1）
- close_vol_ratio:   尾盤 30 分鐘成交量 / 全日總量（= sum(vol[13:01-13:30]) / sum(vol_all)）
- intraday_high_pos: 收盤位置（= (close - low_all) / (high_all - low_all)）
- vwap_dev:          收盤偏離 VWAP 程度（= (close_last - vwap) / vwap）
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
    fetch_dataset_by_stocks,
    fetch_stock_list,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawKBarDaily, Stock

DATASET = "TaiwanStockKBar"
UPDATE_COLS = ["morning_ret", "close_vol_ratio", "intraday_high_pos", "vwap_dev"]

# 台股開收盤時間
MORNING_START = "09:01"
MORNING_END   = "09:30"
CLOSE_START   = "13:01"
CLOSE_END     = "13:30"


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawKBarDaily.trading_date)).scalar()
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


def _compute_kbar_features(df: pd.DataFrame, allowed_stock_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """從分鐘 K 線計算四個日內特徵。"""
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
    time_col  = next((c for c in ["Time", "time", "datetime"] if c in df.columns), None)
    open_col  = next((c for c in ["Open", "open"] if c in df.columns), None)
    high_col  = next((c for c in ["High", "high"] if c in df.columns), None)
    low_col   = next((c for c in ["Low", "low"] if c in df.columns), None)
    close_col = next((c for c in ["Close", "close"] if c in df.columns), None)
    vol_col   = next((c for c in ["Volume", "volume", "vol"] if c in df.columns), None)

    missing = [name for name, col in [
        ("Time", time_col), ("Open", open_col), ("High", high_col),
        ("Low", low_col), ("Close", close_col), ("Volume", vol_col)
    ] if col is None]
    if missing:
        raise FinMindError(f"TaiwanStockKBar missing columns: {missing}; got: {df.columns.tolist()}")

    # 標準化欄位名稱
    df = df.rename(columns={
        time_col: "t", open_col: "open", high_col: "high",
        low_col: "low", close_col: "close", vol_col: "volume",
    })
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 從 Time 欄位提取 HH:MM 格式
    df["t_str"] = df["t"].astype(str).str.extract(r"(\d{2}:\d{2})")[0].fillna("")

    results = []
    for (stock_id, trading_date), grp in df.groupby(["stock_id", "trading_date"]):
        total_vol = grp["volume"].sum()
        if total_vol <= 0:
            continue

        # 全日最高/最低（用於 intraday_high_pos）
        day_high = grp["high"].max()
        day_low  = grp["low"].min()
        last_close = grp.sort_values("t_str").iloc[-1]["close"]

        # morning_ret: 09:01-09:30 的報酬
        morning = grp[(grp["t_str"] >= MORNING_START) & (grp["t_str"] <= MORNING_END)].sort_values("t_str")
        if len(morning) >= 2:
            morning_ret = float(morning.iloc[-1]["close"]) / float(morning.iloc[0]["open"]) - 1 \
                if morning.iloc[0]["open"] > 0 else 0.0
        elif len(morning) == 1:
            morning_ret = 0.0
        else:
            morning_ret = 0.0

        # close_vol_ratio: 尾盤成交量佔比
        close_vol = grp[(grp["t_str"] >= CLOSE_START) & (grp["t_str"] <= CLOSE_END)]["volume"].sum()
        close_vol_ratio = float(close_vol) / float(total_vol) if total_vol > 0 else 0.0

        # intraday_high_pos: 收盤在全日振幅中的位置
        day_range = day_high - day_low
        if day_range > 0:
            intraday_high_pos = float(last_close - day_low) / float(day_range)
        else:
            intraday_high_pos = 0.5  # 無振幅時置中

        # vwap_dev: 收盤偏離 VWAP
        # VWAP = Σ(close * volume) / Σ(volume)（以 close 近似 typical price）
        vwap = float((grp["close"] * grp["volume"]).sum()) / float(total_vol)
        vwap_dev = float(last_close - vwap) / float(vwap) if vwap > 0 else 0.0

        results.append({
            "stock_id": stock_id,
            "trading_date": trading_date,
            "morning_ret": round(morning_ret, 6),
            "close_vol_ratio": round(close_vol_ratio, 6),
            "intraday_high_pos": round(intraday_high_pos, 6),
            "vwap_dev": round(vwap_dev, 6),
        })

    return pd.DataFrame(results)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_kbar_features", commit=True)
    logs: Dict = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        end_date = today

        # TaiwanStockKBar 限制：
        #   - 不支援 batch data_id（逗號分隔多股）
        #   - 不支援日期範圍（每次只能查單日，否則 "size too large"）
        #   - 每日增量設計：預設只抓最近 2 天，避免歷史回補耗盡 API 配額
        #     （5年歷史 ≈ 2000股 × 1250天 = 250萬 API calls，不切實際）
        default_start = today - timedelta(days=2)
        max_db = db_session.query(func.max(RawKBarDaily.trading_date)).scalar()
        start_date = (max_db + timedelta(days=1)) if max_db is not None else default_start

        logs["start_date"] = start_date.isoformat()
        logs["end_date"]   = end_date.isoformat()

        if start_date > end_date:
            logs["rows"] = 0
            logs["skip_reason"] = "already_up_to_date"
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        print(f"[ingest_kbar_features] {start_date} ~ {end_date}", flush=True)

        allowed_stock_ids = _load_allowed_stock_ids(db_session)
        logs["allowed_stock_ids"] = len(allowed_stock_ids)

        stock_ids = fetch_stock_list(
            config.finmind_token,
            requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
            max_retries=getattr(config, "finmind_retry_max", 3),
            backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
        )
        if not stock_ids:
            logs["warning"] = "無法取得股票清單，跳過抓取"
            logs["rows"] = 0
            finish_job(db_session, job_id, "success", logs=logs)
            return {"rows": 0}

        # 逐日查詢（每日一次），每日再逐股查詢
        all_dates = [
            start_date + timedelta(days=d)
            for d in range((end_date - start_date).days + 1)
            if (start_date + timedelta(days=d)).weekday() < 5  # 跳週末
        ]
        logs["days_total"] = len(all_dates)
        logs["fetch_mode"] = "per_day_per_stock"
        total_rows = 0

        for day_idx, query_date in enumerate(all_dates, 1):
            print(f"  [{day_idx}/{len(all_dates)}] {query_date}", flush=True)
            update_job(db_session, job_id, logs={**logs, "progress": f"{day_idx}/{len(all_dates)}", "rows": total_rows}, commit=True)

            # 逐股查詢（不支援 batch），start=end 同一天
            df = fetch_dataset_by_stocks(
                DATASET,
                query_date,
                query_date,
                stock_ids,
                token=config.finmind_token,
                batch_size=1,
                use_batch_query=False,
                requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
                max_retries=getattr(config, "finmind_retry_max", 3),
                backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
            )
            if df is None or df.empty:
                continue

            feat_df = _compute_kbar_features(df, allowed_stock_ids=allowed_stock_ids or None)
            if feat_df.empty:
                continue

            records: List[Dict] = feat_df.to_dict("records")
            stmt = insert(RawKBarDaily).values(records)
            stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
            db_session.execute(stmt)
            db_session.commit()
            total_rows += len(records)
            print(f"    → {len(records)} 筆寫入", flush=True)

        logs["rows"] = total_rows
        print(f"  ✅ kbar_features: {total_rows} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
