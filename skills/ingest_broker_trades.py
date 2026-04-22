"""Priority 1：分點券商聚合（TaiwanStockTradingDailyReport）

從 FinMind 抓取每日分點進出明細，彙整成每支股票的聚合指標，
寫入 raw_broker_trades 表。

Dataset: TaiwanStockTradingDailyReport
Fields:  date, stock_id, securities_trader_id, securities_trader, buy, sell
頻率：   每日
限制：   Sponsor 計劃；資料量龐大（每日 ~2000 股 × ~500 分點），
         故本模組只存聚合指標，不存原始明細。

聚合指標：
- top5_net:            當日買超量最大前 5 分點的合計淨買超（張）
- top5_concentration:  top5 |淨買超| / 全市場 |淨買超| 比例（0-1）
- buy_broker_count:    當日淨買超分點數
- sell_broker_count:   當日淨賣超分點數
- total_net:           全部分點合計淨買超（張）
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
    date_chunks,
    fetch_dataset_by_stocks,
    fetch_stock_list,
)
from app.job_utils import finish_job, start_job, update_job
from app.models import RawBrokerTrade, Stock

DATASET = "TaiwanStockTradingDailyReport"
UPDATE_COLS = ["top5_net", "top5_concentration", "buy_broker_count", "sell_broker_count", "total_net"]


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawBrokerTrade.trading_date)).scalar()
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


def _aggregate_broker(df: pd.DataFrame, allowed_stock_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """將原始分點明細彙整成每日每股聚合指標。"""
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

    # buy / sell 欄位容錯（FinMind 可能叫 buy / sell 或 Buy / Sell）
    buy_col = next((c for c in ["buy", "Buy", "buy_volume"] if c in df.columns), None)
    sell_col = next((c for c in ["sell", "Sell", "sell_volume"] if c in df.columns), None)
    if buy_col is None or sell_col is None:
        raise FinMindError(f"TaiwanStockTradingDailyReport missing buy/sell columns; got: {df.columns.tolist()}")

    df["buy"] = pd.to_numeric(df[buy_col], errors="coerce").fillna(0)
    df["sell"] = pd.to_numeric(df[sell_col], errors="coerce").fillna(0)
    df["net"] = df["buy"] - df["sell"]

    results = []
    for (stock_id, trading_date), grp in df.groupby(["stock_id", "trading_date"]):
        total_net = int(grp["net"].sum())
        buy_broker_count = int((grp["net"] > 0).sum())
        sell_broker_count = int((grp["net"] < 0).sum())

        # Top-5：取 |net| 最大的 5 個分點
        top5 = grp.nlargest(5, "net")
        top5_net = int(top5["net"].sum())
        total_abs = grp["net"].abs().sum()
        top5_abs = top5["net"].abs().sum()
        top5_concentration = float(top5_abs / total_abs) if total_abs > 0 else 0.0

        results.append({
            "stock_id": stock_id,
            "trading_date": trading_date,
            "top5_net": top5_net,
            "top5_concentration": round(top5_concentration, 6),
            "buy_broker_count": buy_broker_count,
            "sell_broker_count": sell_broker_count,
            "total_net": total_net,
        })

    return pd.DataFrame(results)


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_broker_trades", commit=True)
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

        print(f"[ingest_broker_trades] {start_date} ~ {end_date}", flush=True)

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

        # 分點資料量大，使用較小的 chunk（30 天）避免單次請求超時
        chunk_days = min(getattr(config, "chunk_days", 180), 30)
        chunk_ranges = list(date_chunks(start_date, end_date, chunk_days=chunk_days))
        logs["chunks_total"] = len(chunk_ranges)
        total_rows = 0

        for i, (chunk_start, chunk_end) in enumerate(chunk_ranges, 1):
            update_job(db_session, job_id, logs={**logs, "progress": f"{i}/{len(chunk_ranges)}"}, commit=True)

            df = fetch_dataset_by_stocks(
                DATASET,
                chunk_start,
                chunk_end,
                stock_ids,
                token=config.finmind_token,
                batch_size=200,  # 分點資料每批次少一點，避免逾時
                use_batch_query=True,
                requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
                max_retries=getattr(config, "finmind_retry_max", 3),
                backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
            )
            if df is None or df.empty:
                continue

            agg_df = _aggregate_broker(df, allowed_stock_ids=allowed_stock_ids or None)
            if agg_df.empty:
                continue

            records: List[Dict] = agg_df.to_dict("records")
            stmt = insert(RawBrokerTrade).values(records)
            stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
            db_session.execute(stmt)
            db_session.commit()
            total_rows += len(records)
            print(f"  chunk {i}/{len(chunk_ranges)}: {len(records)} 筆", flush=True)

        logs["rows"] = total_rows
        print(f"  ✅ broker_trades: {total_rows} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
