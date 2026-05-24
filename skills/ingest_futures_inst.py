"""Stage 11.0：Ingest 三大法人期貨持倉到 raw_futures_inst。

從 FinMind TaiwanFuturesInstitutionalInvestors dataset 抓取。
取 contract_id="TX" 大台指期貨對應的三大法人多空淨額。

三大法人期貨持倉是領先指標：機構部位反映「對下個 5-10 個交易日的預期」，
比現貨買賣超更前瞻。
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.orm import Session

from app.finmind import fetch_dataset
from app.job_utils import finish_job, start_job
from app.models import RawFuturesInst

logger = logging.getLogger(__name__)

DATASET = "TaiwanFuturesInstitutionalInvestors"
DEFAULT_CONTRACT = "TX"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """FinMind 三大法人格式：
    columns: date, institutional_investors (外資/投信/自營商), long_open_interest_volume,
             short_open_interest_volume, futures_id, ...
    需要 pivot：每 trading_date × 機構 → 多/空 OI → 拼成單一 row
    """
    if df.empty:
        return df
    # 只取 TX 大台指
    df = df[df["futures_id"] == DEFAULT_CONTRACT].copy()
    if df.empty:
        return df

    # 機構名稱對應（FinMind 中文或英文皆可能）
    inst_map = {
        "外資": "foreign", "外資自營商": "foreign",
        "投信": "trust",
        "自營商": "dealer", "自營商(避險)": "dealer", "自營商(自行買賣)": "dealer",
        "foreign": "foreign", "investment_trust": "trust", "dealer": "dealer",
    }
    df["inst_norm"] = df["institutional_investors"].map(inst_map)
    df = df.dropna(subset=["inst_norm"])

    # 同 date × inst_norm 加總（避免 dealer 避險/自行買賣兩 row）
    grouped = df.groupby(["date", "inst_norm"], as_index=False).agg({
        "long_open_interest_volume": "sum",
        "short_open_interest_volume": "sum",
    })
    grouped["net"] = grouped["long_open_interest_volume"] - grouped["short_open_interest_volume"]

    # pivot：每 date 一 row，columns = foreign_long/short/net + trust_* + dealer_*
    pivot = grouped.pivot_table(
        index="date", columns="inst_norm",
        values=["long_open_interest_volume", "short_open_interest_volume", "net"],
        aggfunc="sum",
    )
    # 重新命名 MultiIndex columns: (value_name, inst_norm) → "inst_long_oi" 等
    new_cols = []
    for kind, inst in pivot.columns:
        if kind == "long_open_interest_volume":
            new_cols.append(f"{inst}_long_oi")
        elif kind == "short_open_interest_volume":
            new_cols.append(f"{inst}_short_oi")
        elif kind == "net":
            new_cols.append(f"{inst}_net_oi")
        else:
            new_cols.append(f"{inst}_{kind}")
    pivot.columns = new_cols
    pivot = pivot.reset_index()

    pivot["contract_id"] = DEFAULT_CONTRACT
    pivot["trading_date"] = pd.to_datetime(pivot["date"]).dt.date

    # 確保所有 9 個 oi columns 存在（缺則填 0）
    keep_cols = [
        "foreign_long_oi", "foreign_short_oi", "foreign_net_oi",
        "trust_long_oi", "trust_short_oi", "trust_net_oi",
        "dealer_long_oi", "dealer_short_oi", "dealer_net_oi",
    ]
    for c in keep_cols:
        if c not in pivot.columns:
            pivot[c] = 0
        pivot[c] = pd.to_numeric(pivot[c], errors="coerce").fillna(0).astype("int64")

    final_cols = ["contract_id", "trading_date"] + keep_cols
    out = pivot[final_cols]
    out = out.astype(object).where(out.notna(), None)
    return out


def _upsert(session: Session, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = df.to_dict(orient="records")
    stmt = insert(RawFuturesInst).values(records)
    upd = {c.name: stmt.inserted[c.name] for c in RawFuturesInst.__table__.columns
           if c.name not in ("contract_id", "trading_date")}
    stmt = stmt.on_duplicate_key_update(**upd)
    result = session.execute(stmt)
    session.commit()
    return result.rowcount or len(records)


def run(config, session: Session, days_back: Optional[int] = None) -> dict:
    job_id = start_job(session, "ingest_futures_inst")
    try:
        last_date = session.execute(
            select(func.max(RawFuturesInst.trading_date))
            .where(RawFuturesInst.contract_id == DEFAULT_CONTRACT)
        ).scalar_one_or_none()

        if last_date is None:
            start_date = date.today() - timedelta(days=days_back or 3650)
        else:
            start_date = last_date + timedelta(days=1)

        end_date = date.today()
        if start_date > end_date:
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

        norm = _normalize(df)
        n_rows = _upsert(session, norm)
        logger.info("[ingest_futures_inst] 寫入 %d 筆", n_rows)
        finish_job(session, job_id, "success", logs={"rows": int(n_rows)})
        return {"rows": int(n_rows)}

    except Exception as exc:
        logger.error("[ingest_futures_inst] 失敗: %s", exc)
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
