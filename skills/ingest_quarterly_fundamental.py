"""Priority 8：季報財務摘要（多 dataset 聚合）

從 FinMind 抓取三個季報 dataset，彙整成關鍵財務指標，
寫入 raw_quarterly_fundamental 表。

來源 dataset：
  TaiwanStockBalanceSheet     — 資產負債表（負債/資產 → debt_ratio，ROE）
  TaiwanStockFinancialStatements — 損益表（營業利益率、稅後淨利率）
  TaiwanStockCashFlowsStatement  — 現金流量表（FCF/股）

Fields（原始）：
  date, stock_id, type, value   ← 長格式，需 pivot

時間延遲：
  季報在季底後約 60 天公告（3 月底 → 5 月中）。
  build_features.py 使用 available_date = report_date + 60d 做 merge_asof，
  避免前向洩漏（與月營收 45 天延遲同樣機制）。
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
from app.models import RawQuarterlyFundamental, Stock

# 資產負債表欄位：type 對應的 value 欄位名稱（FinMind 原始文字）
BS_ROW_MAP = {
    "TotalAssets": "total_assets",
    "TotalLiabilities": "total_liabilities",
    "EquityAttributableToOwnersOfParent": "equity",
    "RetainedEarnings": "retained_earnings",
}
IS_ROW_MAP = {
    "OperatingIncome": "operating_income",
    "NetIncome": "net_income",
    "Revenue": "revenue",
}
CF_ROW_MAP = {
    "CashFlowsFromOperatingActivities": "cfo",
    "PurchaseOfPropertyPlantAndEquipment": "capex",
}

UPDATE_COLS = ["roe", "roa", "debt_ratio", "operating_margin", "net_margin", "fcf_per_share"]
PUBLICATION_DELAY_DAYS = 60   # 季報公告延遲（天）
CHUNK_DAYS = 365              # 每次查詢跨度（天）


def _resolve_start_date(session: Session, default_start: date) -> date:
    max_date = session.query(func.max(RawQuarterlyFundamental.report_date)).scalar()
    if max_date is None:
        return default_start
    # 往前 90 天重算以補齊最新季報
    return max(default_start, max_date - timedelta(days=90))


def _load_stock_ids(session: Session) -> List[str]:
    rows = (
        session.query(Stock.stock_id)
        .filter(Stock.is_listed == True)
        .filter(Stock.security_type == "stock")
        .order_by(Stock.stock_id)
        .all()
    )
    return [row[0] for row in rows]


def _pivot_long(df: pd.DataFrame, row_map: Dict[str, str]) -> pd.DataFrame:
    """將 FinMind 長格式財報（date, stock_id, type, value）pivot 成寬格式。"""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df.rename(columns={"date": "report_date"})
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    df["stock_id"] = df["stock_id"].astype(str)
    df = df.dropna(subset=["stock_id", "report_date"])
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]

    type_col = next((c for c in ["type", "Type", "statement_type"] if c in df.columns), None)
    val_col = next((c for c in ["value", "Value", "amount"] if c in df.columns), None)
    if type_col is None or val_col is None:
        return pd.DataFrame()

    df["value_n"] = pd.to_numeric(df[val_col], errors="coerce")
    df_filtered = df[df[type_col].isin(row_map.keys())].copy()
    df_filtered["field"] = df_filtered[type_col].map(row_map)

    wide = (
        df_filtered.groupby(["stock_id", "report_date", "field"])["value_n"]
        .mean()
        .unstack("field")
        .reset_index()
    )
    return wide


def _compute_metrics(bs: pd.DataFrame, is_: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    """合併三張報表並計算關鍵財務指標。"""
    if bs.empty and is_.empty and cf.empty:
        return pd.DataFrame()

    key_cols = ["stock_id", "report_date"]

    # 合併三表（inner join on stock_id + report_date）
    merged = bs.copy() if not bs.empty else pd.DataFrame()
    if not is_.empty:
        if merged.empty:
            merged = is_.copy()
        else:
            merged = merged.merge(is_, on=key_cols, how="outer")
    if not cf.empty:
        if merged.empty:
            merged = cf.copy()
        else:
            merged = merged.merge(cf, on=key_cols, how="outer")

    if merged.empty:
        return pd.DataFrame()

    def safe(col):
        return merged[col] if col in merged.columns else pd.Series([None] * len(merged), dtype=float)

    total_assets = pd.to_numeric(safe("total_assets"), errors="coerce")
    total_liabs = pd.to_numeric(safe("total_liabilities"), errors="coerce")
    equity = pd.to_numeric(safe("equity"), errors="coerce")
    revenue = pd.to_numeric(safe("revenue"), errors="coerce")
    op_income = pd.to_numeric(safe("operating_income"), errors="coerce")
    net_income = pd.to_numeric(safe("net_income"), errors="coerce")
    cfo = pd.to_numeric(safe("cfo"), errors="coerce")
    capex = pd.to_numeric(safe("capex"), errors="coerce").abs()  # capex 原始為負值

    # ── 財務指標計算 ──
    merged["roe"] = net_income / equity.replace(0, float("nan")) * 100
    merged["roa"] = net_income / total_assets.replace(0, float("nan")) * 100
    merged["debt_ratio"] = total_liabs / total_assets.replace(0, float("nan")) * 100
    merged["operating_margin"] = op_income / revenue.replace(0, float("nan")) * 100
    merged["net_margin"] = net_income / revenue.replace(0, float("nan")) * 100

    # FCF / 股（若無股數資訊，用絕對值 /1000 估計；後續可接 shares outstanding）
    fcf = cfo - capex
    # FinMind 財務數字單位為「千元」，除以 1000 轉成「百萬元」，再 /1000 為每千股（張）
    # 簡化：保留原始 FCF 值（千元）供後續相對比較
    merged["fcf_per_share"] = fcf / 1000.0  # 轉換成百萬元，相對指標

    result = merged[["stock_id", "report_date",
                     "roe", "roa", "debt_ratio", "operating_margin", "net_margin", "fcf_per_share"]].copy()
    # 合理性 clip（避免極端值）
    result["roe"] = result["roe"].clip(-500, 500)
    result["roa"] = result["roa"].clip(-200, 200)
    result["debt_ratio"] = result["debt_ratio"].clip(0, 200)
    result["operating_margin"] = result["operating_margin"].clip(-200, 200)
    result["net_margin"] = result["net_margin"].clip(-200, 200)

    return result.dropna(subset=["stock_id", "report_date"])


def _fetch_one_dataset(dataset: str, stock_id: str, start: date, end: date, config) -> pd.DataFrame:
    """抓取單一 dataset（容錯，失敗回傳空 DataFrame）"""
    try:
        return fetch_dataset(
            dataset=dataset,
            start_date=start,
            end_date=end,
            token=config.finmind_token,
            data_id=stock_id,
            requests_per_hour=getattr(config, "finmind_requests_per_hour", 600),
            max_retries=getattr(config, "finmind_retry_max", 3),
            backoff_seconds=getattr(config, "finmind_retry_backoff", 5),
            timeout=120,
        ) or pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def run(config, db_session: Session, **kwargs) -> Dict:
    job_id = start_job(db_session, "ingest_quarterly_fundamental", commit=True)
    logs: Dict = {}
    try:
        today = datetime.now(ZoneInfo(config.tz)).date()
        default_start = max(
            today - timedelta(days=365 * config.train_lookback_years),
            date(2013, 1, 1),   # FinMind 季報從 2013 年起完整
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
        print(f"[ingest_quarterly_fundamental] {start_date} ~ {end_date}，{len(stock_ids)} 檔", flush=True)

        total_rows = 0
        commit_buffer: List[Dict] = []

        for i, stock_id in enumerate(stock_ids, 1):
            if i % 100 == 0:
                update_job(
                    db_session, job_id,
                    logs={**logs, "progress": f"{i}/{len(stock_ids)}", "rows": total_rows},
                    commit=True,
                )
                print(f"  [{i}/{len(stock_ids)}] rows={total_rows}", flush=True)

            # 每檔股票一次抓全期（季報資料量小，一年只有 4 筆）
            bs_raw = _fetch_one_dataset("TaiwanStockBalanceSheet", stock_id, start_date, end_date, config)
            is_raw = _fetch_one_dataset("TaiwanStockFinancialStatements", stock_id, start_date, end_date, config)
            cf_raw = _fetch_one_dataset("TaiwanStockCashFlowsStatement", stock_id, start_date, end_date, config)

            bs = _pivot_long(bs_raw, BS_ROW_MAP)
            is_ = _pivot_long(is_raw, IS_ROW_MAP)
            cf = _pivot_long(cf_raw, CF_ROW_MAP)

            metrics = _compute_metrics(bs, is_, cf)
            if metrics.empty:
                continue

            commit_buffer.extend(metrics.to_dict("records"))

            if len(commit_buffer) >= 2000:
                _flush(db_session, commit_buffer)
                total_rows += len(commit_buffer)
                commit_buffer.clear()

        if commit_buffer:
            _flush(db_session, commit_buffer)
            total_rows += len(commit_buffer)

        logs["rows"] = total_rows
        print(f"  ✅ ingest_quarterly_fundamental: {total_rows} 筆", flush=True)
        finish_job(db_session, job_id, "success", logs=logs)
        return {"rows": total_rows}

    except Exception as exc:
        db_session.rollback()
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise


def _flush(session: Session, buffer: List[Dict]) -> None:
    stmt = insert(RawQuarterlyFundamental).values(buffer)
    stmt = stmt.on_duplicate_key_update({col: stmt.inserted[col] for col in UPDATE_COLS})
    session.execute(stmt)
    session.commit()
