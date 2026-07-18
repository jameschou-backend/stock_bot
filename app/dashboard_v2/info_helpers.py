"""Dashboard v2 資訊瀏覽頁共用 loaders（總覽 / 個股K線 / 研究裁決）。

設計原則：
- 一律讀自家 DB / artifacts（零 API 額度）；唯一例外是個股頁的「FinMind 補抓」
  按鈕（使用者顯式點擊才打 API，見 ``finmind_backfill_stock``）。
- DB 查詢一律帶 stock_id + 日期範圍條件（PK (stock_id, trading_date) 與
  idx_*_trading_date 索引可用），單股查詢限 3 年內（頁面端 clamp）。
- 檔案缺失回 None / 空 DataFrame，由頁面端顯示提示（不 raise 炸頁）。
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

NAV_FILE = PROJECT_ROOT / "artifacts" / "paper_nav" / "nav.jsonl"
IPO_DIR = PROJECT_ROOT / "artifacts" / "ipo_lottery"
DISPOSITION_DIR = PROJECT_ROOT / "artifacts" / "disposition"
REVENUE_LEDGER = PROJECT_ROOT / "artifacts" / "revenue_announcements" / "announcements.parquet"
DOCS_DIR = PROJECT_ROOT / "docs"

# 誠實橫幅（總覽頁頂部 + 各處引用同一句，單一真相源）
HONEST_BANNER_TEXT = (
    "A 線各口徑均無可執行 alpha（v2.2），picks 僅紙上追蹤——詳 docs/prereg_*"
)


# ──────────────────────────────────────────────────────────────
# Paper NAV（artifacts/paper_nav/nav.jsonl）
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_nav_history() -> pd.DataFrame:
    """讀 nav.jsonl → DataFrame(date, nav, holdings_n, config_version, notes)。

    檔案缺失 / 空檔回空 DataFrame。行內 JSON 損毀時跳過該行（不炸頁）。
    """
    if not NAV_FILE.exists():
        return pd.DataFrame()
    rows = []
    for line in NAV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def nav_config_segments(df: pd.DataFrame) -> list[dict]:
    """把 NAV 依 config_version 連續段切分（供分段畫線 + 分段標記）。"""
    if df.empty or "config_version" not in df.columns:
        return []
    segments = []
    seg_id = (df["config_version"] != df["config_version"].shift()).cumsum()
    for _, seg in df.groupby(seg_id):
        segments.append({
            "config_version": str(seg["config_version"].iloc[0]),
            "df": seg,
        })
    return segments


# ──────────────────────────────────────────────────────────────
# IPO 申購掃描（artifacts/ipo_lottery/scan_*.json）
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_latest_ipo_scan() -> Optional[dict]:
    """最新 scan_YYYY-MM-DD.json（含 scan_date / items）。無檔回 None。"""
    files = sorted(IPO_DIR.glob("scan_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def ipo_items_dataframe(scan: dict) -> pd.DataFrame:
    """scan items → 顯示用 DataFrame，折價% 降冪排序。"""
    items = scan.get("items", [])
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df = df.sort_values("discount", ascending=False, na_position="last")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 處置股（artifacts/disposition/YYYY-MM-DD.json）
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_disposition_latest_pair() -> tuple[Optional[dict], Optional[dict]]:
    """回傳（最新, 前一份）處置股快取 JSON；不足兩份時對應位置為 None。"""
    files = sorted(DISPOSITION_DIR.glob("20??-??-??.json"))
    latest = prev = None
    if files:
        try:
            latest = json.loads(files[-1].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            latest = None
    if len(files) >= 2:
        try:
            prev = json.loads(files[-2].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prev = None
    return latest, prev


def disposition_name_map(dispo: dict) -> dict[str, str]:
    """records → {stock_id: name}（僅四碼股票）。"""
    out: dict[str, str] = {}
    for rec in dispo.get("records", []):
        sid = str(rec.get("stock_id", ""))
        if re.fullmatch(r"\d{4}", sid):
            out.setdefault(sid, str(rec.get("name", "")))
    return out


# ──────────────────────────────────────────────────────────────
# 月營收 ledger（artifacts/revenue_announcements/announcements.parquet）
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def revenue_ledger_stats() -> Optional[dict]:
    """月營收公告 ledger 統計（累積筆數 / 最新公告日）。無檔回 None。"""
    if not REVENUE_LEDGER.exists():
        return None
    try:
        df = pd.read_parquet(REVENUE_LEDGER)
    except Exception:
        return None
    if df.empty:
        return {"rows": 0, "max_announcement": None, "stocks": 0}
    return {
        "rows": int(len(df)),
        "max_announcement": str(pd.to_datetime(df["announcement_date"]).max().date()),
        "stocks": int(df["stock_id"].nunique()),
    }


# ──────────────────────────────────────────────────────────────
# Jobs（系統狀態）
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_recent_jobs(_engine, limit: int = 15) -> pd.DataFrame:
    q = text("""
        SELECT job_name, status, started_at, ended_at, error_text
        FROM jobs
        ORDER BY started_at DESC
        LIMIT :n
    """)
    return pd.read_sql(q, _engine, params={"n": limit})


# ──────────────────────────────────────────────────────────────
# 個股 K 線資料（一律帶 stock_id + 日期範圍條件，走既有索引）
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_ohlcv_adj(_engine, stock_id: str, start: date, end: date) -> pd.DataFrame:
    """單股 OHLCV + 官方還原 factor（LEFT JOIN price_adjust_factors，缺值=1.0）。"""
    q = text("""
        SELECT p.trading_date, p.open, p.high, p.low, p.close, p.volume,
               COALESCE(f.adj_factor, 1.0) AS adj_factor
        FROM raw_prices p
        LEFT JOIN price_adjust_factors f
          ON f.stock_id = p.stock_id AND f.trading_date = p.trading_date
        WHERE p.stock_id = :sid
          AND p.trading_date BETWEEN :start AND :end
        ORDER BY p.trading_date
    """)
    df = pd.read_sql(q, _engine, params={"sid": stock_id, "start": start, "end": end})
    if df.empty:
        return df
    for col in ("open", "high", "low", "close", "adj_factor"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    return df


@st.cache_data(ttl=300)
def fetch_foreign_net(_engine, stock_id: str, start: date, end: date) -> pd.DataFrame:
    q = text("""
        SELECT trading_date, foreign_net
        FROM raw_institutional
        WHERE stock_id = :sid AND trading_date BETWEEN :start AND :end
        ORDER BY trading_date
    """)
    df = pd.read_sql(q, _engine, params={"sid": stock_id, "start": start, "end": end})
    if not df.empty:
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        df["foreign_net"] = pd.to_numeric(df["foreign_net"], errors="coerce")
    return df


@st.cache_data(ttl=300)
def fetch_margin_balance(_engine, stock_id: str, start: date, end: date) -> pd.DataFrame:
    q = text("""
        SELECT trading_date, margin_purchase_balance, short_sale_balance
        FROM raw_margin_short
        WHERE stock_id = :sid AND trading_date BETWEEN :start AND :end
        ORDER BY trading_date
    """)
    df = pd.read_sql(q, _engine, params={"sid": stock_id, "start": start, "end": end})
    if not df.empty:
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        for col in ("margin_purchase_balance", "short_sale_balance"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=120)
def fetch_stock_name(_engine, stock_id: str) -> str:
    q = text("SELECT name FROM stocks WHERE stock_id = :sid")
    row = pd.read_sql(q, _engine, params={"sid": stock_id})
    return str(row["name"].iloc[0]) if not row.empty else ""


@st.cache_data(ttl=120)
def stock_freshness(_engine, stock_id: str) -> dict:
    """該股 DB 最新日期，與落後今天的「交易日」數。

    交易日數 = 全市場 distinct trading_date > 該股最新日（DB 內精確）
             + DB 全市場最新日之後到今天的平日數（DB 外近似，假日視為交易日、偏保守）。
    """
    with _engine.connect() as conn:
        stock_max = conn.execute(
            text("SELECT MAX(trading_date) FROM raw_prices WHERE stock_id = :sid"),
            {"sid": stock_id},
        ).scalar()
        market_max = conn.execute(
            text("SELECT MAX(trading_date) FROM raw_prices")
        ).scalar()
        if stock_max is None:
            return {"stock_max": None, "lag_trading_days": None}
        n_after = conn.execute(
            text("SELECT COUNT(DISTINCT trading_date) FROM raw_prices WHERE trading_date > :d"),
            {"d": stock_max},
        ).scalar() or 0
    # DB 之外（market_max, today] 的平日數
    extra = 0
    if market_max is not None:
        d = market_max + timedelta(days=1)
        today = date.today()
        while d <= today:
            if d.weekday() < 5:
                extra += 1
            d += timedelta(days=1)
    return {"stock_max": stock_max, "lag_trading_days": int(n_after) + extra}


def finmind_backfill_stock(stock_id: str, start: date, end: date) -> dict:
    """FinMind 單股補抓 raw_prices 缺日（僅供頁面按鈕顯式觸發，預設不自動打 API）。

    FINMIND_TOKEN 從 env / .env 讀（load_config），不可硬編碼。
    寫入沿用 ingest_prices 的 normalize + on_duplicate_key_update upsert。
    """
    from app.config import load_config
    from app.db import get_session
    from app.finmind import fetch_dataset
    from app.models import RawPrice
    from skills.ingest_prices import _normalize_prices
    from sqlalchemy.dialects.mysql import insert

    cfg = load_config()
    if not cfg.finmind_token:
        raise RuntimeError("FINMIND_TOKEN 未設定（.env），無法補抓")

    raw = fetch_dataset(
        "TaiwanStockPrice", start, end,
        token=cfg.finmind_token, data_id=stock_id,
        requests_per_hour=cfg.finmind_requests_per_hour,
        max_retries=cfg.finmind_retry_max,
        backoff_seconds=cfg.finmind_retry_backoff,
    )
    if raw.empty:
        return {"rows": 0, "min_date": None, "max_date": None}

    df = _normalize_prices(raw)
    if df.empty:
        return {"rows": 0, "min_date": None, "max_date": None}
    # NaN → None（MySQL 不接受 NaN）
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    records = df.to_dict("records")

    with get_session() as session:
        stmt = insert(RawPrice).values(records)
        update_cols = {c: stmt.inserted[c] for c in ["open", "high", "low", "close", "volume"]}
        stmt = stmt.on_duplicate_key_update(**update_cols)
        session.execute(stmt)
        session.commit()

    return {
        "rows": len(records),
        "min_date": str(min(df["trading_date"])),
        "max_date": str(max(df["trading_date"])),
    }


# ──────────────────────────────────────────────────────────────
# 研究裁決（docs/prereg_*.md）
# ──────────────────────────────────────────────────────────────

def list_prereg_docs() -> list[dict]:
    """掃 docs/prereg_*.md，依檔名日期新→舊排序。"""
    out = []
    for p in sorted(DOCS_DIR.glob("prereg_*.md")):
        m = re.search(r"(\d{8})", p.name)
        doc_date = None
        if m:
            try:
                doc_date = datetime.strptime(m.group(1), "%Y%m%d").date()
            except ValueError:
                doc_date = None
        out.append({"path": p, "date": doc_date, "name": p.name})
    return sorted(out, key=lambda d: (d["date"] or date.min), reverse=True)


# 裁決摘要（硬編碼自 memory/decisions.md 已知結果；新增裁決時同步維護）
PREREG_VERDICTS = [
    {
        "臂": "PEAD rank/winsorize（Arm B 零股）",
        "日期": "2026-07-11",
        "裁決": "FAIL",
        "關鍵數字": "Arm B Sharpe -0.89 / MDD -84.8%（雙觸發）；gross +168pp 首見組合層超額但 DSR p 0.107",
        "文件": "prereg_pead_rank_arm_20260711.md",
    },
    {
        "臂": "PEAD 月營收事件臂（Arm A naive top-N）",
        "日期": "2026-07-11",
        "裁決": "FAIL",
        "關鍵數字": "超額 -163pp；超額 Sharpe -0.54，95% CI [-1.619, +0.289] 下界 < 0",
        "文件": "prereg_pead_arm_20260711.md",
    },
    {
        "臂": "零股（odd-lot）口徑三臂",
        "日期": "2026-07-11",
        "裁決": "FAIL",
        "關鍵數字": "悲觀臂 Sharpe 0.083 / MDD -58.4%（雙觸發）；全窗 excess CI 上界為負",
        "文件": "prereg_odd_lot_arm_20260711.md",
    },
    {
        "臂": "personal-baseline 三臂（個人可執行口徑）",
        "日期": "2026-07-10",
        "裁決": "FAIL",
        "關鍵數字": "悲觀臂 Sharpe 0.305 < 0.50；個人口徑超額 -82.5pp",
        "文件": "prereg_personal_baseline_20260710.md",
    },
    {
        "臂": "Strategy D 修復後 10y 重驗（誠實時序 --signal-lag 1）",
        "日期": "2026-07-10",
        "裁決": "FAIL",
        "關鍵數字": "Sharpe 0.952 但 MDD -60.9% 觸發；兩臂差揭露 T 日盤後籌碼 lookahead 值 123× 累積",
        "文件": "prereg_d_revalidation_20260710.md",
    },
]
