"""Page 4: System Health — 資料覆蓋率、pipeline status、模型版本。"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from app.dashboard_v2.helpers import (
    apply_style, render_top_banner, render_kpi, get_engine,
    COLOR_ACCENT, COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING,
    COLOR_TEXT_PRIMARY, COLOR_GRID, COLOR_PANEL, PLOTLY_LAYOUT,
)


apply_style()
render_top_banner("⚕️ System Health", "Data coverage · Pipeline status · Model versions")

engine = get_engine()


@st.cache_data(ttl=60)
def fetch_coverage(_engine, days: int = 30) -> pd.DataFrame:
    q = text(f"""
        SELECT
          'raw_prices' AS dataset, trading_date, COUNT(DISTINCT stock_id) AS n
        FROM raw_prices
        WHERE trading_date >= DATE_SUB(CURDATE(), INTERVAL {days} DAY)
        GROUP BY trading_date
        UNION ALL
        SELECT 'raw_institutional', trading_date, COUNT(DISTINCT stock_id)
        FROM raw_institutional
        WHERE trading_date >= DATE_SUB(CURDATE(), INTERVAL {days} DAY)
        GROUP BY trading_date
        UNION ALL
        SELECT 'raw_margin_short', trading_date, COUNT(DISTINCT stock_id)
        FROM raw_margin_short
        WHERE trading_date >= DATE_SUB(CURDATE(), INTERVAL {days} DAY)
        GROUP BY trading_date
        UNION ALL
        SELECT 'features', trading_date, COUNT(DISTINCT stock_id)
        FROM features
        WHERE trading_date >= DATE_SUB(CURDATE(), INTERVAL {days} DAY)
        GROUP BY trading_date
        ORDER BY trading_date DESC
    """)
    return pd.read_sql(q, _engine)


@st.cache_data(ttl=60)
def fetch_recent_jobs(_engine, limit: int = 30) -> pd.DataFrame:
    q = text(f"""
        SELECT id, name, status, started_at, finished_at,
               TIMESTAMPDIFF(SECOND, started_at, finished_at) AS duration_sec
        FROM jobs
        ORDER BY id DESC LIMIT {limit}
    """)
    return pd.read_sql(q, _engine)


@st.cache_data(ttl=60)
def fetch_sponsor_coverage(_engine) -> pd.DataFrame:
    """檢查 sponsor datasets 是否有資料。"""
    tables = [
        ("raw_holding_dist", "trading_date"),
        ("raw_broker_trades", "trading_date"),
        ("raw_kbar_daily", "trading_date"),
        ("raw_gov_bank", "trading_date"),
        ("raw_fear_greed", "date"),
        ("raw_per", "trading_date"),
        ("raw_securities_lending", "trading_date"),
        ("raw_quarterly_fundamental", "report_date"),
        ("raw_futures_daily", "trading_date"),
        ("raw_futures_inst", "trading_date"),
    ]
    rows = []
    for t, dcol in tables:
        try:
            q = text(f"SELECT COUNT(*) AS n, MAX({dcol}) AS latest FROM {t}")
            r = pd.read_sql(q, _engine).iloc[0]
            rows.append({
                "Dataset": t,
                "總筆數": int(r["n"]) if pd.notna(r["n"]) else 0,
                "最新日期": r["latest"] or "—",
                "狀態": "✅" if (r["n"] and r["n"] > 0) else "❌",
            })
        except Exception:
            rows.append({"Dataset": t, "總筆數": 0, "最新日期": "—", "狀態": "❌"})
    return pd.DataFrame(rows)


# ── Core datasets coverage ──
st.markdown('<div class="section-header">📦 核心 datasets coverage（近 30 天）</div>',
            unsafe_allow_html=True)
cov = fetch_coverage(engine)

if not cov.empty:
    pivot = cov.pivot_table(index="trading_date", columns="dataset", values="n", aggfunc="sum").fillna(0)
    pivot = pivot.tail(30)

    fig = go.Figure()
    for col, color in zip(pivot.columns, [COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER]):
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col], name=col,
            line=dict(color=color, width=2),
            mode="lines+markers",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        title="每日股票覆蓋數",
        xaxis_title="Date", yaxis_title="股票數",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Sponsor datasets coverage ──
st.markdown('<div class="section-header">⭐ Sponsor / 進階資料集</div>',
            unsafe_allow_html=True)
sponsor_df = fetch_sponsor_coverage(engine)
st.dataframe(sponsor_df, use_container_width=True, hide_index=True)

# ── Recent jobs ──
st.markdown('<div class="section-header">⚙️ 最近 30 個 jobs</div>',
            unsafe_allow_html=True)
jobs = fetch_recent_jobs(engine)
if not jobs.empty:
    # 狀態統計
    status_count = jobs["status"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("成功", str(status_count.get("success", 0)), flavor="success")
    with c2:
        render_kpi("失敗", str(status_count.get("failed", 0)),
                   flavor="danger" if status_count.get("failed", 0) > 0 else "default")
    with c3:
        render_kpi("執行中", str(status_count.get("running", 0)),
                   flavor="warning" if status_count.get("running", 0) > 0 else "default")
    with c4:
        avg_dur = jobs["duration_sec"].dropna().mean() if not jobs["duration_sec"].isna().all() else 0
        render_kpi("平均耗時", f"{avg_dur:.0f}s")

    # 顯示 table
    display = jobs.copy()
    display["duration_sec"] = display["duration_sec"].fillna(0).astype(int)
    st.dataframe(display, use_container_width=True, hide_index=True, height=400)
