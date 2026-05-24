"""Dashboard v2 共用 data loaders + plot helpers + 樣式。"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────────────────────────────────────────
# 顏色系統（暗色 + 高 contrast accents）
# ──────────────────────────────────────────────────────────────
COLOR_BG = "#0E1117"
COLOR_PANEL = "#1A1F2E"
COLOR_ACCENT = "#1ABC9C"  # teal
COLOR_DANGER = "#E74C3C"  # red
COLOR_WARNING = "#F39C12"  # orange
COLOR_SUCCESS = "#2ECC71"  # green
COLOR_TEXT_PRIMARY = "#FAFAFA"
COLOR_TEXT_SECONDARY = "#A0A8B0"
COLOR_GRID = "#2A3142"


PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLOR_BG,
    plot_bgcolor=COLOR_BG,
    font=dict(color=COLOR_TEXT_PRIMARY, family="Inter, system-ui, sans-serif"),
    xaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
    yaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
    margin=dict(l=30, r=30, t=40, b=30),
    hoverlabel=dict(bgcolor=COLOR_PANEL, font_color=COLOR_TEXT_PRIMARY),
)


CUSTOM_CSS = f"""
<style>
    /* 全頁深色 */
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT_PRIMARY};
    }}

    /* Hero KPI cards */
    .kpi-card {{
        background: linear-gradient(135deg, {COLOR_PANEL} 0%, #232938 100%);
        padding: 18px 20px;
        border-radius: 12px;
        border-left: 4px solid {COLOR_ACCENT};
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .kpi-card.danger {{ border-left-color: {COLOR_DANGER}; }}
    .kpi-card.warning {{ border-left-color: {COLOR_WARNING}; }}
    .kpi-card.success {{ border-left-color: {COLOR_SUCCESS}; }}
    .kpi-label {{
        font-size: 13px;
        color: {COLOR_TEXT_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .kpi-value {{
        font-size: 32px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        margin: 4px 0;
    }}
    .kpi-delta {{
        font-size: 14px;
        color: {COLOR_ACCENT};
    }}
    .kpi-delta.negative {{ color: {COLOR_DANGER}; }}

    /* Pick card */
    .pick-card {{
        background: {COLOR_PANEL};
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid {COLOR_GRID};
    }}
    .pick-card .rank {{
        font-size: 11px;
        color: {COLOR_TEXT_SECONDARY};
    }}
    .pick-card .stock-id {{
        font-size: 18px;
        font-weight: 700;
        color: {COLOR_ACCENT};
    }}
    .pick-card .stock-name {{
        font-size: 13px;
        color: {COLOR_TEXT_SECONDARY};
    }}
    .pick-card .score {{
        font-size: 13px;
        color: {COLOR_TEXT_PRIMARY};
    }}

    /* Regime tag */
    .regime-tag {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }}
    .regime-bull {{ background: {COLOR_SUCCESS}; color: #000; }}
    .regime-bear {{ background: {COLOR_DANGER}; color: white; }}
    .regime-sideways {{ background: {COLOR_WARNING}; color: #000; }}

    /* Section header */
    .section-header {{
        font-size: 22px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid {COLOR_ACCENT};
    }}

    /* Header banner */
    .top-banner {{
        background: linear-gradient(90deg, {COLOR_PANEL} 0%, #1F2738 100%);
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .banner-title {{
        font-size: 24px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        margin: 0;
    }}
    .banner-subtitle {{
        font-size: 13px;
        color: {COLOR_TEXT_SECONDARY};
        margin: 4px 0 0;
    }}

    /* Hide streamlit default header */
    [data-testid="stHeader"] {{ background: transparent; }}
</style>
"""


def apply_style():
    """套用 v2 dashboard 樣式（每頁 page 起手呼叫）。"""
    st.set_page_config(
        page_title="台股 ML Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_top_banner(title: str, subtitle: str = ""):
    st.markdown(
        f"""<div class="top-banner">
            <div>
                <p class="banner-title">{title}</p>
                <p class="banner-subtitle">{subtitle}</p>
            </div>
            <div style="text-align:right;">
                <p class="banner-subtitle">📅 {date.today().isoformat()}</p>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, delta: str = "", flavor: str = "default"):
    """flavor: default / danger / warning / success。"""
    klass = f"kpi-card {flavor}" if flavor != "default" else "kpi-card"
    delta_klass = "kpi-delta negative" if delta.startswith("-") else "kpi-delta"
    delta_html = f'<div class="{delta_klass}">{delta}</div>' if delta else ""
    st.markdown(
        f"""<div class="{klass}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


def render_regime_tag(regime: str):
    klass = {"bull": "regime-bull", "bear": "regime-bear", "sideways": "regime-sideways"}.get(
        regime.lower(), "regime-sideways"
    )
    icon = {"bull": "🟢", "bear": "🔴", "sideways": "🟡"}.get(regime.lower(), "🟡")
    label = {"bull": "BULL", "bear": "BEAR", "sideways": "SIDEWAYS"}.get(regime.lower(), regime.upper())
    return f'<span class="regime-tag {klass}">{icon} {label}</span>'


# ──────────────────────────────────────────────────────────────
# Data loaders（cached for speed，跨 page 共用）
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_engine():
    """單一 engine 跨 page 共用。"""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.db import get_engine as _get
    return _get()


@st.cache_data(ttl=60)
def fetch_latest_picks(_engine) -> pd.DataFrame:
    """取最新 pick_date 的 picks。"""
    q = text("""
        SELECT p.stock_id, p.score, p.pick_date, s.name, s.industry_category, s.market
        FROM picks p
        LEFT JOIN stocks s ON s.stock_id = p.stock_id
        WHERE p.pick_date = (SELECT MAX(pick_date) FROM picks)
        ORDER BY p.score DESC
    """)
    return pd.read_sql(q, _engine)


@st.cache_data(ttl=300)
def fetch_recent_prices(_engine, stock_ids: tuple, days: int = 5) -> pd.DataFrame:
    """取 stock_ids 最近 N 日 close。"""
    if not stock_ids:
        return pd.DataFrame()
    placeholders = ",".join([f"'{s}'" for s in stock_ids])
    q = text(f"""
        SELECT stock_id, trading_date, close, volume
        FROM raw_prices
        WHERE stock_id IN ({placeholders})
          AND trading_date >= DATE_SUB(CURDATE(), INTERVAL {days} DAY)
        ORDER BY stock_id, trading_date DESC
    """)
    return pd.read_sql(q, _engine)


@st.cache_data(ttl=300)
def fetch_latest_price_per_stock(_engine, stock_ids: tuple) -> dict:
    """取 stock_ids 最新一筆 close（map: stock_id → close）。"""
    if not stock_ids:
        return {}
    placeholders = ",".join([f"'{s}'" for s in stock_ids])
    q = text(f"""
        SELECT stock_id, close FROM raw_prices
        WHERE (stock_id, trading_date) IN (
          SELECT stock_id, MAX(trading_date) FROM raw_prices
          WHERE stock_id IN ({placeholders})
          GROUP BY stock_id
        )
    """)
    df = pd.read_sql(q, _engine)
    return dict(zip(df["stock_id"].astype(str), df["close"].astype(float)))


@st.cache_data(ttl=600)
def fetch_market_regime(_engine) -> dict:
    """取近期大盤狀態（等權市場指數、200ma、20d vol）。"""
    q = text("""
        SELECT trading_date, AVG(close) AS mkt_close
        FROM raw_prices
        WHERE trading_date >= DATE_SUB(CURDATE(), INTERVAL 250 DAY)
          AND stock_id REGEXP '^[0-9]{4}$'
        GROUP BY trading_date
        ORDER BY trading_date
    """)
    df = pd.read_sql(q, _engine)
    if df.empty:
        return {"regime": "unknown", "trend_20": None, "trend_60": None,
                "above_200ma": None, "vol_20": None}
    s = df["mkt_close"].astype(float)
    last = s.iloc[-1]
    ma200 = s.rolling(200, min_periods=40).mean().iloc[-1]
    trend_20 = float(s.iloc[-1] / s.iloc[-21] - 1) if len(s) > 21 else None
    trend_60 = float(s.iloc[-1] / s.iloc[-61] - 1) if len(s) > 61 else None
    above_200ma = bool(last > ma200) if not pd.isna(ma200) else None
    rets = s.pct_change().dropna()
    vol_20 = float(rets.iloc[-20:].std() * np.sqrt(252)) if len(rets) >= 20 else None

    # regime classification
    if above_200ma and trend_60 is not None and trend_60 > 0:
        regime = "bull"
    elif (above_200ma is False) and trend_60 is not None and trend_60 < -0.05:
        regime = "bear"
    else:
        regime = "sideways"
    return {
        "regime": regime,
        "trend_20": trend_20, "trend_60": trend_60,
        "above_200ma": above_200ma, "vol_20": vol_20,
        "last_close": float(last),
    }


@st.cache_data(ttl=120)
def load_backtest_result(json_path: str) -> Optional[dict]:
    p = Path(json_path)
    if not p.exists():
        return None
    return json.load(open(p))


def list_backtest_artifacts() -> list:
    """列出 artifacts/optuna_10y/ artifacts/stage10_* 等 reference backtest results。"""
    paths = []
    for sub in ["optuna_10y", "stage10_10y", "stage10_d1", "stage10_d2"]:
        d = PROJECT_ROOT / "artifacts" / sub
        if d.exists():
            for j in sorted(d.glob("*.json")):
                paths.append(str(j))
    return paths


def equity_curve_from_periods(periods: list) -> pd.DataFrame:
    """從 periods 算 cum equity + 大盤 cum。"""
    if not periods:
        return pd.DataFrame()
    rows = []
    s_eq, b_eq = 1.0, 1.0
    for p in periods:
        r = p.get("return")
        br = p.get("benchmark_return")
        if r is None:
            continue
        s_eq *= (1 + float(r))
        if br is not None:
            b_eq *= (1 + float(br))
        rb = p.get("rebalance_date") or p.get("date")
        rows.append({
            "date": pd.to_datetime(rb),
            "strategy": s_eq,
            "benchmark": b_eq,
        })
    return pd.DataFrame(rows).set_index("date")
