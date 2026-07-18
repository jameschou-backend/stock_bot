"""Page 8: 個股蠟燭圖 — K 線（raw / 還原價）+ MA + 成交量 + 外資買賣超 + 融資餘額。

- 資料一律讀自家 DB（零 API 額度）；查詢帶 stock_id + 日期範圍（走既有索引）。
- 台股慣例：漲紅跌綠（與美股相反）。
- 「FinMind 補抓」按鈕：僅當該股 DB 最新日期落後今天 > 3 交易日時可按（顯式觸發，
  預設不自動打 API）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.dashboard_v2.helpers import (
    apply_style, render_top_banner,
    get_engine,
    COLOR_ACCENT, COLOR_WARNING, COLOR_GRID, COLOR_TEXT_SECONDARY,
    COLOR_BG, COLOR_PANEL, COLOR_TEXT_PRIMARY,
)
from app.dashboard_v2.info_helpers import (
    fetch_ohlcv_adj, fetch_foreign_net, fetch_margin_balance,
    fetch_stock_name, stock_freshness, finmind_backfill_stock,
)

apply_style()
render_top_banner("🕯️ 個股蠟燭圖", "K 線 / MA5·20·60 / 成交量 / 外資買賣超 / 融資餘額（台股慣例：漲紅跌綠）")

engine = get_engine()

# 台股慣例：漲紅跌綠
UP_COLOR = "#E74C3C"    # 漲 = 紅
DOWN_COLOR = "#2ECC71"  # 跌 = 綠

MAX_RANGE_DAYS = 365 * 3   # 效能上限：單股查詢限 3 年內
MA_WARMUP_CAL_DAYS = 130   # MA60 需要 ~60 交易日暖身（以日曆天抓 130 天緩衝）
BACKFILL_LAG_TRADING_DAYS = 3  # 落後 > 3 交易日才開放 FinMind 補抓

# ── 輸入列 ──
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.6])
with c1:
    sid_raw = st.text_input("股票代號（四碼）", value="2330", max_chars=4)
with c2:
    default_start = date.today() - timedelta(days=365)
    start_input = st.date_input("起日", value=default_start, format="YYYY-MM-DD")
with c3:
    end_input = st.date_input("迄日", value=date.today(), format="YYYY-MM-DD")
with c4:
    price_mode = st.radio(
        "價格口徑", ["還原價（close × 官方 factor）", "raw close（未還原）"],
        horizontal=False,
        help="還原價 = 各 OHLC × price_adjust_factors 官方累積還原因子；"
             "除權息日不會出現假跳空。post-sponsor 期間新除權息事件的 factor 可能未更新。",
    )
use_adj = price_mode.startswith("還原")

sid = sid_raw.strip()
if not re.fullmatch(r"\d{4}", sid):
    st.error("股票代號格式錯誤：只接受四碼台股（例：2330）。")
    st.stop()

if start_input >= end_input:
    st.error("起日必須早於迄日。")
    st.stop()

if (end_input - start_input).days > MAX_RANGE_DAYS:
    start_input = end_input - timedelta(days=MAX_RANGE_DAYS)
    st.warning(f"⚠️ 查詢範圍限 3 年內（效能保護），起日已 clamp 至 {start_input}。")

name = fetch_stock_name(engine, sid)

# ── 資料新鮮度 + FinMind 補抓（顯式觸發；預設不打 API）──
fresh = stock_freshness(engine, sid)
if fresh["stock_max"] is None:
    st.error(f"DB 內無 {sid} 的價格資料（raw_prices）。")
    lag = None
else:
    lag = fresh["lag_trading_days"]

fc1, fc2 = st.columns([3, 1])
with fc1:
    if fresh["stock_max"] is not None:
        icon = "✅" if (lag or 0) <= BACKFILL_LAG_TRADING_DAYS else "⚠️"
        st.caption(f"{icon} DB 最新日期：**{fresh['stock_max']}**（落後今天約 {lag} 交易日）")
with fc2:
    can_backfill = lag is not None and lag > BACKFILL_LAG_TRADING_DAYS
    if st.button(
        "🔄 FinMind 補抓",
        disabled=not can_backfill,
        help=(f"僅當該股 DB 最新日期落後今天 > {BACKFILL_LAG_TRADING_DAYS} 交易日時可按；"
              "點擊才會呼叫 FinMind API 補最近缺日（消耗 API 額度）。"),
    ):
        with st.spinner("FinMind 補抓中..."):
            try:
                res = finmind_backfill_stock(
                    sid, fresh["stock_max"] + timedelta(days=1), date.today())
                if res["rows"] > 0:
                    st.success(
                        f"✅ 寫入 raw_prices {res['rows']} 列"
                        f"（{res['min_date']} ~ {res['max_date']}），重新整理載入新資料。")
                    st.cache_data.clear()
                else:
                    st.info("FinMind 無新資料可補（該區間無交易日或 API 無回傳）。")
            except Exception as exc:  # 顯示失敗原因，不炸頁
                st.error(f"❌ 補抓失敗：{exc}")

if fresh["stock_max"] is None:
    st.stop()

# ── 抓資料（含 MA 暖身段，畫圖前再裁掉）──
fetch_start = start_input - timedelta(days=MA_WARMUP_CAL_DAYS)
px = fetch_ohlcv_adj(engine, sid, fetch_start, end_input)
if px.empty or px[px["trading_date"] >= pd.Timestamp(start_input)].empty:
    st.warning(f"{sid} 在 {start_input} ~ {end_input} 區間無價格資料。")
    st.stop()

# 價格口徑：還原價 = OHLC × factor（乘法調整對 OHLC 一致適用）
for col in ("open", "high", "low", "close"):
    px[f"p_{col}"] = px[col] * px["adj_factor"] if use_adj else px[col]

# MA（在選定口徑上算，含暖身段）
for w in (5, 20, 60):
    px[f"ma_{w}"] = px["p_close"].rolling(w, min_periods=w).mean()

# 裁掉暖身段
df = px[px["trading_date"] >= pd.Timestamp(start_input)].reset_index(drop=True)

inst = fetch_foreign_net(engine, sid, start_input, end_input)
margin = fetch_margin_balance(engine, sid, start_input, end_input)

# ── 標題 KPI ──
last = df.iloc[-1]
prev_close = df["p_close"].iloc[-2] if len(df) > 1 else last["p_close"]
day_chg = float(last["p_close"] / prev_close - 1) if prev_close else 0.0
st.markdown(
    f"### {sid} {name}　"
    f"<span style='color:{UP_COLOR if day_chg >= 0 else DOWN_COLOR};'>"
    f"{float(last['p_close']):,.2f}（{day_chg:+.2%}）</span>　"
    f"<span style='color:{COLOR_TEXT_SECONDARY};font-size:14px;'>"
    f"{last['trading_date'].date()}｜{'還原價' if use_adj else 'raw close'}</span>",
    unsafe_allow_html=True,
)

# ── 非交易日 rangebreaks（週末 + 區間內缺日=假日/停牌）──
present = set(df["trading_date"].dt.date)
all_weekdays = pd.bdate_range(df["trading_date"].min(), df["trading_date"].max())
holidays = [d for d in all_weekdays if d.date() not in present]

# ── 四列子圖：K 線 / 成交量 / 外資買賣超 / 融資餘額 ──
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    row_heights=[0.52, 0.16, 0.16, 0.16], vertical_spacing=0.02,
    subplot_titles=("", "成交量", "外資買賣超（股）", "融資餘額（張）"),
)

fig.add_trace(go.Candlestick(
    x=df["trading_date"],
    open=df["p_open"], high=df["p_high"], low=df["p_low"], close=df["p_close"],
    increasing_line_color=UP_COLOR, increasing_fillcolor=UP_COLOR,
    decreasing_line_color=DOWN_COLOR, decreasing_fillcolor=DOWN_COLOR,
    name="K 線", showlegend=False,
), row=1, col=1)

for w, color in ((5, "#F1C40F"), (20, COLOR_ACCENT), (60, "#9B59B6")):
    fig.add_trace(go.Scatter(
        x=df["trading_date"], y=df[f"ma_{w}"],
        mode="lines", name=f"MA{w}",
        line=dict(color=color, width=1.3),
    ), row=1, col=1)

# 成交量（顏色跟 K 棒同向：收 >= 開 紅、收 < 開 綠）
vol_colors = [UP_COLOR if c >= o else DOWN_COLOR
              for o, c in zip(df["p_open"], df["p_close"])]
fig.add_trace(go.Bar(
    x=df["trading_date"], y=df["volume"],
    marker_color=vol_colors, name="成交量", showlegend=False,
), row=2, col=1)

# 外資買賣超（買超紅 / 賣超綠，台股慣例）
if not inst.empty:
    fn_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in inst["foreign_net"].fillna(0)]
    fig.add_trace(go.Bar(
        x=inst["trading_date"], y=inst["foreign_net"],
        marker_color=fn_colors, name="外資買賣超", showlegend=False,
    ), row=3, col=1)
else:
    fig.add_annotation(text="（無外資買賣超資料）", showarrow=False,
                       xref="x3 domain", yref="y3 domain", x=0.5, y=0.5,
                       font=dict(color=COLOR_TEXT_SECONDARY))

# 融資餘額
if not margin.empty:
    fig.add_trace(go.Scatter(
        x=margin["trading_date"], y=margin["margin_purchase_balance"],
        mode="lines", name="融資餘額",
        line=dict(color=COLOR_WARNING, width=1.5),
        fill="tozeroy", fillcolor="rgba(243,156,18,0.15)",
        showlegend=False,
    ), row=4, col=1)
else:
    fig.add_annotation(text="（無融資融券資料）", showarrow=False,
                       xref="x4 domain", yref="y4 domain", x=0.5, y=0.5,
                       font=dict(color=COLOR_TEXT_SECONDARY))

fig.update_layout(
    paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
    font=dict(color=COLOR_TEXT_PRIMARY, family="Inter, system-ui, sans-serif"),
    hoverlabel=dict(bgcolor=COLOR_PANEL, font_color=COLOR_TEXT_PRIMARY),
    margin=dict(l=40, r=30, t=30, b=30),
    height=820,
    hovermode="x unified",          # hover 統一（四列子圖共用十字）
    xaxis_rangeslider_visible=False,  # rangeslider 關閉
    legend=dict(orientation="h", yanchor="bottom", y=1.01),
    bargap=0.15,
)
fig.update_xaxes(
    gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID,
    rangebreaks=[
        dict(bounds=["sat", "mon"]),            # 週末
        dict(values=[d.isoformat() for d in holidays]),  # 假日 / 停牌缺日
    ],
)
fig.update_yaxes(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "資料來源：自家 DB（raw_prices / price_adjust_factors / raw_institutional / raw_margin_short），"
    "不打外部 API。還原價使用官方累積還原因子（adj = close × factor）；"
    "post-sponsor 期間（2026-06-24 後）的新除權息事件 factor 可能未更新。"
)
