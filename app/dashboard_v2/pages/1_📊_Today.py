"""Page 1: Today — 投資者每日操作視圖。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.dashboard_v2.helpers import (
    apply_style, render_top_banner, render_kpi, render_regime_tag,
    get_engine, fetch_latest_picks, fetch_market_regime,
    fetch_latest_price_per_stock,
    COLOR_ACCENT, COLOR_DANGER, COLOR_SUCCESS, COLOR_GRID,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_PANEL, PLOTLY_LAYOUT,
)
from app.dashboard_v2.portfolio import load_portfolio, compute_pnl, compute_picks_alignment


apply_style()

engine = get_engine()
picks_df = fetch_latest_picks(engine)
regime = fetch_market_regime(engine)

pick_date = picks_df["pick_date"].iloc[0] if not picks_df.empty else "N/A"
render_top_banner(
    "📊 Today's Operations",
    f"Pick date: {pick_date} · 大盤 regime: {regime['regime'].upper()}",
)

# ── 大盤 regime banner ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">市場狀態</div>
            <div style="margin:12px 0;">{render_regime_tag(regime['regime'])}</div>
        </div>""", unsafe_allow_html=True,
    )
with c2:
    render_kpi(
        "大盤近 20 日報酬",
        f"{regime['trend_20']:+.2%}" if regime['trend_20'] is not None else "N/A",
        flavor="success" if regime['trend_20'] and regime['trend_20'] > 0 else "danger",
    )
with c3:
    render_kpi(
        "大盤近 60 日報酬",
        f"{regime['trend_60']:+.2%}" if regime['trend_60'] is not None else "N/A",
    )
with c4:
    render_kpi(
        "大盤 20 日波動率",
        f"{regime['vol_20']:.1%}" if regime['vol_20'] is not None else "N/A",
        flavor="warning" if regime['vol_20'] and regime['vol_20'] > 0.25 else "default",
    )

# ── Today's picks ──
st.markdown('<div class="section-header">🎯 Today\'s 30 Picks</div>', unsafe_allow_html=True)

if picks_df.empty:
    st.warning("尚無 picks 資料，請先跑 `make pipeline`")
else:
    # 取最新報價算每檔建議部位
    stock_ids = tuple(picks_df["stock_id"].astype(str).tolist())
    price_map = fetch_latest_price_per_stock(engine, stock_ids)

    # 100 萬資金，30 檔等權，每檔 3.33 萬
    capital = st.sidebar.number_input("總資金 (萬)", min_value=10, max_value=10000, value=100, step=10)
    capital_per_stock = capital * 10000 / len(picks_df)
    st.sidebar.caption(f"每檔配額: ${capital_per_stock:,.0f}")
    st.sidebar.caption(f"基於 {len(picks_df)} 檔等權")

    rows = []
    for _, p in picks_df.iterrows():
        sid = str(p["stock_id"])
        px = price_map.get(sid)
        shares = int(capital_per_stock / px / 1000) * 1000 if px else 0
        actual_cost = shares * px if px else 0
        rows.append({
            "排名": "",
            "代碼": sid,
            "名稱": p.get("name") or "",
            "產業": p.get("industry_category") or "",
            "Score": f"{p['score']:.4f}",
            "現價": f"${px:,.2f}" if px else "N/A",
            "建議股數": f"{shares:,}" if shares else "N/A",
            "金額": f"${actual_cost:,.0f}" if actual_cost else "N/A",
        })
    out_df = pd.DataFrame(rows)
    out_df.index = range(1, len(out_df) + 1)
    out_df["排名"] = out_df.index

    st.dataframe(
        out_df,
        use_container_width=True,
        height=600,
        column_config={
            "排名": st.column_config.NumberColumn(width="small"),
            "代碼": st.column_config.TextColumn(width="small"),
            "名稱": st.column_config.TextColumn(width="medium"),
            "產業": st.column_config.TextColumn(width="medium"),
        },
    )

    # ── 產業分布 ──
    st.markdown('<div class="section-header">🏭 產業分布</div>', unsafe_allow_html=True)
    ind_count = picks_df["industry_category"].value_counts().head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ind_count.values, y=ind_count.index,
        orientation="h", marker_color=COLOR_ACCENT,
        text=ind_count.values, textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        title="Top 10 產業（檔數）",
        showlegend=False,
        xaxis_title="檔數", yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── 我的持倉 (portfolio.json) ──
st.markdown('<div class="section-header">💼 我的持倉</div>', unsafe_allow_html=True)
portfolio = load_portfolio()
if portfolio is None:
    st.info(
        "未設定 portfolio.json。複製 portfolio.example.json → portfolio.json 並編輯。\n\n"
        "```bash\ncp portfolio.example.json portfolio.json\nvim portfolio.json\n```"
    )
else:
    # 取所有持倉的最新價
    hold_ids = tuple(str(k) for k in portfolio.get("positions", {}).keys())
    if hold_ids:
        hold_prices = fetch_latest_price_per_stock(engine, hold_ids)
        result = compute_pnl(portfolio, hold_prices)
        totals = result["totals"]

        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi("總資產", f"${totals['total_assets']:,.0f}",
                       flavor="success")
        with c2:
            render_kpi("持倉市值", f"${totals['total_market_value']:,.0f}",
                       f"成本 ${totals['total_cost']:,.0f}")
        with c3:
            pnl = totals["unreal_pnl"]
            pnl_pct = totals["unreal_pnl_pct"]
            render_kpi(
                "未實現損益", f"${pnl:+,.0f}",
                f"{pnl_pct:+.2%}",
                flavor="success" if pnl >= 0 else "danger",
            )
        with c4:
            render_kpi("現金", f"${totals['cash']:,.0f}",
                       f"持倉 {totals['n_positions']} 檔")

        # 持倉 table
        pos_df = result["positions"]
        if not pos_df.empty:
            pos_df["建議"] = pos_df["stock_id"].apply(
                lambda s: "✅ 持續" if s in picks_df["stock_id"].astype(str).tolist() else "❌ 該賣"
            )
            display = pos_df[["stock_id", "shares", "entry_price",
                              "current_price", "unreal_pnl", "return_pct",
                              "entry_date", "建議"]].copy()
            display["entry_price"] = display["entry_price"].map(lambda x: f"${x:,.2f}")
            display["current_price"] = display["current_price"].map(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
            display["unreal_pnl"] = display["unreal_pnl"].map(lambda x: f"${x:+,.0f}")
            display["return_pct"] = display["return_pct"].map(lambda x: f"{x:+.2%}")
            st.dataframe(display, use_container_width=True, hide_index=True)

            # 持倉 vs picks 對齊
            alignment = compute_picks_alignment(portfolio, picks_df)
            ca1, ca2, ca3 = st.columns(3)
            with ca1:
                render_kpi("符合今日 picks", str(len(alignment["in_picks"])),
                           f"alignment {alignment['alignment_pct']:.0%}",
                           flavor="success")
            with ca2:
                render_kpi("該賣（不在 picks）", str(len(alignment["missing"])),
                           "下個再平衡日換出",
                           flavor="warning" if alignment["missing"] else "default")
            with ca3:
                render_kpi("該買（picks 中未持有）", str(len(alignment["new_picks"])),
                           "下個再平衡日買入",
                           flavor="warning" if alignment["new_picks"] else "default")

            if alignment["missing"]:
                st.warning(f"**🔴 該賣**：{', '.join(alignment['missing'][:10])}{'...' if len(alignment['missing']) > 10 else ''}")
            if alignment["new_picks"]:
                st.info(f"**🟢 該買**：{', '.join(alignment['new_picks'][:10])}{'...' if len(alignment['new_picks']) > 10 else ''}")
    else:
        st.info("portfolio.json 內無 positions")

# ── 操作提醒 ──
st.markdown('<div class="section-header">📋 今日操作提醒</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    st.info(
        "**🟢 進場時機**\n\n"
        "月初再平衡日當日**收盤**等權買入 top 30。\n\n"
        f"建議下單時間：每月第 1 個交易日 13:25 之前。"
    )
with col_b:
    st.warning(
        "**🔴 出場時機**\n\n"
        "下個再平衡日全平倉換股（持有約 20 個交易日）。\n\n"
        "**無停損**（歷史驗證 -7% 停損反而 cum 退到 +69%）。"
    )
