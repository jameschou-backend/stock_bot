"""Page 6: D Replay — Strategy D 回測持倉時間軸視覺化。

支援：
- Gantt-style 持倉甘特圖（每檔股票進出場區間）
- 月度持倉表（每月持有的 stock_id 清單）
- 完整 trades log 表格（可排序/篩選）
- 出場原因分布 + 各原因平均報酬
- 累積報酬 vs Drawdown 曲線
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import text

from app.dashboard_v2.helpers import (
    apply_style, render_top_banner, render_kpi, get_engine,
    COLOR_ACCENT, COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_GRID, COLOR_PANEL,
    PLOTLY_LAYOUT,
)


apply_style()
render_top_banner("📅 Strategy D Replay", "回測持倉時間軸 · 換股軌跡 · 出場分析")

# ── 1. 選 backtest 檔 ──
D_REPLAY_DIR = PROJECT_ROOT / "artifacts" / "d_replay"
if not D_REPLAY_DIR.exists():
    st.error(f"找不到 {D_REPLAY_DIR}，請先跑 D backtest")
    st.stop()

json_files = sorted(D_REPLAY_DIR.glob("*.json"))
if not json_files:
    st.error("artifacts/d_replay/ 內無 json 檔")
    st.stop()


def _label(p: Path) -> str:
    return p.stem


selected = st.sidebar.selectbox(
    "選擇 backtest",
    options=json_files,
    format_func=_label,
    index=len(json_files) - 1,
)


@st.cache_data(ttl=300)
def _load(path: str) -> dict:
    return json.load(open(path))


data = _load(str(selected))
trades = data.get("trades_log", [])
equity_pts = data.get("equity_curve", [])
summary = data.get("summary", {})

if not trades:
    st.warning("此 backtest 無 trades_log")
    st.stop()

# ── stock_id → name lookup ──
engine = get_engine()


@st.cache_data(ttl=3600)
def _name_map(_engine) -> dict:
    q = text("SELECT stock_id, name, industry_category FROM stocks")
    df = pd.read_sql(q, _engine)
    return {
        str(r["stock_id"]): {"name": r["name"] or "", "industry": r["industry_category"] or ""}
        for _, r in df.iterrows()
    }


name_map = _name_map(engine)

# 建 trades DataFrame
trades_df = pd.DataFrame(trades)
trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
trades_df["stock_id"] = trades_df["stock_id"].astype(str)
trades_df["name"] = trades_df["stock_id"].map(lambda s: name_map.get(s, {}).get("name", ""))
trades_df["industry"] = trades_df["stock_id"].map(lambda s: name_map.get(s, {}).get("industry", ""))
trades_df["label"] = trades_df["stock_id"] + " " + trades_df["name"]
trades_df["return_pct"] = trades_df["realized_pnl_pct"].astype(float)
trades_df["is_win"] = trades_df["return_pct"] > 0


# ── 2. KPI Cards ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("累積報酬", f"{summary.get('total_return', 0)*100:+.1f}%",
               f"年化 {summary.get('annualized_return', 0)*100:+.1f}%",
               flavor="success" if summary.get('total_return', 0) > 0 else "danger")
with c2:
    render_kpi("Sharpe", f"{summary.get('sharpe_ratio', 0):.2f}",
               f"Calmar {summary.get('calmar_ratio', 0):.2f}")
with c3:
    render_kpi("MDD", f"{summary.get('max_drawdown', 0)*100:.1f}%",
               f"Win {summary.get('win_rate', 0)*100:.0f}%",
               flavor="danger")
with c4:
    render_kpi("總交易數", f"{len(trades_df)}",
               f"平均持有 {summary.get('avg_hold_days', 0):.1f}d")


# ── 3. 持倉甘特圖（Gantt）──
st.markdown('<div class="section-header">📊 持倉甘特圖（每檔股票進出場區間）</div>', unsafe_allow_html=True)

# 過濾器
fcol1, fcol2 = st.columns([2, 3])
with fcol1:
    min_ret = st.slider("只看單筆報酬 ≥", -50, 100, -50, step=5,
                        help="-50 = 全部；+30 = 只看 +30% 以上贏家") / 100
with fcol2:
    exit_filter = st.multiselect(
        "出場原因 filter",
        options=sorted(trades_df["exit_reason"].unique().tolist()),
        default=sorted(trades_df["exit_reason"].unique().tolist()),
    )

filtered = trades_df[
    (trades_df["return_pct"] >= min_ret) & (trades_df["exit_reason"].isin(exit_filter))
].copy()

if filtered.empty:
    st.info("無符合 filter 條件的交易")
else:
    # 按 entry_date 排序 + y-axis labels（按進場時間順序，由上到下）
    filtered = filtered.sort_values("entry_date").reset_index(drop=True)
    filtered["y_idx"] = filtered.index

    fig = go.Figure()

    # 每筆交易一條 bar
    for _, t in filtered.iterrows():
        color = COLOR_SUCCESS if t["return_pct"] > 0 else COLOR_DANGER
        hover = (
            f"<b>{t['stock_id']} {t['name']}</b><br>"
            f"進場：{t['entry_date'].date()} @ ${t['entry_price']:.2f}<br>"
            f"出場：{t['exit_date'].date()} @ ${t['exit_price']:.2f}<br>"
            f"持有：{t['days_held']}d · 報酬 {t['return_pct']*100:+.1f}%<br>"
            f"原因：{t['exit_reason']} · Score {t['score']:.3f}"
        )
        fig.add_trace(go.Scatter(
            x=[t["entry_date"], t["exit_date"]],
            y=[t["y_idx"], t["y_idx"]],
            mode="lines+markers",
            line=dict(color=color, width=6),
            marker=dict(size=8, color=color,
                        line=dict(color=COLOR_TEXT_PRIMARY, width=1)),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))
        # 末端報酬 label
        fig.add_annotation(
            x=t["exit_date"], y=t["y_idx"],
            text=f" {t['return_pct']*100:+.0f}%",
            showarrow=False,
            font=dict(color=color, size=10),
            xanchor="left",
        )

    # 持倉 label（y-axis）— 不用 **PLOTLY_LAYOUT，避免 yaxis/xaxis kwarg 衝突
    n_show = min(len(filtered), 80)  # 超過 80 筆只顯示部分 tick label，避免擠
    tick_step = max(1, len(filtered) // n_show)
    fig.update_layout(
        paper_bgcolor=PLOTLY_LAYOUT["paper_bgcolor"],
        plot_bgcolor=PLOTLY_LAYOUT["plot_bgcolor"],
        font=PLOTLY_LAYOUT["font"],
        margin=PLOTLY_LAYOUT["margin"],
        hoverlabel=PLOTLY_LAYOUT["hoverlabel"],
        height=max(400, min(1200, 20 * len(filtered) + 100)),
        title=f"{len(filtered)}/{len(trades_df)} 筆交易",
        yaxis=dict(
            tickvals=list(range(0, len(filtered), tick_step)),
            ticktext=[filtered.iloc[i]["label"] for i in range(0, len(filtered), tick_step)],
            showgrid=True,
            gridcolor=COLOR_GRID,
            autorange="reversed",
        ),
        xaxis=dict(showgrid=True, gridcolor=COLOR_GRID),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 4. 月度持倉變化 ──
st.markdown('<div class="section-header">📅 月度持倉變化（哪幾檔在 portfolio）</div>', unsafe_allow_html=True)

# 每月月初檢視持倉
months = pd.period_range(
    trades_df["entry_date"].min(),
    trades_df["exit_date"].max(),
    freq="M",
)
monthly_data = []
for m in months:
    m_start = m.start_time
    held = trades_df[(trades_df["entry_date"] <= m_start) & (trades_df["exit_date"] >= m_start)]
    held_list = [f"{r['stock_id']} {r['name']}" for _, r in held.iterrows()]
    monthly_data.append({
        "月份": str(m),
        "持倉數": len(held),
        "持倉清單": " / ".join(held_list) if held_list else "—",
    })

monthly_df = pd.DataFrame(monthly_data)
st.dataframe(monthly_df, use_container_width=True, height=400, hide_index=True)


# ── 5. 出場原因分布 + 各原因平均報酬 ──
st.markdown('<div class="section-header">🚪 出場原因分析</div>', unsafe_allow_html=True)

exit_stats = trades_df.groupby("exit_reason").agg(
    次數=("return_pct", "count"),
    平均報酬=("return_pct", "mean"),
    中位數=("return_pct", "median"),
    勝率=("is_win", "mean"),
    平均持有=("days_held", "mean"),
).round(4).reset_index()
exit_stats["平均報酬"] = exit_stats["平均報酬"].map(lambda x: f"{x*100:+.2f}%")
exit_stats["中位數"] = exit_stats["中位數"].map(lambda x: f"{x*100:+.2f}%")
exit_stats["勝率"] = exit_stats["勝率"].map(lambda x: f"{x*100:.0f}%")
exit_stats["平均持有"] = exit_stats["平均持有"].map(lambda x: f"{x:.1f}d")
exit_stats = exit_stats.sort_values("次數", ascending=False)

ecol1, ecol2 = st.columns([2, 3])
with ecol1:
    st.dataframe(exit_stats, use_container_width=True, hide_index=True)

with ecol2:
    # 各出場原因 box plot
    fig = go.Figure()
    for reason in trades_df["exit_reason"].unique():
        rets = trades_df[trades_df["exit_reason"] == reason]["return_pct"] * 100
        fig.add_trace(go.Box(
            y=rets,
            name=reason,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(size=5),
        ))
    fig.update_layout(
        paper_bgcolor=PLOTLY_LAYOUT["paper_bgcolor"],
        plot_bgcolor=PLOTLY_LAYOUT["plot_bgcolor"],
        font=PLOTLY_LAYOUT["font"],
        margin=PLOTLY_LAYOUT["margin"],
        hoverlabel=PLOTLY_LAYOUT["hoverlabel"],
        xaxis=PLOTLY_LAYOUT["xaxis"],
        yaxis=dict(
            **PLOTLY_LAYOUT["yaxis"],
            title="單筆報酬 (%)",
        ),
        height=400,
        title="各出場原因 報酬分布（box + scatter）",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 6. 累積報酬 + Drawdown 曲線 ──
st.markdown('<div class="section-header">📈 累積報酬 + Drawdown</div>', unsafe_allow_html=True)

if equity_pts:
    eq_df = pd.DataFrame(equity_pts)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df["equity"] = eq_df["equity"].astype(float)
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] / eq_df["peak"] - 1) * 100  # %

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("累積資產", "Drawdown (%)"),
    )
    fig.add_trace(
        go.Scatter(x=eq_df["date"], y=eq_df["equity"],
                   line=dict(color=COLOR_ACCENT, width=2), name="Equity"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=eq_df["date"], y=eq_df["dd"],
                   fill="tozeroy", line=dict(color=COLOR_DANGER, width=1),
                   fillcolor="rgba(231,76,60,0.3)", name="DD"),
        row=2, col=1,
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=550, showlegend=False)
    fig.update_yaxes(gridcolor=COLOR_GRID, row=1, col=1)
    fig.update_yaxes(gridcolor=COLOR_GRID, row=2, col=1)
    fig.update_xaxes(gridcolor=COLOR_GRID, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ── 7. 完整 Trades Log（可排序/篩選）──
st.markdown('<div class="section-header">📋 完整 Trades Log</div>', unsafe_allow_html=True)

display_df = trades_df[[
    "stock_id", "name", "industry", "entry_date", "exit_date",
    "entry_price", "exit_price", "return_pct", "days_held",
    "exit_reason", "score",
]].copy()
display_df["entry_date"] = display_df["entry_date"].dt.strftime("%Y-%m-%d")
display_df["exit_date"] = display_df["exit_date"].dt.strftime("%Y-%m-%d")
display_df["entry_price"] = display_df["entry_price"].map(lambda x: f"${x:,.2f}")
display_df["exit_price"] = display_df["exit_price"].map(lambda x: f"${x:,.2f}")
display_df["return_pct"] = display_df["return_pct"].map(lambda x: f"{x*100:+.2f}%")
display_df["score"] = display_df["score"].map(lambda x: f"{x:.4f}")
display_df = display_df.rename(columns={
    "stock_id": "代碼", "name": "名稱", "industry": "產業",
    "entry_date": "進場日", "exit_date": "出場日",
    "entry_price": "進場價", "exit_price": "出場價",
    "return_pct": "報酬", "days_held": "持有天",
    "exit_reason": "出場原因", "score": "Score",
})
st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)
