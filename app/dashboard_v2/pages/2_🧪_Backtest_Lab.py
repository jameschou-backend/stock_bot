"""Page 2: Backtest Lab — 回測結果視覺化。"""
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
    apply_style, render_top_banner, render_kpi,
    list_backtest_artifacts, load_backtest_result, equity_curve_from_periods,
    COLOR_ACCENT, COLOR_DANGER, COLOR_SUCCESS, COLOR_WARNING,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_GRID, PLOTLY_LAYOUT,
)


apply_style()
render_top_banner("🧪 Backtest Lab", "歷史回測結果視覺化 · 跨實驗對照")

# ── Selector ──
artifacts = list_backtest_artifacts()
if not artifacts:
    st.warning("無 backtest result，請先跑：`python scripts/run_backtest.py --months 120 ...`")
    st.stop()

# 友善 label
def _label(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"

selected = st.multiselect(
    "選 1-3 個 backtest 對照",
    options=artifacts,
    format_func=_label,
    default=artifacts[:2] if len(artifacts) >= 2 else artifacts,
)

if not selected:
    st.info("選一個 backtest 開始")
    st.stop()

# ── Load + summary cards ──
results = {}
for path in selected:
    r = load_backtest_result(path)
    if r:
        results[_label(path)] = r

if not results:
    st.warning("無有效 result")
    st.stop()

# ── KPI 對照 ──
st.markdown('<div class="section-header">📊 KPI 對照</div>', unsafe_allow_html=True)

n = len(results)
cols = st.columns(n)
for col, (name, r) in zip(cols, results.items()):
    s = r.get("summary", {})
    with col:
        # cum 從 periods 算（若 summary 沒 cumulative_return）
        cum = s.get("cumulative_return")
        if cum is None and r.get("periods"):
            c = 1.0
            for p in r["periods"]:
                if p.get("return") is not None:
                    c *= (1 + p["return"])
            cum = c - 1
        st.markdown(f"#### {name}")
        render_kpi("Sharpe", f"{s.get('sharpe_ratio', 0):.2f}", flavor="success" if s.get('sharpe_ratio', 0) > 1.3 else "default")
        render_kpi("MDD", f"{s.get('max_drawdown', 0):.2%}", flavor="warning" if s.get('max_drawdown', 0) < -0.30 else "default")
        render_kpi("Calmar", f"{s.get('calmar_ratio', 0):.2f}")
        if cum is not None:
            render_kpi("累積", f"{cum:.1%}")
        if s.get("max_drawdown") and "hedge_metrics" in r:
            hm = r["hedge_metrics"]
            render_kpi(
                f"Hedged Sharpe (h={hm.get('hedge_ratio', 0):.1f})",
                f"{hm.get('hedged_sharpe', 0):.2f}",
                flavor="success",
            )

# ── Equity curve overlay ──
st.markdown('<div class="section-header">📈 Equity Curve Overlay</div>', unsafe_allow_html=True)

fig = go.Figure()
colors = [COLOR_ACCENT, COLOR_WARNING, COLOR_DANGER, COLOR_SUCCESS, "#9B59B6"]
benchmark_added = False
for i, (name, r) in enumerate(results.items()):
    eq = equity_curve_from_periods(r.get("periods", []))
    if eq.empty:
        continue
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq["strategy"], name=name,
        line=dict(color=colors[i % len(colors)], width=2),
    ))
    if not benchmark_added:
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["benchmark"], name="大盤 (TAIEX-equiv)",
            line=dict(color=COLOR_TEXT_SECONDARY, width=1, dash="dash"),
        ))
        benchmark_added = True

fig.update_layout(
    **PLOTLY_LAYOUT,
    height=500,
    yaxis_type="log",
    yaxis_title="Equity (log)",
    xaxis_title="Date",
    legend=dict(orientation="h", y=1.05, x=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── 月度報酬 heatmap（第 1 個 backtest）──
first_name = list(results.keys())[0]
first_r = results[first_name]
st.markdown(f'<div class="section-header">📅 月度報酬 Heatmap — {first_name}</div>', unsafe_allow_html=True)

periods = first_r.get("periods", [])
if periods:
    rows = []
    for p in periods:
        r = p.get("return")
        d = p.get("rebalance_date") or p.get("date")
        if r is None or not d:
            continue
        td = pd.to_datetime(d)
        rows.append({"year": td.year, "month": td.month, "return": float(r)})
    df = pd.DataFrame(rows)
    if not df.empty:
        pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
        fig2 = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{m:02d}" for m in pivot.columns],
            y=pivot.index,
            colorscale=[[0, COLOR_DANGER], [0.5, "#1A1F2E"], [1, COLOR_SUCCESS]],
            zmid=0,
            text=[[f"{v:+.1%}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            hovertemplate="%{y} 月 %{x}: %{z:.2%}<extra></extra>",
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            xaxis_title="月", yaxis_title="年",
        )
        st.plotly_chart(fig2, use_container_width=True)
