"""Page 3: Features — SHAP 特徵歸因。"""
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
    apply_style, render_top_banner,
    COLOR_ACCENT, COLOR_DANGER, COLOR_WARNING, COLOR_SUCCESS,
    COLOR_TEXT_PRIMARY, COLOR_GRID, COLOR_PANEL, PLOTLY_LAYOUT,
)


apply_style()
render_top_banner("🎯 Feature Insights", "SHAP 特徵歸因 · 重要性排序 · Regime breakdown")

SHAP_DIR = PROJECT_ROOT / "artifacts/shap_analysis"
if not (SHAP_DIR / "global_importance.csv").exists():
    st.warning(
        "尚無 SHAP 分析結果。請先跑：\n"
        "```bash\npython scripts/shap_v2.py --sample 5000\n```"
    )
    st.stop()

# ── Global importance ──
st.markdown('<div class="section-header">📊 Global Importance（Top 20）</div>', unsafe_allow_html=True)
gi = pd.read_csv(SHAP_DIR / "global_importance.csv")
top20 = gi.head(20).iloc[::-1]  # reverse for horizontal bar bottom-up

fig = go.Figure()
fig.add_trace(go.Bar(
    x=top20["mean_abs_shap"],
    y=top20["feature"],
    orientation="h",
    marker=dict(
        color=top20["mean_abs_shap"],
        colorscale=[[0, COLOR_PANEL], [1, COLOR_ACCENT]],
    ),
    text=[f"{p:.1f}%" for p in top20["pct"]],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>mean|SHAP|: %{x:.6f}<br>占比: %{text}<extra></extra>",
))
fig.update_layout(
    **PLOTLY_LAYOUT,
    height=600,
    title="Top 20 features by mean |SHAP|",
    xaxis_title="mean |SHAP|", yaxis_title="",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# ── 累積貢獻 + 剪枝點 ──
st.markdown('<div class="section-header">🌳 累積貢獻 + 剪枝建議</div>', unsafe_allow_html=True)
gi_sorted = gi.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
gi_sorted["rank"] = gi_sorted.index + 1

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=gi_sorted["rank"], y=gi_sorted["cum_pct"],
    mode="lines+markers",
    line=dict(color=COLOR_ACCENT, width=2),
    marker=dict(size=4),
    hovertemplate="Rank %{x}: %{y:.2f}% cum<extra></extra>",
))
# 80% / 95% / 99% 標線
for pct, color, label in [(80, COLOR_WARNING, "80%"), (95, COLOR_SUCCESS, "95%"), (99, COLOR_DANGER, "99%")]:
    fig2.add_hline(y=pct, line_dash="dash", line_color=color, opacity=0.5,
                   annotation_text=f"{label} cum", annotation_position="right")

fig2.update_layout(
    **PLOTLY_LAYOUT,
    height=400,
    title="Cumulative SHAP importance vs feature rank",
    xaxis_title="Feature Rank", yaxis_title="Cumulative %",
)
st.plotly_chart(fig2, use_container_width=True)

n_80 = (gi_sorted["cum_pct"] <= 80).sum() + 1
n_95 = (gi_sorted["cum_pct"] <= 95).sum() + 1
n_99 = (gi_sorted["cum_pct"] <= 99).sum() + 1

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="kpi-card success">
        <div class="kpi-label">前 N 個達 80%</div>
        <div class="kpi-value">{n_80}</div>
        <div class="kpi-delta">剩 {len(gi_sorted) - n_80} 可能剪枝</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card warning">
        <div class="kpi-label">前 N 個達 95%</div>
        <div class="kpi-value">{n_95}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">前 N 個達 99%</div>
        <div class="kpi-value">{n_99}</div>
    </div>""", unsafe_allow_html=True)

# ── Regime breakdown ──
if (SHAP_DIR / "regime_breakdown.csv").exists():
    st.markdown('<div class="section-header">🌗 Regime Breakdown (Bull vs Bear)</div>', unsafe_allow_html=True)
    rb = pd.read_csv(SHAP_DIR / "regime_breakdown.csv").head(15)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Bear", x=rb["feature"], y=rb["bear_mean_abs_shap"],
        marker_color=COLOR_DANGER,
    ))
    fig3.add_trace(go.Bar(
        name="Bull", x=rb["feature"], y=rb["bull_mean_abs_shap"],
        marker_color=COLOR_SUCCESS,
    ))
    fig3.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        barmode="group",
        title="Top 15 features: bull vs bear importance",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Redundant pairs ──
if (SHAP_DIR / "redundant_pairs.csv").exists():
    st.markdown('<div class="section-header">🔄 Redundant Pairs (|corr| > 0.85)</div>', unsafe_allow_html=True)
    rd = pd.read_csv(SHAP_DIR / "redundant_pairs.csv")
    rd["建議刪除"] = rd.apply(
        lambda r: r["f1"] if r["f1_shap"] < r["f2_shap"] else r["f2"], axis=1
    )
    display = rd[["f1", "f2", "corr", "f1_shap", "f2_shap", "建議刪除"]].copy()
    display["corr"] = display["corr"].map(lambda x: f"{x:+.4f}")
    display["f1_shap"] = display["f1_shap"].map(lambda x: f"{x:.6f}")
    display["f2_shap"] = display["f2_shap"].map(lambda x: f"{x:.6f}")
    st.dataframe(display, use_container_width=True, hide_index=True)
