"""Dashboard v2 entry point — 多頁系統。

跑：
    streamlit run app/dashboard_v2/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.dashboard_v2.helpers import (
    apply_style, render_top_banner, render_kpi, render_regime_tag,
    get_engine, fetch_latest_picks, fetch_market_regime,
)


apply_style()

render_top_banner(
    "📈 台股 ML 選股系統 v2",
    "基準 v2.1 · Sharpe 0.613 · A 線僅紙上追蹤（各口徑無可執行 alpha，詳 docs/prereg_*）",
)

st.markdown(
    """
    ### 👋 歡迎使用 Dashboard v2

    左側 sidebar 可切換視圖：

    | Page | 用途 | 目標讀者 |
    |------|------|---------|
    | **📊 Today** | 今日 picks + 持倉 + 大盤環境 | 投資者每日操作 |
    | **🧪 Backtest Lab** | 歷史回測結果視覺化、對照 | 評估策略表現 |
    | **🎯 Features** | SHAP 特徵歸因 + 重要性排序 | 研究模型行為 |
    | **⚕️ System Health** | data coverage / pipeline status | 監控系統健康 |
    | **📈 Experiments** | MLflow / Optuna 實驗追蹤 | 實驗管理 |
    | **🏠 總覽** | paper NAV + picks + 申購機會 + 處置股 + 系統狀態 | 每日資訊瀏覽 |
    | **🕯️ 個股K線** | 蠟燭圖（raw/還原價）+ 法人 + 融資 | 個股查詢 |
    | **📜 研究裁決** | prereg 預登記文件 + 裁決摘要 | 研究誠實紀錄 |
    """
)

# Quick summary
engine = get_engine()
col1, col2, col3 = st.columns(3)

with col1:
    picks = fetch_latest_picks(engine)
    pick_date = picks["pick_date"].iloc[0] if not picks.empty else "N/A"
    render_kpi("今日 picks", str(len(picks)), f"{pick_date}", "success")

with col2:
    regime = fetch_market_regime(engine)
    regime_html = render_regime_tag(regime["regime"])
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">市場 regime</div>
            <div style="margin-top:8px;">{regime_html}</div>
            <div class="kpi-delta">200ma {'✓' if regime['above_200ma'] else '✗'} · vol {regime['vol_20']:.1%}</div>
        </div>""" if regime["vol_20"] is not None else
        f'<div class="kpi-card"><div class="kpi-label">市場 regime</div><div>計算中...</div></div>',
        unsafe_allow_html=True,
    )

with col3:
    # 最新 backtest 結果
    from app.dashboard_v2.helpers import load_backtest_result, PROJECT_ROOT as PR
    bt = load_backtest_result(str(PR / "artifacts/stage10_10y/topn30.json"))
    if bt:
        s = bt.get("summary", {})
        render_kpi(
            "Production 10y Sharpe",
            f"{s.get('sharpe_ratio', 0):.2f}",
            f"MDD {s.get('max_drawdown', 0):.1%}",
            "success",
        )
    else:
        render_kpi("Production 10y Sharpe", "N/A", "missing", "warning")

st.markdown("---")
st.caption(
    "Dashboard v2 · Stage 10.1 (topn 20→30 POSITIVE) + Stage 10.6 (Beta-hedge analysis)"
)
