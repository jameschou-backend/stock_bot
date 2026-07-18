"""Page 9: 研究裁決 — docs/prereg_*.md 預登記文件渲染 + 裁決摘要表。

預登記制度：判準在跑數字之前 commit（git history 為證），跑完不得回改判準。
本頁把所有 prereg 文件按日期新→舊渲染，頂部給一行式裁決摘要。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.dashboard_v2.helpers import (
    apply_style, render_top_banner, render_kpi, COLOR_DANGER,
)
from app.dashboard_v2.info_helpers import (
    HONEST_BANNER_TEXT, PREREG_VERDICTS, list_prereg_docs,
)

apply_style()
render_top_banner("📜 研究裁決", "預登記（pre-registration）文件與裁決摘要——判準先寫死，跑完不得回改")

st.markdown(
    f"""<div style="background:rgba(231,76,60,0.12);border:1px solid {COLOR_DANGER};
        border-left:6px solid {COLOR_DANGER};border-radius:8px;padding:14px 18px;margin-bottom:18px;">
        <b style="color:{COLOR_DANGER};">⚠️ 誠實聲明</b>&nbsp;&nbsp;
        <span>{HONEST_BANNER_TEXT}</span>
    </div>""",
    unsafe_allow_html=True,
)

# ── 裁決摘要表（硬編碼自 memory/decisions.md 已知結果）──
st.markdown('<div class="section-header">⚖️ 裁決摘要</div>', unsafe_allow_html=True)

n_fail = sum(1 for v in PREREG_VERDICTS if v["裁決"] == "FAIL")
c1, c2 = st.columns([1, 3])
with c1:
    render_kpi("裁決", f"{n_fail}/{len(PREREG_VERDICTS)} FAIL",
               "主動 alpha 調查窮盡（2026-07-11）", flavor="danger")
with c2:
    st.info(
        "**終局定調**：機構口徑無 alpha（v2.2 超額 -41pp）、個人整股 / 個人零股口徑均 FAIL、"
        "營收 PEAD 訊號存在且獨立但成交成本結構性 > alpha（容量天花板，換通路 / 換訊號都不能解）。"
        "資金配置：0050 / 現金 + 申購抽籤；A 線 picks 僅紙上追蹤。"
    )

summary_df = pd.DataFrame(PREREG_VERDICTS)
summary_df["裁決"] = summary_df["裁決"].map(lambda v: f"❌ {v}" if v == "FAIL" else f"✅ {v}")
st.dataframe(
    summary_df[["日期", "臂", "裁決", "關鍵數字", "文件"]],
    use_container_width=True, hide_index=True,
    column_config={
        "日期": st.column_config.TextColumn(width="small"),
        "臂": st.column_config.TextColumn(width="medium"),
        "裁決": st.column_config.TextColumn(width="small"),
        "關鍵數字": st.column_config.TextColumn(width="large"),
        "文件": st.column_config.TextColumn(width="medium"),
    },
)

# ── 預登記文件全文（按日期新→舊）──
st.markdown('<div class="section-header">📄 預登記文件（新 → 舊）</div>', unsafe_allow_html=True)

docs = list_prereg_docs()
if not docs:
    st.info("docs/ 下找不到 prereg_*.md。")
else:
    verdict_by_file = {v["文件"]: v for v in PREREG_VERDICTS}
    for d in docs:
        v = verdict_by_file.get(d["name"])
        badge = f"｜裁決：{v['裁決']}" if v else ""
        label = f"{d['date'] or '?'}　{d['name']}{badge}"
        with st.expander(label, expanded=False):
            try:
                content = d["path"].read_text(encoding="utf-8")
            except OSError as exc:
                st.error(f"讀取失敗：{exc}")
                continue
            st.markdown(content)
