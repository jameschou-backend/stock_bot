"""Page 7: 總覽 — paper NAV / 今日 picks / 申購機會 / 處置股警示 / 系統狀態。

一律讀自家 DB + artifacts（零 API 額度）。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.dashboard_v2.helpers import (
    apply_style, render_top_banner, render_kpi,
    get_engine, fetch_latest_picks,
    COLOR_ACCENT, COLOR_DANGER, COLOR_WARNING, COLOR_SUCCESS,
    COLOR_TEXT_SECONDARY, PLOTLY_LAYOUT,
)
from app.dashboard_v2.info_helpers import (
    HONEST_BANNER_TEXT,
    load_nav_history, nav_config_segments,
    load_latest_ipo_scan, ipo_items_dataframe,
    load_disposition_latest_pair, disposition_name_map,
    revenue_ledger_stats, fetch_recent_jobs,
)

apply_style()

render_top_banner("🏠 總覽", "paper NAV · A 線 picks · 申購機會 · 處置股 · 系統狀態")

# ── 誠實橫幅（最高優先級，放頂部）──
st.markdown(
    f"""<div style="background:rgba(231,76,60,0.12);border:1px solid {COLOR_DANGER};
        border-left:6px solid {COLOR_DANGER};border-radius:8px;padding:14px 18px;margin-bottom:18px;">
        <b style="color:{COLOR_DANGER};">⚠️ 誠實聲明</b>&nbsp;&nbsp;
        <span>{HONEST_BANNER_TEXT}</span>
    </div>""",
    unsafe_allow_html=True,
)

engine = get_engine()

# ══════════════════════════════════════════════════════════════
# 1) Paper NAV 淨值曲線（config_version 分段標記）
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📈 Paper NAV（訊號價紙上淨值）</div>', unsafe_allow_html=True)

nav_df = load_nav_history()
if nav_df.empty:
    st.info("尚無 paper NAV 紀錄（artifacts/paper_nav/nav.jsonl）——由 daily_run 的 paper_nav 步驟產生。")
else:
    latest = nav_df.iloc[-1]
    first = nav_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        chg = float(latest["nav"]) - 1.0
        render_kpi("最新 NAV", f"{float(latest['nav']):.4f}",
                   f"{chg:+.2%}（起算日至今）",
                   flavor="success" if chg >= 0 else "danger")
    with c2:
        render_kpi("最新日期", str(latest["date"].date()),
                   f"起算 {first['date'].date()}")
    with c3:
        render_kpi("持股數", str(int(latest.get("holdings_n", 0))),
                   "月頻換股（等權）")
    with c4:
        render_kpi("config_version", str(latest.get("config_version", "?")),
                   f"共 {nav_df['config_version'].nunique()} 段配置")

    fig = go.Figure()
    seg_colors = [COLOR_ACCENT, COLOR_WARNING, COLOR_SUCCESS, "#9B59B6", "#3498DB"]
    for i, seg in enumerate(nav_config_segments(nav_df)):
        sdf = seg["df"]
        fig.add_trace(go.Scatter(
            x=sdf["date"], y=sdf["nav"],
            mode="lines+markers",
            name=seg["config_version"],
            line=dict(color=seg_colors[i % len(seg_colors)], width=2),
            marker=dict(size=5),
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=COLOR_TEXT_SECONDARY,
                  annotation_text="NAV 1.0")
    fig.update_layout(
        **PLOTLY_LAYOUT, height=340,
        title="訊號價 paper NAV（無成本無滑價、raw close 未還原 → 系統性低估）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"最新行 notes：{latest.get('notes', '')}")

# ══════════════════════════════════════════════════════════════
# 2) 最新 A 線 picks（紙上追蹤）
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎯 最新 A 線 picks（紙上追蹤）</div>', unsafe_allow_html=True)

picks_df = fetch_latest_picks(engine)
if picks_df.empty:
    st.warning("picks 表為空——請先跑 `make pipeline`。")
else:
    pick_date = picks_df["pick_date"].iloc[0]
    st.caption(
        f"pick_date = **{pick_date}**，共 {len(picks_df)} 檔"
        "（大盤過濾 / 季節性降倉時有效 topN 會低於 30）——僅紙上追蹤，非投資建議。"
    )
    show = picks_df[["stock_id", "name", "industry_category", "market", "score"]].copy()
    show["score"] = show["score"].astype(float).map(lambda x: f"{x:.4f}")
    show.columns = ["代碼", "名稱", "產業", "市場", "分數"]
    show.index = range(1, len(show) + 1)
    st.dataframe(show, use_container_width=True, height=min(700, 40 + 36 * len(show)))

# ══════════════════════════════════════════════════════════════
# 3) 今日申購機會（唯一可行動項）
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🎟️ 公開申購抽籤機會（折價% 排序）</div>', unsafe_allow_html=True)

scan = load_latest_ipo_scan()
if not scan:
    st.info("尚無申購掃描結果（artifacts/ipo_lottery/scan_*.json）——跑 `python scripts/ipo_lottery_scan.py`。")
else:
    scan_date = scan.get("scan_date", "?")
    items_df = ipo_items_dataframe(scan)
    stale = str(scan_date) != date.today().isoformat()
    if stale:
        st.warning(f"⚠️ 掃描日期 {scan_date} 非今日——資料可能過期，申購期間請以公告為準。")
    if items_df.empty:
        st.info(f"掃描日 {scan_date}：目前無進行中 / 即將開始的申購案。")
    else:
        n_good = int((items_df["discount"].fillna(0) >= 0.10).sum())
        render_kpi("折價 ≥ 10% 案件", str(n_good),
                   f"掃描日 {scan_date}（申購抽籤與選股策略零相關，為唯一可行動項）",
                   flavor="success" if n_good else "default")
        disp = pd.DataFrame({
            "代碼": items_df["stock_id"],
            "名稱": items_df["name"],
            "類型": items_df["market_type"],
            "狀態": items_df["status"],
            "申購期間": items_df["sub_start"].astype(str) + " ~ " + items_df["sub_end"].astype(str),
            "抽籤日": items_df["draw_date"],
            "承銷價": items_df["effective_price"].map(
                lambda x: f"{x:,.1f}" if pd.notna(x) else "—"),
            "市價": items_df["market_price"].map(
                lambda x: f"{x:,.1f}" if pd.notna(x) else "—"),
            "折價%": items_df["discount"].map(
                lambda x: f"{x:+.1%}" if pd.notna(x) else "n/a"),
            "折價額": items_df["discount_amount"].map(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "—"),
            "保本中籤率": items_df["breakeven_win_rate"].map(
                lambda x: f"{x:.3%}" if pd.notna(x) else "—"),
        })
        disp.index = range(1, len(disp) + 1)
        st.dataframe(disp, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 4) 處置股警示
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🚧 處置股警示</div>', unsafe_allow_html=True)

dispo, dispo_prev = load_disposition_latest_pair()
if not dispo:
    st.info("尚無處置股快取（artifacts/disposition/）——由 daily_pick 的處置股過濾產生。")
else:
    dispo_ids = [s for s in dispo.get("disposition", [])]
    att_ids = [s for s in dispo.get("attention", [])]
    names = disposition_name_map(dispo)
    new_ids = []
    if dispo_prev:
        new_ids = sorted(set(dispo_ids) - set(dispo_prev.get("disposition", [])))

    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi("處置股", str(len(dispo_ids)), f"as_of {dispo.get('as_of', '?')}",
                   flavor="warning" if dispo_ids else "default")
    with c2:
        render_kpi("本次新增", str(len(new_ids)),
                   "vs 前一份快取" if dispo_prev else "（無前一份可比）",
                   flavor="danger" if new_ids else "default")
    with c3:
        render_kpi("注意股", str(len(att_ids)), "僅記錄，不剔除")

    if new_ids:
        st.error("🆕 新增處置股：" + "、".join(f"{s} {names.get(s, '')}" for s in new_ids))

    # picks ∩ 處置股（執行衛生）
    if not picks_df.empty:
        overlap = sorted(set(picks_df["stock_id"].astype(str)) & set(dispo_ids))
        if overlap:
            st.error("⚠️ 今日 picks 內含處置股（分盤交易，月底換股可能出不掉）："
                     + "、".join(f"{s} {names.get(s, '')}" for s in overlap))
        else:
            st.success("今日 picks 無處置股 ✅")

    with st.expander(f"處置股完整名單（{len(dispo_ids)} 檔）"):
        st.write("、".join(f"{s} {names.get(s, '')}" for s in dispo_ids) or "（空）")

# ══════════════════════════════════════════════════════════════
# 5) 系統狀態
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⚙️ 系統狀態</div>', unsafe_allow_html=True)

jobs_df = fetch_recent_jobs(engine, limit=15)
rev = revenue_ledger_stats()

c1, c2, c3, c4 = st.columns(4)
with c1:
    if jobs_df.empty:
        render_kpi("最近 daily_run", "無 job 紀錄", flavor="warning")
    else:
        latest_job = jobs_df.iloc[0]
        n_failed = int((jobs_df["status"] == "failed").sum())
        ok = latest_job["status"] == "success" and n_failed == 0
        render_kpi(
            "最近 daily_run",
            "✅ 成功" if ok else ("❌ 有失敗" if n_failed else f"{latest_job['status']}"),
            f"最新 job：{latest_job['job_name']} @ {latest_job['started_at']}",
            flavor="success" if ok else "danger",
        )
with c2:
    if not jobs_df.empty:
        dp = jobs_df[jobs_df["job_name"] == "daily_pick"]
        if not dp.empty:
            render_kpi("daily_pick", str(dp.iloc[0]["status"]),
                       f"@ {dp.iloc[0]['started_at']}",
                       flavor="success" if dp.iloc[0]["status"] == "success" else "danger")
        else:
            render_kpi("daily_pick", "近期無紀錄", flavor="warning")
with c3:
    if rev is None:
        render_kpi("月營收 ledger", "無檔案", "artifacts/revenue_announcements/", "warning")
    else:
        render_kpi("月營收 ledger", f"{rev['rows']:,} 筆",
                   f"{rev['stocks']} 檔｜最新公告 {rev['max_announcement']}")
with c4:
    nav_last = str(nav_df["date"].max().date()) if not nav_df.empty else "無"
    render_kpi("paper NAV 最新日", nav_last,
               "daily_run 每日追加",
               flavor="default" if not nav_df.empty else "warning")

with st.expander("最近 15 個 jobs"):
    if jobs_df.empty:
        st.write("（jobs 表無資料）")
    else:
        jshow = jobs_df.copy()
        jshow["status"] = jshow["status"].map(
            lambda s: {"success": "✅ success", "failed": "❌ failed", "running": "⏳ running"}.get(s, s))
        st.dataframe(jshow, use_container_width=True, hide_index=True)
