"""Prospective portfolio with the original evidence retained as an archive."""
import json
import streamlit as st
from app import forward_journal as journal, forward_service as service
from app.forward_portfolio_ui import render as render_portfolio


def render():
    from app.forward_simulation_ui import render as render_simulation
    render_simulation()
    render_portfolio()
    from app.forward_comparison_ui import render as render_comparison
    render_comparison()
    with st.expander('原策略訊號封存與舊版證據（v1）'):
        data=journal.summary()
        st.caption(f"已封存 {data['prospective_days']} 個訊號日；舊版委託 {data['orders']} 筆、成交 {data['confirmed_fills']} 筆。新版不匯入舊委託或假定成交。")
        if st.button('封存今日原策略訊號',key='forward_freeze'):
            try:
                event=service.freeze_today()
                st.success('訊號已封存。') if event['kind']=='signal' else st.warning('資料未完整，已保留缺口。')
            except (ValueError,OSError) as exc: st.error(str(exc))
        if st.button('記錄0050最新報價',key='forward_quote'):
            try:
                service.capture_quotes();st.success('已記錄；此報價不作為零股成交證據。')
            except Exception as exc: st.error(f'{type(exc).__name__}: {exc}')
        st.caption('新委託與成交請使用上方新版資產帳本；原始訊號、程式與回報均保留。每次即時報價最多1個全市場請求，10秒快取，共用Sponsor限額。')
        st.download_button('下載舊版封存紀錄',json.dumps(data['rows'],ensure_ascii=False,indent=2),
                           'forward-v1-evidence.json','application/json',key='forward_download')
