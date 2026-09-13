"""Read-only five-axis comparison and previously frozen next-month forecasts."""
import json
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st
from app.cash_risk_ui import ROOT,render as render_comparison

REPORT=ROOT/'artifacts/forward_simulation/five_axis_delivery_20260913.json'


def render(path=REPORT):
    render_comparison(path,title='五項研究：進場、排序、配置與營收預期',key_prefix='five_axis')
    if not path.is_file():return
    # All performance and code validation is handled by the comparison renderer.
    # Forecasts remain a separate artifact; they are never presented as profits.
    try:
        report=json.loads(path.read_text())
        if not report['offline_identical'] or any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest
                                                 for name,digest in report['research_code_sha256'].items()):
            return
        forecast=report['future_revenue_forecast']
    except (KeyError,ValueError,OSError):return
    with st.expander('已封存的下期營收估計（等待實際公布）'):
        st.caption(f"營收月份 {forecast['revenue_month']}；估計封存於 {forecast['frozen_at']}。這些是營收金額估計，尚未形成交易績效。")
        frame=pd.DataFrame([{'股票':r['stock_id']+' '+r.get('name',''),
            '預估營收（億元）':round(r['expected_revenue_twd']/100_000_000,2)
                if r['expected_revenue_twd'] is not None else None,
            '狀態':'等待實際公布' if r['expected_revenue_twd'] is not None else '歷史資料不足'}
            for r in forecast['rows']])
        query=st.text_input('搜尋股票代號或名稱',key='five_axis_forecast_filter')
        if query:frame=frame[frame['股票'].str.contains(query,regex=False)]
        st.dataframe(frame,hide_index=True,use_container_width=True)
        st.download_button('下載封存營收預估',json.dumps(forecast,ensure_ascii=False),
                           'revenue-forecast.json','application/json',key='five_axis_forecast_download')
