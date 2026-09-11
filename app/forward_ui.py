"""Plain-language prospective evidence panel, separate from historical returns."""
import json
import pandas as pd
import streamlit as st
from app import forward_journal as journal, forward_service as service


def render():
    st.subheader('從今天開始，記錄真正看得到的證據')
    st.caption('先封存訊號，再記報價與委託。歷史回測、今日模型排名及實際成交紀錄分開保留。')
    try:
        data = journal.summary()
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    cols = st.columns(3)
    cols[0].metric('事前封存訊號日', data['prospective_days'])
    cols[1].metric('紙上委託', data['orders'])
    cols[2].metric('已登錄紙上成交', data['confirmed_fills'])
    st.warning('驗證尚未完成；沒有成交證據，不顯示模擬獲利。')
    left, right = st.columns(2)
    if left.button('封存今日資料與訊號', key='forward_freeze', use_container_width=True):
        try:
            event = service.freeze_today()
            if event['kind']=='blocked': st.warning('已記錄資料缺口：'+'；'.join(event['body']['reasons']))
            else: st.success('今日原策略訊號已封存，後續不能覆寫。')
        except Exception as exc:
            st.error(f'封存未完成：{type(exc).__name__}: {exc}')
    if right.button('記錄0050最新報價', key='forward_quote', use_container_width=True):
        try:
            event = service.capture_quotes()
            st.success('已記錄報價；報價不是成交回報。')
            st.dataframe(pd.DataFrame(event['body']['quotes']), hide_index=True)
        except Exception as exc:
            st.error(f'報價未取得：{type(exc).__name__}: {exc}')
    st.caption('每次報價操作最多一個全市場請求，10秒內重用快取；共用 Sponsor 額度。零股即時深度尚未接入。')
    if st.button('建立已封存候選的紙上委託', key='forward_plan'):
        try:
            planned=service.plan_frozen_candidates()
            st.success(f'已記錄 {len(planned)} 筆預定委託；不會送給券商。')
        except ValueError as exc: st.error(str(exc))
    orders=[r for r in data['rows'] if r['kind']=='order']
    if orders:
        table=[dict(代號=r['body']['stock_id'], 預定日期=r['body']['entry_date'],
            交易別='零股' if r['body']['channel']=='odd' else '整張',
            股數=r['body']['qty'], 限價=r['body']['limit_price'],
            狀態='已取消' if any(e['kind']=='cancel' and e['body']['order_hash']==r['hash'] for e in data['rows']) else '預定，尚待成交證據') for r in orders]
        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
        with st.expander('登錄成交回報或取消未成交委託'):
            selected=st.selectbox('預定委託',orders,format_func=lambda r:f"{r['body']['stock_id']} / {r['body']['channel']} / {r['body']['qty']}股",key='forward_order_select')
            reason=st.text_input('取消原因',value='當日未成交',key='forward_cancel_reason')
            if st.button('記錄取消',key='forward_cancel'):
                try:
                    with journal.connection() as con: journal.cancel(con,selected['hash'],reason)
                    st.success('已保留取消紀錄，原委託仍可查閱。')
                except ValueError as exc: st.error(str(exc))
            report=st.file_uploader('紙上成交回報 JSON',type=['json'],key='forward_fill_report')
            st.caption('格式：qty、price、fee、tax、executed_at（含時區）、evidence（source=paper_execution_report、唯一report_id）。回報為使用者提供，尚未由券商認證。')
            if st.button('儲存紙上成交回報',disabled=report is None,key='forward_fill'):
                try:
                    payload=json.loads(report.getvalue())
                    with journal.connection() as con:
                        journal.record_fill(con,selected['hash'],**payload)
                    st.success('已登錄，重複回報不會重複入帳。')
                except (TypeError,KeyError,ValueError) as exc: st.error(str(exc))
    if data['rows']:
        records = [dict(時間=r['recorded_at'], 類型=r['kind'], 編號=r['seq'],
            說明='；'.join(r['body'].get('reasons', []))) for r in data['rows']]
        with st.expander('驗證紀錄與目前限制'):
            st.dataframe(pd.DataFrame(records),hide_index=True,use_container_width=True)
            for limitation in data['limitations']: st.write('• '+limitation)
            st.download_button('下載封存紀錄', json.dumps(data['rows'],ensure_ascii=False,indent=2),
                               'forward-evidence.json', 'application/json', key='forward_download')
