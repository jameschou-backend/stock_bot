"""Explicit refresh only: rendering this panel never calls FinMind."""
import json
import pandas as pd
import streamlit as st
from app import forward_corporate_audit as audit
from app.finmind import FinMindError


def render(path, account):
    with st.expander('公司行動待辦：先檢查股息與股數', expanded=True):
        st.caption('查詢持股、已成交股票及待成交計畫；只做核對，不自動入帳。每批最多12次，1小時內重用已取得資料。')
        if st.button('更新公司行動檢查', key='corp_refresh_' + account):
            try:
                result = audit.refresh(path)
                st.success(f"本批呼叫共用查詢 {result['calls']} 次、重用 {result['reused']} 筆。未完成項目可再次按更新續查。")
            except (ValueError, OSError, FinMindError) as exc:
                st.error(str(exc))
        try:
            report = audit.inspect(path)
        except (ValueError, OSError) as exc:
            st.error('公司行動證據無法讀取：' + str(exc))
            return
        if report['blocked']:
            st.warning('有待核對項目，尚不能封存新的收盤資產。')
        elif report['scope']['stock_ids']:
            st.info('本次資料未發現已知阻擋；仍需人工核對官方公告，不能視為完整通過。')
        else:
            st.info('目前沒有持股、成交或待成交計畫需要檢查。')
        for item in report['issues']:
            st.write(f"{item['stock_id']} {item['date']}｜{item['message']}")
        if report['events']:
            st.dataframe(pd.DataFrame(report['events']).rename(columns={
                'stock_id':'代號','ex_date':'除息日','cash_per_share':'每股股息',
                'payment_date':'公告發放日','eligible_qty':'除息前股數','paid':'已登錄交付'}),hide_index=True)
        st.caption(report['limitation'])
        st.download_button('下載公司行動核對報告', json.dumps(report, ensure_ascii=False, indent=2),
                           'corporate-review.json', 'application/json', key='corp_download_' + account)

    from app.forward_corporate_resolution_ui import render as render_resolution
    render_resolution(path, account, report)
