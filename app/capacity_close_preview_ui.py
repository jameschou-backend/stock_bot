"""Show actionable closing prerequisites without writing an approval."""
import json
import pandas as pd
import streamlit as st
from app.capacity_close_preview import inspect
from app.capacity_source_guard import NAME


def render(root, role):
    with st.container(border=True):
        st.markdown('**今晚能結算嗎？先看缺什麼**')
        if not (root / NAME).is_file():
            st.info('逐筆來源檢查尚未啟用，結算前清單暫不可用。'); return
        try:
            report = inspect(root)
        except TimeoutError:
            st.info('模擬正在更新帳本，稍後重新整理即可查看。'); return
        except (ValueError, OSError) as exc:
            st.error('結算前核對未完成：'+str(exc)); return
        book = report['books'][role]
        st.caption('讀取時間：'+pd.Timestamp(report['observed_at']).tz_convert('Asia/Taipei').strftime('%Y/%m/%d %H:%M:%S'))
        if book['already_closed']:
            st.success('今天的結算紀錄已存在；以下顯示目前核對狀態。')
        st.caption(report['note'])
        st.dataframe(pd.DataFrame(book['checks']).rename(columns={'item':'項目','status':'狀態','detail':'下一步／說明'}), hide_index=True)
        cash = book['cash']
        st.write(f"今日帳列費稅 {float(cash['fees_and_tax']):,.2f} 元；現金收支差額 {float(cash['difference']):,.2f} 元。")
        if book['orders']:
            frame = pd.DataFrame(book['orders'])
            frame['channel'] = frame['channel'].replace({'board':'整股','odd':'零股'})
            frame['side'] = frame['side'].replace({'buy':'買進','sell':'賣出'})
            st.dataframe(frame[['stock_id','side','channel','planned','filled','unfilled','cancelled','fees_and_tax']].rename(
                columns={'stock_id':'股票','side':'買賣','channel':'盤別','planned':'委託股數','filled':'已成交',
                         'unfilled':'未成交','cancelled':'剩餘已取消','fees_and_tax':'費稅'}),hide_index=True)
        if any(c['status']=='需你核對' for c in book['checks']):
            st.info('下一步：下方展開「有持股後：顯示每日公司行動核對」，核對後保存。每份帳本分別記錄；資料有修訂時需重新核對。')
        st.download_button('下載三帳本結算前清單', json.dumps(report, ensure_ascii=False, indent=2),
                           'capacity-close-preview.json', 'application/json', key='capacity_close_preview_download')
