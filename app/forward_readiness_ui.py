"""A small operational checklist; opening it performs no network requests."""
import json
import pandas as pd
import streamlit as st
from app import forward_readiness as readiness
from app.finmind import FinMindError


def render(root):
    with st.expander('依序完成：資料核對 → 每日流程 → 成交比對', expanded=True):
        st.caption('只讀帳本；按更新才查詢FinMind。兩份帳本合計每批最多12次，沿用共用快取與限流。')
        if st.button('更新兩份帳本的公司行動資料', key='cash_readiness_refresh'):
            try:
                result = readiness.refresh(root)
                st.success(f"查詢 {result['calls']} 次、重用 {result['reused']} 筆。未完成項目保留，可下次續查。")
            except (ValueError, OSError, FinMindError, TimeoutError) as exc:
                st.error(str(exc))
        try:
            report = readiness.inspect(root)
        except (ValueError, OSError) as exc:
            st.error('檢查未完成：'+str(exc));return
        st.write('下一步：'+report['next_action'])
        for role, b in report['books'].items():
            label = '策略' if role=='strategy' else '0050基準'
            status = '來源缺漏／衝突' if b['source_blocked'] else '未發現已知來源衝突'
            review = '今日核對已保存' if b['review_current'] else '尚未保存今日人工核對'
            st.write(f"{label}：{status}；{review}。")
        st.caption('來源回傳空資料不代表公告完整。成交後持股的每日結算，仍需在下方保存當日核對；此按鈕不會代勾或登錄權益。')
        st.write(f"完整交易日流程：已觀察 {report['completed_days']} 天。尚無紀錄時不算通過。")
        if report['days']:
            st.dataframe(pd.DataFrame(report['days'][-10:]), hide_index=True)
        if st.checkbox('查看成交比例、未成交數量與費用', key='cash_execution_details'):
            entries = [dict(帳本='策略' if role=='strategy' else '0050基準', **x)
                       for role,b in report['books'].items() for x in b['execution']]
            if entries:
                st.dataframe(pd.DataFrame(entries).rename(columns={'stock_id':'代號','channel':'交易別',
                    'planned':'計畫股數','filled':'已成交','unfilled':'未成交','cancelled':'已結束',
                    'fill_ratio':'成交比例','average_price':'平均成交價','fees_and_tax':'費稅',
                    'adverse_vs_limit_bps':'相對限價不利差（基點）'}), hide_index=True)
            st.caption('這裡只有模擬成交；無成交時均價與價差為未知。相對限價價差不能當成真實滑價，券商成交驗證尚未完成。')
        st.caption('歷史資料對帳與未見區間績效仍待驗證；本清單不授予實盤資格。')
        st.download_button('下載前向待辦與成交診斷', json.dumps(report, ensure_ascii=False, indent=2),
                           'forward-readiness.json', 'application/json', key='cash_readiness_download')
