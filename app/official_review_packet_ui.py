"""Review aids are separate from the human attestation control."""
import json
import streamlit as st
import pandas as pd
from app import official_review_packet as packet
from app import capacity_forward as policy, forward_portfolio as p, forward_automation as auto


def render(root, role):
    holdings = p.state(policy.verify(root / (role+'.sqlite3'), role))['holdings']
    if not holdings:
        return
    st.markdown('**持股官方公告核對入口**')
    if st.button('更新官方公告快照（最多1次）', key='capacity_official_refresh'):
        try:
            result = packet.capture()
            if result['status'] == 'ok':
                st.success('已重用30分鐘內快照。' if result['reused'] else '已取得官方公告快照。')
            else:
                st.error('官方快照取得或格式核對失敗；保留失敗紀錄，30分鐘內不重複查詢。')
        except (ValueError, OSError, TimeoutError) as exc:
            st.error('官方公告更新未完成：'+str(exc))
    try:
        report = packet.inspect(auto.markets(sorted(holdings)))
    except (ValueError, OSError) as exc:
        st.error('官方公告摘要無法核對：'+str(exc)); return
    st.caption(report['note']+'。查看不抓資料；按更新才查詢，不會代勾人工核對。')
    if report['snapshot_at']:
        st.caption('公告快照取得時間：'+pd.Timestamp(report['snapshot_at']).tz_convert('Asia/Taipei').strftime('%Y/%m/%d %H:%M:%S'))
    for stock in report['stocks']:
        st.write(stock['stock_id']+'｜'+stock['scope'])
        if stock['disclosures']:
            for row in stock['disclosures']:
                st.text(row['published_at']+' '+row['title'])
                st.text(row['explanation'])
        elif report['ready'] and stock['scope']=='上市公司重大訊息':
            st.write('這份快照沒有命中公告；仍須核對即時公告與停復牌。')
    st.markdown('[公開資訊觀測站：即時重大訊息](https://mops.twse.com.tw/) · '
                '[證交所：即時市場公告](https://mis.twse.com.tw/) · '
                '[暫停交易歷史查詢](https://www.twse.com.tw/zh/trading/historical/twtawu.html)')
    if '0050' in holdings:
        st.markdown('[0050 元大官方基金資訊](https://www.yuantafunds.com/myfund/information/1066) · '
                    '[0050 交易所配息紀錄](https://www.twse.com.tw/en/ETFortune-institute/dividendList?startDate=&stkNo=0050)')
    st.caption('盤中快照無法涵蓋稍後公告；歷史停牌頁不能代替即時市場公告。')
    st.download_button('下載持股官方公告核對摘要', json.dumps(report, ensure_ascii=False, indent=2),
                       'official-review-packet.json', 'application/json', key='capacity_official_packet')
