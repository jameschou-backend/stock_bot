"""Dahu Tou manual receipt reconciliation, without brokerage actions."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from app.manual_execution_audit import audit

ROOT=Path(__file__).resolve().parents[1]


def render():
    with st.expander('大戶投：賣出／撤單後可以補位嗎？',expanded=False):
        st.write('依序核對：事前買賣計畫 → 成交或撤單成功回報 → 更新可用額度 → 下一筆買進。')
        st.caption('按下刪單不等於已撤單。部分成交、剩1股及待交付股票權利仍占名額；賣款不會自動變成可用買進額度。')
        st.info('此工具只檢查你提供的紀錄，不操作大戶投。尚未取得你的券商紀錄，因此尚未完成真實執行驗證。')
        st.markdown('[查看回報準備方式與欄位說明](https://www.sinotrade.com.tw/richclub/manual.pdf)（永豐官方操作手冊）')
        st.caption('目前使用本專案的轉錄格式；尚未確認大戶投原生匯出欄位，不能直接套用其他券商CSV。金額單位為分，股數單位為股。請移除帳號、姓名及其他個資。')
        template=ROOT/'docs/examples/dahu_manual_receipts.json'
        st.download_button('下載核對範本（虛構示例）',template.read_bytes(),
            file_name='dahu-manual-receipts-example.json',mime='application/json',key='dahu_template')
        upload=st.file_uploader('上傳已核對的回報紀錄',type=['json'],key='dahu_receipts')
        if upload is not None:
            try:
                if upload.size>5_000_000:
                    raise ValueError('單日回報檔案請小於5MB')
                result=audit(json.loads(upload.getvalue().decode('utf-8-sig')))
                st.success('提供的時序與資金數字一致；尚未驗證原始回報真實性及完整性。')
                st.write(result['note'])
                rows=[{'回報':r['id'],'事件':r['kind'],'現金（元）':r['cash_cents']/100,
                    '可用買進額度（元）':r['available_cents']/100,'預留（元）':r['reserved_cents']/100,
                    '持股':str(r['holdings'])} for r in result['decisions']]
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
                if result['open_orders']:
                    st.warning('仍未終結的委託：'+', '.join(result['open_orders']))
                st.download_button('下載本次核對結果',json.dumps(result,ensure_ascii=False,indent=2),
                    'dahu-audit.json','application/json',key='dahu_result')
            except (ValueError,KeyError,TypeError,UnicodeError) as exc:
                st.error('核對未通過：'+str(exc))
