"""Dahu Tou manual receipt reconciliation, without brokerage actions."""
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import streamlit as st
from app.manual_execution_audit import audit, TZ
from app.manual_receipt_entry import new_document, append_event, cents, shares

ROOT=Path(__file__).resolve().parents[1]


def render_entry():
    st.write('沒有JSON也可以逐筆整理。先填事前計畫，再依實際收到順序加入回報。')
    st.caption('這是人工轉錄，不能證明計畫確實事前存在。草稿只保留在目前瀏覽工作階段，請下載保存；原始證據仍由你保管。')
    st.caption('可核對四碼台股代號及00631L。接受代號只表示可整理紀錄，不代表已確認商品交易資格；費稅按券商實際回報填寫。')
    draft = st.session_state.get('dahu_entry_draft')
    if draft is None:
        today = datetime.now(TZ).date()
        with st.form('dahu_opening_form'):
            day = st.date_input('核對交易日',today,max_value=today)
            cash = st.text_input('期初策略現金（元）','1000000')
            slots = st.number_input('最多持股檔數',min_value=1,max_value=30,value=3,step=1)
            st.caption('期初持股／待交付股票：沒有就保留空白。股數包含零股，整股委託1000股就是填1000。')
            held = st.data_editor(pd.DataFrame({'股票代號':pd.Series(dtype=str),'股數':pd.Series(dtype=int)}),
                num_rows='dynamic',hide_index=True,key='dahu_opening_holdings')
            st.write('已除權、尚未交付的股票')
            rights = st.data_editor(pd.DataFrame({'股票代號':pd.Series(dtype=str),'股數':pd.Series(dtype=int)}),
                num_rows='dynamic',hide_index=True,key='dahu_opening_rights')
            st.write('事前買賣計畫（加入第一筆回報後，計畫固定）')
            plans = st.data_editor(pd.DataFrame([{'委託代號':'','股票代號':'','買賣':'買進','盤別':'零股',
                '股數':1,'限價（元）':'','預算（元）':'','訊號日':str(today-timedelta(days=1))}]),
                column_config={'買賣':st.column_config.SelectboxColumn(options=['買進','賣出'],required=True),
                    '盤別':st.column_config.SelectboxColumn(options=['整股','零股'],required=True)},
                num_rows='dynamic',hide_index=True,key='dahu_opening_plans')
            st.caption('買進預算須包含限價總額與費用；賣出預算填0。訊號日必須早於交易日。')
            if st.form_submit_button('建立當日草稿'):
                try:
                    st.session_state['dahu_entry_draft'] = new_document(str(day),cash,int(slots),
                        held.to_dict('records'),rights.to_dict('records'),plans.to_dict('records'))
                    st.rerun()
                except (ValueError,KeyError,TypeError) as exc:
                    st.error('計畫未建立：'+str(exc))
        return
    st.write(f"核對日：{draft['session']}，已整理 {len(draft['events'])} 筆回報。")
    st.caption('變更期初資料或事前計畫須清除草稿重填，避免把事後想法補成事前計畫。')
    kinds={'可用額度快照':'funds_snapshot','已送出委託':'submit','成交回報':'fill',
           '提出撤單':'cancel_request','撤單成功':'cancel_ack','撤單失敗':'cancel_rejected'}
    kind=kinds[st.selectbox('加入哪種紀錄',list(kinds),key='dahu_entry_kind')]
    with st.form('dahu_event_form',clear_on_submit=False):
        eid=st.text_input('本筆紀錄編號',f"E{len(draft['events'])+1}")
        occurred=st.text_input('券商事件時間（台北 HH:MM:SS，可加小數秒）','09:00:00')
        received=st.text_input('你實際收到／看到回報的時間','09:00:01')
        proof=st.file_uploader('附上已遮蔽帳號與個資的證據檔',type=['png','jpg','jpeg','pdf','csv','json','txt'],key='dahu_event_proof')
        fields={}
        if kind=='funds_snapshot':
            available=st.text_input('畫面顯示的可用买進額度（元）','0')
            choices=['期初，尚無委託']+[e['id'] for e in draft['events'] if e['kind']!='funds_snapshot']
            covers=st.selectbox('確認這個額度已包含哪一筆最新回報',choices)
            fields['covers_through']=None if covers==choices[0] else covers
        else:
            fields['order_id']=st.selectbox('對應事前委託', [p['order_id'] for p in draft['plans']])
        if kind=='fill':
            qty=st.text_input('本筆成交股數','1');price=st.text_input('成交價（元）','0')
            fee=st.text_input('本筆手續費（元）','0');tax=st.text_input('本筆交易稅（元）','0')
        if kind in ('cancel_ack','cancel_rejected'):
            fields['request_id']=st.text_input('對應的提出撤單紀錄編號')
        if kind=='cancel_ack':
            filled=st.text_input('券商回報的累計成交股數','0')
            cancelled=st.text_input('券商確認取消的剩餘股數','1')
        if st.form_submit_button('核對並加入回報'):
            try:
                if kind=='funds_snapshot':fields['available_cents']=cents(available)
                if kind=='fill':fields.update(qty=shares(qty,1),price_cents=cents(price),fee_cents=cents(fee),tax_cents=cents(tax))
                if kind=='cancel_ack':fields.update(cumulative_filled_qty=shares(filled),cancelled_qty=shares(cancelled,1))
                st.session_state['dahu_entry_draft']=append_event(draft,kind,eid,occurred,received,
                    proof.getvalue() if proof is not None else b'',fields)
                st.rerun()
            except (ValueError,KeyError,TypeError) as exc:
                st.error('未加入這筆回報：'+str(exc))
    if draft['events']:
        result=audit(draft)
        st.success('目前提供的時序及金額一致；尚未驗證券商原始回報真實性與完整性。')
        st.dataframe(pd.DataFrame([{'編號':r['id'],'事件':next(k for k,v in kinds.items() if v==r['kind']),
            '現金（元）':r['cash_cents']/100,'可用額度（元）':r['available_cents']/100,
            '預留（元）':r['reserved_cents']/100} for r in result['decisions']]),hide_index=True)
        if result['open_orders']:st.warning('尚未終結的委託：'+', '.join(result['open_orders']))
        st.download_button('下載可供再次核對的回報JSON',json.dumps(draft,ensure_ascii=False,indent=2),
            'dahu-manual-receipts.json','application/json',key='dahu_entry_export')
        if st.button('移除最後一筆，重新更正',key='dahu_entry_undo'):
            st.session_state['dahu_entry_draft']={**draft,'events':draft['events'][:-1]};st.rerun()
    if st.button('清除草稿，重新填寫',key='dahu_entry_reset'):
        del st.session_state['dahu_entry_draft'];st.rerun()


def render():
    with st.expander('大戶投：賣出／撤單後可以補位嗎？',expanded=False):
        st.write('依序核對：事前買賣計畫 → 成交或撤單成功回報 → 更新可用額度 → 下一筆買進。')
        st.caption('按下刪單不等於已撤單。部分成交、剩1股及待交付股票權利仍占名額；賣款不會自動變成可用買進額度。')
        st.info('此工具只檢查你提供的紀錄，不操作大戶投。真實執行驗證仍需完整的事前計畫、成交／撤單時序和可用額度證據；僅有庫存或損益截圖不足以完成。')
        st.markdown('[查看回報準備方式與欄位說明](https://www.sinotrade.com.tw/richclub/manual.pdf)（永豐官方操作手冊）')
        method=st.radio('整理紀錄方式',['中文逐筆填寫','上傳整理好的JSON'],horizontal=True,key='dahu_input_method')
        if method=='中文逐筆填寫':
            render_entry()
            return
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
