"""Reviewable evidence workflows; all writes require an explicit save button."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from app import forward_journal as j, forward_portfolio as p, forward_restatement as restatement, forward_halts as halts, forward_odd_lot as odd


def select_version(original,account):
    options=[dict(path=str(original),created_at='原始前向紀錄')]+restatement.versions(original)
    selected=st.selectbox('帳本版本',options,format_func=lambda x:'原始帳本' if x['path']==str(original) else '更正版本｜'+x['created_at'],key='evidence_version_'+account)
    if selected['path']!=str(original):st.warning('目前是更正版本：供會計對帳及後續紙上記帳使用，不能當成原始前向績效。原始版本仍保留。')
    return Path(selected['path'])


def proof_fields(prefix):
    return dict(reference=st.text_input('回報／文件編號',key=prefix+'ref'),reviewer=st.text_input('核對人',key=prefix+'reviewer'),
        reason=st.text_input('核對／更正原因',key=prefix+'reason'),text=st.text_area('證據文字（至少20字；請勿貼登入資訊）',key=prefix+'text'))


def render_restatement(path,account):
    with st.expander('帳本更正：先比較差異，再另存版本'):
        st.write('用於遲到成交、漏登權益或誤填費稅。系統會重算整段帳本；不會自動補造成交、公告或價格。')
        data=restatement.read(path)
        if any(r['kind']=='restatement_lineage' for r in data):
            st.info('可以更正此版本及其後續成交；將再另存新版本，來源鏈與後續紀錄都保留。')
        candidates=[r for r in data if r['kind'] in restatement.EDITABLE]
        if not candidates:st.info('尚無可核對的成交或收盤紀錄。');return
        mode=st.radio('更正方式',['更正一筆','作廢一筆','補登／多筆更正'],horizontal=True,key='repair_mode_'+account)
        if mode!='補登／多筆更正':
            target=st.selectbox('原始紀錄',candidates,format_func=lambda r:f"{r['kind']}｜{r['event_key']}｜{r['recorded_at']}",key='repair_target_'+account)
            if mode=='更正一筆':
                raw=st.text_area('更正後完整內容 JSON',json.dumps(target['body'],ensure_ascii=False,indent=2),height=220,key='repair_body_'+account+target['hash'])
            else:raw=None
        else:
            st.caption('每項操作填 replace／void／insert；補登使用 before 指定原始紀錄 hash，必須在相應收盤之前。')
            template=[dict(op='insert',before=candidates[-1]['hash'],kind='fill',body=dict(order_id='請填既有委託編號',qty=1,price='1',fee='20',tax='0',executed_at='請填實際時間及時區',evidence=dict(source='paper_execution_report',report_id='請填實際回報編號')))]
            raw=st.text_area('更正操作清單 JSON',json.dumps(template,ensure_ascii=False,indent=2),height=240,key='repair_ops_'+account)
        proof=proof_fields('repair_'+account)
        key='repair_preview_'+account
        if st.button('預覽現金、成本及每日資產差異',key='repair_preview_btn_'+account):
            st.session_state.pop(key,None)
            try:
                ops=json.loads(raw) if mode=='補登／多筆更正' else [dict(op='void',target=target['hash']) if mode=='作廢一筆' else dict(op='replace',target=target['hash'],body=json.loads(raw))]
                st.session_state[key]=restatement.preview(path,ops,proof)
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
        preview=st.session_state.get(key)
        if preview and preview['command']['source_path']==str(path.resolve()):
            st.caption('下方保存的是這次已產生的預覽；修改上方輸入後需重新預覽。')
            st.dataframe(pd.DataFrame([dict(版本='更正前',**{k:preview['before'][k] for k in ('cash','nav','costs','realized_pnl')}),dict(版本='更正後',**{k:preview['after'][k] for k in ('cash','nav','costs','realized_pnl')})]).rename(columns={'cash':'現金','nav':'總資產','costs':'費稅','realized_pnl':'已實現損益'}),hide_index=True)
            st.json(preview,expanded=False)
            st.download_button('下載更正預覽與操作證據',json.dumps(preview,ensure_ascii=False,indent=2),'restatement-preview.json','application/json',key='repair_download_'+account)
            if st.button('另存更正版本（保留原始帳本）',key='repair_save_'+account):
                try:
                    result=restatement.materialize(path,preview['command']);st.session_state.pop(key,None)
                    st.success('已另存：'+str(result)+'。在上方「帳本版本」選擇查看。')
                except (ValueError,KeyError,OSError) as exc:st.error(str(exc))


def render_halts(path,account):
    with st.expander('停牌公告與估值來源'):
        with j.connection(path) as con:rows=j.read_events(con)
        marks=halts.estimated(rows)
        if marks:st.warning('目前總資產含停牌估值；不可把沿用價格拿來觸發交易。');st.json(marks)
        active=list(halts.notices(rows).values())
        if active:st.dataframe(pd.DataFrame([dict(代號=r['body']['stock_id'],停牌起日=r['body']['halt_start'],恢復日期=r['body']['resume_date'] or '未定',已撤回=r['body']['withdrawn']) for r in active]),hide_index=True)
        st.caption('僅處理完整交易日停牌。請核對證交所、櫃買或公司公告；一般缺價不能勾成停牌。恢復日不確定可留白，恢復時新增修訂公告。')
        with st.form('halt_form_'+account):
            sid=st.text_input('停牌股票代號（四碼）')
            start=st.text_input('停止交易起日 YYYY-MM-DD')
            resume=st.text_input('恢復交易日 YYYY-MM-DD（未定留白）')
            withdrawn=st.checkbox('撤回此股票先前停牌紀錄（需更正公告證據）')
            url=st.text_input('公告 HTTPS 連結')
            title=st.text_input('公告標題')
            published=st.text_input('公告發布時間（含時區）')
            reviewer=st.text_input('公告核對人')
            text=st.text_area('停牌相關公告原文（至少20字）')
            save=st.form_submit_button('保存停牌公告／修訂')
        if save:
            try:
                halts.save_notice(path,dict(stock_id=sid,halt_start=start,resume_date=resume or None,withdrawn=withdrawn),dict(url=url,title=title,published_at=published,reviewer=reviewer,text=text));st.success('已保存公告證據，不改寫既有收盤或成交。');st.rerun()
            except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))


def render_odd(path,account):
    with st.expander('零股行情與成交證據比對'):
        data=p.summary(path)
        orders=[x for x in data['orders'] if x['channel']=='odd']
        st.caption('直接查證交所 MIS 盤中零股五檔；上市／上櫃分開。15秒共用快取，不耗 FinMind 額度。過期行情只供查閱，揭示量不保證成交。')
        if not orders:st.info('尚無零股委託可比對。');return
        order=st.selectbox('要比對的零股委託',orders,format_func=lambda x:f"{x['stock_id']}｜{x['order_id']}｜{x['session']}",key='odd_order_'+account)
        market=st.radio('上市或上櫃',['上市','上櫃'],horizontal=True,key='odd_market_'+account)
        market_code='tse' if market=='上市' else 'otc'
        if st.button('取得此股票零股五檔',key='odd_refresh_'+account):
            try:
                with st.spinner('查詢官方零股行情…'):odd.refresh(market_code,order['stock_id'])
            except (ValueError,OSError) as exc:st.error(str(exc))
        sources=[r for r in odd.history() if r['body'].get('query')==[market_code,order['stock_id']]]
        latest=sources[-1] if sources else None
        if latest:
            if latest['body']['status']!='ok':st.error('最近行情取得失敗：'+latest['body']['error'])
            else:
                quote=latest['body']['quote'];assessment=odd.assess(quote,order,j.now().isoformat())
                st.write('行情揭示：'+quote['quote_at']+'；本機取得：'+quote['retrieved_at'])
                st.metric('委託限價內可見股數',f"{assessment['visible_shares_at_limit']:,} 股")
                if assessment['reasons']:st.warning('；'.join(assessment['reasons']))
                st.dataframe(pd.DataFrame([dict(買賣='賣方',價格=x['price'],股數=x['shares']) for x in quote['asks']]+[dict(買賣='買方',價格=x['price'],股數=x['shares']) for x in quote['bids']]),hide_index=True)
        report=odd.review(path)
        if report['fills']:st.dataframe(pd.DataFrame(report['fills']).drop(columns=['fill_hash']),hide_index=True)
        fills=[r for r in data['rows'] if r['kind']=='fill' and r['body']['order_id']==order['order_id']]
        successful=[r for r in sources if r['body']['status']=='ok']
        if fills and successful:
            fill=st.selectbox('已登錄成交',fills,format_func=lambda r:r['body']['evidence']['report_id'],key='odd_fill_'+account)
            source=st.selectbox('比對哪次零股行情',successful,format_func=lambda r:r['body']['quote']['quote_at']+'｜'+r['hash'][:8],key='odd_source_'+account)
            proof=proof_fields('oddproof_'+account)
            if st.button('附上成交憑證並比對（不新增成交）',key='odd_attach_'+account):
                try:
                    odd.attach(path,fill['hash'],source['hash'],proof);st.success('已保存證據比對；現金與持股不變。');st.rerun()
                except (ValueError,KeyError,TypeError) as exc:st.error(str(exc))
        st.download_button('下載零股執行比對',json.dumps(report,ensure_ascii=False,indent=2),'odd-lot-execution-review.json','application/json',key='odd_review_download_'+account)
