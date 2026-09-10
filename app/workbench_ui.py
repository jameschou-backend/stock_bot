"""Streamlit workbench: decisions, recorded fills and explicit research jobs."""
from datetime import date, datetime, time as daytime, timezone
from zoneinfo import ZoneInfo
from uuid import uuid4
import pandas as pd
import streamlit as st
from app import workbench_service as service, workbench_ledger as ledger, workbench_jobs as jobs
from app.db import get_session


@st.cache_data(ttl=30,show_spinner=False)
def status_data(): return service.data_status()


@st.cache_data(ttl=30,show_spinner=False)
def candidate_data(): return service.candidates()


def percent(value): return '尚未計算' if value is None else f'{value:.1%}'


def money(value): return '待更新' if value is None else f'{value:,.0f}'


def action(fn,*args,**kwargs):
    try:
        with get_session() as session:
            result=fn(session,*args,**kwargs)
        st.success('已儲存')
        return True,result
    except ValueError as exc:
        st.error(str(exc))
        return False,None


def render():
    st.set_page_config(page_title='台股投資工作台',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
    st.markdown('''<style>
    .block-container {max-width:1220px;padding-top:4rem;padding-bottom:3rem}
    h1 {font-size:2rem!important;letter-spacing:-.04em}
    [data-testid="stMetric"] {border:1px solid #dde4eb;border-radius:14px;padding:16px 20px}
    [data-testid="stMetricLabel"] {font-size:.85rem}
    [data-testid="stMetricValue"] {font-size:1.5rem;white-space:normal}
    [data-testid="stMetricValue"] > div {white-space:normal;overflow:visible}
    [data-testid="stTabs"] button {font-size:1rem;padding-left:18px;padding-right:18px}
    .wb-eyebrow {color:#197b80;font-weight:700;letter-spacing:.12em;font-size:12px}
    .wb-note {color:#637487;font-size:14px;line-height:1.6}
    </style>''',unsafe_allow_html=True)
    st.markdown('<div class="wb-eyebrow">STOCK BOT · 台股波段研究</div>',unsafe_allow_html=True)
    title,refresh=st.columns([5,1])
    with title:
        st.title('讓每一筆交易，都有依據')
        st.caption('先看資料，再訂計畫；成交後，把成本與結果記清楚。')
    with refresh:
        if st.button('重新整理',use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    try:
        status=status_data()
        candidates=candidate_data()
    except Exception as exc:
        st.error(f'資料服務尚未就緒 ({type(exc).__name__})。請先啟動 MySQL，並執行 make workbench-init。')
        st.stop()
    quota=status['quota']
    cols=st.columns(4)
    cols[0].metric('股價更新至',status['price_date'] or '尚無資料')
    cols[1].metric('近 60 分鐘已用請求',f"{quota['requests_in_window']:,}",help='本機共用帳本，包含重試；其他裝置的使用量無法在此看到。')
    cols[2].metric('系統剩餘額度',f"{quota['remaining_requests']:,}",help='Sponsor 上限 6,000，系統預設使用 5,400，保留 600 次緩衝。')
    cols[3].metric('策略狀態','研究驗證中',help='尚未有通過新驗證的實盤策略。')
    if quota['retry_after_seconds']>0:
        st.warning(f"FinMind 暫停請求，約 {quota['retry_after_seconds']/60:.0f} 分鐘後可重試；已取得的資料仍可查看。")
    if status['problems']:
        st.warning('；'.join(status['problems']))
    today,holdings,news,flow,research=st.tabs(['今日觀察','持倉與成交','新聞與題材','族群資金','策略驗證'])
    with today:
        st.subheader('先挑值得研究的股票')
        st.caption('排名是模型排序，不是上漲機率。目前名單供研究與紙上追蹤。')
        if candidates:
            frame=pd.DataFrame(candidates)
            st.dataframe(frame[['rank','stock_id','name','market','price','price_date','signal_date']].rename(columns={
                'rank':'排名','stock_id':'代號','name':'股票','market':'市場','price':'最近收盤','price_date':'報價日期','signal_date':'訊號日期'}),
                hide_index=True,use_container_width=True)
        else:
            st.info('還沒有候選名單。先完成資料更新與選股流程。')
        st.divider()
        st.subheader('把想法變成交易計畫')
        st.caption('計畫只會預留帳本資金，不會送出券商委託。估算停損仍可能遇到跳空與滑價。')
        account=st.radio('使用哪份帳本',['paper','real'],format_func=lambda v:'紙上練習' if v=='paper' else '實際成交紀錄',horizontal=True,key='plan_account')
        book=service.portfolio(account)
        if not book['initialized']:
            st.info('請先到「持倉與成交」設定這份帳本的期初現金。')
        else:
            c1,c2,c3=st.columns(3)
            options=[c['stock_id'] for c in candidates]
            names={c['stock_id']:c['name'] for c in candidates}
            sid=c1.selectbox('研究標的',options or ['2330'],format_func=lambda x:f"{x} {names.get(x,'')}")
            latest=next((c['price'] for c in candidates if c['stock_id']==sid),None)
            entry=c2.number_input('預計買價',min_value=.01,value=float(latest or 100),step=.5,key=f'entry_{sid}')
            stop=c3.number_input('離場警戒價',min_value=.01,value=round(float(latest or 100)*.9,2),step=.5,key=f'stop_{sid}')
            c1,c2,c3=st.columns(3)
            risk=c1.number_input('每筆最多承擔資金的 (%)',min_value=.1,max_value=5.,value=1.,step=.1)
            allocation=c2.number_input('單檔最多占資金 (%)',min_value=1.,max_value=100.,value=15.,step=1.)
            lot=c3.selectbox('交易單位',[1,1000],format_func=lambda x:'零股（1 股）' if x==1 else '整張（1,000 股）')
            st.caption('上方風險比例是試算初始值，可自行調整；不代表已確認你的投資限制。')
            try:
                if book['equity'] is None:
                    st.warning('持股報價缺漏，暫時無法按總資產計算新部位。')
                else:
                    exposure=sum(p['qty']*p['price'] for p in book['positions'] if p['stock_id']==sid)
                    exposure+=sum((p['qty']-p['filled_qty'])*p['entry_price'] for p in book['plans']
                                  if p['stock_id']==sid and p['status']=='open')
                    preview=ledger.preview_plan(book['available_cash'],book['equity'],entry,stop,risk,allocation,lot,exposure)
                    c1,c2,c3=st.columns(3)
                    c1.metric('試算數量',f"{preview['qty']:,} 股")
                    c2.metric('預計占用現金',money(preview['estimated_cash']))
                    c3.metric('估計停損與成本',money(preview['estimated_loss']))
                    reason=st.text_input('記下進場理由',max_chars=500,placeholder='例如：營收改善，等待回檔確認')
                    if st.button('儲存研究計畫',type='primary',disabled=preview['qty']<=0 or not status['data_ready']):
                        ok,_=action(ledger.create_plan,account,sid,entry,stop,preview['qty'],reason)
                        if ok: st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
        with st.expander('資料狀態與更新'):
            st.dataframe(pd.DataFrame(status['markets']).rename(columns={'market':'市場','date':'最新日期','stocks':'股票數'}),hide_index=True,use_container_width=True)
            st.caption(status['adjustment_note'])
            st.caption('Sponsor 更新使用 FinMind 股價路徑；已完成區間會跳過，月營收按月份抓取。')
            if st.button('更新資料與候選名單'):
                try:
                    job=jobs.submit(jobs.WorkRequest(kind='update_data'))
                    st.success(f"工作已啟動：{job['job_id'][:8]}。可到「策略驗證」查看進度。")
                except (ValueError,TimeoutError) as exc: st.warning(str(exc))
    with holdings:
        render_ledger()
    with research:
        render_research(status)
    with news:
        render_news_research()
    with flow:
        render_chain_flow()
    st.divider()
    render_jobs()


def render_ledger():
    st.subheader('看見扣除成本後的結果')
    account=st.radio('帳本',['paper','real'],format_func=lambda v:'紙上練習' if v=='paper' else '實際成交紀錄',horizontal=True,key='ledger_account')
    book=service.portfolio(account)
    if not book['initialized']:
        st.info('填入這份帳本的期初現金，從第一筆成交開始記錄。兩份帳本的資金與損益分開計算。')
        with st.form(f'account_{account}'):
            capital=st.number_input('期初現金（元）',min_value=1.,max_value=1e10,value=None,placeholder='輸入你要追蹤的金額')
            if st.form_submit_button('建立帳本',type='primary'):
                if capital is None: st.error('請填寫期初現金')
                else:
                    ok,_=action(ledger.initialize_account,account,capital)
                    if ok: st.rerun()
        return
    c=st.columns(4)
    c[0].metric('可用現金',money(book['available_cash']))
    c[1].metric('已實現損益',money(book['realized_pnl']))
    c[2].metric('未實現損益',money(book['unrealized_pnl']))
    c[3].metric('合計淨損益',money(book['net_pnl']))
    st.caption(f"累計手續費與稅：{money(book['fees_and_tax'])} 元 · 未完成計畫預留：{money(book['reserved_cash'])} 元。未實現損益依表內報價日期估值，尚未扣未來賣出成本。")
    st.caption('目前帳本追蹤現金買賣；現金股利、股票分割與後續入出金尚未納入。')
    if book['missing_quotes']: st.warning('以下持股缺少報價，合計估值暫不顯示：'+', '.join(book['missing_quotes']))
    if book['funding_shortfall']: st.warning('部分未完成計畫的預留資金不足，請取消或重新規劃。')
    if book['positions']:
        st.dataframe(pd.DataFrame(book['positions']).rename(columns={'stock_id':'代號','qty':'股數','average_cost':'含費平均成本','price':'收盤價','price_date':'報價日期','unrealized_pnl':'未實現損益','cost_basis':'剩餘成本'}),hide_index=True,use_container_width=True)
    else: st.info('尚無持倉。下方記錄成交後會更新。')
    open_plans=[p for p in book['plans'] if p['status']=='open']
    if open_plans:
        with st.expander(f'未完成計畫（{len(open_plans)}）',expanded=True):
            for p in open_plans:
                c1,c2=st.columns([5,1])
                c1.write(f"{p['stock_id']} · {p['entry_price']:g} 元 · 已完成 {p['filled_qty']:,} / {p['qty']:,} 股 · 警戒 {p['stop_price']:g}")
                if c2.button('取消剩餘',key=p['plan_id']):
                    ok,_=action(ledger.cancel_plan,account,p['plan_id'])
                    if ok: st.rerun()
    st.subheader('記錄一筆成交')
    st.caption('填入券商成交回報或紙上模擬的數字。這個功能只記帳，不會下單。')
    fid_key=f'fill_id_{account}'
    if fid_key not in st.session_state: st.session_state[fid_key]=uuid4().hex
    saved_key=f'fill_saved_{account}'
    if st.session_state.get(saved_key):
        st.success('上一筆成交已入帳。核對上方損益後，再開始下一筆。')
        if st.button('記錄下一筆成交',key=f'next_fill_{account}'):
            st.session_state[fid_key]=uuid4().hex
            st.session_state[saved_key]=False
            st.rerun()
        render_fills(book,account)
        return
    with st.form(f'fill_{account}',clear_on_submit=False):
        c1,c2,c3=st.columns(3)
        sid=c1.text_input('股票代號',max_chars=4,value=st.session_state.get('last_fill_stock',''))
        side=c2.selectbox('成交方向',['buy','sell'],format_func=lambda x:'買進' if x=='buy' else '賣出')
        qty=c3.number_input('成交股數',min_value=1,step=1,value=100)
        c1,c2,c3=st.columns(3)
        price=c1.number_input('成交價格',min_value=.01,value=100.,step=.1)
        fee=c2.number_input('實付手續費',min_value=0.,value=0.,step=1.)
        tax=c3.number_input('實付交易稅',min_value=0.,value=0.,step=1.)
        c1,c2,c3=st.columns(3)
        now=datetime.now(ZoneInfo('Asia/Taipei'))
        day=c1.date_input('成交日期',value=now.date(),max_value=now.date())
        clock=c2.time_input('成交時間（台北）',value=now.time().replace(second=0,microsecond=0))
        plan_id=c3.selectbox('對應計畫（買進選填）',[None]+[p['plan_id'] for p in open_plans],format_func=lambda x:'無對應計畫' if x is None else next(f"{p['stock_id']} 剩 {p['qty']-p['filled_qty']} 股" for p in open_plans if p['plan_id']==x))
        if st.form_submit_button('儲存成交紀錄',type='primary'):
            executed=datetime.combine(day,clock,tzinfo=ZoneInfo('Asia/Taipei'))
            ok,_=action(ledger.record_fill,account,st.session_state[fid_key],sid,side,int(qty),price,fee,tax,executed,plan_id)
            if ok:
                st.session_state[saved_key]=True
                st.session_state.last_fill_stock=sid
                st.rerun()
    render_fills(book,account)


def render_fills(book,account):
    if book['fills']:
        with st.expander('成交明細與匯出'):
            frame=pd.DataFrame(book['fills'])
            st.dataframe(frame.rename(columns={'stock_id':'代號','side':'方向','qty':'股數','price':'價格','fee':'手續費','tax':'交易稅','executed_at':'成交時間 UTC'}),hide_index=True,use_container_width=True)
            st.download_button('下載成交明細',frame.to_csv(index=False).encode('utf-8-sig'),file_name=f'{account}-fills.csv',mime='text/csv')


def render_research(status):
    from app.capacity_research_ui import render as render_capacity
    from app.regime_switch_ui import render as render_regime_switch
    from app.diffusion_research_ui import render as render_diffusion
    from app.guidance_research_ui import render as render_guidance
    st.subheader('策略能不能用，讓證據回答')
    st.info('目前沒有通過新驗證的實盤策略。下方回測用來檢查假設，不會自動啟用策略。')
    render_capacity()
    st.divider()
    render_regime_switch()
    st.divider()
    render_diffusion()
    st.divider()
    render_guidance()
    st.divider()
    render_event_group_research()
    st.divider()
    render_revenue_research()
    st.divider()
    render_theme_research()
    st.divider()
    render_flow_research()
    st.divider()
    if st.checkbox('顯示前一輪價格研究（尚未排除上市櫃前行情）'):
        render_rule_research()
    st.divider()
    st.subheader('模型回測')
    with st.form('research'):
        c1,c2,c3=st.columns(3)
        months=c1.selectbox('驗證區間（月）',[3,6,12,24,60,120],index=2)
        topn=c2.selectbox('持有檔數',[4,6,10,20,30],index=2)
        stop=c3.slider('停損比例 (%)',min_value=5,max_value=30,value=12)
        quick=st.checkbox('先做快速試跑（較少模型樹數，結果不能與正式回測混用）')
        st.caption('固定：訊號延遲 1 個交易日、含來回稅費與滑價、滾動訓練窗 730 天。相同訓練問題會重用模型；首次完整回測仍需時間。')
        if st.form_submit_button('開始背景回測',type='primary'):
            try:
                job=jobs.submit(jobs.WorkRequest(kind='backtest',months=months,topn=topn,stoploss=-stop/100,quick=quick))
                st.success(f"工作 {job['job_id'][:8]} 已啟動，可繼續使用其他頁面。")
            except (ValueError,TimeoutError) as exc: st.warning(str(exc))
    with st.expander('目前還需要確認的證據'):
        evidence=service.strategy_evidence()
        for item in evidence['validation_requirements']: st.write('• '+item)
        st.caption(status['adjustment_note'])
        st.write('已有研究紀錄：')
        for item in evidence['documents'][-8:]: st.caption(item['name'])


def render_news_research():
    from app.news_research import overview
    from skills.news_radar import THEMES, EVENTS, STATUS
    st.subheader('從新聞找線索，再核對公司是否受惠')
    st.caption('規則版：辨識 11 組題材、供給、報價、接單、量產與獲利線索；不是全文 AI 判讀，也不是買進排名。')
    mode=st.radio('新聞研究方式',['scan','review'],horizontal=True,
                  format_func=lambda m:'看最近題材' if m=='scan' else '回到歷史日期')
    if mode=='scan':
        a,b=st.columns(2)
        fetch=a.button('更新近 7 天並分析',type='primary')
        offline=b.button('只分析本機近 7 天')
        st.caption('更新正常最多 7 次 FinMind 請求；一小時內重按優先重用日快取。分析本機資料為 0 次請求。')
        if fetch or offline:
            try:
                job=jobs.submit(jobs.WorkRequest(kind='news_scan',news_days=7,fetch_news=fetch))
                st.success(f"新聞工作 {job['job_id'][:8]} 已啟動；下方可看進度，完成後按頁首「重新整理」。")
            except (ValueError,TimeoutError) as exc: st.warning(str(exc))
    else:
        with st.form('historical_news'):
            a,b,c=st.columns(3)
            sid=a.text_input('回顧股票代號',value='2408',max_chars=4)
            cutoff=b.date_input('回到哪一天判讀',value=date(2025,7,10),max_value=datetime.now(ZoneInfo('Asia/Taipei')).date())
            days=c.selectbox('往前查看幾天',[30,90,100,180,365],index=2)
            st.caption('只用截止日前的新聞；當日因時區未確認而排除。價格欄只看報導前，不拿後來漲幅替新聞打分。')
            if st.form_submit_button('重建新聞時間線',type='primary'):
                try:
                    job=jobs.submit(jobs.WorkRequest(kind='news_review',news_stock_id=sid,news_end=cutoff,news_days=days))
                    st.success(f"歷史判讀 {job['job_id'][:8]} 已啟動；完成後按頁首「重新整理」。")
                except (ValueError,TimeoutError) as exc: st.warning(str(exc))
    report=overview(mode)
    if not report['available']:
        st.info(report['note'])
        return
    st.divider()
    description=(f"{report['stock_id']} {report['stock_name']} · 判讀截止 {report['cutoff']}" if mode=='review' else '最近一次完成的新聞分析')
    st.write('**目前顯示：** '+description)
    st.caption(f"新聞日期 {report['start']}～{report['end']} · 分析完成 {report['analyzed_at']} · 耗時 {report['elapsed_seconds']:.2f} 秒")
    if mode=='scan' and (datetime.now(ZoneInfo('Asia/Taipei')).date()-date.fromisoformat(report['end'])).days>1:
        st.warning('這份近期新聞結果已過期，請更新；股價新鮮不代表新聞也已更新。')
    c=st.columns(3)
    c[0].metric('去重後新聞',f"{report['unique_articles']:,}")
    c[1].metric('合併重複列',f"{report['duplicates_collapsed']:,}")
    c[2].metric('涵蓋題材',len(report['themes']))
    st.info(report['evidence_note'])
    st.caption(report['time_note'])
    if not report['themes']:
        st.warning('這個日期範圍沒有可辨識題材；不能據此認定市場沒有題材。')
    if report['stories']:
        table=[{'題材':t['name'],'新聞數':t['articles'],'營運線索':t['operating_clues'],
                '預期／概念':t['expectations'],'負面／正反並存':t['negative_or_mixed'],
                '股價評論':t['price_commentary'],'關聯公司':len(t['stock_ids'])} for t in report['themes']]
        if table: st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
        selected=st.selectbox('挑選要讀的題材',['all']+[t['id'] for t in report['themes']],
                              format_func=lambda x:'全部新聞（含未分類）' if x=='all' else THEMES[x][0],key=f'news_topic_{mode}')
        focus=st.checkbox('先看營運、預期與負面線索',value=True,key=f'news_focus_{mode}')
        query=st.text_input('搜尋日期、公司或新聞文字',key=f'news_search_{mode}',
                            placeholder='例如：2025-05-16、南亞科、漲價').strip().casefold()
        stories=[s for s in report['stories'] if (selected=='all' or selected in s['themes'])
                 and (not focus or s['status'] in ('operating_clue','expectation','negative_or_mixed'))
                 and (not query or query in ' '.join([s['source_date'],s['title'],*s['stock_ids'],
                       *(report['names'].get(sid,'') for sid in s['stock_ids'])]).casefold())]
        if mode=='scan': stories=list(reversed(stories))
        rows=[]
        for s in stories:
            item={'日期':s['source_date'],'新聞':s['title'],'判讀':STATUS[s['status']],
                  '線索':'、'.join(EVENTS[e][0] for e in s['events']),
                  '標題點名':'、'.join(sid+' '+report['names'].get(sid,'') for sid in s['headline_named_ids']),
                  '供應商關聯':'、'.join(s['stock_ids']),'來源':'、'.join(s['sources'])}
            if mode=='review':
                p=s.get('price_context')
                item['報導前20日超額']=f"{p['excess_20d']*100:+.1f} 個百分點" if p else '行情不足'
            rows.append(item)
        st.caption(f'符合篩選 {len(rows)} 篇，表格先列 {min(100,len(rows))} 篇；完整結果可下載。同題材和供應商關聯不是直接受惠證據。')
        if rows: st.dataframe(pd.DataFrame(rows[:100]),hide_index=True,use_container_width=True)
        if stories:
            story_id=st.selectbox('查看一篇新聞的依據',[s['id'] for s in stories[:100]],
                                 format_func=lambda x:next(s['source_date']+' '+s['title'][:65] for s in stories if s['id']==x),
                                 key=f'news_story_{mode}_{selected}_{focus}')
            s=next(s for s in stories if s['id']==story_id)
            st.text(s['title'])
            st.write('**觸發詞：** '+'、'.join(s['matched_terms']))
            st.write('**判讀：** '+STATUS[s['status']]+'；應再核對原文的主詞、時程、否定語句與數字。')
            st.caption(f"供應商時間 {s['provider_datetime']}；首次本機記錄 {s['first_recorded_at']}。")
            for i,url in enumerate(s['links'][:3]): st.link_button(f'開啟新聞來源 {i+1}',url)
            if mode=='review':
                p=s.get('price_context')
                if p:
                    st.write(f"截至報導前 {p['as_of']}：股票 20 日報酬 {percent(p['stock_return_20d'])}，0050 {percent(p['benchmark_return_20d'])}。")
                    st.caption('報導前已相對 0050 上漲至少 10 個百分點；需注意價格可能已先反應。' if p['already_outperforming_10pp']
                               else '報導前超額未達 10 個百分點；這不能證明股價尚未反映題材。')
                else: st.caption('固定價格快照不足，這篇不顯示價格先後判讀。')
    with st.expander('覆蓋範圍、未分類題材與使用限制'):
        st.write(f"舊新聞庫最後一筆：{report['source']['legacy_latest'] or '無'}；本次取用 {len(report['source']['local_days'])} 天日快取。")
        st.caption(report['source']['coverage_note'])
        if report['unclassified_topic_phrases']:
            st.write('尚待人工命名的題材詞：'+ '、'.join(p['phrase'] for p in report['unclassified_topic_phrases']))
        st.caption('去重只處理相同標題與來源尾綴；不同標題仍可能在轉載同一事件。來源家數不等於獨立證據數。')
        if mode=='review': st.caption(report['price_source'].get('note',''))
        st.caption('目前沒有全文抽取、事件因果證明或自動選股下單；標題規則可能誤判否定句與受惠主體。')
    st.download_button('下載這次新聞研究',__import__('json').dumps(report,ensure_ascii=False),
                       file_name=f'news-{mode}.json',mime='application/json',key=f'news_download_{mode}')


def render_chain_flow():
    from app.chain_flow_research import overview
    st.subheader('題材有熱度，資金有跟上嗎？')
    st.caption('先看成交占比是否增加，再看法人方向、收紅家數與龍頭集中度。成交金額不等於淨流入。')
    a,b=st.columns(2)
    update=a.button('更新族群資金',type='primary')
    offline=b.button('只重算族群快取')
    st.caption('初次最多 32 次請求；後續新增交易日通常約 3 次。已取得的歷史日快照保留，開頁面不抓資料。')
    if update or offline:
        try:
            job=jobs.submit(jobs.WorkRequest(kind='chain_flow',fetch_flow=update))
            st.success(f"族群研究 {job['job_id'][:8]} 已啟動；完成後按頁首「重新整理」。")
        except (ValueError,TimeoutError) as exc: st.warning(str(exc))
    report=overview()
    if not report['available']:
        st.info(report['note']);return
    st.write(f"**資料截至 {report['as_of']}** · 分析 {report['elapsed_seconds']:.2f} 秒")
    st.caption(f"近 5 日 {report['recent_start']}～{report['as_of']}，對照自 {report['start']} 起的前 20 日。產業鏈成分更新至 {report['members_update_max']}。")
    if (datetime.now(ZoneInfo('Asia/Taipei')).date()-date.fromisoformat(report['as_of'])).days>3:
        st.warning('這份族群資料已超過 3 個日曆日，請更新確認。')
    st.info('供應商部分產業鏈含興櫃；下方公司與法人只分析本機上市櫃普通股。成分或金額覆蓋不足時標示「資料待核對」，不能解讀成沒有資金。')
    focus=st.checkbox('先看記憶體、被動元件、光通訊、伺服器、散熱與衛星相關',value=True)
    groups=[g for g in report['groups'] if g['available'] and (not focus or g.get('topic') or g['name']=='被動元件')]
    rows=[]
    for g in groups:
        rows.append({'族群':g['name'],'近5日成交占比':f"{g['share_5d_pct']:.2f}%",
                     '前20日':f"{g['share_previous20_pct']:.2f}%",'占比變化':f"{g['share_change_pp']:+.2f} 個百分點",
                     '當日收紅':percent(g['intraday_up_fraction']),'最大一檔占比':percent(g['top1_share']),
                     '判讀':g['reading']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    if not groups: return
    chosen=st.selectbox('查看族群資金細節',[g['name'] for g in groups])
    g=next(x for x in groups if x['name']==chosen)
    a,b,c=st.columns(3)
    def estimated_cash(value):
        if value is None: return '資料不足'
        return f'{value/1e4:+,.1f} 萬' if abs(value)<1e8 else f'{value/1e8:+,.1f} 億'
    a.metric('當日族群成交額',f"{g['today_money']/1e8:,.1f} 億")
    b.metric('投信近5日估計買賣超',estimated_cash(g['trust_observed_net_est_5d']))
    c.metric('外資近5日估計買賣超',estimated_cash(g['foreign_observed_net_est_5d']))
    st.caption(f"法人數值只含五日完整成分，按淨股數 × 當日收盤價估算；投信覆蓋 {percent(g['trust_coverage'])}，外資覆蓋 {percent(g['foreign_coverage'])}。正值偏買，負值偏賣。")
    gap='未知' if g['money_reconciliation_gap'] is None else f"{g['money_reconciliation_gap']:.2%}"
    st.caption(f"當日行情涵蓋 {g['quoted_members']}/{g['provider_traded_members']} 家；成交金額對帳差異 {gap}。收紅指收盤高於開盤。")
    st.line_chart(pd.DataFrame(g['share_history']).set_index('date').rename(columns={'share_pct':'成交占比 (%)'}))
    st.write('**成交最多的成分股**')
    leaders=[{'代號':x['stock_id'],'公司':x['name'],'成交額（億）':round(x['money']/1e8,2),
              '開盤至收盤':percent(x['intraday_return'])} for x in g['leaders']]
    st.dataframe(pd.DataFrame(leaders),hide_index=True,use_container_width=True)
    st.caption(f"目前保存的新聞中，日期不晚於 {report['as_of']}、符合對應題材且標題點名成分股的文章有 {g['news_named_articles']} 篇。新聞分析時間：{report['news_analyzed_at'] or '尚無分析'}。一般產業總計未加題材詞篩選。")
    with st.expander('計算口徑與缺漏'):
        for note in report['limitations']: st.write('• '+note)
        st.write('供應商成分中未納入本機上市櫃普通股：'+'、'.join(g['excluded_provider_ids']))
        st.write('本機成分缺當日有效行情：'+'、'.join(g['missing_quote_ids']))
    st.download_button('下載族群資金研究',__import__('json').dumps(report,ensure_ascii=False),
                       file_name='chain-flow-research.json',mime='application/json')


def render_theme_research():
    st.subheader('題材出現之後，買進能贏 0050 嗎？')
    report=service.theme_research_overview()
    if not report['available']:
        st.info(report['note'])
        return
    st.caption(f"整理於 {report['observed_at']} · 歷史行情截至 {report['source']['last_date']} · 12 組回放 {report['elapsed_seconds']:.2f} 秒 · 0 次 FinMind 請求")
    st.warning('這是三組事後挑選的歷史案例，尚未建成自動掃描全市場的題材策略；表內不是今天的買進名單。')
    scenario=st.radio('題材回放成本',['stress','base'],horizontal=True,key='theme_scenario',
                     format_func=lambda s:'較高滑價（每邊 0.45%）' if s=='stress' else '基本滑價（每邊 0.30%）')
    policy=st.radio('題材持有方式',['hold','risk_exit'],horizontal=True,key='theme_policy',
                   format_func=lambda p:'買入後持有半年' if p=='hold' else '半年內可提前退出')
    results=[r for r in report['results'] if r['scenario']==scenario and r['policy']==policy]
    cases={c['id']:c for c in report['cases']}
    rows=[]
    for r in results:
        c=cases[r['case_id']]; a=r['summary']; b=r['benchmark_summary']
        rows.append({'題材':c['theme'],'比較期間':a['start']+' ～ '+a['end'],
                     '扣成本報酬':percent(a['total_return']),'同期 0050':percent(b['total_return']),
                     '領先／落後':f"{r['excess_return']*100:+.2f} 個百分點",'最大跌幅':percent(a['max_drawdown'])})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption('各案例獨立本金、期間不同且重疊，不能把三個報酬相加或平均當成一套策略績效。')
    selected=st.selectbox('查看題材證據與受惠候選',list(cases),format_func=lambda x:cases[x]['theme'])
    c=cases[selected]; r=next(x for x in results if x['case_id']==selected)
    st.write('**受惠候選：** '+'、'.join(m['stock_id']+' '+m['name'] for m in c['members']))
    st.markdown(f"**事件來源：**[{c['source_title']}]({c['source_url']}) · {c['source_date']}")
    st.caption(c['evidence_level'])
    st.write('**支持理由：** '+c['positive'])
    st.write('**不利證據：** '+c['counter'])
    st.write('**還需確認：** '+c['needs_verification'])
    st.write(f"**何時買：** 公告日期之後首個交易日（{r['summary']['start']}）收盤，各檔等額；當天不能成交的份額留現金。")
    st.write('**何時退：** '+('預先排定持有 126 個交易日間隔，期末收盤賣出。' if policy=='hold'
             else '最晚持有 126 個交易日間隔；收盤相對進場價下跌至少 15%，或較持有高點回落至少 20%，次日收盤賣出。遇無法成交逐日重試；賣後不再買回。'))
    curve=pd.DataFrame(r['equity_curve']).set_index('date').rename(columns={'strategy':'題材案例','benchmark':'0050'})
    st.line_chart(curve,use_container_width=True)
    st.caption('各以 1 元起算，已扣稅費與滑價；收盤觸發停損，並不保證成交價或最大虧損。')
    if selected=='passive':
        action=report['price_audit']['corporate_action']
        st.markdown(f"國巨有 7 個交易日因面額變更停牌，期間以前值估值、不模擬成交；每股換 4 股的分割已反映在還原價格。[核對依據]({action['source_url']})")
    with st.expander('逐股結果、成交紀錄與資料限制'):
        names={m['stock_id']:m['name'] for m in c['members']}
        st.dataframe(pd.DataFrame([{'代號':p['stock_id'],'公司':names[p['stock_id']],
                                   '原始份額':percent(p['initial_weight']),'該份額淨報酬':percent(p['allocated_return']),
                                   '對案例貢獻':f"{p['pnl_contribution']*100:+.2f} 個百分點",
                                   '已買進':p['entered'],'期末未售':p['unliquidated']} for p in r['per_stock']]),
                     hide_index=True,use_container_width=True)
        a=r['summary']
        st.write(f"平均現金 {percent(a['average_cash_fraction'])}；稅費及滑價合計／期初本金 {percent(a['cost_per_initial_capital'])}；雙邊成交額／期初本金 {a['two_way_turnover']:.2f} 倍。")
        st.write(f"未買進 {a['blocked_entries']} 筆、賣出受阻 {a['blocked_exit_days']} 檔日、持有缺價 {a['held_missing_price_days']} 檔日、期末未售 {a['unliquidated_positions']} 檔；持有時還原價單日變動逾 50% 共 {len(r['large_move_exposures'])} 次。")
        trade_rows=[{'代號':t['stock_id'],'日期':t['date'],'方向':'買進' if t['side']=='buy' else '賣出',
                     '依據日期':t['signal_date'],'原因':{'event':'題材事件','scheduled_horizon':'半年到期',
                         'entry_stop':'跌破進場停損','trailing_stop':'從高點回落'}[t['reason']]} for t in r['trades']]
        st.dataframe(pd.DataFrame(trade_rows),hide_index=True,use_container_width=True)
        st.caption(c['date_basis']+' 系統到整理日才記錄此案例，不能冒充過去即時訊號。')
        st.caption('價格抽查：'+report['price_audit']['scope'])
        for note in report['limitations']: st.caption('• '+note)
    st.download_button('下載題材證據與回放結果',__import__('json').dumps(report,ensure_ascii=False,indent=2),
                       file_name='theme-research.json',mime='application/json')


def render_revenue_research():
    from app.revenue_research import overview
    st.subheader('營收變好，選股有比較準嗎？')
    report=overview()
    if not report['available']:
        st.info(report['note'])
        return
    st.caption(f"30 組比較 {report['elapsed_seconds']:.1f} 秒 · 重算 0 次 FinMind 請求 · 行情截至 {report['source']['last_date']}")
    st.warning('歷史重播：營收尚缺完整公告時間與修訂版本，還原價也仍有對帳差異。以下結果不能當作已驗證的可實現報酬。')
    view=st.selectbox('營收比較情境',['主要比較','再延後 15 天','基本滑價','只看上市','只看上櫃'],key='revenue_view')
    lag,scenario,market={'主要比較':(45,'stress','ALL'),'再延後 15 天':(60,'stress','ALL'),
                         '基本滑價':(45,'base','ALL'),'只看上市':(45,'stress','TWSE'),
                         '只看上櫃':(45,'stress','TPEX')}[view]
    selected=[r for r in report['results'] if (r['lag_days'],r['scenario'],r['market'])==(lag,scenario,market)]
    base=selected[0];bm=base['benchmark_summary']
    st.caption(f"{bm['start']}～{bm['end']} · 營收資料月份的次月 1 日再等 {lag} 天 · 訊號後一交易日成交 · 每邊滑價 {'0.45%' if scenario=='stress' else '0.30%'}，另計稅費")
    table=[{'選股方式':r['name'],'累積淨報酬':percent(r['summary']['total_return']),
            '年化報酬':percent(r['summary']['annualized_return']),
            '最大跌幅':percent(r['summary']['max_drawdown']),
            '平均留現金':percent(r['diagnostics']['average_cash_fraction'])} for r in selected]
    table.append({'選股方式':'0050 買入持有','累積淨報酬':percent(bm['total_return']),
                  '年化報酬':percent(bm['annualized_return']),'最大跌幅':percent(bm['max_drawdown']),
                  '平均留現金':'持有至期末'})
    st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
    winners=[r['name'] for r in selected if r['summary']['total_return']>bm['total_return']]
    st.write('全期超過 0050：'+('、'.join(winners) if winners else '這組比較沒有')+'。')
    st.write('營收加速：近三個月營收年增至少 20%，且高於前三個月的年增率；營收優先則直接按營收成長排序。每月選最多 10 檔，不足留現金。')
    with st.expander('看分年、市場差異、成本與資料核對'):
        years=sorted(bm['annual_returns'])
        annual=[{'選股方式':r['name'],**{y:percent(r['summary']['annual_returns'][y]) for y in years}} for r in selected]
        annual.append({'選股方式':'0050 買入持有',**{y:percent(bm['annual_returns'][y]) for y in years}})
        st.dataframe(pd.DataFrame(annual),hide_index=True,use_container_width=True)
        st.caption('2026 為截至快照的部分年度；歷史已研究過，分年比較不等於未見樣本外驗證。')
        diagnostics=[{'選股方式':r['name'],
                      '滾動一年贏0050':percent(r['rolling']['252']['win_fraction']),
                      '年均單邊換手':f"{r['diagnostics']['annual_one_way_turnover_using_end_day_nav']:.1f} 倍",
                      '累計成本／期初資金':percent(r['summary']['fees_initial_equity'])} for r in selected]
        st.dataframe(pd.DataFrame(diagnostics),hide_index=True,use_container_width=True)
        st.caption('滾動窗口重疊；累計成本以期初資金為分母，因多年反覆交易可能超過 100%，不是年費率。')
        audit=report['revenue_inputs']['audit']
        st.write(f"資料抽查：3 家公司共 {sum(c['matched'] for c in audit['finmind_db_checks'])} 筆與 FinMind 相符；台積電 {audit['official_check']['months']} 個月與公司 SEC 申報相符。抽查不能證明全市場正確或歷史版本完整。")
        for item in report['limitations']: st.caption('• '+item)
    st.download_button('下載營收交叉驗證摘要',__import__('json').dumps(report,ensure_ascii=False,indent=2),
                       file_name='revenue-research.json',mime='application/json')


def render_flow_research():
    st.subheader('加投信、加放量，真的有比較好嗎？')
    report=service.flow_research_overview()
    if not report['available']:
        st.info(report['note'])
        return
    st.caption(f"資料截至 {report['source']['last_date']} · 八組比較 {report['elapsed_seconds']:.1f} 秒 · 重算使用 0 次 FinMind 請求")
    st.warning('已排除上市櫃前行情與存託憑證，但當前名冊仍有存活者偏差，還原價也尚待對帳。這輪尚未加入營收、獲利與產品出貨事件。')
    scenario=st.radio('投信／放量比較成本',['stress','base'],horizontal=True,key='flow_scenario',
                     format_func=lambda x:'較高滑價（每邊 0.45%）' if x=='stress' else '基本滑價（每邊 0.30%）')
    results=[r for r in report['results'] if r['scenario']==scenario]
    base=next(r for r in results if r['rule']=='price')
    rows=[]
    for r in results:
        rows.append({'選股方式':r['name'],'全期累積報酬':percent(r['summary']['total_return']),
                     '年化報酬':percent(r['summary']['annualized_return']),
                     '最大跌幅':percent(r['summary']['max_drawdown']),
                     '平均留現金':percent(r['diagnostics']['average_cash_fraction'])})
    bm=base['benchmark_summary']
    rows.append({'選股方式':'0050 買入持有','全期累積報酬':percent(bm['total_return']),
                 '年化報酬':percent(bm['annualized_return']),'最大跌幅':percent(bm['max_drawdown']),
                 '平均留現金':'買入後持有至期末'})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption(f"期間：{bm['start']}～{bm['end']}；月末選股、次交易日收盤模擬成交，最多 10 檔，不足留現金。已計入稅費與滑價。")
    improved=[r['name'] for r in results if r['rule']!='price'
              and r['summary']['total_return']>base['summary']['total_return']]
    st.write('比單用價格條件報酬更高：'+('、'.join(improved) if improved else '本輪沒有')+'。')
    winners=[r['name'] for r in results if r['summary']['total_return']>bm['total_return']]
    st.write('全期報酬超過 0050：'+('、'.join(winners) if winners else '本輪沒有')+'。這是歷史探索結果，不能視為已通過實盤驗證。')
    with st.expander('看不同期間、現金影響與選股條件'):
        details=[]
        for r in results:
            details.append({'選股方式':r['name'],
                            '2018–2022':percent(r['segments']['2018_2022']['strategy']['total_return']),
                            '2023–2025':percent(r['segments']['2023_2025']['strategy']['total_return']),
                            '2026 至快照':percent(r['segments']['2026_partial']['strategy']['total_return']),
                            '滾動一年贏0050比例':percent(r['rolling']['252']['win_fraction']),
                            '滾動三年贏0050比例':percent(r['rolling']['756']['win_fraction']),
                            '平均持股數':f"{r['summary']['average_positions']:.1f}"})
        details.append({'選股方式':'0050 買入持有',
                        '2018–2022':percent(base['segments']['2018_2022']['benchmark']['total_return']),
                        '2023–2025':percent(base['segments']['2023_2025']['benchmark']['total_return']),
                        '2026 至快照':percent(base['segments']['2026_partial']['benchmark']['total_return'])})
        st.dataframe(pd.DataFrame(details),hide_index=True,use_container_width=True)
        st.caption('滾動一年／三年按 252／756 個交易日計算；區間重疊，不能當成獨立勝率。')
        st.write('投信：20 日淨買占成交量至少 1%，近 5 日淨買超且至少 3 天買超；缺資料不算通過。')
        st.write('放量：近 5 日均量至少為之前 20 日的 1.5 倍、5 日上漲，且收盤位於當日高低區間的上方 30%。')
        st.caption('加條件可能讓股票不足，留下較多現金；不能把少投入資金的影響全部當成選股能力。')
        for r in results:
            if r['summary']['unliquidated_positions'] or r['diagnostics']['large_move_exposures']:
                st.caption(f"{r['name']}：期末未平倉 {r['summary']['unliquidated_positions']} 檔；持有時還原價單日變動超過 50% 共 {len(r['diagnostics']['large_move_exposures'])} 次，仍需核對。")
        for item in report['limitations']: st.caption('• '+item)
    st.download_button('下載投信與放量實測摘要',__import__('json').dumps(report,ensure_ascii=False,indent=2),
                       file_name='flow-research.json',mime='application/json')


def render_rule_research():
    st.subheader('不用模型，也可以比較選股規則')
    report=service.rule_research_overview()
    if not report['available']:
        st.info(report['note'])
        return
    st.caption(f"歷史快照截至 {report['source']['last_date']} · 六組比較計算 {report['elapsed_seconds']:.1f} 秒 · 重算不呼叫 FinMind")
    st.warning('這是方向探索：歷史上市日期與還原價尚未完成對帳，可能混入當時興櫃股票。不能把表內報酬當作可實現績效。')
    scenario=st.radio('交易成本情境',['stress','base'],horizontal=True,
                     format_func=lambda x:'較高滑價（每邊 0.45%）' if x=='stress' else '基本滑價（每邊 0.30%）')
    results=[r for r in report['results'] if r['scenario']==scenario]
    rows=[]
    for r in results:
        rows.append({'選股方式':r['name'],
                     '2023–2025 報酬':percent(r['segments']['2023_2025']['strategy']['total_return']),
                     '2026 年至快照報酬':percent(r['segments']['2026_partial']['strategy']['total_return']),
                     '全期累積報酬':percent(r['summary']['total_return']),
                     '全期最大跌幅':percent(r['summary']['max_drawdown'])})
    bm=results[0]
    rows.append({'選股方式':'0050 買入持有（比較基準）',
                 '2023–2025 報酬':percent(bm['segments']['2023_2025']['benchmark']['total_return']),
                 '2026 年至快照報酬':percent(bm['segments']['2026_partial']['benchmark']['total_return']),
                 '全期累積報酬':percent(bm['benchmark_summary']['total_return']),
                 '全期最大跌幅':percent(bm['benchmark_summary']['max_drawdown'])})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption(f"全期：{bm['summary']['start']}～{bm['summary']['end']}。均含手續費、交易稅與滑價；分段是同一組合的連續表現。")
    unsettled=[f"{r['name']} {r['summary']['unliquidated_positions']} 檔" for r in results
               if r['summary']['unliquidated_positions']]
    if unsettled:
        st.caption('期末仍無法賣出：'+'、'.join(unsettled)+'；該部位按最後可得價格估值，未扣未來賣出成本。')
    passed=[r['name'] for r in results if all(r['segments'][p]['excess_return']>0
                                           for p in ('2023_2025','2026_partial'))]
    if not passed:
        st.write('目前沒有一組在 2023–2025 與 2026 年兩段都勝過 0050，尚無證據支持用這些規則取代基準。')
    else:
        st.write('兩段皆勝過基準的方向：'+ '、'.join(passed)+'；仍需先排除資料缺陷，再作新的向前驗證。')
    with st.expander('這三種方法怎麼選股？'):
        st.write('中期動能：挑中期漲幅較強、股價高於半年均線的股票。')
        st.write('波動調整動能：相同資格，再把漲幅除以波動，降低忽上忽下股票的排名。')
        st.write('接近一年新高：挑接近一年高點、均線向上，且近三個月仍上漲的股票。')
        st.caption('共同條件：近 20 日平均成交額至少 5,000 萬元；月末決定名單，次一交易日收盤模擬成交；最多 10 檔、不足留現金。')
        for item in report['limitations']: st.caption('• '+item)
    st.download_button('下載規則比較摘要',__import__('json').dumps(report,ensure_ascii=False,indent=2),
                       file_name='rule-research.json',mime='application/json')


@st.fragment(run_every='5s')
def render_jobs():
    st.caption('工作進度每 5 秒自動更新。')
    running=jobs.recent_jobs()
    labels={'queued':'準備中','running':'執行中','completed':'已完成','failed':'未完成'}
    if not running: st.caption('尚未從工作台啟動工作。')
    for job in running:
        kind_names={'update_data':'資料更新','backtest':'策略回測','news_scan':'新聞題材','news_review':'歷史新聞判讀','chain_flow':'族群資金研究'}
        with st.expander(f"{labels[job['status']]} · {kind_names.get(job['request']['kind'],'研究工作')} · {job['job_id'][:8]}",expanded=job['status'] in ('running','failed')):
            st.write(job['message'])
            if job.get('elapsed_seconds') is not None: st.caption(f"執行耗時 {job['elapsed_seconds']:.1f} 秒")
            summary=job.get('summary')
            if summary and job['request']['kind'] in ('news_scan','news_review'):
                st.write(f"去重後 {summary.get('unique_articles',0):,} 篇；合併 {summary.get('duplicates_collapsed',0):,} 筆重複列。")
            elif summary and job['request']['kind']=='chain_flow':
                st.write(f"已整理 {summary.get('groups',0)} 組產業鏈與子產業。")
            elif summary:
                c=st.columns(3)
                c[0].metric('回測累積報酬',percent(summary.get('total_return')))
                c[1].metric('最大回撤',percent(summary.get('max_drawdown')))
                c[2].metric('股票池等權基準',percent(summary.get('benchmark_total_return')))
                st.caption(f"實際區間：{summary.get('backtest_start')} ~ {summary.get('backtest_end')} · 價格口徑：{summary.get('pnl_convention','尚未註明')}")
                curve=pd.DataFrame(job.get('equity_curve',[]))
                if not curve.empty and {'date','equity'}.issubset(curve.columns): st.line_chart(curve.set_index('date')[['equity']])
                st.download_button('下載驗證摘要',__import__('json').dumps(job,ensure_ascii=False,indent=2),file_name=f"{job['job_id']}.json",key=f"dl_{job['job_id']}")


def render_event_group_research():
    from app.event_group_research import overview
    st.subheader('營運轉好＋題材有錢，能拿來選股嗎？')
    report=overview()
    if not report['available']:
        st.info(report['note'])
        return
    audit=report['source_audit']; inputs=report['inputs']
    st.error('新聞日期核對未通過：目前不能用這份歷史資料判定策略是否贏 0050。')
    st.write('已完成三種規則：營運事件、題材群轉強、兩者同時成立。群體確認排除個股自己，檢查同群漲勢、均線及成交比重。')
    st.caption(f"掃描本機 {inputs['provenance']['news']['db_news_rows']:,} 筆新聞 · 抽出 {inputs['counts']['accepted']:,} 筆待核對營運線索 · 題材關聯不是已確認受惠")
    st.write('例如華東：FinMind 記為 2022-01-01，原文卻是 2020-08-14。已直接重查 API，仍回傳相同日期；不是本機轉換造成。')
    with st.expander('查看原文日期核對與失敗原因'):
        labels={'date_conflict':'明確錯置','calendar_date_difference':'日期差異待釐清',
                'same_calendar_date_only':'日期相同，版本未證實','unverified':'原文未取得'}
        st.dataframe(pd.DataFrame([{'代號':e['stock_id'],'標題':e['title'],'供應商日期':e['source_date'],
            '原文日期':e['original_publication_date'] or '未知','核對':labels[e['date_check']],'原文':e['link']}
            for e in audit['samples']]),hide_index=True,use_container_width=True,
            column_config={'原文':st.column_config.LinkColumn('原文')})
        st.caption('每種事件按時間取最早三筆，共 12 筆；選樣時未看事後股價。這不是隨機抽樣，不能推算全體錯誤率。')
        st.write('下一步需要把原文發佈、更新、供應商日期與首次抓到時間分開保存；日期未核對的新聞不進正式策略驗證。')
        for note in report['limitations']: st.caption('• '+note)
    st.caption(f"效能：訊號快照 {inputs['elapsed_seconds']:.1f} 秒；30 組診斷 {report['elapsed_seconds']:.1f} 秒。重算不呼叫 FinMind。本輪補缺口 16 次、來源核對 2 次，共 18 次；16 個缺日回應皆空，仍標為未知。")
    if st.checkbox('展開程式診斷數字（不能作為策略績效）',key='event_diagnostic'):
        st.warning('以下保留原供應商日期，用來檢查交易引擎、成本及價格口徑。新聞時序未過，數字不支持策略優劣或可實現報酬。')
        view=st.selectbox('事件診斷情境',['官方參考價・壓力成本','舊還原價・壓力成本','再晚一天進場','官方參考價・基本成本','舊還原價・基本成本'],key='event_view')
        basis,scenario,delay={'官方參考價・壓力成本':('official','stress',0),'舊還原價・壓力成本':('snapshot','stress',0),
            '再晚一天進場':('official','stress',1),'官方參考價・基本成本':('official','base',0),
            '舊還原價・基本成本':('snapshot','base',0)}[view]
        horizon=st.radio('事件持有上限',[63,126],format_func=lambda v:f'{v} 個交易日',horizontal=True,key='event_horizon')
        selected=[r for r in report['results'] if (r['basis'],r['scenario'],r['delay'],r['horizon'])==(basis,scenario,delay,horizon)]
        bm=selected[0]['benchmark_summary']
        st.caption(f"{bm['start']}～{bm['end']}；新訊號截止 2025-11-30，後續只退出持股。每邊滑價 {'0.45%' if scenario=='stress' else '0.30%'}，另計稅費；一般訊號後一交易日收盤執行。")
        table=[{'診斷方式':r['name'],'模擬累積淨報酬':percent(r['summary']['total_return']),
            '最大跌幅':percent(r['summary']['max_drawdown']),'平均留現金':percent(r['diagnostics']['average_cash_fraction']),
            '完成交易':r['summary']['completed_trades']} for r in selected]
        table.append({'診斷方式':'0050 同期持有','模擬累積淨報酬':percent(bm['total_return']),
            '最大跌幅':percent(bm['max_drawdown']),'平均留現金':'持有至期末','完成交易':1})
        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
        st.caption('每股初始配置最多前日資產 10%，最多 10 檔，不足留現金。跌破進場價 15% 或自高點跌 20%，下一交易日才嘗試退出；跳空可能超過停損幅度。')
        with st.expander('分年、成本與贏輸交易'):
            years=sorted(bm['annual_returns'])
            annual=[{'方式':r['name'],**{y:percent(r['summary']['annual_returns'][y]) for y in years}} for r in selected]
            annual.append({'方式':'0050',**{y:percent(bm['annual_returns'][y]) for y in years}})
            st.dataframe(pd.DataFrame(annual),hide_index=True,use_container_width=True)
            st.dataframe(pd.DataFrame([{'方式':r['name'],'平均持有交易日':r['summary']['average_holding_sessions'],
                '累計成本／期初資金':percent(r['summary']['fees_initial_equity']),
                '受阻單數':r['summary']['blocked_orders'],'缺價持有日':r['summary']['missing_hold_days']}
                for r in selected]),hide_index=True,use_container_width=True)
            st.caption('2026 為部分年度；累計成本不是年費率。以下保留最好與最差已完成交易，不可相加成投資組合績效。')
            examples=[{'方式':r['name'],'類型':label,'代號':t['stock_id'],'買入':t['entry_date'],
                       '賣出':t['exit_date'],'單筆模擬淨報酬':percent(t['net_return'])}
                      for r in selected for key,label in [('best_trades','較好'),('worst_trades','較差')] for t in r[key][:3]]
            st.dataframe(pd.DataFrame(examples),hide_index=True,use_container_width=True)
    st.download_button('下載事件診斷與日期稽核',__import__('json').dumps(report,ensure_ascii=False,indent=2),
                       file_name='event-group-diagnostic.json',mime='application/json')
