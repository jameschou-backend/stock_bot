"""Streamlit workbench: decisions, recorded fills and explicit research jobs."""
from datetime import datetime, time as daytime, timezone
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
    today,holdings,research=st.tabs(['今日觀察','持倉與成交','策略驗證'])
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
    st.subheader('策略能不能用，讓證據回答')
    st.info('目前沒有通過新驗證的實盤策略。下方回測用來檢查假設，不會自動啟用策略。')
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
    render_jobs()
    with st.expander('目前還需要確認的證據'):
        evidence=service.strategy_evidence()
        for item in evidence['validation_requirements']: st.write('• '+item)
        st.caption(status['adjustment_note'])
        st.write('已有研究紀錄：')
        for item in evidence['documents'][-8:]: st.caption(item['name'])


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
        with st.expander(f"{labels[job['status']]} · {'資料更新' if job['request']['kind']=='update_data' else '策略回測'} · {job['job_id'][:8]}",expanded=job['status'] in ('running','failed')):
            st.write(job['message'])
            if job.get('elapsed_seconds') is not None: st.caption(f"執行耗時 {job['elapsed_seconds']:.1f} 秒")
            summary=job.get('summary')
            if summary:
                c=st.columns(3)
                c[0].metric('回測累積報酬',percent(summary.get('total_return')))
                c[1].metric('最大回撤',percent(summary.get('max_drawdown')))
                c[2].metric('股票池等權基準',percent(summary.get('benchmark_total_return')))
                st.caption(f"實際區間：{summary.get('backtest_start')} ~ {summary.get('backtest_end')} · 價格口徑：{summary.get('pnl_convention','尚未註明')}")
                curve=pd.DataFrame(job.get('equity_curve',[]))
                if not curve.empty and {'date','equity'}.issubset(curve.columns): st.line_chart(curve.set_index('date')[['equity']])
                st.download_button('下載驗證摘要',__import__('json').dumps(job,ensure_ascii=False,indent=2),file_name=f"{job['job_id']}.json",key=f"dl_{job['job_id']}")
