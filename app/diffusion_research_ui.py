"""Read-only leader-to-peer research, with the actual selection sequence."""
import json

import pandas as pd
import streamlit as st

from app.diffusion_research import ROOT, overview


def pct(value):
    return f'{value:.2%}' if value is not None else '未知'


def render():
    st.subheader('領先股漲了，等同群接力再買，有幫助嗎？')
    report = overview()
    if not report['available']:
        st.info(report['note'])
        return
    st.write('每月用過去股價找出一起波動的股票群。先找突破、放量的領先股，再等其他成員變強，才選接力股。')
    st.warning('歷史探索，尚未取得實盤資格。價格同行不代表已證實的產業題材；目前名冊也有存活者偏差。')
    st.caption(f"期間 {report['start']}～{report['end']}；新領先訊號截止 {report['signal_end']}。最多同時 3 個事件，每次最多約 1/3 資產，持有 63 個交易日；其餘持有 0050。")
    choices = {
        '官方價格・壓力成本': ('official','official','stress',0),
        '官方價格・基本成本': ('official','official','base',0),
        '舊價格與舊訊號・壓力成本': ('snapshot','snapshot','stress',0),
        '舊價格與舊訊號・基本成本': ('snapshot','snapshot','base',0),
        '官方價格・再晚一天買': ('official','official','stress',1),
        '官方訊號固定・換舊價格對帳': ('snapshot','official','stress',0),
    }
    selected = st.selectbox('接力實驗比較情境', list(choices), key='diffusion_scenario')
    basis, signal_basis, scenario, delay = choices[selected]
    rows = [r for r in report['results'] if (r['basis'],r['signal_basis'],r['scenario'],r['delay']) == choices[selected]]
    bm = next(r for r in report['baselines'] if (r['basis'],r['scenario']) == (basis,scenario))
    primary = next(r for r in rows if r['rule']=='follower_after')
    basket = next(r for r in rows if r['rule']=='basket_after')
    a,b,c = st.columns(3)
    a.metric('接力股：累積試算淨報酬',pct(primary['summary']['total_return']))
    b.metric('0050 同期持有',pct(bm['summary']['total_return']))
    c.metric('同群整籃持有',pct(basket['summary']['total_return']))
    unresolved = primary['valuation_audit']['finding_count']
    if unresolved:
        st.error(f'接力股投組有 {unresolved} 筆持有估值疑點。下列收益保留供對帳，不能解讀成已核實績效。')
    elif primary['excess_vs_0050'] <= 0:
        st.info('這組等待接力後買入的試算沒有贏過 0050，暫不支持用它取代持有基準。')
    elif primary['excess_vs_basket'] <= 0:
        st.info('這組雖贏過 0050，卻未贏同群整籃；尚未證明挑接力股有額外幫助。')
    else:
        st.info('這組歷史試算贏過兩個比較，但仍需看價格對帳、不同成本及未來資料，才能判斷是否可重複。')
    if not primary['summary']['final_liquidation_complete']:
        st.warning('期末有無法賣出的持股；報酬含未平倉估值，並非全部可提領現金。')
    table = []
    for row in rows + [bm]:
        s = row['summary']
        table.append({'方法':row.get('name','0050 持有'), '累積試算淨報酬':pct(s['total_return']),
            '年化':pct(s['cagr']), '最大跌幅':pct(s['max_drawdown']),
            '買入／完成事件':f"{s['entered_cohorts']}／{s['completed_cohorts']}",
            '平均股票比重':pct(s['mean_active_weight']),
            '可疑估值筆數':row.get('valuation_audit',{}).get('finding_count',0),
            '期末': '已清算' if s['final_liquidation_complete'] else '含未平倉估值'})
    st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
    st.caption(f"每邊滑價 {'0.45%' if scenario=='stress' else '0.30%'}，另扣每邊 0.1425% 手續費及賣出稅，包含每次由 0050 換股再換回的成本。領先就買包含後來未接力的事件；確認後各臂也可能因重疊持股或無法成交而有不同實際持倉。")
    leader = next(r for r in rows if r['rule']=='leader_now')
    if leader['excess_vs_0050'] > 0 and not leader['valuation_audit']['finding_count']:
        st.info('本情境「領先出現就買」值得再驗證。它使用全部領先事件，沒有等後來接力成功才挑出來；目前仍是同一段歷史上的候選方向。')
        losing_years = [year for year,value in leader['summary']['annual_returns'].items()
                        if year < report['end'][:4] and value < bm['summary']['annual_returns'][year]]
        if losing_years:
            st.caption('領先就買仍在 '+ '、'.join(losing_years)+' 年落後 0050。價格、成本情境重用同一段歷史，不能當成多次獨立驗證。')
    if choices[selected] == ('official','official','stress',0):
        with st.expander('看資產曲線、各年報酬與成本'):
            names = {'leader_now':'領先就買','follower_after':'接力股','leader_after':'原領先股','basket_after':'同群整籃','0050':'0050'}
            frame = pd.concat([pd.Series({r['date']:r['nav'] for r in report['charts'][key]},name=name)
                               for key,name in names.items()],axis=1)
            frame.index = pd.to_datetime(frame.index)
            st.line_chart(frame)
            yearly = [{'方法':r.get('name','0050 持有'),**{y:pct(v) for y,v in r['summary']['annual_returns'].items()}}
                      for r in rows+[bm]]
            st.dataframe(pd.DataFrame(yearly),hide_index=True,use_container_width=True)
            st.caption('2026 只到 6 月 23 日。曲線起始資金設為 1，含稅費與滑價；尚未核實的估值也保留在曲線中。')
    with st.expander('為什麼選這檔？看領先與接力的先後'):
        companies = pd.read_parquet(ROOT/'.cache/diffusion-research/companies.parquet')
        names = dict(zip(companies.stock_id,companies.name))
        def stock(sid):
            return f'{sid} {names.get(sid, "")}' if sid else '無'
        statuses = {'confirmed':'出現接力','no_confirmation':'10 日內未接力',
                    'data_insufficient':'有資料不足日，未確認','window_incomplete':'觀察窗口未完成'}
        events = report['events'][signal_basis]
        stats = report['signal_info']['stats'][signal_basis]
        st.write(f"{stats['months']} 個月形成 {stats['groups']} 個月度群組，{stats['leaders']} 個領先事件；其中 {stats['statuses'].get('confirmed',0)} 個確認接力。")
        event_filter = st.selectbox('事件狀態', ['全部',*statuses.values()],key='diffusion_event_status')
        shown = [e for e in events if event_filter=='全部' or statuses.get(e['status'])==event_filter]
        st.dataframe(pd.DataFrame([{'領先日':e['leader_date'],'領先股':stock(e['leader_id']),
            '同群檔數':len(e['members']), '確認日':e['confirmation_date'] or '無',
            '接力股':stock(e['follower_id']), '領先時同群變強':pct(e['leader_peer_breadth']),
            '確認時同群變強':pct(e['confirmation_peer_breadth']), '結果':statuses.get(e['status'],e['status'])}
            for e in shown]),hide_index=True,use_container_width=True)
        if shown:
            event_map = {e['event_id']:e for e in shown}
            event_id = st.selectbox('選一個事件看依據',list(event_map),key='diffusion_event',
                format_func=lambda key: f"{event_map[key]['leader_date']} {stock(event_map[key]['leader_id'])} → {stock(event_map[key]['follower_id'])}")
            event = event_map[event_id]
            st.write('當時凍結的同群：'+ '、'.join(stock(sid) for sid in event['members']))
            st.caption(f"分群資料截止 {event['group_cutoff_date']}；領先股突破前 60 日高點，20 日報酬 {pct(event['leader_return20'])}（0050 {pct(event['benchmark_return20'])}），成交量為此前 20 日均量 {event['leader_volume_ratio']:.2f} 倍。")
            reasons = {'confirmed':'接力確認','missing_peer_or_benchmark_prices':'必要行情不足',
                'breadth_or_new_responders_below_threshold':'同群變強比例或新接力檔數不足',
                'missing_turnover_share_window':'成交占比資料不足',
                'peer_turnover_share_not_rising':'同群成交占比未增加',
                'leader_or_follower_quality_window_invalid':'領先或接力股行情品質未過關'}
            st.dataframe(pd.DataFrame([{'觀察日':r['date'],'領先後第幾天':r['session_after_leader'],
                '其他成員變強比例':pct(r['peer_breadth']),
                '新接力股': '、'.join(stock(sid) for sid in (r['new_responders'] or [])),
                '同群近 5 日成交占比':pct(r['turnover_share']['last5_mean']),
                '此前 20 日成交占比':pct(r['turnover_share']['prior20_mean']),
                '判定':reasons.get(r['reason'],r['reason'])} for r in event['confirmation_checks']]),
                hide_index=True,use_container_width=True)
            st.caption('同群變強與成交占比均排除領先股；成交額是收盤價×成交股數估值，不能當成淨流入。確認當日完成觀察，最快下一交易日才買。')
    with st.expander('看實際模擬持股、未成交原因與估值疑點'):
        methods = {r['name']:r for r in rows}
        method = st.selectbox('檢查哪一種持股方法',list(methods),index=2,key='diffusion_trade_method')
        detail = methods[method]
        s = detail['summary']
        st.write(f"{method}收到 {detail['signal_count']} 個訊號，實際買入 {s['entered_cohorts']} 次、完成 {s['completed_cohorts']} 次；拒絕 {s['rejected_event_count']} 次。")
        st.dataframe(pd.DataFrame([{'事件':t['event_id'], '持股':'、'.join(t['members']),
            '買入':t['entry_date'],'賣出':t.get('exit_date','尚未賣出'),
            '持有交易日':t.get('holding_sessions'),
            '含轉換成本盈虧／期初資金':pct(t.get('cycle_net_pnl')),
            '退出受阻天數':t['blocked_exit_sessions']} for t in detail['cohorts']]),
            hide_index=True,use_container_width=True)
        completed = [t for t in detail['cohorts'] if t.get('cycle_net_pnl') is not None]
        if completed:
            best,max_loss = max(completed,key=lambda t:t['cycle_net_pnl']),min(completed,key=lambda t:t['cycle_net_pnl'])
            st.caption(f"單事件最佳 {best['event_id']}：{pct(best['cycle_net_pnl'])}；最差 {max_loss['event_id']}：{pct(max_loss['cycle_net_pnl'])}（均相對期初資金，不含放棄 0050 的機會成本，不能視為單股報酬率）。")
        reasons = {'overlapping_member':'與現有持股重疊','slots_full':'同時持有事件已滿',
            'entry_instruments_not_tradable':'必要標的無法成交','no_benchmark_funding':'0050 可轉出資金不足',
            'terminal_session':'期末不再開倉','outside_window':'超出研究期間'}
        if detail['rejections']:
            st.dataframe(pd.DataFrame([{'事件':r['event_id'],'預定買入':r['entry_date'],
                '原因':reasons.get(r['reason'],r['reason'])} for r in detail['rejections']]),
                hide_index=True,use_container_width=True)
        if detail['valuation_audit']['findings']:
            st.dataframe(pd.DataFrame(detail['valuation_audit']['findings']).rename(columns={
                'date':'日期','stock_id':'代號','observed_return_since_last_quote':'距前次有效報價漲跌',
                'price_basis_difference':'兩價格版本差異'}),hide_index=True,use_container_width=True)
        st.dataframe(pd.DataFrame([{'方法':r.get('name','0050 持有'),
            '累計成本／期初資金':pct(r['summary']['total_cost']),
            '累計雙向成交額／期初資金':f"{r['summary']['turnover']:.2f} 倍",
            '買賣成交筆數':r['summary']['trade_count']} for r in rows+[bm]]),
            hide_index=True,use_container_width=True)
    with st.expander('看每月分群及研究限制'):
        st.dataframe(pd.DataFrame([{'月份':g['month'],'資料截止':g['cutoff_date'],
            '合格股票':g['eligible_count'],'分群輸入檔數':g['selected_count'],
            '保留群數':len(g['clusters']),'結果':g['status']} for g in report['groups'][signal_basis]]),
            hide_index=True,use_container_width=True)
        for note in report['limitations']:
            st.caption('• '+note)
    st.caption(f"效能：價量準備 {report['inputs']['elapsed_seconds']:.1f} 秒，兩套分群與訊號 {report['signal_info']['elapsed_seconds']:.1f} 秒，24 組投組比較含稽核 {report['elapsed_seconds']:.1f} 秒。研究使用 0 次 FinMind、0 次預測模型重訓。切換情境只讀結果。")
    st.download_button('下載接力選股實驗',json.dumps(report,ensure_ascii=False,indent=2),
        file_name='diffusion-research.json',mime='application/json',key='diffusion_download')
