"""Plain-language, read-only official-guidance pilot panel."""
import json
import pandas as pd
import streamlit as st

from app.guidance_research import overview


def pct(value):
    return f'{value:.2%}' if value is not None else '未知'


def render():
    st.subheader('官方說得更好，等股價確認再買，有幫助嗎？')
    report = overview()
    if not report['available']:
        st.info(report['note'])
        return
    st.write('先用台積電 12 場連續季度公告試驗：實績有沒有超過三個月前的指引，再看公告後是否價漲、量增、強過 0050。')
    st.warning('這是單公司歷史試驗，還不能當成台股選股策略。年度展望逐字稿有版本時間疑點，另列診斷。')
    st.caption(f"共同期間 {report['start']}～{report['end']}。平常持有0050；合格時以至多30%資產轉入台積電，63個交易日後換回。這是研究配置，並非你的實際持倉。")
    choices = {'官方價格・壓力成本':('official','stress',0), '官方價格・基本成本':('official','base',0),
               '舊價格・壓力成本':('snapshot','stress',0), '舊價格・基本成本':('snapshot','base',0),
               '再晚一天進場':('official','stress',1)}
    selected = st.selectbox('官方指引比較情境', list(choices), key='guidance_scenario')
    basis, scenario, delay = choices[selected]
    include_timing = st.checkbox('顯示年度展望的時間假設診斷', key='guidance_timing')
    results = [r for r in report['results'] if (r['basis'],r['scenario'],r['delay'])==(basis,scenario,delay)
               and (include_timing or not r['timing_diagnostic'])]
    baselines = [r for r in report['baselines'] if (r['basis'],r['scenario'])==(basis,scenario)]
    bm = next(r for r in baselines if r['mode']=='benchmark')
    mix = next(r for r in baselines if r['mode']=='static_mix')
    primary = next(r for r in results if r['rule']=='beat_confirm')
    a,b,c = st.columns(3)
    a.metric('實績＋確認：累積淨報酬',pct(primary['summary']['total_return']))
    b.metric('0050 同期持有',pct(bm['summary']['total_return']))
    c.metric('固定 70%0050＋30%台積電',pct(mix['summary']['total_return']))
    if primary['excess_vs_0050'] <= 0:
        st.info('這組「實績＋確認」沒有跑贏 0050，暫時不支持增加操作。')
    elif primary['excess_vs_static_mix'] <= 0:
        st.info('這組雖跑贏 0050，仍輸給固定混合持有；尚未證明等待訊號比直接持有更好。')
    else:
        st.info('這組歷史試驗勝過兩個基準，但只有單公司少量事件，仍需擴大樣本與向前驗證。')
    table = []
    for row in results + baselines:
        s = row['summary']
        label = row.get('name') or {'benchmark':'0050 持有','static_mix':'固定 70/30 混合持有'}[row['mode']]
        table.append({'方法':label, '累積報酬（扣費）':pct(s['total_return']), '年化報酬':pct(s['cagr']),
                      '最大跌幅':pct(s['max_drawdown']), '股票持有次數':s['active_trade_count'],
                      '平均台積電權重':pct(s['mean_active_weight'])})
    st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
    st.caption(f"每邊滑價 {'0.45%' if scenario=='stress' else '0.30%'}，另扣每邊0.1425%手續費及賣出稅。每次由0050換股、再換回，兩邊成本均計入。")
    if include_timing:
        st.warning('年度上修與組合使用推定文件可用日期；延後已知修正版仍不能證明首版內容，這兩列不能解讀成可實現績效。')
    if (basis,scenario,delay)==('official','stress',0):
        with st.expander('看資產變化與成本'):
            names = {'beat_confirm':'實績＋確認','benchmark':'0050','static_mix':'固定混合'}
            chart = pd.concat([pd.Series({r['date']:r['nav'] for r in report['charts'][key]},name=label)
                               for key,label in names.items()],axis=1)
            chart.index = pd.to_datetime(chart.index)
            st.line_chart(chart)
            st.caption('起始資金設為1；總報酬單位是研究計算，不等於券商可成交股數。')
            st.dataframe(pd.DataFrame([{'方法':r.get('name',r.get('mode')),
                '累計成本／期初資金':pct(r['summary']['total_cost']),
                '買賣成交筆數':r['summary']['trade_count'],'到期受阻交易日':r['summary']['blocked_exit_sessions']}
                for r in results+baselines]),hide_index=True,use_container_width=True)
    with st.expander('逐季看：原本說多少，後來做到多少'):
        events = report['events'][basis]
        rows = []
        for event in events:
            comp = event['comparison']
            if event['status']=='warmup':
                continue
            rows.append({'季度':event['reported_quarter'],'公布日':event['meeting_date'],
                '原營收區間（十億美元）':f"{comp.get('prior_usd_low')}～{comp.get('prior_usd_high')}" if comp['status']=='matched' else '未知',
                '實際營收（十億美元）':comp.get('actual_usd_billion'),
                '實績過關': '是' if comp['beat'] else '否／資料不足',
                '價量確認日':event.get('confirmation_date') or '未確認',
                '預定買入':event['entries'].get('beat_confirm','無'),
                '原指引':comp.get('prior_source_url'), '當季實績':event.get('source_url')})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,
                     column_config={'原指引':st.column_config.LinkColumn(),'當季實績':st.column_config.LinkColumn()})
        st.caption('實績過關：美元營收高於原上限且毛利率不低於原下限。季營收可能已由月營收預知，不能把全部差異視為法說當天的新資訊。預定買入仍可能因重疊持股或無法成交而拒絕。')
    with st.expander('看實際模擬交易與拒絕原因'):
        trades = [{**t,'方法':r['name']} for r in results for t in r['trades']]
        if trades:
            st.dataframe(pd.DataFrame([{'方法':t['方法'],'事件':t['event_id'],'買入':t['entry_date'],
                '賣出':t['exit_date'],'持有交易日':t['holding_sessions'],
                '股票單筆淨報酬':pct(t['stock_net_return'])} for t in trades]),hide_index=True,use_container_width=True)
        rejected = [{**x,'方法':r['name']} for r in results for x in r['rejections']]
        if rejected:
            labels={'overlapping_position':'已有持股，不加碼／延期','entry_instruments_not_tradable':'當日無法換股',
                    'terminal_session':'期末不再開倉','outside_window':'超出期間'}
            st.dataframe(pd.DataFrame([{'方法':r['方法'],'事件':r['event_id'], '日期':r['entry_date'],
                '原因':labels.get(r['reason'],r['reason'])} for r in rejected]),hide_index=True,use_container_width=True)
        st.caption('單筆股票報酬不包含放棄0050的機會成本，也不能相加成投組報酬；上方整體結果已含每次轉換成本。')
    with st.expander('全年展望與研究限制'):
        labels={'new_period':'新年度，不跨年比较','up':'上修','down':'下修','maintained':'維持','implied':'推算改善','unknown':'未知'}
        st.dataframe(pd.DataFrame([{'公布日':e['meeting_date'],'分類':labels[e['annual_direction']],
            '官方文字':e.get('annual_wording'),'推定可用日':e.get('annual_signal_date_assumed') or '不產生上修訊號',
            '來源':e.get('annual_source_url')} for e in report['events'][basis]]),hide_index=True,use_container_width=True,
            column_config={'來源':st.column_config.LinkColumn()})
        for note in report['limitations']:
            st.caption('• '+note)
    st.caption(f"效能：兩檔行情切片 {report['inputs']['preparation_seconds']:.1f} 秒；20 組離線比較 {report['elapsed_seconds']:.1f} 秒。研究使用 0 次 FinMind、0 次模型重訓。開頁與切換情境不重跑。")
    st.download_button('下載官方指引試驗',json.dumps(report,ensure_ascii=False,indent=2),
                       file_name='guidance-pilot.json',mime='application/json',key='guidance_download')
