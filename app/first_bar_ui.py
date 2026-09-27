"""Sealed early-entry timing results; price statistics are not account returns."""
from io import BytesIO
from pathlib import Path
import json
from app.backtest_tool_ui import verified_bytes

ROOT=Path(__file__).resolve().parents[1]
PUBLICATION='artifacts/forward_simulation/first_bar_20260927.json'


def case_names():
    return {f'{arm}_lag{lag}_{channel}_{stress}' for arm in ('first','wait') for lag in (8,15)
        for channel in ('mixed','board') for stress in ('control','combined')} | {
            f'benchmark_{channel}_{stress}' for channel in ('mixed','board') for stress in ('control','combined')}


def load(root=ROOT):
    root=Path(root);path=root/PUBLICATION
    pub=json.loads(verified_bytes(dict(path=PUBLICATION,sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    r=json.loads(verified_bytes(pub['report'],root,'.json'))
    if (pub.get('schema')!='first_bar_publication_v1' or r.get('schema')!='first_bar_v1'
            or r.get('completed') is not True or any(r.get(k) is not False for k in
            ('live_qualified','adopted','unseen_validation','historical_first_publication_verified',
             'theme_filter_included','price_statistics_are_account_returns'))):
        raise ValueError('第一根研究範圍或資格不符')
    proof=pub['reproducibility']
    if proof.get('passed') is not True or len(proof['runs'])!=2 or proof['runs'][0]['path']==proof['runs'][1]['path']:
        raise ValueError('缺少獨立重算')
    found=False
    for desc in proof['runs']:
        m=json.loads(verified_bytes(desc,root,'.json'))
        if m['source_sha256']!=r['source_sha256'] or {k:v for k,v in m['files_sha256'].items() if k!='report.json'}!=proof['files_sha256']:
            raise ValueError('重算來源或結果不同')
        folder=Path(desc['path']).parent
        if str(folder/'report.json')==pub['report']['path']:
            found=m['files_sha256']['report.json']==pub['report']['sha256']
            for case in r['cases'].values():
                p=Path(case['result']['path'])
                if p.parent!=folder or m['files_sha256'].get(p.name)!=case['result']['sha256']:
                    raise ValueError('帳戶與研究結果未連結')
    if not found or set(r['cases'])!=case_names():raise ValueError('報告封存或20個帳戶不完整')
    if r['all_accounts_completed']!=all(c['completed'] for c in r['cases'].values()):
        raise ValueError('帳戶完成狀態不符')
    for c in r['cases'].values():
        if not c['completed'] and ('summary' in c or c.get('excess_return') is not None):
            raise ValueError('未完成帳戶不可發布績效')
    return r,pub


def timing_rows(report,lag,horizon,phase):
    labels={'all_events':'全部第一根','chip_known':'籌碼可判斷','concentrated':'籌碼集中','not_concentrated':'未符合集中'}
    pct=lambda v:'未知' if v is None else f'{v:.2%}'
    return [{'條件':labels[r['group']],'事件數':r['events'],'配對已知':r['paired_known'],
        '第一根後平均漲幅':pct(r['paired_first_mean']),'等突破平均漲幅':pct(r['paired_wait_mean']),
        '提早進場差距':pct(r['paired_difference_mean']),'第一根相對0050':pct(r['first_excess']),
        '五日內跌回原點':pct(r['false_start5']),'未知':r['unknown']}
        for r in report['statistics'] if (r['lag'],r['horizon'],r['phase'])==(lag,horizon,phase)]


def render():
    import pandas as pd
    import streamlit as st
    st.subheader('籌碼集中後，第一根帶量紅 K 要不要先買？')
    try:r,pub=load()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.error('尚無有效封存結果：'+str(exc));return
    st.write('前20日整理幅度不超過15%；出現2倍量、漲至少2%、收在高檔的紅K，且前10日沒有同類紅K。第一根收盤確認，隔天才進場。')
    st.info('比較同一批事件：第一根後進場，或最多等20日突破。主統計排除指定十檔贏家，沒有加入新聞題材篩選；仍屬已見歷史研究。')
    phase=st.radio('期間',['replication','discovery'],format_func=lambda x:'2025–2026' if x=='replication' else '2022–2024',horizontal=True,key='first_bar_phase')
    horizon=st.radio('固定結果窗口',[20,60],format_func=lambda n:f'{n}個交易日',horizontal=True,key='first_bar_horizon')
    lag=st.radio('集保延遲',[8,15],format_func=lambda n:f'觀察日後{n}天',horizontal=True,key='first_bar_lag')
    st.caption('下表為同一段期間的個股價格統計，未扣成本、沒有資金配置，不能當成帳戶報酬。等待期間保持現金；未出現突破的事件也保留。')
    st.dataframe(pd.DataFrame(timing_rows(r,lag,horizon,phase)),hide_index=True,use_container_width=True)
    selected=next(x for x in r['statistics'] if (x['lag'],x['horizon'],x['phase'],x['group'])==(lag,horizon,phase,'concentrated'))
    st.write(f"集中組 {selected['events']} 次：{selected['same_day']} 次第一根已符合突破、{selected['later']} 次後來才突破、{selected['never']} 次20日未突破、{selected['wait_unknown']} 次無法判斷。")
    if selected['later_median'] is not None:st.caption(f"後來才突破的事件，等待中位數 {selected['later_median']:.0f} 個交易日。五日跌回原點指跌破第一根前一天收盤，不等於已實現停損。")
    st.write('**100萬元完整帳戶驗證**')
    st.caption('最多五檔、閒錢現金、12%收盤停損／63日持有上限。資料不足就停止，不顯示部分回測報酬；只整張是另一種成交限制，不能替代零股結果。')
    rows=[]
    for channel,label in [('mixed','整股＋零股'),('board','只整張診斷')]:
        for arm,title in [('first','第一根後'),('wait','等突破')]:
            for stress,cost in [('control','一般'),('combined','合併壓力')]:
                c=r['cases'][f'{arm}_lag{lag}_{channel}_{stress}']
                rows.append({'進場':title,'成交模式':label,'成本':cost,
                    '完整帳戶淨報酬':f"{c['summary']['total_return']:.2%}" if c['completed'] else '未完成',
                    '最大回撤':f"{c['summary']['max_drawdown']:.2%}" if c['completed'] else '—',
                    '原因':c.get('reason','完成；仍不具實戰資格')})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    folder=Path(pub['report']['path']).parent
    with st.expander('指定個股：有哪些第一根？'):
        from skills.stock_launch import TARGETS
        names=dict(TARGETS,**{'3491':'昇達科'})
        sid=st.selectbox('個股',list(names),format_func=lambda x:x+' '+names[x],key='first_bar_stock')
        descriptor=dict(path=str(folder/'outcomes.csv'),sha256=pub['reproducibility']['files_sha256']['outcomes.csv'])
        table=pd.read_csv(BytesIO(verified_bytes(descriptor,ROOT,'.csv')),dtype={'stock_id':str})
        chosen=table[table.stock_id.eq(sid)&table.lag.eq(lag)&table.horizon.eq(horizon)&table.phase.eq(phase)]
        shown=chosen[['signal_date','concentration','volume_ratio','wait_signal_date','wait_sessions','first_return','wait_return','false_start5']].copy()
        shown=shown.rename(columns={'signal_date':'第一根訊號日','concentration':'符合集中','volume_ratio':'量比',
            'wait_signal_date':'突破訊號日','wait_sessions':'等待交易日','first_return':'第一根後價格變化',
            'wait_return':'等突破價格變化','false_start5':'五日跌回原點'})
        for key in ('第一根後價格變化','等突破價格變化'):
            shown[key]=shown[key].map(lambda v:'未知' if pd.isna(v) else f'{v:.2%}')
        st.dataframe(shown,hide_index=True,use_container_width=True)
        st.caption('指定十檔僅作案例，已從主統計與帳戶候選排除；訊號日不是當日成交，缺值保留未知。')
    for name,label in [('events.csv','全部第一根事件'),('outcomes.csv','完整配對結果'),('statistics.csv','分組統計')]:
        descriptor=dict(path=str(folder/name),sha256=pub['reproducibility']['files_sha256'][name])
        st.download_button('下載'+label,verified_bytes(descriptor,ROOT,'.csv'),file_name=name,mime='text/csv',key='first_bar_'+name)
    st.caption(f"兩輪離線重算相符；事件與進場通過{len(r['causality_checks'])}組未來改寫／截斷檢查，每組含兩種集保延遲與進場方式。0次新FinMind請求。")
