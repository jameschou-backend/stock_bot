"""Sealed ten-stock institutional/MA evidence, including negative outcomes."""
from pathlib import Path
import json
from app.backtest_tool_ui import verified_bytes
from app.stock_launch_ui import read_table

ROOT=Path(__file__).resolve().parents[1]
PUBLICATION='artifacts/forward_simulation/launch_flows_20260927.json'
LABELS={'above20':'站上20日線','above60':'站上60日線','ma_stack':'多頭排列且均線上揚',
    'foreign_positive5':'外資5日淨買','trust_positive5':'投信5日淨買','foreign_streak3':'外資連買3日',
    'trust_streak3':'投信連買3日','both_positive5':'外資投信5日都淨買',
    'streak_and_trend':'任一連買3日＋均線多頭','concentrated_both':'集中＋外資投信都淨買'}


def load(root=ROOT):
    root=Path(root);path=root/PUBLICATION
    pub=json.loads(verified_bytes(dict(path=PUBLICATION,sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    r=json.loads(verified_bytes(pub['report'],root,'.json'))
    if (pub.get('schema')!='launch_flows_publication_v1' or r.get('schema')!='launch_flows_v1'
        or r.get('completed') is not True or r.get('strategy_net_return') is not None
        or any(r.get(k) is not False for k in ('live_qualified','adopted','unseen_validation','portfolio_returns_computed','historical_first_publication_verified'))):
        raise ValueError('法人研究資格或範圍不符')
    proof=pub['reproducibility'];hashes={k:v['sha256'] for k,v in r['artifacts'].items()}
    if proof.get('passed') is not True or len(proof['runs'])!=2 or proof['runs'][0]['path']==proof['runs'][1]['path'] or proof['csv_sha256']!=hashes:
        raise ValueError('缺少兩輪獨立重算')
    found=False
    for desc in proof['runs']:
        m=json.loads(verified_bytes(desc,root,'.json'))
        if m['source_sha256']!=r['source_sha256'] or {k:v['sha256'] for k,v in m['files'].items()}!=hashes:
            raise ValueError('獨立結果或來源不一致')
        if str(Path(desc['path']).with_name('report.json'))==pub['report']['path']:
            found=m['report_sha256']==pub['report']['sha256'] and m['files']==r['artifacts']
    if not found:raise ValueError('報告未連結封存檔案')
    return r


def fmt(value,kind='number'):
    import pandas as pd
    if pd.isna(value):return '未知'
    if kind=='percent':return f'{value:+.2%}'
    if kind=='bool':return '是' if value else '否'
    if kind=='streak':return '至少20' if value>=20 else str(int(value))
    return f'{value/1000:+,.1f}'


def summary_rows(report,horizon,phase,lag,scope):
    return [{'條件':LABELS[x['rule']],'符合條件':'是' if x['value'] else '否','事件':x['events'],
        '結果已知':x['known'],'急漲比例':fmt(x['surge_rate'],'percent'),
        '個股平均漲幅':fmt(x['mean_return'],'percent'),'相對0050':fmt(x['mean_excess'],'percent'),
        '條件未知':x['condition_unknown'],'結果未知':x['outcome_unknown']}
        for x in report['statistics'] if (x['horizon'],x['phase'],x['flow_lag'],x['scope'])==(horizon,phase,lag,scope)]


def render():
    import pandas as pd
    import streamlit as st
    st.subheader('這些股票起漲前，法人與均線有什麼不同？')
    try:r=load()
    except (OSError,ValueError,KeyError,TypeError) as exc:st.error('尚無有效法人研究：'+str(exc));return
    st.info('逐股看十檔；驗證比較排除這十檔。法人覆蓋的是以前策略選過的283檔子集，不是完整全市場；沒有成本、成交或帳戶報酬。')
    st.caption('主要觀察第一根前一天的法人資料，和訊號當日均線。外資含外資自營商；自營商另拆自行買賣／避險。缺分類、缺日、尚未走完結果都保留未知。')
    st.caption('多頭且上揚＝收盤>20日線>60日線，20日線高於5日前、60日線高於20日前；不是只要站上均線就算。')
    with st.expander('十檔最後一段急漲的事前狀態',expanded=True):
        st.caption('以下日期是事後用20日漲30%、且超過0050 20百分點定位的回溯觀察日；不是當時已知買點。每檔取最後一個事件，不挑最高報酬。')
        latest=read_table(r,'latest_retrospective');timeline=read_table(r,'retrospective_timeline')
        rows=[]
        for e in latest.itertuples():
            t=timeline[timeline.event_id.eq(e.event_id)];p=t[t.offset.eq(0)].iloc[0];f=t[t.offset.eq(-1)].iloc[0]
            rows.append({'股票':e.stock_id+' '+r['targets'][e.stock_id],'回溯觀察日':e.signal_date,
                '外資5日淨額（張）':fmt(f.foreign_net5),'投信5日淨額（張）':fmt(f.trust_net5),
                '自營商5日淨額（張）':fmt(f.dealer_net5),'外資連買':fmt(f.foreign_buy_streak,'streak'),
                '投信連買':fmt(f.trust_buy_streak,'streak'),'站上20日線':fmt(p.above20,'bool'),
                '站上60日線':fmt(p.above60,'bool'),'多頭且上揚':fmt(p.ma_stack,'bool')})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.write('**逐次訊號與啟動前的變化**')
    kind=st.radio('事件類型',['first','retrospective'],format_func=lambda x:'第一根訊號（含失敗）' if x=='first' else '事後急漲觀察日',horizontal=True,key='flow_kind')
    sid=st.selectbox('股票',list(r['targets']),format_func=lambda x:x+' '+r['targets'][x],key='flow_stock')
    horizon=st.radio('後續價格窗口',[20,60],horizontal=True,key='flow_horizon')
    data=read_table(r,'named_first_timeline' if kind=='first' else 'retrospective_timeline')
    data=data[data.stock_id.eq(sid)&data.horizon.eq(horizon)]
    if data.empty:st.info('這個範圍沒有事件。')
    else:
        choices=data[['event_id','signal_date']].drop_duplicates().sort_values('signal_date');labels=dict(zip(choices.event_id,choices.signal_date))
        event=st.selectbox('日期',choices.event_id.tolist(),index=len(choices)-1,format_func=labels.get,key='flow_event')
        rows=[];chosen=data[data.event_id.eq(event)]
        for x in chosen.itertuples():
            rows.append({'距事件交易日':x.offset,'資料截至':x.feature_date,
                '外資5日（張）':fmt(x.foreign_net5),'投信5日（張）':fmt(x.trust_net5),
                '自營商自行5日（張）':fmt(x.dealer_self_net5),'自營商避險5日（張）':fmt(x.dealer_hedging_net5),
                '外資連買':fmt(x.foreign_buy_streak,'streak'),'投信連買':fmt(x.trust_buy_streak,'streak'),
                '距20日線':fmt(x.distance_ma20,'percent'),'距60日線':fmt(x.distance_ma60,'percent'),
                '多頭且上揚':fmt(x.ma_stack,'bool')})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        x=chosen.iloc[0];ret=x.first_return if kind=='first' else x.forward_return
        st.caption('後續價格變化：'+fmt(ret,'percent')+'。訊號次日收盤起算，未扣成本；當日法人僅視為收盤後可用，不能反推當日下單。')
    st.write('**同一條件，成功與失敗都比較**')
    phase=st.radio('期間',['replication','discovery'],format_func=lambda x:'2025–2026' if x=='replication' else '2022–2024',horizontal=True,key='flow_phase')
    lag=st.radio('法人資料截至',[1,0],format_func=lambda x:'第一根前一天' if x else '第一根當天（同步反應）',horizontal=True,key='flow_lag')
    scope=st.radio('比較範圍',['all_first_bars','within_trend'],format_func=lambda x:'所有第一根事件' if x=='all_first_bars' else '只在均線多頭組內比較法人',horizontal=True,key='flow_scope')
    st.caption('這裡排除指定十檔。每條件的「是／否」使用同一條件已知母體；未達急漲門檻不等於虧損。60日窗口重疊，全部是已研究歷史。')
    st.dataframe(pd.DataFrame(summary_rows(r,horizon,phase,lag,scope)),hide_index=True,use_container_width=True)
    for key,label in [('named_first_timeline','十檔全部訊號明細'),('retrospective_timeline','十檔急漲前時序'),('statistics','完整條件比較')]:
        st.download_button('下載'+label,verified_bytes(r['artifacts'][key],ROOT,'.csv'),file_name=key+'.csv',mime='text/csv',key='flow_'+key)
    st.caption(f"新增資料請求{r['collection_requests']}次；研究離線重算兩次相符，{len(r['causality_checks'])}組未來截斷／改寫檢查通過。實戰資格未改變。")
