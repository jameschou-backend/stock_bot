"""Read-only individual-stock case anatomy, explicitly separate from account returns."""
from pathlib import Path
from io import BytesIO
import json
from app.backtest_tool_ui import verified_bytes

ROOT=Path(__file__).resolve().parents[1]
PUBLICATION='artifacts/forward_simulation/stock_launch_20260927.json'
LABELS={'all_eligible':'所有合格股票','relative_strength':'近20日領先0050至少10百分點',
        'breakout60':'突破前60日高點','turnover_heat':'近5日成交額升溫1.5倍',
        'breakout_strength':'突破＋相對強勢','money_strength':'成交額升溫＋強於0050',
        'early_rotation':'近高點＋成交升溫＋60日漲幅不超過30%'}


def load(root=ROOT):
    root=Path(root);path=root/PUBLICATION
    publication=json.loads(verified_bytes(dict(path=PUBLICATION,
        sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    report=json.loads(verified_bytes(publication['report'],root,'.json'))
    if (publication.get('schema')!='stock_launch_publication_v1' or report.get('schema')!='stock_launch_v1'
            or report.get('completed') is not True or report.get('strategy_net_return') is not None
            or any(report.get(k) is not False for k in ('live_qualified','adopted','unseen_validation','portfolio_returns_computed'))):
        raise ValueError('研究範圍或資格不符')
    proof=publication['reproducibility']
    if proof.get('passed') is not True or len(proof['runs'])!=2 or proof['runs'][0]['path']==proof['runs'][1]['path']:
        raise ValueError('缺少兩輪獨立比對')
    expected={key:row['sha256'] for key,row in report['artifacts'].items()}
    if proof['csv_sha256']!=expected:raise ValueError('比對結果不符')
    manifest_paths=[]
    for descriptor in proof['runs']:
        manifest=json.loads(verified_bytes(descriptor,root,'.json'));manifest_paths.append(descriptor['path'])
        if {key:row['sha256'] for key,row in manifest['files'].items()}!=expected:
            raise ValueError('封存研究的結果不一致')
        if manifest['source_sha256']!=report['source_sha256']:raise ValueError('兩輪研究來源不同')
        if (str(Path(descriptor['path']).with_name('report.json'))==publication['report']['path']
                and manifest['report_sha256']!=publication['report']['sha256']):
            raise ValueError('報告與封存雜湊不符')
    if str(Path(publication['report']['path']).with_name('manifest.json')) not in manifest_paths:
        raise ValueError('報告未連結封存研究')
    return report


def read_table(report,name,root=ROOT):
    import pandas as pd
    return pd.read_csv(BytesIO(verified_bytes(report['artifacts'][name],root,'.csv')),dtype={'stock_id':str,'case_stock_id':str})


def case_rows(report,horizon):
    rows=[]
    for row in report['latest_cases']:
        if row['horizon']!=horizon:continue
        if not row['event_found']:
            rows.append({'股票':row['stock_id']+' '+row['name'],'回溯觀察日':'無符合事件'});continue
        rows.append({'股票':row['stock_id']+' '+row['name'],'回溯觀察日':row['signal_date'],
            '後續價格漲幅':f"{row['forward_return']:.1%}",'當時近20日漲幅':f"{row['momentum20']:.1%}",
            '領先0050':f"{row['relative20']*100:.1f}百分點",'近5日成交額比':f"{row['turnover_ratio']:.2f}倍",
            '當日突破60日高點':'是' if row['breakout60'] else '否','失敗對照數':row['matched_controls']})
    return rows


def render():
    import pandas as pd
    import streamlit as st
    st.subheader('個股啟動前研究')
    st.caption('台達電、禾伸堂、國巨、華新科、華邦電、南亞科、南亞、聯電、群創｜0050僅作比較基準')
    try:report=load()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.error('個股研究尚無有效封存報告：'+str(exc));return
    st.info('這裡回答大漲前有哪些線索。回溯觀察日由事後結果定位，不能當成當時已知的買點；命中率也不是投資報酬率。')
    horizon=st.radio('觀察上漲速度',[20,60],format_func=lambda x:'約1個月：漲30%且超過0050 20百分點' if x==20 else '約3個月：漲50%且超過0050 30百分點',horizontal=True)
    st.caption(f"行情至{report['source_end']}。每檔列最後一個合格事件，完整歷史事件與資料缺口保留。")
    st.dataframe(pd.DataFrame(case_rows(report,horizon)),hide_index=True,use_container_width=True)
    st.write('**同樣條件放到其他股票，多少次真的急漲？**')
    phase=st.radio('檢查期間',['replication','discovery'],format_func=lambda x:'2025–2026' if x=='replication' else '2022–2024',horizontal=True)
    stats=[r for r in report['rule_statistics'] if r['horizon']==horizon and r['cohort']=='excluding_named_nine' and r['scope']==phase]
    rows=[{'條件':LABELS[r['rule']],'命中率':f"{r['precision']:.1%}" if r['precision'] is not None else '未知',
           '急漲／已知結果':f"{r['tp']}／{r['tp']+r['fp']}",'未達急漲門檻':r['fp'],
           '涵蓋急漲比例':f"{r['recall']:.1%}" if r['recall'] is not None else '未知',
           '結果未知':r['outcome_unknown_triggers']} for r in stats]
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.caption('已排除指定九檔，避免用指定贏家算勝率。「未達門檻」不等於虧損。固定每21個交易日觀察一次；60日窗口會重疊。所有期間都已研究過。')
    with st.expander('逐檔查看觀察日前20、10、5日與當日'):
        sid=st.selectbox('股票',list(report['targets']),format_func=lambda s:s+' '+report['targets'][s])
        try:
            events=read_table(report,'events');events=events[events.horizon.eq(horizon)&events.stock_id.eq(sid)]
            if events.empty:st.info('固定定義下沒有合格事件。');return
            event_labels={row.case_id:row.signal_date+' 觀察日' for row in events.itertuples()}
            cid=st.selectbox('歷史事件',events.case_id.tolist(),index=len(events)-1,format_func=event_labels.get)
            series=read_table(report,'trajectories');series=series[series.case_id.eq(cid)&series.role.eq('case')]
            view=series[['offset','signal_date','momentum20','relative20','turnover_ratio','breakout60']].copy()
            view['momentum20']=view.momentum20.map(lambda v:f'{v:.1%}' if pd.notna(v) else '未知')
            view['relative20']=view.relative20.map(lambda v:f'{v*100:.1f}百分點' if pd.notna(v) else '未知')
            view['turnover_ratio']=view.turnover_ratio.map(lambda v:f'{v:.2f}倍' if pd.notna(v) else '未知')
            view.columns=['距觀察日','日期','近20日漲幅','相對0050強度','成交額比','突破60日高點']
            st.dataframe(view,hide_index=True,use_container_width=True)
            plot=series.set_index('offset')[['momentum20','relative20']]*100
            plot.columns=['近20日漲幅（%）','領先0050（百分點）'];st.line_chart(plot)
            controls=read_table(report,'matched_controls')
            st.write('同日、同市場／產業／流動性，卻未達急漲門檻的對照：')
            control_view=controls[controls.case_id.eq(cid)][['stock_id','name','signal_date','forward_return','turnover_ratio']].copy()
            control_view['forward_return']=control_view.forward_return.map(lambda v:f'{v:.1%}' if pd.notna(v) else '未知')
            control_view['turnover_ratio']=control_view.turnover_ratio.map(lambda v:f'{v:.2f}倍' if pd.notna(v) else '未知')
            control_view.columns=['代號','名稱','觀察日','後續價格漲幅','成交額比']
            st.dataframe(control_view,hide_index=True,use_container_width=True)
        except (OSError,ValueError,KeyError,TypeError) as exc:st.error('事件資料驗證未通過：'+str(exc))
    with st.expander('資料完整性與重跑記錄'):
        st.dataframe(pd.DataFrame([r for r in report['named_coverage'] if r['horizon']==horizon]),hide_index=True,use_container_width=True)
        st.caption(f"兩輪CSV結果一致；6項截斷／變更未來檢查通過；離線研究耗時{report['elapsed_seconds']:.1f}秒，FinMind請求0次。")
        st.caption('名冊與產業分類是目前版本；缺行情／價格版本衝突保留未知。法人、財報與新聞尚未納入這份量化訊號。尚無實戰資格。')
