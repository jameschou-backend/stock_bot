"""Read-only presentation of sealed concentration research and news gaps."""
from pathlib import Path
import json
from app.backtest_tool_ui import verified_bytes
from app.stock_launch_ui import read_table

ROOT=Path(__file__).resolve().parents[1]
PUBLICATION='artifacts/forward_simulation/theme_chips_20260927.json'
LABELS={'all_eligible':'同一籌碼已知母體','concentration':'籌碼集中',
        'relative_strength':'相對強勢','both':'集中＋相對強勢',
        'concentration_only':'集中、未相對強勢','strength_only':'相對強勢、未集中','neither':'兩者未符合'}


def load(root=ROOT):
    root=Path(root);path=root/PUBLICATION
    publication=json.loads(verified_bytes(dict(path=PUBLICATION,
        sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    report=json.loads(verified_bytes(publication['report'],root,'.json'))
    if (publication.get('schema')!='theme_chips_publication_v1' or report.get('schema')!='theme_chips_v1'
            or report.get('completed') is not True or report.get('strategy_net_return') is not None
            or any(report.get(k) is not False for k in ('live_qualified','adopted','unseen_validation',
                'portfolio_returns_computed','theme_factorial_identifiable'))):
        raise ValueError('研究範圍或資格不符')
    proof=publication['reproducibility']
    if proof.get('passed') is not True or len(proof['runs'])!=2 or proof['runs'][0]['path']==proof['runs'][1]['path']:
        raise ValueError('缺少兩輪獨立比對')
    expected={key:row['sha256'] for key,row in report['artifacts'].items()}
    if proof['csv_sha256']!=expected:raise ValueError('比對結果不符')
    paths=[]
    for descriptor in proof['runs']:
        manifest=json.loads(verified_bytes(descriptor,root,'.json'));paths.append(descriptor['path'])
        if {k:r['sha256'] for k,r in manifest['files'].items()}!=expected:
            raise ValueError('封存研究的結果不一致')
        if manifest['source_sha256']!=report['source_sha256']:raise ValueError('兩輪研究來源不同')
        if (str(Path(descriptor['path']).with_name('report.json'))==publication['report']['path']
                and manifest['report_sha256']!=publication['report']['sha256']):
            raise ValueError('報告與封存雜湊不符')
    if str(Path(publication['report']['path']).with_name('manifest.json')) not in paths:
        raise ValueError('報告未連結封存研究')
    return report


def summary_rows(report,horizon,lag,phase):
    selected=[r for r in report['statistics'] if r['horizon']==horizon and r['lag']==lag
              and r['threshold']==.005 and r['scope']==phase]
    def pct(v):return '未知' if v is None else f'{v:.2%}'
    return [{'條件':LABELS[r['rule']],'命中／已知結果':f"{r['tp']}／{r['tp']+r['fp']}",
             '急漲命中率':pct(r['precision']),'個股平均漲幅':pct(r['mean_return']),
             '個股中位漲幅':pct(r['median_return']),'平均超過0050':pct(r['mean_excess']),
             '結果未知':r['outcome_unknown_triggers']} for r in selected]


def render():
    import pandas as pd
    import streamlit as st
    st.subheader('題材與籌碼：起漲前有沒有線索？')
    try:report=load()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.error('籌碼研究尚無有效封存報告：'+str(exc));return
    st.info('已完成全市場籌碼對照。新聞日期與「沒有題材」的覆蓋尚未驗證，因此不能把下面表格稱為題材策略回測；也沒有實戰資格。')
    st.caption('集中＝四週千張以上持股增加至少0.5百分點、百張以下減少，且大戶級距股數增加。這是帳戶分布，不代表已知主力意圖。')
    horizon=st.radio('急漲觀察窗口',[20,60],horizontal=True,format_func=lambda x:'約1個月' if x==20 else '約3個月')
    phase=st.radio('研究期間',['replication','discovery'],horizontal=True,
        format_func=lambda x:'2025–2026' if x=='replication' else '2022–2024',key='theme_chip_phase')
    lag=st.radio('集保資料延遲',[8,15],horizontal=True,
        format_func=lambda x:f'觀察日後{x}個日曆日才使用')
    st.caption('急漲門檻：'+('20交易日漲30%，且超過0050 20百分點。' if horizon==20 else '60交易日漲50%，且超過0050 30百分點。')+
        '使用T日以前資訊，T+1收盤起算；排除指定九檔，兩組都只用同一籌碼已知母體。')
    st.dataframe(pd.DataFrame(summary_rows(report,horizon,lag,phase)),hide_index=True,use_container_width=True)
    inc=next(r for r in report['incremental'] if r['horizon']==horizon and r['lag']==lag
             and r['threshold']==.005 and r['phase']==phase)
    if inc['difference'] is not None:
        st.write(f"已相對強勢的股票，加上集中條件與未集中相比，急漲機率差 **{inc['difference']*100:+.2f}百分點**。")
        if inc['ci_low'] is not None:
            st.caption(f"按日期重抽的描述性95%區間：{inc['ci_low']*100:+.2f}～{inc['ci_high']*100:+.2f}百分點。未完全校正跨期相關、多次試驗或產業組成。")
    cov=next(r for r in report['coverage'] if r['horizon']==horizon and r['lag']==lag and r['phase']==phase)
    st.caption(f"籌碼可判斷 {cov['chip_known']:,}／{cov['rows']:,} 筆。缺失保留未知。個股平均漲幅不含成本、成交與资金配置，不能相加成策略報酬。")
    with st.expander('九檔案例：公告與籌碼的先後順序',expanded=True):
        cases=read_table(report,'named_events');cases=cases[cases.horizon.eq(horizon)&cases.lag.eq(lag)]
        sid=st.selectbox('查看個股',list(report['targets']),format_func=lambda s:s+' '+report['targets'][s])
        selected=cases[cases.stock_id.eq(sid)].sort_values('signal_date')
        if selected.empty:st.info('沒有符合固定門檻的回溯事件。')
        else:
            rows=[]
            for r in selected.itertuples():
                pp=lambda x:'未知' if pd.isna(x) else f'{x*100:+.2f}百分點'
                rows.append({'回溯觀察日':r.signal_date,'後續漲幅':f'{r.forward_return:.1%}',
                    '可用集保觀察日':r.observed_date,'大戶四週變化':pp(r.large_pct_delta4),
                    '小戶四週變化':pp(r.small_pct_delta4),
                    '集中條件':'未知' if pd.isna(r.concentration) else '符合' if r.concentration else '不符合',
                    '已有較早材料':'有已核對文件日期' if r.theme_observed is True else '未建立完整證據'})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        for source in report['news_sources']['events']:
            if source['stock_id']!=sid:continue
            st.markdown(f"[{source['title']}]({source['url']})")
            st.write(source['finding']);st.caption(source['caveat'])
        st.caption('事件日是事後定位，非預告買點。文件日期不等於首次上網日期；有較早材料也不代表它是上漲原因，未找到材料不等於沒有題材。')
    with st.expander('敏感度與驗證紀錄'):
        rows=[{'延遲日數':r['lag'],'增加門檻（百分點）':r['threshold']*100,
               '集中且強勢樣本':r['concentrated_rows'],'強勢未集中樣本':r['other_rows'],
               '命中率差（百分點）':None if r['difference'] is None else r['difference']*100}
              for r in report['incremental'] if r['horizon']==horizon and r['phase']==phase]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption(f"採集 {report['chip_source_audit']['requests']} 次請求，{report['chip_source_audit']['market_weeks']} 個有效週別；兩輪CSV一致，12項未來截斷／變更檢查通過。離線研究每輪約{report['elapsed_seconds']:.0f}秒。")
        st.caption('所有期間已研究過；現有公司名冊有存活者偏差；歷史集保首次發布／修訂尚未證實。新聞×籌碼四組沒有可核對的負例，暫不計勝率。')
