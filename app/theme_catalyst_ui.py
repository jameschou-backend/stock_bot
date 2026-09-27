"""Read-only comparison and exports of dated theme-strategy accounts."""
from pathlib import Path
import json
from app.backtest_tool_ui import verified_bytes

ROOT=Path(__file__).resolve().parents[1]
PUBLICATION='artifacts/forward_simulation/theme_catalyst_20260927.json'


def case_names():
    return {f'{scope}_delay{delay}_{arm}_{stress}' for scope in ('primary','augmented')
        for delay in (0,5) for arm in ('watchlist_breakout','confirmed_catalyst')
        for stress in ('control','combined')} | {'benchmark_control','benchmark_combined'}


def load(root=ROOT):
    root=Path(root);path=root/PUBLICATION
    publication=json.loads(verified_bytes(dict(path=PUBLICATION,
        sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    if publication.get('schema')!='theme_catalyst_publication_v1':raise ValueError('研究發布版本不符')
    report=json.loads(verified_bytes(publication['report'],root,'.json'))
    if (report.get('schema')!='theme_catalyst_v1' or report.get('completed') is not True
            or set(report['cases'])!=case_names()
            or any(report.get(k) is not False for k in ('live_qualified','adopted','unseen_validation',
                'historical_first_publication_verified','valid_unbiased_strategy_evidence'))):
        raise ValueError('研究範圍或資格標記不符')
    proof=publication['reproducibility']
    if (proof.get('passed') is not True or len(proof['runs'])!=2
            or proof['runs'][0]['path']==proof['runs'][1]['path']):raise ValueError('缺少兩輪獨立重算')
    found=False
    for descriptor in proof['runs']:
        manifest=json.loads(verified_bytes(descriptor,root,'.json'))
        if manifest['source_sha256']!=report['source_sha256']:raise ValueError('重算來源不同')
        comparable={k:v for k,v in manifest['files_sha256'].items() if k!='report.json'}
        if comparable!=proof['files_sha256']:raise ValueError('重算帳戶或表格不同')
        folder=Path(descriptor['path']).parent
        if str(folder/'report.json')==publication['report']['path']:
            if manifest['files_sha256']['report.json']!=publication['report']['sha256']:raise ValueError('報告未封存')
            found=True
            for row in report['cases'].values():
                p=Path(row['result']['path'])
                if p.parent!=folder or manifest['files_sha256'].get(p.name)!=row['result']['sha256']:
                    raise ValueError('帳戶未連結報告')
    if not found:raise ValueError('報告沒有對應重算紀錄')
    if report['all_accounts_completed']!=all(c['completed'] for c in report['cases'].values()):
        raise ValueError('帳戶完成狀態不符')
    for name,row in report['cases'].items():
        if not row['completed'] and ('summary' in row or row.get('excess_return') is not None):
            raise ValueError('未完成帳戶不能顯示全期報酬')
        if not name.startswith('benchmark') and row['completed']:
            bm=report['cases']['benchmark_'+name.rsplit('_',1)[1]]
            if bm['completed'] and (row['benchmark_return']!=bm['summary']['total_return']
                    or abs(row['excess_return']-(row['summary']['total_return']-row['benchmark_return']))>1e-12):
                raise ValueError('基準或超額報酬不符')
    return report


def comparison_rows(report,delay):
    result=[]
    for scope,label in [('primary','原始七檔'),('augmented','七檔＋昇達科')]:
        for arm,title in [('watchlist_breakout','題材名單＋突破'),('confirmed_catalyst','營運證據＋突破')]:
            base=report['cases'][f'{scope}_delay{delay}_{arm}_control']
            stress=report['cases'][f'{scope}_delay{delay}_{arm}_combined']
            pct=lambda r,key:'資料不足，未完成' if not r['completed'] else f"{r['summary'][key]:.2%}"
            result.append({'名單':label,'規則':title,'一般淨報酬':pct(base,'total_return'),
                '壓力淨報酬':pct(stress,'total_return'),'一般最大回撤':pct(base,'max_drawdown'),
                '與0050差距':f"{base['excess_return']*100:+.2f}百分點" if base.get('excess_return') is not None else '未比較',
                '候選訊號':base['candidate_count'],
                '平均現金':f"{base['average_cash_fraction']:.1%}" if base['completed'] else '—'})
    return result


def read_case(report,name,root=ROOT):
    data=json.loads(verified_bytes(report['cases'][name]['result'],Path(root),'.json'))
    if data['strategy_case']!=name or data['completed']!=report['cases'][name]['completed']:
        raise ValueError('個別帳戶與摘要不同')
    if data['completed'] and data['summary']!=report['cases'][name]['summary']:
        raise ValueError('帳戶報酬與摘要不同')
    return data


def render():
    import pandas as pd
    import streamlit as st
    st.subheader('題材營運策略：有業績支持，再等突破')
    try:report=load()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.error('尚無有效封存結果：'+str(exc));return
    st.info('這是已見歷史上的策略實驗。文件日期採保守延遲假設，尚未證明當時已取得該版本；不具實戰資格。')
    st.write('本金100萬、最多5檔、每檔以當時淨值20%為上限，閒錢留現金。訊號後下一交易日才嘗試買入，含整股／零股、稅費、滑價與成交量限制。')
    st.caption(f"期間 {report['start']}～{report['end']}。突破＝高於前20日最高收盤、60日均線，且20日漲幅領先0050；平均成交金額至少5,000萬。收盤跌12%或持有63交易日後才發出出場指示。")
    delay=st.radio('公告時間敏感度',[0,5],horizontal=True,
        format_func=lambda v:'文件日後開始使用' if v==0 else '再延後5個交易日',key='catalyst_delay')
    st.dataframe(pd.DataFrame(comparison_rows(report,delay)),hide_index=True,use_container_width=True)
    bm=report['cases']['benchmark_control']
    if bm['completed']:st.caption(f"同期間0050扣成本報酬 {bm['summary']['total_return']:.2%}。未完成的對照組不顯示全期報酬；0%且0訊號代表全現金，不能當成有效策略。")
    st.warning('七檔官方材料多未拆分低軌衛星貢獻，屬未知；不能把其他AI業績當成衛星收入。昇達科是指定贏家的增補案例，不能冒充事前全市場選出。')
    choices=[n for n in sorted(report['cases']) if not n.startswith('benchmark') and f'delay{delay}_' in n]
    names={n:('七檔＋昇達科' if n.startswith('augmented') else '原始七檔')+'｜'+
        ('營運證據＋突破' if 'confirmed_catalyst' in n else '題材名單＋突破')+'｜'+
        ('一般成本' if n.endswith('control') else '合併壓力') for n in choices}
    preferred=f'augmented_delay{delay}_confirmed_catalyst_control'
    selected=st.selectbox('檢查交易與資產',choices,index=choices.index(preferred),format_func=lambda n:names[n])
    try:case=read_case(report,selected)
    except (OSError,ValueError,KeyError) as exc:st.error(str(exc));return
    if not case['completed']:
        st.error('此帳戶未完成：'+case['reason'])
    else:
        account=case['account'];daily=pd.DataFrame(account['daily'])
        daily['date']=pd.to_datetime(daily['date'])
        nav, profit, drawdown=st.columns(3)
        nav.metric('期末總資產',f"{case['summary']['final_nav']:,.0f} 元")
        profit.metric('扣成本累積報酬',f"{case['summary']['total_return']:.2%}")
        drawdown.metric('最大回撤',f"{case['summary']['max_drawdown']:.2%}")
        st.line_chart(daily.set_index('date')[['nav','cash']].rename(columns={'nav':'總資產','cash':'現金'}))
        trades=pd.DataFrame(account['trades'])
        if trades.empty:st.write('沒有成交，資金留現金。')
        else:
            fields=[k for k in ('date','signal_date','stock_id','name','side','channel','qty','reference_price','gross','total_cost','reason') if k in trades]
            shown=trades[fields].copy().replace({'side':{'buy':'買進','sell':'賣出'},
                'channel':{'odd':'零股','board':'整股'},
                'reason':{'leader_entry':'進場條件成立','time63':'持有63交易日','loss12':'收盤停損條件'}})
            shown=shown.rename(columns={'date':'成交日','signal_date':'訊號日','stock_id':'代號','name':'股票',
                'side':'買賣','channel':'市場','qty':'股數','reference_price':'參考價',
                'gross':'成交金額','total_cost':'稅費與滑價','reason':'原因'})
            st.dataframe(shown,hide_index=True,use_container_width=True)
            st.caption('滑價已單獨列入成本；參考價不是保證可成交價格。下載檔保留原始欄位供核帳。')
        for key,label in [('trades','逐筆成交'),('daily','每日資產'),('cash_ledger','現金帳本')]:
            st.download_button('下載'+label,pd.DataFrame(account[key]).to_csv(index=False).encode('utf-8-sig'),
                file_name=selected+'-'+key+'.csv',mime='text/csv',key='catalyst_'+key)
    with st.expander('公司原始資料與驗證範圍'):
        sid=st.selectbox('公司',list(report['cohort'])+['3491'],key='catalyst_issuer')
        for e in report['events']:
            if e['stock_id']!=sid:continue
            st.markdown(f"{e['source_date']}｜[{e['title']}]({e['source_url']})")
            st.caption('可作歷史日期假設' if e['signal_eligible'] else '排除：文件日期或版本不符合')
            st.write(e['facts'])
        st.caption(f"4個截斷日期、12項未來價格／文件檢查通過；嚴格按實際收集日模式，歷史訊號為0。離線重算約{report['elapsed_seconds']:.1f}秒，0次FinMind。")
        for limitation in report['limitations']:st.write('• '+limitation)
