"""Read-only, source-verified execution and single-stock research displays."""
from functools import lru_cache
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st
from app.exit_research import _read,_file,_stamp

ROOT=Path(__file__).resolve().parents[1]
LABELS={'control':'原策略','depth':'零股加對手量限制','quote':'零股按對手買賣價',
    'slip90':'每邊滑價0.90%','entry_delay':'進場再晚一天','exit_delay':'出場再晚一天','combined':'全部壓力合併'}


def signature(folder):
    path=ROOT/'.cache'/folder/'manifest.json'
    stamp=_stamp(path);meta=_read(path)
    if meta.get('offline_identical') is not True or meta.get('live_qualified') is not False:
        raise ValueError('研究尚未完成離線核對')
    files=meta.get('files_sha256',{})
    required={f'.cache/{folder}/summary.json','docs/prereg_execution_holder_20260911.md'}
    if folder=='execution-stress':
        required|={'skills/execution_stress.py','scripts/research_execution_stress.py',
            '.cache/execution-stress/cases/control.json','.cache/execution-stress/cases/control_benchmark.json'}
    elif folder=='holder-analysis-2492':
        required|={'skills/holder_case.py','scripts/research_holder_2492.py',
            '.cache/holder-analysis-2492/weekly.csv','.cache/holder-case-2492/TaiwanStockHoldingSharesPer.parquet'}
    else:
        raise ValueError('未知研究目錄')
    if not required.issubset(files):
        raise ValueError('研究缺少必要來源索引')
    state=(stamp,tuple((n,h,_stamp(_file(ROOT,n))) for n,h in sorted(files.items())))
    if _stamp(path)!=stamp:
        raise ValueError('來源正在變更')
    return state


@lru_cache(maxsize=2)
def verified(folder,state):
    for name,expected,_ in state[1]:
        digest=hashlib.sha256()
        with _file(ROOT,name).open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):
                digest.update(block)
        if digest.hexdigest()!=expected:
            raise ValueError('研究來源已變更：'+name)
    value=_read(ROOT/'.cache'/folder/'summary.json')
    if value.get('live_qualified') is not False or signature(folder)!=state:
        raise ValueError('研究資格或來源不一致')
    return value


def load(folder):
    try:
        return verified(folder,signature(folder))
    except (OSError,KeyError,TypeError,ValueError) as exc:
        st.info('本機尚無可驗證結果：'+str(exc))
        return None


def render():
    st.subheader('原策略真的買得到嗎？')
    result=load('execution-stress')
    if result:
        st.caption('2022/1/3–2026/9/9｜100萬元複利｜每組對照相同成交條件的0050。')
        st.warning('原歷史報酬對成交深度與進場時間很敏感，目前不能視為已驗證的可實現優勢。')
        rows=[]
        for mode,label in LABELS.items():
            c,b=result['cases'][mode],result['cases'][mode+'_benchmark']
            if not c['completed'] or not b['completed']:
                rows.append({'方法':label,'狀態':'證據不足，未完成'});continue
            r,br=c['summary'],b['summary']
            rows.append({'方法':label,'淨報酬（%）':round(r['total_return']*100,2),
                '同條件0050（%）':round(br['total_return']*100,2),
                '超額（百分點）':round((r['total_return']-br['total_return'])*100,2),
                '最大回撤（%）':round(r['max_drawdown']*100,2),'期末資產（元）':round(r['final_nav'])})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        options=[m for m in LABELS if result['cases'][m]['completed'] and result['cases'][m+'_benchmark']['completed']]
        if options:
            mode=st.selectbox('查看成交壓力情境',options,format_func=LABELS.get,key='execution_case')
            base=ROOT/'.cache/execution-stress/cases'
            account=_read(base/(mode+'.json'))['account']
            compare=_read(base/(mode+'_benchmark.json'))['account']
            chart=pd.DataFrame(account['daily']).set_index('date')[['nav']].rename(columns={'nav':LABELS[mode]})
            chart['同條件0050']=pd.DataFrame(compare['daily']).set_index('date').nav
            st.line_chart(chart)
            st.download_button('下載這組全部成交CSV',pd.DataFrame(account['trades']).to_csv(index=False).encode('utf-8-sig'),
                file_name='execution_'+mode+'_trades.csv',mime='text/csv',key='execution_trades')
            st.download_button('下載這組每日資產CSV',pd.DataFrame(account['daily']).to_csv(index=False).encode('utf-8-sig'),
                file_name='execution_'+mode+'_nav.csv',mime='text/csv',key='execution_nav')
        st.caption('最後對手量只是保守快照代理，不是盤中完整撮合；延遲會改變買到的股票、後續名額及複利。')
    st.divider()
    st.subheader('華新科2492：小股東真的變少嗎？')
    holder=load('holder-analysis-2492')
    if holder:
        st.caption('股價2026/4/2–9/11；持股觀測4/2–9/4。單股事後研究，不是帳戶回測。')
        st.info('上漲段曾出現小股東減少、大戶集中；回跌後小股東人數增加。起迄相比持股占比下降、人數卻增加，提前判斷能力仍不穩定。')
        a,b=holder['first'],holder['last']
        cols=st.columns(3)
        cols[0].metric('100張以下持股占比',f"{b['small_pct']:.2f}%",f"{holder['small_pct_change']:+.2f}個百分點",delta_color='off')
        cols[1].metric('100張以下人數',f"{b['small_people']:,.0f}",f"{b['small_people']-a['small_people']:+,.0f}人",delta_color='off')
        cols[2].metric('超過1000張持股占比',f"{b['large_pct']:.2f}%",f"{holder['large_pct_change']:+.2f}個百分點",delta_color='off')
        weekly=pd.read_csv(ROOT/'.cache/holder-analysis-2492/weekly.csv')
        indexed=weekly.set_index('date')
        st.line_chart(indexed[['small_pct','large_pct']].rename(columns={'small_pct':'100張以下持股（%）','large_pct':'超過1000張持股（%）'}))
        st.dataframe(weekly[['date','raw_close','small_pct','large_pct','small_people','large_people']].rename(columns={
            'date':'持股觀測日','raw_close':'當日股價','small_pct':'100張以下持股（%）','large_pct':'超過1000張持股（%）',
            'small_people':'100張以下人數','large_people':'超過1000張人數'}),hide_index=True,use_container_width=True)
        st.download_button('下載華新科每週比較CSV',weekly.to_csv(index=False).encode('utf-8-sig'),
            file_name='holder_2492_20260402.csv',mime='text/csv',key='holder_weekly')
        st.caption('比例按集保庫存計算；100張以下包含剛好100張。公布時間採7／14日假設，不能用4/2觀測提前交易。')
