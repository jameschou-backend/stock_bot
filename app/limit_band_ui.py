"""Human-readable, hash-verified fixed buy/sell execution comparison."""
from copy import deepcopy
from pathlib import Path
import json
import threading

from app.backtest_full_pass_ui import _file_signature
from app.backtest_tool_ui import verified_bytes
from scripts.research_exit_scenarios import sha,summarize
from scripts.research_limit_bands import CASES,analyze

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/limit_bands_20260928.json'
LABELS={0:'原限價',1:'只放寬買進 2%',2:'只降低賣出限價 2%',3:'買賣同時調整 2%'}
_CACHE={};_LOCK=threading.RLock()


def validate(path,root):
    if sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('限價比較報告已變更')
    value=json.loads(path.read_text())
    if (value.get('schema')!='limit_bands_v1' or value.get('offline_identical') is not True
        or value.get('compared_cases')!=12 or set(value['cases'])!=set(CASES)
        or value.get('live_qualified') is not False or value.get('unseen_validation') is not False
        or value.get('network_calls')!=0):raise ValueError('缺少十二組完整離線重播證據')
    for name,digest in value['source_sha256'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root) or sha(target)!=digest:raise ValueError('實驗來源已變更：'+name)
    cases={}
    for name,row in value['cases'].items():
        result=json.loads(verified_bytes(row['result'],root,'.json'));cases[name]=result
        if row['config']!=CASES[name] or row['completed']!=result['completed'] or row['summary']!=result['summary']:
            raise ValueError('顯示組別或結果不一致')
        if row['completed']:
            if row['summary']!=summarize(result['account']) or not all(result['audit'].get(k) is True for k in (
                'unknown_liquidity_rejected','fixed_band_rule_rebuilt','tick_fills_rebuilt')):
                raise ValueError('完整帳戶與稽核未核對')
        elif row['summary'] is not None or 'account' in result:raise ValueError('資料中止不可發表報酬')
    if value['all_completed']!=all(c['completed'] for c in cases.values()) or value['analysis']!=analyze(cases):
        raise ValueError('跨組比較與完整帳戶不一致')
    return value


def load(path=REPORT,root=ROOT):
    path,root=Path(path).resolve(),Path(root).resolve()
    value=json.loads(path.read_text())
    refs=set(value['source_sha256'])|{str(path.relative_to(root)),str(path.with_suffix('.sha256').relative_to(root))}
    refs.update(r['result']['path'] for r in value['cases'].values())
    signature=lambda:tuple((p,_file_signature(root/p,root)) for p in sorted(refs))
    before=signature();key=(str(path),str(root))
    with _LOCK:
        if key in _CACHE and _CACHE[key][0]==before:return deepcopy(_CACHE[key][1])
        result=validate(path,root)
        if before!=signature():raise ValueError('驗證期間來源改變')
        _CACHE[key]=(before,deepcopy(result));return result


def render():
    import pandas as pd
    import streamlit as st
    with st.expander('買賣限價分開改善，結果如何？',expanded=True):
        if not REPORT.exists():
            st.info('固定四種限價規則，待完整重播核對後公布。');return
        try:value=load()
        except (OSError,ValueError,KeyError,TypeError) as exc:
            st.error('本輪比較目前不可採信：'+str(exc));return
        st.caption('454 個固定訊號，2022/01/03～2026/09/09；100萬元複利、5檔個股、整張、閒置現金。')
        mode=st.radio('成交情境',['normal','stress'],format_func=lambda k:'一般成本與量' if k=='normal' else '較高成本、較少可成交量',
                      horizontal=True,key='limit_band_mode')
        rows=[]
        for mask,label in LABELS.items():
            name=f'strategy_{mask}_{mode}';case=value['cases'][name];analysis=value['analysis'][name]
            benchmark=value['cases'][f'benchmark_{mask&1}_{mode}']
            summary=case['summary'];b=benchmark['summary']
            rows.append({'限價規則':label,'個股總報酬':f"{summary['total_return']:+.2%}" if summary else '資料中止',
                '0050同條件':f"{b['total_return']:+.2%}" if b else '資料中止',
                '領先／落後':f"{analysis['excess_return']*100:+.2f} 個百分點" if analysis['completed'] else '—',
                '最大回撤':f"{summary['max_drawdown']:.2%}" if summary else '—',
                '原因':case.get('reason') or ''})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.write('放寬買價：前日參考價 × 1.02；降低賣價：前日參考價 × 0.98。都在看當天成交前固定，'
                 '買進仍受預留現金限制；同價排隊量不計入，穿價量不足就部分成交或不成交。')
        st.warning('這是同一段已研究歷史的執行改善比較；不等於未見樣本勝出，也不代表已具備實戰資格。')
        mask=st.selectbox('查看完整買賣與穩健性',list(LABELS),format_func=LABELS.get,key='limit_band_case')
        name=f'strategy_{mask}_{mode}';analysis=value['analysis'][name]
        if analysis['completed']:
            rolling=analysis['rolling_252']
            st.write(f"252日滾動窗口勝過0050：{rolling['win_fraction']:.1%}；"
                     f"退出有等待：{analysis['delayed_exit_events']} 次，最長 {analysis['max_wait_sessions']} 個交易日。")
            annual=pd.DataFrame(analysis['annual']).rename(columns={'year':'年度','strategy':'個股報酬','benchmark':'0050報酬','excess':'超額報酬'})
            for col in ('個股報酬','0050報酬','超額報酬'):annual[col]=annual[col].map(lambda n:f'{n:+.2%}')
            st.dataframe(annual,hide_index=True,use_container_width=True)
            st.caption('滾動窗口有重疊；2026年只計到9月9日。')
        result=json.loads(verified_bytes(value['cases'][name]['result'],ROOT,'.json'))
        st.download_button('下載這組全部計畫、委託、成交及資產',json.dumps(result,ensure_ascii=False,indent=2),
                           name+'.json','application/json',key='limit_band_download')
