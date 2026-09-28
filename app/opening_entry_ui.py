"""Verified opening-entry research, kept separate from live qualification."""
from copy import deepcopy
from pathlib import Path
import json
import threading

from app.backtest_full_pass_ui import _file_signature
from app.backtest_tool_ui import verified_bytes
from scripts.research_exit_scenarios import sha,summarize
from scripts.research_opening_entry import CASES,analyze

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/opening_entry_20260928.json'
_CACHE={};_LOCK=threading.RLock()


def validate(path,root):
    if sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('開盤研究報告已變更')
    value=json.loads(path.read_text())
    if (value.get('schema')!='opening_entry_v1' or value.get('offline_identical') is not True
        or value.get('compared_cases')!=4 or set(value['cases'])!=set(CASES)
        or value.get('live_qualified') is not False or value.get('unseen_validation') is not False
        or value.get('opening_auction_inferred') is not True or value.get('cancellation_latency_verified') is not False
        or value.get('network_calls')!=0):raise ValueError('缺少四組離線重播或開盤限制說明')
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
                'unknown_liquidity_rejected','opening_buy_rule_rebuilt','unchanged_sell_rule_rebuilt','tick_fills_rebuilt')):
                raise ValueError('開盤完整帳戶與稽核未核對')
        elif row['summary'] is not None or 'account' in result:raise ValueError('資料中止不可發表報酬')
    controls={k:json.loads(verified_bytes(v,root,'.json')) for k,v in value['controls'].items()}
    if value['all_completed']!=all(c['completed'] for c in cases.values()) or value['analysis']!=analyze(cases,controls):
        raise ValueError('開盤比較與帳本不一致')
    return value


def load(path=REPORT,root=ROOT):
    path,root=Path(path).resolve(),Path(root).resolve()
    value=json.loads(path.read_text())
    refs=set(value['source_sha256'])|{str(path.relative_to(root)),str(path.with_suffix('.sha256').relative_to(root))}
    refs.update(r['result']['path'] for r in value['cases'].values())
    refs.update(r['path'] for r in value['controls'].values())
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
    with st.expander('改成隔天開盤買，會比較好嗎？',expanded=True):
        if not REPORT.exists():
            st.info('開盤買進版本正在核對；未完成前不顯示部分期間報酬。');return
        try:value=load()
        except (OSError,ValueError,KeyError,TypeError) as exc:
            st.error('開盤研究目前不可採信：'+str(exc));return
        st.caption('2022/01/03～2026/09/09，454 個固定訊號；100 萬元複利、5 檔個股、整張、閒錢留現金。')
        st.write('今天收盤選好股票 → 明天開盤前固定股數與預算 → 按開盤成交價和開盤量估計成交。賣出規則維持原版。')
        mode=st.radio('開盤成交情境',['normal','stress'],horizontal=True,key='opening_entry_mode',
            format_func=lambda k:'一般成本與量' if k=='normal' else '較高成本、較少可成交量')
        name='strategy_'+mode;case=value['cases'][name];analysis=value['analysis'][name]
        benchmark=value['cases']['benchmark_'+mode]
        rows=[]
        for label,row in [('個股：隔天開盤買',case),('0050：同條件開盤買',benchmark)]:
            summary=row['summary']
            rows.append({'版本':label,'總報酬':f"{summary['total_return']:+.2%}" if summary else '資料中止',
                '最大回撤':f"{summary['max_drawdown']:.2%}" if summary else '—','原因':row.get('reason') or ''})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        if analysis['completed']:
            control=analysis['control_summary']
            st.write(f"原前收盤限價版：{control['total_return']:+.2%}；開盤版相對0050：{analysis['excess_return']*100:+.2f} 個百分點。")
            annual=pd.DataFrame(analysis['annual']).rename(columns={'year':'年度','strategy':'個股報酬','benchmark':'0050報酬','excess':'超額報酬'})
            for col in ('個股報酬','0050報酬','超額報酬'):annual[col]=annual[col].map(lambda n:f'{n:+.2%}')
            st.dataframe(annual,hide_index=True,use_container_width=True)
        st.warning('開盤量以最早同時間逐筆批次推定，尚非完整委託簿證明。剩餘委託撤單延遲未驗證；此為已看過歷史的研究，不代表實戰資格。')
        st.caption('開盤前以漲停價上限預留現金，成交按開盤價另扣成本；開盤漲停不算買到。只用開盤批次的1%量，壓力情境0.5%，不以全日量補足。')
        result=json.loads(verified_bytes(case['result'],ROOT,'.json'))
        st.download_button('下載開盤版全部計畫、成交及資產',json.dumps(result,ensure_ascii=False,indent=2),
            name+'_opening.json','application/json',key='opening_entry_download')
