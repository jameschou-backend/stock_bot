"""Mixed-channel research explicitly separated from verified odd-lot execution."""
from copy import deepcopy
from pathlib import Path
import json
import threading
from app.backtest_full_pass_ui import _file_signature
from app.backtest_tool_ui import verified_bytes
from scripts.research_exit_scenarios import sha,summarize
from scripts.research_mixed_odd import CASES,analyze

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/mixed_odd_20260928.json'
_CACHE={};_LOCK=threading.RLock()


def validate(path,root):
    if sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('零股研究報告已變更')
    v=json.loads(path.read_text())
    if (v.get('schema')!='mixed_odd_v1' or v.get('offline_identical') is not True
        or v.get('compared_cases')!=4 or set(v['cases'])!=set(CASES)
        or v.get('live_qualified') is not False or v.get('unseen_validation') is not False
        or v.get('odd_tick_verified') is not False or v.get('odd_execution_evidence')!='daily_envelope_estimate'
        or v.get('opening_auction_inferred') is not True or v.get('cancellation_latency_verified') is not False
        or v.get('network_calls')!=0):raise ValueError('零股日估算不可冒充撮合驗證')
    for name,digest in v['source_sha256'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root) or sha(target)!=digest:raise ValueError('零股來源已變更：'+name)
    cases={}
    for name,row in v['cases'].items():
        result=json.loads(verified_bytes(row['result'],root,'.json'));cases[name]=result
        if row['config']!=CASES[name] or row['completed']!=result['completed'] or row['summary']!=result['summary']:
            raise ValueError('零股顯示與帳本不一致')
        if row['completed']:
            if row['summary']!=summarize(result['account']) or not all(result['audit'].get(k) is True for k in (
                'precommitted_mixed_plans','channel_fills_rebuilt','independent_odd_prices','separate_channel_costs','daily_odd_estimate_only')):
                raise ValueError('缺少零股帳務核對')
        elif row['summary'] is not None or 'account' in result:raise ValueError('中止回測不得顯示部分報酬')
    controls={k:json.loads(verified_bytes(ref,root,'.json')) for k,ref in v['controls'].items()}
    if v['all_completed']!=all(c['completed'] for c in cases.values()) or v['analysis']!=analyze(cases,controls):
        raise ValueError('零股比較結果不一致')
    return v


def load(path=REPORT,root=ROOT):
    path,root=Path(path).resolve(),Path(root).resolve();v=json.loads(path.read_text())
    refs=set(v['source_sha256'])|{str(path.relative_to(root)),str(path.with_suffix('.sha256').relative_to(root))}
    refs.update(r['result']['path'] for r in v['cases'].values());refs.update(r['path'] for r in v['controls'].values())
    signature=lambda:tuple((p,_file_signature(root/p,root)) for p in sorted(refs))
    before=signature();key=(str(path),str(root))
    with _LOCK:
        if key in _CACHE and _CACHE[key][0]==before:return deepcopy(_CACHE[key][1])
        result=validate(path,root)
        if before!=signature():raise ValueError('零股驗證期間來源改變')
        _CACHE[key]=(before,deepcopy(result));return result


def render():
    import pandas as pd
    import streamlit as st
    with st.expander('加入零股：讓每檔預算不足一張也能買',expanded=True):
        st.write('整張走原開盤規則，剩餘 1～999 股另下零股單；高價股可以只買零股，賣出及配股殘股同樣處理。')
        st.warning('這一版是零股日行情估算，尚非歷史逐次撮合驗證。零股價格、量與佣金獨立計算，不套用整張開盤價。')
        if not REPORT.exists():st.info('完整比較尚在核對，暫不顯示報酬。');return
        try:v=load()
        except (OSError,ValueError,KeyError,TypeError) as exc:st.error('零股報告目前不可採信：'+str(exc));return
        rows=[]
        for name,row in v['cases'].items():
            s=row['summary'];label=('0050 基準' if name.startswith('benchmark') else '五檔個股')+('／一般' if name.endswith('normal') else '／壓力')
            rows.append({'帳戶':label,'結果':'完整日行情估算' if row['completed'] else '缺資料，中止',
                         '淨報酬':f"{s['total_return']:+.2%}" if s else '尚無完整報酬','缺少的資料':row.get('reason') or ''})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        if not v['all_completed']:st.info('零股規則與核帳已加入；缺官方零股日表的帳戶不以普通盤資料代替，也不把中途資產當最終報酬。')
        name=st.selectbox('查看零股計畫與資料缺口',list(v['cases']),key='mixed_odd_case')
        result=json.loads(verified_bytes(v['cases'][name]['result'],ROOT,'.json'))
        st.download_button('下載零股計畫與核對紀錄',json.dumps(result,ensure_ascii=False,indent=2),
            name+'_mixed_odd.json','application/json',key='mixed_odd_download')
