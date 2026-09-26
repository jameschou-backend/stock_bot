"""Bounded rerun controls and verified results for a published ETF account."""
import hashlib
import json
import math
from pathlib import Path
from app.backtest_tool_ui import verified_bytes
from app.index_continuous_ui import ROOT,load,detail,ARMS
from skills.backtest_case_cache import content_digest


def load_result(path,*,root=ROOT,request=None,expected_sha256=None):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root/'.cache'):
        raise ValueError('回測結果不在本機快取範圍')
    raw=path.read_bytes()
    if expected_sha256 and hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('工作完成檔已改變')
    value=json.loads(raw)
    if (value.get('schema')!='index_case_job_v1' or value.get('completed') is not True
            or any(value.get(k) is not False for k in ('live_qualified','unseen_validation','strict_data_ready'))
            or value.get('full_account_and_decisions_reproduced') is not True):
        raise ValueError('回測未完成或資格標記不符')
    arm,mask=value['arm'],value['factor_mask']
    if arm not in ARMS or type(mask) is not int or mask not in range(8):
        raise ValueError('回測規則或情境不符')
    if request is not None and (request.get('kind')!='index_backtest'
            or request.get('index_rule','equal')!=arm or request.get('index_mask',0)!=mask):
        raise ValueError('回測結果與所選工作不符')
    metrics=value['metrics']
    for key in ('network_calls','database_writes','executed_cases','reused_cases'):
        if type(metrics.get(key)) is not int:raise ValueError('回測執行計數不完整')
    if (metrics['network_calls']!=0 or metrics['database_writes']!=0
            or (metrics['executed_cases'],metrics['reused_cases']) not in ((1,0),(0,1))):
        raise ValueError('回測執行計數不符')
    if request and request.get('index_fresh') and metrics['executed_cases']!=1:
        raise ValueError('要求重新計算但只回傳快取')
    for key in ('source_validation_seconds','elapsed_seconds'):
        if type(metrics.get(key)) not in (int,float) or not math.isfinite(metrics[key]) or metrics[key]<0:
            raise ValueError('回測耗時紀錄不符')
    if metrics['elapsed_seconds']<metrics['source_validation_seconds']:raise ValueError('回測耗時順序不符')
    pub=load(root);name=f'{arm}_{mask}'
    benchmark_key='combined' if mask&1 else 'control'
    for key in ('start','end','initial_cash','data_quality','limitations'):
        if value[key]!=pub[key]:raise ValueError('回測範圍或限制與封存研究不符')
    if (value['canonical_case']!=pub['cases'][name]['result']
            or value['benchmark']!=pub['benchmarks'][benchmark_key]['result']):
        raise ValueError('回測未連結所選封存帳戶')
    expected,benchmark=detail(pub,name,root)
    case=json.loads(verified_bytes(value['account'],root,'.json'))
    fields=('account','decisions','plans','pending','audit','summary','config','completed','live_qualified','unseen_validation')
    if (set(case)!=set(fields) or any(case[k]!=expected[k] for k in fields)
            or value['summary']!=case['summary'] or value['benchmark_summary']!=benchmark['summary']):
        raise ValueError('完整帳戶、判斷或摘要與封存結果不符')
    record=json.loads(verified_bytes(value['source_identity'],root,'.json'))
    identity=record['identity']
    if (record['identity_digest']!=content_digest(identity) or identity['schema']!='index_case_identity_v1'
            or identity['period']!=[pub['start'],pub['end']] or identity['initial_cash']!=pub['initial_cash']):
        raise ValueError('來源身分紀錄不符')
    sources=json.loads(verified_bytes(pub['source_identity'],root,'.json'))
    sources[pub['source_identity']['path']]=pub['source_identity']['sha256']
    for ref in (value['canonical_case'],value['benchmark']):sources[ref['path']]=ref['sha256']
    if any(identity['sources'].get(p)!=digest for p,digest in sources.items()):
        raise ValueError('回測來源未連結封存資料')
    return value,case


def render(arm,mask):
    import streamlit as st
    from app import workbench_jobs as jobs
    st.write('驗證所選帳戶')
    st.caption('固定100萬元、2016–2026期間。每次核對全部來源；首次計算，之後可重用一致結果。不抓取行情、不建立排程。')
    fresh=st.checkbox('忽略已完成結果，重新計算',key='index_case_fresh')
    if st.button('再次驗證這組回測',key='index_case_run'):
        try:
            job=jobs.submit(jobs.WorkRequest(kind='index_backtest',index_rule=arm,index_mask=mask,index_fresh=fresh))
            st.session_state['index_case_job']=job['job_id']
            st.success('已提交一次背景驗證，工作完成後可更新狀態。')
        except (OSError,ValueError,TimeoutError) as exc:st.error(str(exc))
    candidates=[j for j in jobs.recent_jobs(100) if j['request']['kind']=='index_backtest'
                and j['request'].get('index_rule','equal')==arm and j['request'].get('index_mask',0)==mask]
    if not candidates:return
    job=candidates[0]
    st.write('目前工作：'+job['message'])
    if job['status'] in ('queued','running'):
        st.caption('切換頁面不會停止工作；請勿重複提交。')
        if st.button('更新回測狀態',key='index_case_refresh'):st.rerun()
    elif job['status']=='completed':
        try:
            if not job.get('result_sha256'):raise ValueError('缺少工作完成檔指紋')
            value,case=load_result(job['result_path'],request=job['request'],expected_sha256=job['result_sha256'])
            m=value['metrics']
            st.success(f"完整帳戶與所有判斷一致｜重新計算 {m['executed_cases']} 組｜重用 {m['reused_cases']} 組｜{m['elapsed_seconds']:.1f} 秒")
            st.caption(f"其中來源內容核對 {m['source_validation_seconds']:.1f} 秒；網路請求 0 次。完成重現仍不代表取得實戰資格。")
            st.download_button('下載本次驗證完整帳戶',json.dumps(case,ensure_ascii=False),file_name=f'{arm}_{mask}-verified-account.json',mime='application/json',key='index_case_result')
        except (OSError,ValueError,KeyError,TypeError) as exc:st.error('回測結果核對失敗：'+str(exc))
