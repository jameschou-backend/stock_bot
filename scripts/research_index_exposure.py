#!/usr/bin/env python3
"""Run all fixed index exposures offline and preserve every attempted account."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read,write,sha,summarize
from scripts.research_volatility_budget import metrics
from skills.index_exposure_inputs import load,BASE,BENCHMARK
from skills.index_exposure_replay import IndexExposureReplay
from skills.index_exposure_audit import audit_index
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.strategy_validation_gate import assess_family
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

ARMS={'equal':0,'trend200':200,'trend180':180,'trend220':220}
CODE=[Path(__file__),*[ROOT/p for p in (
    'skills/index_exposure_inputs.py','skills/index_exposure_replay.py','skills/index_exposure_audit.py',
    'skills/strategy_validation_gate.py','skills/trial_registry.py','scripts/research_volatility_budget.py',
    'docs/prereg_index_exposure_20260927.md','docs/index_exposure_execution_contract_20260927.md')]]

def screen(cases):return assess_family(cases,central='trend200',neighbors=('trend180','trend220'))

def run(output):
    output=Path(output).resolve();tick=time.monotonic()
    if output.exists() or not output.is_relative_to(BASE) or output==BASE:
        raise ValueError('Choose a new immutable directory under the index study cache')
    with offline_only():
        data=load();refs=dict(data['sources']);refs.update(file_identities(CODE,ROOT))
        write(output/'identity.json',refs);write(output/'quality.json',data['quality'])
        for key,bench in data['benchmarks'].items():
            if summarize(bench['account'])!=bench['summary']:raise ValueError('Benchmark summary cannot be reproduced')
            validate_completed_account(bench['account'],data['days'],data['days'][0],data['days'][-1])
        rows={};trials=[]
        for arm,window in ARMS.items():
            for mask in range(8):
                name=f'{arm}_{mask}';status='failed';error=None;engine=None
                print('running',name,flush=True)
                try:
                    engine=IndexExposureReplay(data,window,mask)
                    account=engine.run()
                    value=dict(completed=False,config=dict(window=window,factor_mask=mask,benchmark=False,board_only=True,
                        position_count=1,target_weight=.75),account=account,decisions=engine.decisions,
                        plans=engine.plans,pending=engine.pending,live_qualified=False,unseen_validation=False)
                    write(output/'cases'/(name+'.json'),value)
                    validate_completed_account(account,data['days'],data['days'][0],data['days'][-1])
                    value.update(audit=audit_index(value,data),summary=summarize(account),completed=True)
                    benchmark=data['benchmarks']['combined' if mask&1 else 'control']
                    for key in ('initial_cash','commission','minimum_fee','participation','odd_participation','slippage'):
                        if account['settings'][key]!=benchmark['account']['settings'][key]:
                            raise ValueError('Benchmark cost/capital mismatch '+key)
                    # These fields only serve existing generic concentration metrics.
                    value['volatility_decisions']=[]
                    value['stock_pnl']={'00631L':dict(profit=value['summary']['profit'])}
                    value['metrics']=metrics(value,benchmark)
                    write(output/'cases'/(name+'.json'),value);status='completed'
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}'
                    write(output/'errors'/(name+'.json'),dict(case=name,error=error))
                    raise
                finally:
                    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='index_exposure_20260927',
                        command=' '.join(sys.argv),case=name,window=window,factor_mask=mask,
                        result_path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),status=status,error=error)
                    append_trial_registry(record);trials.append(record);write(output/'trials.json',trials)
                rows[name]=dict(completed=True,config=value['config'],summary=value['summary'],metrics=value['metrics'],
                    result=dict(path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),
                                sha256=sha(output/'cases'/(name+'.json'))))
                print(name,value['summary']['total_return'],flush=True)
        if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Index research source changed')
    report=dict(schema='index_exposure_research_v1',start=data['days'][0],end=data['days'][-1],initial_cash=1000000,
        position_count=1,all_completed=True,cases=rows,validation=screen(rows),data_quality=data['quality'],
        elapsed_seconds=round(time.monotonic()-tick,3),trial_count=len(trials),benchmark_cache_reuses=2,
        network_calls=0,database_writes=0,live_qualified=False,strict_data_ready=False,unseen_validation=False)
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json') if p.name!='manifest.json'}))
    return report

def verify(left,right,output):
    left,right,output=(Path(p).resolve() for p in (left,right,output))
    if left==right or output.exists():raise ValueError('Two separate runs and a new proof file required')
    reports=[];identities=[]
    expected={f'{arm}_{m}' for arm in ARMS for m in range(8)}
    with offline_only():
        data=load()
        for folder in (left,right):
            manifest=read(folder/'manifest.json')
            for name,digest in manifest['files_sha256'].items():
                if sha(folder/name)!=digest:raise ValueError('Index artifact changed '+name)
            refs=read(folder/'identity.json');identities.append(refs)
            if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Index source changed')
            report=read(folder/'report.json');reports.append(report)
            trials=read(folder/'trials.json')
            if set(report['cases'])!=expected or len(trials)!=32 or {r['case'] for r in trials}!=expected or any(r['status']!='completed' for r in trials):
                raise ValueError('Incomplete registered index research')
            for name in expected:
                row=report['cases'][name];case=read(ROOT/row['result']['path'])
                if sha(ROOT/row['result']['path'])!=row['result']['sha256']:raise ValueError('Case digest differs')
                if audit_index(case,data)!=case['audit'] or summarize(case['account'])!=case['summary']:
                    raise ValueError('Case independent audit differs')
            if report['validation']!=screen(report['cases']):raise ValueError('Validation changed')
        if identities[0]!=identities[1]:raise ValueError('Run inputs differ')
        for name in expected:
            if read(left/'cases'/(name+'.json'))!=read(right/'cases'/(name+'.json')):
                raise ValueError('Full account and decisions did not reproduce '+name)
    proof=dict(schema='index_exposure_reproduction_v1',passed=True,compared_cases=32,all_completed=True,
        newly_executed_cases=64,benchmark_cache_reuses=4,network_calls=0,independent_audit_repeated=True,
        runs=[dict(path=str((f/'manifest.json').relative_to(ROOT)),sha256=sha(f/'manifest.json')) for f in (left,right)],
        live_qualified=False,strict_data_ready=False)
    write(output,proof);output.with_suffix('.sha256').write_text(sha(output)+'\n');return proof

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path,nargs=2);args=parser.parse_args()
    with file_lock(BASE/'.run.lock',timeout=0):
        result=verify(*args.compare,args.output) if args.compare else run(args.output)
    print('complete',result['all_completed'],flush=True)
