#!/usr/bin/env python3
"""Replay one funded account across 2016-2026 with fixed rules and corrected ETF bounds."""
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
from skills.index_continuous_inputs import load,BASE,START,END
from skills.index_continuous_benchmark import run_benchmark,audit_benchmark
from skills.index_exposure_replay import IndexExposureReplay
from skills.index_exposure_audit import audit_index
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

ARMS={'equal':0,'trend200':200,'trend180':180,'trend220':220}
CODE=[Path(__file__),*[ROOT/p for p in ('skills/index_continuous_inputs.py','skills/index_continuous_benchmark.py',
    'skills/index_earlier_inputs.py','skills/index_exposure_inputs.py',
    'skills/index_exposure_replay.py','skills/index_exposure_audit.py','skills/trial_registry.py','skills/etf_limit_audit.py',
    'docs/prereg_index_continuous_20260927.md','docs/index_continuous_limit_contract_20260927.md')]]

def screen(cases):
    result={}
    for arm in ARMS:
        rows=[cases[f'{arm}_{m}'] for m in range(8)]
        wins=sum(r['metrics']['excess_return']>0 for r in rows)
        worst=min(r['summary']['max_drawdown'] for r in rows)
        rolling=all(rows[m]['metrics']['rolling252_win_rate']>=.6 for m in (0,7))
        years=[sum(r['excess']>0 for r in rows[m]['metrics']['annual_excess'] if 2016<=int(r['year'])<=2025) for m in (0,7)]
        result[arm]=dict(benchmark_winning_stresses=wins,worst_drawdown=worst,
            historical_stress_pass=wins==8 and worst>=-.5,endpoint_rolling252_pass=rolling,
            endpoint_positive_full_years=years,endpoint_annual_pass=all(n>=6 for n in years))
    central=result['trend200'];neighbors=all(result[a]['historical_stress_pass'] for a in ('trend180','trend220'))
    return dict(central_arm='trend200',arms=result,neighbor_stability_pass=neighbors,
        historical_robust_candidate=central['historical_stress_pass'] and central['endpoint_rolling252_pass'] and central['endpoint_annual_pass'] and neighbors,
        live_qualified=False,unseen_validation=False)

def sealed_cases(path,refs):
    if sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Sealed publication changed')
    refs.update(file_identities([path,path.with_suffix('.sha256')],ROOT));publication=read(path);out={}
    for group in ('cases','benchmarks'):
        for name,row in publication.get(group,{}).items():
            ref=row['result'];p=ROOT/ref['path']
            if sha(p)!=ref['sha256']:raise ValueError('Sealed comparison account changed')
            refs[ref['path']]=ref['sha256'];out[('benchmark_' if group=='benchmarks' else '')+name]=read(p)
    return out

def prefix_audit(account,expected,end):
    for key in ('daily','trades','orders','cash_ledger','holdings','corporate_actions'):
        prefix=[r for r in account[key] if r['date']<=end]
        if prefix!=expected[key]:raise ValueError('Continuous prefix differs: '+key)
    return dict(daily_and_all_journal_prefixes_identical=True,end=end)

def period_data(data,days):
    return dict(data,days=days)

def run(output):
    output=Path(output).resolve();tick=time.monotonic()
    if output.exists() or not output.is_relative_to(BASE) or output==BASE:
        raise ValueError('Use a new immutable continuous-period result directory')
    trials=[];cases={};benchmarks={};controls={}
    def execute(name,config,operation):
        error=None;status='failed'
        try:
            value=operation();write(output/'cases'/(name+'.json'),value);status='completed'
            return value
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}';write(output/'errors'/(name+'.json'),dict(error=error,case=name));raise
        finally:
            record=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='index_continuous_20260927',
                command=' '.join(sys.argv),case=name,config=config,status=status,error=error,
                result_path=str((output/'cases'/(name+'.json')).relative_to(ROOT)))
            append_trial_registry(record);trials.append(record);write(output/'trials.json',trials)
    def row(name,value):
        p=output/'cases'/(name+'.json')
        return dict(completed=True,config=value['config'],summary=value['summary'],metrics=value.get('metrics'),
                    result=dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    with offline_only():
        data=load();refs=dict(data['sources']);refs.update(file_identities(CODE,ROOT))
        early=sealed_cases(ROOT/'artifacts/forward_simulation/index_earlier_20260927.json',refs)
        recent=sealed_cases(ROOT/'artifacts/forward_simulation/index_exposure_20260927.json',refs)
        recent.update({'benchmark_'+k:v for k,v in data['current']['benchmarks'].items()})
        write(output/'identity.json',refs);write(output/'quality.json',data['quality']);write(output/'limit-audit.json',data['limit_audit'])
        for period,days,old in (('early',data['early']['days'],early),('recent',data['current']['days'],recent)):
            for key,stress in (('control',False),('combined',True)):
                name='reference_'+period+'_'+key
                value=execute(name,dict(benchmark=True,slippage_stress=stress,period=period),lambda:run_benchmark(period_data(data,days),stress))
                previous=old['benchmark_'+key]
                controls[name]=dict(row(name,value),account_identical=value['account']==previous['account'],
                    previous_summary=previous['summary'],return_difference=value['summary']['total_return']-previous['summary']['total_return'])
                print(name,'account_identical',controls[name]['account_identical'],'return_difference',controls[name]['return_difference'],flush=True)
        for key,stress in (('control',False),('combined',True)):
            name='benchmark_'+key
            benchmarks[key]=execute(name,dict(benchmark=True,slippage_stress=stress),lambda:run_benchmark(data,stress))
            print(name,benchmarks[key]['summary']['total_return'],flush=True)
        for arm,window in ARMS.items():
            for mask in range(8):
                name=f'{arm}_{mask}';config=dict(window=window,factor_mask=mask,benchmark=False,board_only=True,position_count=1,target_weight=.75)
                def operation():
                    engine=IndexExposureReplay(data,window,mask);account=engine.run()
                    value=dict(completed=False,account=account,config=config,decisions=engine.decisions,
                        plans=engine.plans,pending=engine.pending,live_qualified=False,unseen_validation=False)
                    write(output/'cases'/(name+'.json'),value)
                    validate_completed_account(account,data['days'],START,END)
                    value.update(audit=audit_index(value,data),summary=summarize(account),completed=True,
                        volatility_decisions=[])
                    value['prefix_audit']=prefix_audit(account,early[name]['account'],data['early']['days'][-1])
                    value['stock_pnl']={'00631L':dict(profit=value['summary']['profit'])}
                    benchmark=benchmarks['combined' if mask&1 else 'control']
                    for k in ('initial_cash','commission','minimum_fee','participation','odd_participation','slippage'):
                        if account['settings'][k]!=benchmark['account']['settings'][k]:raise ValueError('Benchmark scope mismatch')
                    value['metrics']=metrics(value,benchmark);return value
                value=execute(name,config,operation);cases[name]=row(name,value)
                print(name,value['summary']['total_return'],flush=True)
        if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Continuous source changed')
    report=dict(schema='index_continuous_research_v1',start=START,end=END,initial_cash=1000000,position_count=1,
        cases=cases,benchmarks={k:row('benchmark_'+k,v) for k,v in benchmarks.items()},reference_controls=controls,all_completed=True,
        validation=screen(cases),data_quality=data['quality'],elapsed_seconds=round(time.monotonic()-tick,3),
        trial_count=38,network_calls=0,database_writes=0,live_qualified=False,strict_data_ready=False,unseen_validation=False)
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json') if p.name!='manifest.json'}))
    return report

def verify(left,right,output):
    left,right,output=(Path(p).resolve() for p in (left,right,output))
    if left==right or output.exists():raise ValueError('Separate runs and new proof required')
    expected={f'{a}_{m}' for a in ARMS for m in range(8)}
    controls={f'reference_{period}_{mode}' for period in ('early','recent') for mode in ('control','combined')}
    all_names=expected|{'benchmark_control','benchmark_combined'}|controls
    identities=[];reports=[]
    with offline_only():
        data=load();early=sealed_cases(ROOT/'artifacts/forward_simulation/index_earlier_20260927.json',{})
        for folder in (left,right):
            for name,digest in read(folder/'manifest.json')['files_sha256'].items():
                if sha(folder/name)!=digest:raise ValueError('Continuous artifact changed')
            refs=read(folder/'identity.json');identities.append(refs)
            if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Continuous source identity changed')
            report=read(folder/'report.json');reports.append(report);trials=read(folder/'trials.json')
            if set(report['cases'])!=expected or set(report['reference_controls'])!=controls or len(trials)!=38 or {t['case'] for t in trials}!=all_names or any(t['status']!='completed' for t in trials):
                raise ValueError('Missing continuous cases or trial registrations')
            for name in all_names:
                case=read(folder/'cases'/(name+'.json'))
                if name in controls:
                    subset=period_data(data,data['early' if '_early_' in name else 'current']['days'])
                    audit=audit_benchmark(case,subset)
                elif name.startswith('benchmark_'):audit=audit_benchmark(case,data)
                else:
                    audit=audit_index(case,data)
                    if prefix_audit(case['account'],early[name]['account'],data['early']['days'][-1])!=case['prefix_audit']:
                        raise ValueError('Continuous prefix audit changed')
                if audit!=case['audit'] or summarize(case['account'])!=case['summary']:raise ValueError('Continuous audit changed')
            if screen(report['cases'])!=report['validation']:raise ValueError('Continuous validation changed')
        if identities[0]!=identities[1]:raise ValueError('Continuous inputs differ across runs')
        for name in all_names:
            if read(left/'cases'/(name+'.json'))!=read(right/'cases'/(name+'.json')):
                raise ValueError('Continuous full case did not reproduce '+name)
    proof=dict(schema='index_continuous_reproduction_v1',passed=True,compared_cases=38,all_completed=True,
        newly_executed_cases=76,network_calls=0,independent_audit_repeated=True,
        runs=[dict(path=str((f/'manifest.json').relative_to(ROOT)),sha256=sha(f/'manifest.json')) for f in (left,right)],
        live_qualified=False,strict_data_ready=False,unseen_validation=False)
    write(output,proof);output.with_suffix('.sha256').write_text(sha(output)+'\n');return proof

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path,nargs=2);args=parser.parse_args()
    with file_lock(BASE/'.run.lock',timeout=0):
        result=verify(*args.compare,args.output) if args.compare else run(args.output)
    print('complete',result['all_completed'],flush=True)
