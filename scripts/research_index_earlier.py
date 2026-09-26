#!/usr/bin/env python3
"""Replicate fixed ETF rules on 2016-2021, including a real cash-dividend benchmark."""
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
from skills.index_earlier_inputs import load,BASE,START,END
from skills.index_earlier_benchmark import run_benchmark,audit_benchmark
from skills.index_exposure_replay import IndexExposureReplay
from skills.index_exposure_audit import audit_index
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

ARMS={'equal':0,'trend200':200,'trend180':180,'trend220':220}
CODE=[Path(__file__),*[ROOT/p for p in ('skills/index_earlier_inputs.py','skills/index_earlier_benchmark.py',
    'skills/index_exposure_replay.py','skills/index_exposure_audit.py','skills/trial_registry.py',
    'docs/prereg_index_earlier_period_20260927.md','docs/index_earlier_data_contract_20260927.md')]]

def screen(cases):
    result={}
    for arm in ARMS:
        rows=[cases[f'{arm}_{m}'] for m in range(8)]
        wins=sum(r['metrics']['excess_return']>0 for r in rows)
        worst=min(r['summary']['max_drawdown'] for r in rows)
        rolling=all(rows[m]['metrics']['rolling252_win_rate']>=.6 for m in (0,7))
        years=[sum(r['excess']>0 for r in rows[m]['metrics']['annual_excess']) for m in (0,7)]
        result[arm]=dict(benchmark_winning_stresses=wins,worst_drawdown=worst,
            historical_stress_pass=wins==8 and worst>=-.5,endpoint_rolling252_pass=rolling,
            endpoint_positive_full_years=years,endpoint_annual_pass=all(n>=4 for n in years))
    central=result['trend200'];neighbors=all(result[a]['historical_stress_pass'] for a in ('trend180','trend220'))
    return dict(central_arm='trend200',arms=result,neighbor_stability_pass=neighbors,
        historical_robust_candidate=central['historical_stress_pass'] and central['endpoint_rolling252_pass'] and central['endpoint_annual_pass'] and neighbors,
        live_qualified=False,unseen_validation=False)

def run(output):
    output=Path(output).resolve();tick=time.monotonic()
    if output.exists() or not output.is_relative_to(BASE) or output==BASE:
        raise ValueError('Use a new immutable earlier-period result directory')
    trials=[];cases={};benchmarks={}
    def execute(name,config,operation):
        error=None;status='failed'
        try:
            value=operation();write(output/'cases'/(name+'.json'),value);status='completed'
            return value
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}';write(output/'errors'/(name+'.json'),dict(error=error,case=name));raise
        finally:
            record=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='index_earlier_20260927',
                command=' '.join(sys.argv),case=name,config=config,status=status,error=error,
                result_path=str((output/'cases'/(name+'.json')).relative_to(ROOT)))
            append_trial_registry(record);trials.append(record);write(output/'trials.json',trials)
    def row(name,value):
        p=output/'cases'/(name+'.json')
        return dict(completed=True,config=value['config'],summary=value['summary'],metrics=value.get('metrics'),
                    result=dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
    with offline_only():
        data=load();refs=dict(data['sources']);refs.update(file_identities(CODE,ROOT))
        write(output/'identity.json',refs);write(output/'quality.json',data['quality'])
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
                    value['stock_pnl']={'00631L':dict(profit=value['summary']['profit'])}
                    benchmark=benchmarks['combined' if mask&1 else 'control']
                    for k in ('initial_cash','commission','minimum_fee','participation','odd_participation','slippage'):
                        if account['settings'][k]!=benchmark['account']['settings'][k]:raise ValueError('Benchmark scope mismatch')
                    value['metrics']=metrics(value,benchmark);return value
                value=execute(name,config,operation);cases[name]=row(name,value)
                print(name,value['summary']['total_return'],flush=True)
        if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Earlier source changed')
    report=dict(schema='index_earlier_research_v1',start=START,end=END,initial_cash=1000000,position_count=1,
        cases=cases,benchmarks={k:row('benchmark_'+k,v) for k,v in benchmarks.items()},all_completed=True,
        validation=screen(cases),data_quality=data['quality'],elapsed_seconds=round(time.monotonic()-tick,3),
        trial_count=34,network_calls=0,database_writes=0,live_qualified=False,strict_data_ready=False,unseen_validation=False)
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json') if p.name!='manifest.json'}))
    return report

def verify(left,right,output):
    left,right,output=(Path(p).resolve() for p in (left,right,output))
    if left==right or output.exists():raise ValueError('Separate runs and new proof required')
    expected={f'{a}_{m}' for a in ARMS for m in range(8)};all_names=expected|{'benchmark_control','benchmark_combined'}
    identities=[];reports=[]
    with offline_only():
        data=load()
        for folder in (left,right):
            for name,digest in read(folder/'manifest.json')['files_sha256'].items():
                if sha(folder/name)!=digest:raise ValueError('Earlier artifact changed')
            refs=read(folder/'identity.json');identities.append(refs)
            if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Earlier source identity changed')
            report=read(folder/'report.json');reports.append(report);trials=read(folder/'trials.json')
            if set(report['cases'])!=expected or len(trials)!=34 or {t['case'] for t in trials}!=all_names or any(t['status']!='completed' for t in trials):
                raise ValueError('Missing earlier cases or trial registrations')
            for name in all_names:
                case=read(folder/'cases'/(name+'.json'))
                audit=audit_benchmark(case,data) if name.startswith('benchmark_') else audit_index(case,data)
                if audit!=case['audit'] or summarize(case['account'])!=case['summary']:raise ValueError('Earlier audit changed')
            if screen(report['cases'])!=report['validation']:raise ValueError('Earlier validation changed')
        if identities[0]!=identities[1]:raise ValueError('Earlier inputs differ across runs')
        for name in all_names:
            if read(left/'cases'/(name+'.json'))!=read(right/'cases'/(name+'.json')):
                raise ValueError('Earlier full case did not reproduce '+name)
    proof=dict(schema='index_earlier_reproduction_v1',passed=True,compared_cases=34,all_completed=True,
        newly_executed_cases=68,network_calls=0,independent_audit_repeated=True,
        runs=[dict(path=str((f/'manifest.json').relative_to(ROOT)),sha256=sha(f/'manifest.json')) for f in (left,right)],
        live_qualified=False,strict_data_ready=False,unseen_validation=False)
    write(output,proof);output.with_suffix('.sha256').write_text(sha(output)+'\n');return proof

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path,nargs=2);args=parser.parse_args()
    with file_lock(BASE/'.run.lock',timeout=0):
        result=verify(*args.compare,args.output) if args.compare else run(args.output)
    print('complete',result['all_completed'],flush=True)
