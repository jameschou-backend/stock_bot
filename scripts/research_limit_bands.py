#!/usr/bin/env python3
"""Fixed 2x2 buy/sell limits with paired benchmark and historical robustness."""
from pathlib import Path
from copy import deepcopy
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts import research_strict_ticks as strict
from skills.limit_band_replay import factory,audit_bands
from skills.execution_factorial import stock_pnl,path_comparison

old=strict.old
OUTPUT=ROOT/'.cache/limit-bands-20260928'
SPEC=ROOT/'docs/prereg_limit_band_20260928.md'
CODE=[Path(__file__),ROOT/'skills/limit_band_replay.py',SPEC,*strict.CODE,*old.CODE]
CASES={f'{"benchmark" if benchmark else "strategy"}_{mask}_{mode}':dict(benchmark=benchmark,mask=mask,stress=mode=='stress')
       for mode in ('normal','stress') for benchmark in (False,True) for mask in ((0,1) if benchmark else range(4))}


def neutral_account(account):
    value=deepcopy(account);value.pop('opening_tick_plans')
    value['settings'].pop('band_mask');value['settings']['execution']='five_slot_precommitted_ticks_v1'
    for row in value['tick_plans']:row.pop('sizing_budget')
    return value


def evidence(stock,benchmark):
    a,b=stock['account'],benchmark['account']
    if [r['date'] for r in a['daily']]!=[r['date'] for r in b['daily']]:raise ValueError('Different comparison calendars')
    index=pd.to_datetime([r['date'] for r in a['daily']])
    s=pd.Series([r['nav'] for r in a['daily']],index=index)
    base=pd.Series([r['nav'] for r in b['daily']],index=index)
    annual=[];prior_s=prior_b=a['settings']['initial_cash']
    for year in sorted(set(index.year)):
        end_s=float(s.loc[s.index.year==year].iloc[-1]);end_b=float(base.loc[base.index.year==year].iloc[-1])
        annual.append(dict(year=int(year),strategy=end_s/prior_s-1,benchmark=end_b/prior_b-1,
                           excess=end_s/prior_s-end_b/prior_b))
        prior_s,prior_b=end_s,end_b
    rolling=(s/s.shift(252)-base/base.shift(252)).dropna()
    marks={r['stock_id']:dict(price=r['price']) for r in a['holdings'] if r['date']==a['daily'][-1]['date']}
    profits=stock_pnl(a,marks)
    positive=sorted((r['profit'] for r in profits.values() if r['profit']>0),reverse=True)
    positive_sum=sum(positive)
    sessions={r['date']:i for i,r in enumerate(a['daily'])}
    attempts={};fills={}
    for row in a['orders']:
        if row['side']=='sell' and row['channel']=='board' and row['requested_qty']:
            attempts.setdefault(row['event_id'],row)
    for row in a['trades']:
        if row['side']=='sell':fills.setdefault(row['event_id'],row)
    waits=[dict(event_id=eid,stock_id=row['stock_id'],first_attempt=row['date'],
                first_fill=fills[eid]['date'] if eid in fills else None,
                wait_sessions=sessions[fills[eid]['date']]-sessions[row['date']] if eid in fills else None)
           for eid,row in attempts.items()]
    return dict(excess_return=stock['summary']['total_return']-benchmark['summary']['total_return'],
        annual=annual,rolling_252=dict(count=len(rolling),win_fraction=float(rolling.gt(0).mean()),
            minimum=float(rolling.min()),median=float(rolling.median()),maximum=float(rolling.max()),overlapping=True),
        positive_profit_top1_fraction=positive[0]/positive_sum if positive else None,
        positive_profit_top3_fraction=sum(positive[:3])/positive_sum if positive else None,
        stock_profit=profits,sell_waits=waits,
        delayed_exit_events=sum(r['wait_sessions'] is not None and r['wait_sessions']>0 for r in waits),
        unfilled_exit_events=sum(r['wait_sessions'] is None for r in waits),
        max_wait_sessions=max((r['wait_sessions'] or 0 for r in waits),default=0),
        note='Observed history; overlapping windows are not independent or unseen tests.')


def analyze(cases):
    rows={}
    for mode in ('normal','stress'):
        control=cases[f'strategy_0_{mode}']
        for mask in range(4):
            name=f'strategy_{mask}_{mode}';stock=cases[name];benchmark=cases[f'benchmark_{mask&1}_{mode}']
            complete=stock['completed'] and benchmark['completed']
            rows[name]=dict(completed=complete)
            if complete:
                rows[name].update(evidence(stock,benchmark))
                if control['completed']:rows[name]['path_vs_control']=path_comparison(control['account'],stock['account'])
    return rows


def run(output=None,prepare=False):
    output=(strict.PREP/'limit-band-preparation' if prepare else Path(output).resolve())
    if not prepare and (output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT):
        raise ValueError('Choose a new limit-bands output directory')
    started=time.monotonic();publication=old.load_selector()
    refs=dict(publication['source_sha256']);refs.update(old.file_identities(CODE,ROOT))
    if not prepare:old.write(output/'identity.json',refs)
    data,inputs,identity,repairs,repair_refs=strict.repaired_data(publication);refs.update(repair_refs)
    if len(data.entries)!=454:raise ValueError('Frozen signals changed')
    additions=old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]
    cases={}

    def replay():
        for name,config in CASES.items():
            old.record('limit_band_'+name,output,'preparation_started' if prepare else 'started')
            ticks=strict.AdditionalTicks(prepare=prepare)
            try:
                result=strict.case(data,inputs,identity,additions,benchmark=config['benchmark'],stress=config['stress'],
                    ticks=ticks,engine_factory=factory(config['mask']),audit_ticks=audit_bands)
                if not prepare and config['mask']==0 and result['completed']:
                    baseline_name=('benchmark' if config['benchmark'] else 'strategy')+('_stress' if config['stress'] else '_normal')
                    reference=ROOT/f'.cache/strict-ticks-20260928/final-a/cases/{baseline_name}.json'
                    expected=old.read(reference)
                    if not expected['completed'] or neutral_account(result['account'])!=expected['account']:
                        raise ValueError('Zero band did not reproduce corrected account')
                    refs.update(old.file_identities([reference],ROOT))
            except Exception as exc:
                old.record('limit_band_'+name,output,'error',error=str(exc));raise
            status=('preparation_' if prepare else '')+('completed' if result['completed'] else 'blocked')
            old.record('limit_band_'+name,output,status,reason=result.get('reason'),
                       **({} if prepare else dict(summary=result['summary'])))
            if prepare:
                receipt=dict(completed=result['completed'],reason=result.get('reason'),network_calls=ticks.calls,
                             source_sha256=ticks.files,performance_report=False)
                old.write(output/(name+'.json'),receipt)
                print(name,result['completed'],result.get('reason'),'requests',ticks.calls,flush=True)
            else:
                old.write(output/'cases'/(name+'.json'),result)
                print(name,result['summary'] or result.get('reason'),flush=True)
            cases[name]=result;refs.update(result['source_sha256'])
    if prepare:replay()
    else:
        with old.offline_only():replay()
    if prepare:return dict(all_completed=all(r['completed'] for r in cases.values()))
    analysis=analyze(cases)
    if old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Study source changed')
    report=dict(schema='limit_bands_v1',start=data.start,end=data.end,candidate_count=len(data.entries),
        cases={k:dict(completed=v['completed'],summary=v['summary'],reason=v.get('reason'),config=CASES[k],
            result=dict(path=str((output/'cases'/(k+'.json')).relative_to(ROOT)),sha256=old.sha(output/'cases'/(k+'.json'))))
            for k,v in cases.items()},analysis=analysis,repairs=repairs,source_sha256=refs,
        all_completed=all(r['completed'] for r in cases.values()),network_calls=0,database_writes=0,
        elapsed_seconds=round(time.monotonic()-started,3),live_qualified=False,unseen_validation=False)
    old.write(output/'report.json',report);return report


def verify(left,right,output):
    left,right,output=map(lambda p:Path(p).resolve(),(left,right,output))
    if left==right or output.exists():raise ValueError('Use separate replay directories and new publication')
    reports=[old.read(p/'report.json') for p in (left,right)];refs={}
    for folder,report in zip((left,right),reports):
        if set(report['cases'])!=set(CASES):raise ValueError('Missing fixed experiment cases')
        refs.update(report['source_sha256'])
        refs[str((folder/'report.json').relative_to(ROOT))]=old.sha(folder/'report.json')
        for row in report['cases'].values():refs[row['result']['path']]=row['result']['sha256']
    if reports[0]['source_sha256']!=reports[1]['source_sha256'] or reports[0]['analysis']!=reports[1]['analysis']:
        raise ValueError('Independent evidence changed')
    for name in CASES:
        if old.read(ROOT/reports[0]['cases'][name]['result']['path'])!=old.read(ROOT/reports[1]['cases'][name]['result']['path']):
            raise ValueError('Full independent account changed: '+name)
    if old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Source changed before publication')
    value=dict(reports[0],source_sha256=refs,offline_identical=True,compared_cases=12)
    old.write(output,value);output.with_suffix('.sha256').write_text(old.sha(output)+'\n');return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--compare',nargs=2,type=Path);p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare and (args.output or args.compare):p.error('Separate preparation and offline publication')
    if not args.prepare and not args.output:p.error('--output required')
    value=verify(*args.compare,args.output) if args.compare else run(args.output,args.prepare)
    print('all_completed',value['all_completed'],flush=True)
