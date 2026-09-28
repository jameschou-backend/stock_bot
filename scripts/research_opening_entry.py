#!/usr/bin/env python3
"""Paired next-session opening-entry study; freeze plans before opening ticks."""
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_strict_ticks as strict
from scripts.research_limit_bands import evidence
from skills.opening_entry_replay import factory,audit_opening
from skills.execution_factorial import path_comparison
from skills.exit_corporate_completion import load_exit_completion,DOCUMENT

old=strict.old
OUTPUT=ROOT/'.cache/opening-entry-20260928'
SPEC=ROOT/'docs/prereg_opening_entry_20260928.md'
CODE=[Path(__file__),ROOT/'skills/opening_entry_replay.py',ROOT/'scripts/research_limit_bands.py',ROOT/'skills/exit_corporate_completion.py',ROOT/DOCUMENT,SPEC,*strict.CODE,*old.CODE]
CASES={f'{"benchmark" if benchmark else "strategy"}_{mode}':dict(benchmark=benchmark,stress=mode=='stress')
       for mode in ('normal','stress') for benchmark in (False,True)}


def analyze(cases,controls):
    analysis={}
    for mode in ('normal','stress'):
        name='strategy_'+mode;stock,benchmark=cases[name],cases['benchmark_'+mode]
        analysis[name]=dict(completed=stock['completed'] and benchmark['completed'])
        if analysis[name]['completed']:
            analysis[name].update(evidence(stock,benchmark))
            before=controls[name]
            analysis[name]['control_summary']=before['summary']
            analysis[name]['path_vs_control']=path_comparison(before['account'],stock['account'])
    return analysis


def run(output=None,prepare=False):
    output=(strict.PREP/'opening-entry-preparation' if prepare else Path(output).resolve())
    if not prepare and (output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT):
        raise ValueError('Choose a new opening-entry output directory')
    started=time.monotonic();publication=old.load_selector()
    refs=dict(publication['source_sha256']);refs.update(old.file_identities(CODE,ROOT))
    if not prepare:old.write(output/'identity.json',refs)
    data,inputs,identity,repairs,repair_refs=strict.repaired_data(publication);refs.update(repair_refs)
    if len(data.entries)!=454:raise ValueError('Frozen signals changed')
    supplement,supplement_refs=load_exit_completion(ROOT)
    refs.update(supplement_refs)
    additions=old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]|supplement
    audit_feeds=old.ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    def audit(*args):return audit_opening(*args,feeds=audit_feeds)
    cases={}
    def replay():
        for name,config in CASES.items():
            old.record('opening_entry_'+name,output,'preparation_started' if prepare else 'started')
            ticks=strict.AdditionalTicks(prepare=prepare)
            try:
                result=strict.case(data,inputs,identity,additions,**config,ticks=ticks,engine_factory=factory,audit_ticks=audit)
            except Exception as exc:
                old.record('opening_entry_'+name,output,'error',error=str(exc));raise
            old.record('opening_entry_'+name,output,('preparation_' if prepare else '')+('completed' if result['completed'] else 'blocked'),
                reason=result.get('reason'),**({} if prepare else dict(summary=result['summary'])))
            if prepare:
                old.write(output/(name+'.json'),dict(completed=result['completed'],reason=result.get('reason'),
                    network_calls=ticks.calls,source_sha256=ticks.files,performance_report=False))
                print(name,result['completed'],result.get('reason'),'requests',ticks.calls,flush=True)
            else:
                old.write(output/'cases'/(name+'.json'),result)
                print(name,result['summary'] or result.get('reason'),flush=True)
            cases[name]=result;refs.update(result['source_sha256'])
    if prepare:replay()
    else:
        with old.offline_only():replay()
    if prepare:return dict(all_completed=all(r['completed'] for r in cases.values()))
    controls={}
    for mode in ('normal','stress'):
        name='strategy_'+mode
        control=ROOT/f'.cache/strict-ticks-20260928/final-a/cases/{name}.json'
        refs.update(old.file_identities([control],ROOT));controls[name]=old.read(control)
    analysis=analyze(cases,controls)
    if old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Study source changed')
    report=dict(schema='opening_entry_v1',start=data.start,end=data.end,candidate_count=len(data.entries),
        cases={k:dict(completed=v['completed'],summary=v['summary'],reason=v.get('reason'),config=CASES[k],
            result=dict(path=str((output/'cases'/(k+'.json')).relative_to(ROOT)),sha256=old.sha(output/'cases'/(k+'.json'))))
            for k,v in cases.items()},analysis=analysis,repairs=repairs,source_sha256=refs,
        controls={name:dict(path=f'.cache/strict-ticks-20260928/final-a/cases/{name}.json',sha256=old.sha(ROOT/f'.cache/strict-ticks-20260928/final-a/cases/{name}.json')) for name in controls},
        all_completed=all(r['completed'] for r in cases.values()),network_calls=0,database_writes=0,
        elapsed_seconds=round(time.monotonic()-started,3),live_qualified=False,unseen_validation=False,
        opening_auction_inferred=True,cancellation_latency_verified=False)
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
    value=dict(reports[0],source_sha256=refs,offline_identical=True,compared_cases=4)
    old.write(output,value);output.with_suffix('.sha256').write_text(old.sha(output)+'\n');return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--compare',nargs=2,type=Path);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare and (args.output or args.compare):parser.error('Separate preparation and offline publication')
    if not args.prepare and not args.output:parser.error('--output required')
    value=verify(*args.compare,args.output) if args.compare else run(args.output,args.prepare)
    print('all_completed',value['all_completed'],flush=True)
