#!/usr/bin/env python3
"""Frozen board-opening plus whole-session odd daily-estimate study."""
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_strict_ticks as strict
from scripts.research_limit_bands import evidence
from skills.mixed_odd_replay import factory
from skills.mixed_odd_audit import audit_mixed_execution,audit_mixed_resources
from skills.odd_daily_cache import OddDailyCache
from skills.strict_tick_inputs import StrictResidualReplay,StrictResidualBenchmark
from skills.execution_factorial import path_comparison
from skills.exit_corporate_completion import load_exit_completion,DOCUMENT

old=strict.old
OUTPUT=ROOT/'.cache/mixed-odd-20260928'
SPEC=ROOT/'docs/prereg_mixed_odd_20260928.md'
CODE=[Path(__file__),ROOT/'skills/mixed_odd_replay.py',ROOT/'skills/mixed_odd_audit.py',ROOT/'skills/odd_daily_cache.py',ROOT/'scripts/research_limit_bands.py',ROOT/'skills/exit_corporate_completion.py',ROOT/DOCUMENT,SPEC,*strict.CODE,*old.CODE]
CASES={f'{"benchmark" if benchmark else "strategy"}_{mode}':dict(benchmark=benchmark,stress=mode=='stress')
       for mode in ('normal','stress') for benchmark in (False,True)}


def case(data,inputs,identity,additions,*,benchmark,stress,ticks=None,odds=None):
    ticks=ticks or strict.AdditionalTicks()
    feeds=old.ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    overrides=(old.read(old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        old.read(old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp=old.TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
    args=(data.quotes,data.companies,data.days,data.entries,feeds,corp)
    common=dict(start=data.start,end=data.end,ticks=ticks,participation=.005 if stress else .01,
                liquidity_identity=identity,odd_feeds=odds)
    engine=(factory(StrictResidualBenchmark)(*args,**common,stress_mode='slip90' if stress else 'control')
        if benchmark else factory(StrictResidualReplay)(*args,**common,factor_mask=1 if stress else 0,
            residual_policy='release',identity_report=identity,exit_signals=data.features,
            action_dates=list(zip(data.events.stock_id,data.events.event_date))))
    try:
        account=engine.run()
        old.validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        if benchmark:
            audit=old.audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
        else:
            audit=audit_mixed_resources(account,engine.resource_plans,engine.slot_decisions,
                engine.board_decisions,engine.residual_days,data.quotes)
        markets={(r['stock_id'],r['date']):r['market'] for r in [*ticks.queries,*odds.queries]}
        audit.update(audit_mixed_execution(account,ticks,odds,markets,data.quotes,data.days,corp,feeds))
        audit['unknown_liquidity_rejected']=True
        result=dict(completed=True,summary=old.summarize(account),account=account,audit=audit)
    except (old.ReplayDataUnavailable,old.UnresolvedAction) as exc:
        result=dict(completed=False,summary=None,reason=str(exc),partial_diagnostics=dict(
            plans=engine.tick_plans,orders=engine.orders,trades=engine.trades,completed_sessions=len(engine.daily)))
    return dict(result,source_sha256=ticks.files|odds.files,tick_queries=ticks.queries,odd_queries=odds.queries,network_calls=ticks.calls,
                live_qualified=False,unseen_validation=False)


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
    output=(strict.PREP/'mixed-odd-preparation' if prepare else Path(output).resolve())
    if not prepare and (output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT):
        raise ValueError('Choose a new mixed-odd output directory')
    started=time.monotonic();publication=old.load_selector()
    refs=dict(publication['source_sha256']);refs.update(old.file_identities(CODE,ROOT))
    if not prepare:old.write(output/'identity.json',refs)
    data,inputs,identity,repairs,repair_refs=strict.repaired_data(publication);refs.update(repair_refs)
    if len(data.entries)!=454:raise ValueError('Frozen signals changed')
    supplement,supplement_refs=load_exit_completion(ROOT)
    refs.update(supplement_refs)
    additions=old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]|supplement
    odds=OddDailyCache(ROOT,inputs/'execution-feeds')
    cases={}
    def replay():
        for name,config in CASES.items():
            old.record('mixed_odd_'+name,output,'preparation_started' if prepare else 'started')
            ticks=strict.AdditionalTicks(prepare=prepare)
            try:
                result=case(data,inputs,identity,additions,**config,ticks=ticks,odds=odds)
            except Exception as exc:
                old.record('mixed_odd_'+name,output,'error',error=str(exc));raise
            old.record('mixed_odd_'+name,output,('preparation_' if prepare else '')+('completed' if result['completed'] else 'blocked'),
                reason=result.get('reason'),**({} if prepare else dict(summary=result['summary'])))
            if prepare:
                old.write(output/(name+'.json'),dict(completed=result['completed'],reason=result.get('reason'),
                    network_calls=ticks.calls,source_sha256=result['source_sha256'],performance_report=False))
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
        control=ROOT/f'.cache/opening-entry-20260928/final-b/cases/{name}.json'
        refs.update(old.file_identities([control],ROOT));controls[name]=old.read(control)
    analysis=analyze(cases,controls)
    if old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Study source changed')
    report=dict(schema='mixed_odd_v1',start=data.start,end=data.end,candidate_count=len(data.entries),
        cases={k:dict(completed=v['completed'],summary=v['summary'],reason=v.get('reason'),config=CASES[k],
            result=dict(path=str((output/'cases'/(k+'.json')).relative_to(ROOT)),sha256=old.sha(output/'cases'/(k+'.json'))))
            for k,v in cases.items()},analysis=analysis,repairs=repairs,source_sha256=refs,
        controls={name:dict(path=f'.cache/opening-entry-20260928/final-b/cases/{name}.json',sha256=old.sha(ROOT/f'.cache/opening-entry-20260928/final-b/cases/{name}.json')) for name in controls},
        all_completed=all(r['completed'] for r in cases.values()),network_calls=0,database_writes=0,
        elapsed_seconds=round(time.monotonic()-started,3),live_qualified=False,unseen_validation=False,
        opening_auction_inferred=True,cancellation_latency_verified=False,odd_tick_verified=False,
        odd_execution_evidence='daily_envelope_estimate')
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
