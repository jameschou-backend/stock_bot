#!/usr/bin/env python3
"""User-requested high-low midpoint proxy study; no guaranteed-fill claim."""
from pathlib import Path
import sys, argparse, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_high_return as parent
from scripts.research_high_return import (strict,old,audit_high_return_resources,market_routes,
    OddDailyCache,load_exit_completion)
from skills.midpoint_replay import MidpointStock,MidpointBenchmark
from skills.midpoint_audit import audit_midpoint
from skills.trial_registry import append_trial_registry
from datetime import datetime

OUTPUT=ROOT/'.cache/midpoint-20260928'
SPEC=ROOT/'docs/prereg_midpoint_20260928.md'
CODE=[Path(__file__),ROOT/'skills/midpoint_replay.py',ROOT/'skills/midpoint_audit.py',SPEC,*parent.CODE]
CASES={arm:dict(benchmark=arm=='benchmark',ordering='capacity' if arm=='benchmark' else arm)
       for arm in ('original','capacity','capacity_vol','benchmark')}

def case(data,inputs,identity,additions,*,benchmark,stress=False,ticks=None,odds=None,ordering='capacity',position_count=3):
    if stress:raise ValueError('This study fixes the user-requested main scenario')
    ticks=ticks or strict.AdditionalTicks()
    feeds=old.ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    overrides=(old.read(old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        old.read(old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp=old.TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
    args=(data.quotes,data.companies,data.days,data.entries,feeds,corp)
    common=dict(start=data.start,end=data.end,ticks=ticks,participation=.005 if stress else .01,
                liquidity_identity=identity,odd_feeds=odds)
    engine=(MidpointBenchmark(*args,**common,stress_mode='slip90' if stress else 'control')
        if benchmark else MidpointStock(*args,**common,ordering=ordering,position_count=position_count,factor_mask=1 if stress else 0,
            residual_policy='release',identity_report=identity,exit_signals=data.features,
            action_dates=list(zip(data.events.stock_id,data.events.event_date))))
    try:
        account=engine.run()
        old.validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        if benchmark:
            audit=old.audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
        else:
            audit=audit_high_return_resources(account,engine.resource_plans,engine.slot_decisions,
                engine.board_decisions,engine.residual_days,data.quotes)
        markets=market_routes([*ticks.queries,*odds.queries])
        audit.update(audit_midpoint(account,ticks,odds,markets,data.quotes,data.days,corp,feeds))
        audit['unknown_liquidity_rejected']=True
        result=dict(completed=True,summary=old.summarize(account),account=account,audit=audit)
    except (old.ReplayDataUnavailable,old.UnresolvedAction) as exc:
        result=dict(completed=False,summary=None,reason=str(exc),partial_diagnostics=dict(
            plans=engine.tick_plans,orders=engine.orders,trades=engine.trades,completed_sessions=len(engine.daily)))
    return dict(result,source_sha256=ticks.files|odds.files,tick_queries=ticks.queries,odd_queries=odds.queries,network_calls=ticks.calls,
                live_qualified=False,unseen_validation=False)


def run(output,prepare=False):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT:
        raise ValueError('Use a new midpoint output directory')
    started=time.monotonic();pub=old.load_selector()
    data,inputs,identity,repairs,repair_refs=strict.repaired_data(pub)
    if len(data.entries)!=454:raise ValueError('Frozen signals changed')
    extra,extra_refs=load_exit_completion(ROOT)
    additions=old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]|extra
    refs=dict(pub['source_sha256'])|repair_refs|extra_refs|old.file_identities(CODE,ROOT)
    cases={}
    if prepare:
        from scripts.prepare_mixed_odd_authorized import AuthorizedBudget,AuthorizedOdds,CACHE
        from skills.replay_market_feeds import ReplayMarketFeeds
        budget=AuthorizedBudget(CACHE/'budget.json',maximum={'finmind':0,'official':119})
        # Never extend an index already hashed by a published older study.
        provider=ReplayMarketFeeds(OUTPUT/'prepared-feeds',offline=False,http_get=budget.official,official_min_interval=5)
    def replay():
        for name,config in CASES.items():
            odds=(AuthorizedOdds(ROOT,inputs/'execution-feeds',provider) if prepare
                  else OddDailyCache(ROOT,inputs/'execution-feeds'))
            print('start',name,flush=True)
            result=case(data,inputs,identity,additions,**config,ticks=strict.AdditionalTicks(),odds=odds)
            if result['network_calls']:raise ValueError('Midpoint does not fetch ticks')
            path=output/'cases'/(name+'.json');old.write(path,result)
            cases[name]=dict(completed=result['completed'],summary=result['summary'],reason=result.get('reason'),
                config=config,result=dict(path=str(path.relative_to(ROOT)),sha256=old.sha(path)))
            refs.update(result['source_sha256'])
            append_trial_registry(dict(timestamp=datetime.now().isoformat(timespec='seconds'),
                source='midpoint_20260928',command=' '.join(sys.argv),params=config,
                result_path=str(path.relative_to(ROOT)),completed=result['completed'],
                preparation=prepare,live_qualified=False))
            print(name,({k:result['summary'][k] for k in ('total_return','max_drawdown','final_nav')}
                if not prepare and result['completed'] else
                dict(completed=result['completed'],reason=result.get('reason'))),flush=True)
    if prepare:replay()
    else:
        with old.offline_only():replay()
    if not prepare and old.file_identities([ROOT/p for p in refs],ROOT)!=refs:
        raise ValueError('Study source changed')
    report=dict(schema='midpoint_proxy_v1',start=data.start,end=data.end,cases=cases,
        all_completed=all(r['completed'] for r in cases.values()),candidate_count=len(data.entries),
        source_sha256=refs,preparation=prepare,finmind_requests=0,
        elapsed_seconds=round(time.monotonic()-started,3),live_qualified=False,
        unseen_validation=False,actual_fill_verified=False,price_formula='(high+low)/2')
    old.write(output/'report.json',report);return report


def verify(left,right,output):
    left,right,output=map(lambda p:Path(p).resolve(),(left,right,output))
    if left==right or output.exists():raise ValueError('Use independent runs and new publication')
    reports=[old.read(p/'report.json') for p in (left,right)];refs={}
    for folder,report in zip((left,right),reports):
        if report['preparation'] or not report['all_completed'] or set(report['cases'])!=set(CASES):
            raise ValueError('Publication needs four complete offline accounts')
        refs.update(report['source_sha256']);refs[str((folder/'report.json').relative_to(ROOT))]=old.sha(folder/'report.json')
        for row in report['cases'].values():refs[row['result']['path']]=row['result']['sha256']
    if reports[0]['source_sha256']!=reports[1]['source_sha256']:raise ValueError('Inputs changed')
    for name in CASES:
        if old.read(ROOT/reports[0]['cases'][name]['result']['path'])!=old.read(ROOT/reports[1]['cases'][name]['result']['path']):
            raise ValueError('Independent full account differs: '+name)
    if old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Sources changed')
    result=dict(reports[0],source_sha256=refs,offline_identical=True,compared_cases=len(CASES))
    old.write(output,result);output.with_suffix('.sha256').write_text(old.sha(output)+'\n');return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--compare',nargs=2,type=Path)
    args=parser.parse_args()
    if args.prepare and args.compare:parser.error('Preparation cannot publish results')
    if args.compare:verify(*args.compare,args.output)
    else:run(args.output,args.prepare)
