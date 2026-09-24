#!/usr/bin/env python3
"""Predeclared 2x2 slot-policy study in two fixed cash-policy contexts."""
from collections import Counter
from pathlib import Path
import argparse
import itertools
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import load_config
from app.file_lock import file_lock
from scripts import research_execution_resources as parent
from scripts.research_exit_scenarios import read,write,sha,encoded,summarize,TrackedCorporateActions
from scripts.research_chip import ADDITIONS
from skills.slot_reuse_replay import SlotReuseReplay,audit_slots
from skills.execution_resources import ResourceBenchmark,audit_resources
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

OUTPUT=ROOT/'.cache/slot-reuse-20260924'
SPEC=ROOT/'docs/prereg_slot_reuse_20260924.md'


def inventory():
    result=parent.inventory()
    for name,digest in read(parent.OUTPUT/'manifest.json')['files_sha256'].items():
        if sha(parent.OUTPUT/name)!=digest:raise ValueError('Sealed resource evidence changed: '+name)
    for path in (SPEC,Path(__file__),ROOT/'skills/slot_reuse_replay.py',parent.OUTPUT/'manifest.json'):
        result[str(path.relative_to(ROOT))]=sha(path)
    return result


def configurations():
    for stress in ('control','combined'):
        for cash in (False,True):
            for r,f in itertools.product((False,True),repeat=2):
                name=f'capacity_{stress}_cash{int(cash)}_rf{int(r)}{int(f)}'
                yield name,dict(stress=stress,cash=cash,lock_opening_slots=r,lock_failed_slots=f,benchmark=False)
            yield f'benchmark_{stress}_cash{int(cash)}',dict(stress=stress,cash=cash,benchmark=True)


def run_case(data,config,cache,budget,prepare):
    token=load_config().finmind_token if prepare else None
    feeds=ReplayMarketFeeds(cache/'execution-feeds',offline=not prepare,token=token,
        http_get=budget.http,finmind_fetch=budget.finmind)
    overrides=(read(parent.parent.five.parent.cash.OVERRIDES)['overrides'] | read(ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'])
    corp=TrackedCorporateActions(data.events,cache/'dividends',token,offline=True,overrides=overrides)
    args=(data.quotes,data.companies,data.days,data.entries,feeds,corp)
    kw=dict(start=data.start,end=data.end,stress_mode=config['stress'],
            opening_cash_only=config['cash'],lock_unused=config['cash'])
    flags={} if config['benchmark'] else {k:config[k] for k in ('lock_opening_slots','lock_failed_slots')}
    if config['benchmark']:engine=ResourceBenchmark(*args,**kw)
    else:
        engine=SlotReuseReplay(*args,**kw,**flags,exit_signals=data.features,
            action_dates=list(zip(data.events.stock_id,data.events.event_date)))
    try:
        account=engine.run();plans=engine.resource_plans
        if config['benchmark']:
            audit=audit_resources(account,plans,opening_cash_only=config['cash'],lock_unused=config['cash'],lock_slots=False)
            decisions=[]
        else:
            decisions=engine.slot_decisions
            audit=audit_slots(account,plans,decisions,**flags,opening_cash_only=config['cash'],lock_unused=config['cash'])
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):raise
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)
    control=None
    c=int(config['cash'])
    if config['benchmark']:control=f"benchmark_{config['stress']}_{c}0{c}"
    elif flags['lock_opening_slots']==flags['lock_failed_slots']:
        control=f"capacity_{config['stress']}_{c}{int(flags['lock_opening_slots'])}{c}"
    if control and encoded(read(parent.OUTPUT/'cases'/f'{control}.json')['account'])!=encoded(account):
        raise ValueError('Parent account mismatch: '+control)
    return dict(completed=True,config=config,account=account,resource_plans=plans,slot_decisions=decisions,
        audit=audit,summary=summarize(account),parent_control=control,
        attempts=dict(total=len(decisions),executed=sum(r['attempted'] for r in decisions),
            zero_fill=sum(r['attempted'] and not r['filled_qty'] for r in decisions),
            failures=dict(Counter(r['failure'] for r in decisions if r['failure'])),
            blocker_categories=dict(Counter(k for r in decisions for k in r['blocker_categories']))),
        live_qualified=False,unseen_validation=False)


def paired_diagnosis(a,b):
    first=None
    if [d['date'] for d in a['account']['daily']]!=[d['date'] for d in b['account']['daily']]:
        raise ValueError('Paired calendars differ')
    for x,y in zip(a['account']['daily'],b['account']['daily']):
        if abs(x['nav']-y['nav'])>.005:
            first=dict(date=x['date'],from_nav=x['nav'],to_nav=y['nav']);break
    old={c['event_id'] for c in a['account']['cohorts']};new={c['event_id'] for c in b['account']['cohorts']}
    blocked=[d for d in b['slot_decisions'] if d['failure']=='resource_slots_locked' and d['event_id'] in old-new]
    return dict(net_return_difference=b['summary']['total_return']-a['summary']['total_return'],
        first_nav_difference=first,removed_cohorts=sorted(old-new),added_cohorts=sorted(new-old),
        directly_blocked_removed_entries=[{k:d[k] for k in ('date','event_id','stock_id','blocker_categories')} for d in blocked])


def contrasts(output,rows):
    results=[]
    for stress in ('control','combined'):
        for cash in (False,True):
            for index,factor in enumerate(('lock_opening_slots','lock_failed_slots')):
                for other in (0,1):
                    bits=[0,0];bits[1-index]=other
                    a=f'capacity_{stress}_cash{int(cash)}_rf'+''.join(map(str,bits))
                    bits[index]=1;b=f'capacity_{stress}_cash{int(cash)}_rf'+''.join(map(str,bits))
                    if rows[a]['completed'] and rows[b]['completed']:
                        results.append(dict(from_case=a,to_case=b,factor=factor,
                            **paired_diagnosis(read(output/'cases'/f'{a}.json'),read(output/'cases'/f'{b}.json'))))
    return results


def run(output=OUTPUT,offline_replay=False,prepare=False):
    if offline_replay and prepare:raise ValueError('Offline run cannot prepare')
    tick=time.monotonic();source=inventory();identity=output/'identity.json'
    if identity.exists() and read(identity)!=source:raise ValueError('Source/code changed; use a new output')
    if offline_replay:
        for name,digest in read(output/'manifest.json')['files_sha256'].items():
            if sha(output/name)!=digest:raise ValueError('Evidence changed: '+name)
    else:write(identity,source)
    cache=output/'inputs'
    if not cache.exists():
        if offline_replay:raise ValueError('Offline inputs missing')
        shutil.copytree(parent.OUTPUT/'inputs',cache)
    data,_=parent.parent.inputs();budget=parent.Budget(output);rows={}
    for name,config in configurations():
        path=output/'cases'/f'{name}.json'
        if path.exists() and not offline_replay and read(path)['completed']:
            if read(output/'checkpoint.json').get(name)!=sha(path):raise ValueError('Checkpoint changed')
            result=read(path)
        else:
            print('running',name,flush=True);result=run_case(data,config,cache,budget,prepare)
            if offline_replay:
                if encoded(read(path))!=encoded(result):raise ValueError('Offline case differs: '+name)
            else:
                write(path,result);p=output/'checkpoint.json';saved=read(p) if p.exists() else {}
                saved[name]=sha(path);write(p,saved)
        rows[name]={k:v for k,v in result.items() if k not in ('account','resource_plans','slot_decisions')}
        rows[name].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
        print(name,round(result['summary']['total_return']*100,2) if result['completed'] else result['reason'],flush=True)
    for name,config in configurations():
        if config['benchmark'] or not rows[name]['completed']:continue
        base=f"benchmark_{config['stress']}_cash{int(config['cash'])}"
        if rows[base]['completed']:
            rows[name]['benchmark_case']=base;rows[name]['benchmark']=rows[base]['summary']
            rows[name]['rolling252']=parent.parent.rolling_comparison(read(ROOT/rows[name]['path'])['account'],read(ROOT/rows[base]['path'])['account'])
    if inventory()!=source:raise ValueError('Inputs changed during research')
    report=dict(cases=rows,conditional_contrasts=contrasts(output,rows),candidate_count=len(data.entries),
        all_completed=all(r['completed'] for r in rows.values()),network_calls=budget.calls,database_writes=0,
        elapsed_seconds=round(time.monotonic()-tick,3),live_qualified=False,unseen_validation=False,
        limitations=['Daily-order sequence is not verified intraday sequencing',
                     'Failed attempts are not proven confirmed cancellations before subsequent orders',
                     'Conditional account differences include compounded path and membership changes',
                     'Historical universe, publication revisions and odd-lot auctions remain incomplete',
                     'Cash-only or unused-budget-only contexts not expanded in this study'])
    if offline_replay:
        original=read(output/'summary.json')
        if rows!=original['cases'] or report['conditional_contrasts']!=original['conditional_contrasts']:
            raise ValueError('Offline diagnosis differs')
        write(output/'offline.json',dict(all_cases_identical=True,all_completed=report['all_completed'],
            elapsed_seconds=report['elapsed_seconds'],network_calls=0,manifest_sha256=sha(output/'manifest.json')))
    else:
        write(output/'summary.json',report)
        files=[p for folder in ('inputs','cases') for p in (output/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.parquet')]
        files += [identity,output/'checkpoint.json',output/'summary.json']
        write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p) for p in files}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--offline-replay',action='store_true')
    args=p.parse_args()
    with file_lock(ROOT/'.cache/slot-reuse.lock',timeout=0):report=run(args.output,args.offline_replay,args.prepare)
    print('complete',sum(v['completed'] for v in report['cases'].values()),'/',len(report['cases']),flush=True)
