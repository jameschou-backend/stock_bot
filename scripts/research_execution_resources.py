#!/usr/bin/env python3
"""Frozen factorial execution-resource study, with exact legacy controls."""
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
from scripts import research_priority as parent
from scripts import research_reservation_bridge as bridge
from scripts.research_exit_scenarios import read,write,sha,encoded,summarize,TrackedCorporateActions
from scripts.research_chip import ADDITIONS
from skills.execution_resources import ResourceCapacityReplay,ResourceBenchmark,audit_resources
from skills.reservation_replay import ReservedCapacityReplay,audit_reservations
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

OUTPUT=ROOT/'.cache/execution-resources-20260924'
SPEC=ROOT/'docs/prereg_execution_resources_20260924.md'
FLAGS=('opening_cash_only','lock_slots','lock_unused')


def inventory():
    result=parent.inventory()
    for path in (SPEC,Path(__file__),ROOT/'skills/execution_resources.py',ROOT/'skills/reservation_replay.py',
                 ROOT/'docs/intraday_corporate_additions_20260914.json'):
        result[str(path.relative_to(ROOT))]=sha(path)
    extra=read(ROOT/'docs/intraday_corporate_additions_20260914.json')
    for name,digest in extra['evidence_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Corporate source changed: '+name)
        result[name]=digest
    for name,digest in read(bridge.OUTPUT/'manifest.json').items():
        if sha(bridge.OUTPUT/name)!=digest:raise ValueError('Bridge source changed: '+name)
    result[str((bridge.OUTPUT/'manifest.json').relative_to(ROOT))]=sha(bridge.OUTPUT/'manifest.json')
    return result


class Budget(bridge.PreparationBudget):
    def take(self,kind):
        with file_lock(self.path.with_suffix('.lock')):
            counts=read(self.path) if self.path.exists() else dict(finmind=0,official=0)
            if counts[kind]>=(30 if kind=='finmind' else 100):
                raise ReplayDataUnavailable('Resource preparation lifetime ceiling: '+kind)
            counts[kind]+=1;write(self.path,counts)
        self.calls+=1


def configurations():
    for stress in ('control','combined'):
        for benchmark in (False,True):
            for c,s,u in itertools.product((False,True),repeat=3):
                if benchmark and s:continue
                label=('benchmark' if benchmark else 'capacity')+'_'+stress+f'_{int(c)}{int(s)}{int(u)}'
                yield label,dict(stress=stress,benchmark=benchmark,**dict(zip(FLAGS,(c,s,u))))
        yield 'reserved_'+stress,dict(stress=stress,reserved=True,benchmark=False)


def run_case(data,config,cache,budget,prepare):
    token=load_config().finmind_token if prepare else None
    feeds=ReplayMarketFeeds(cache/'execution-feeds',offline=not prepare,token=token,
        http_get=budget.http,finmind_fetch=budget.finmind)
    overrides=(read(parent.five.parent.cash.OVERRIDES)['overrides'] | read(ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'])
    corp=TrackedCorporateActions(data.events,cache/'dividends',token,offline=True,overrides=overrides)
    args=(data.quotes,data.companies,data.days,data.entries,feeds,corp)
    kw=dict(start=data.start,end=data.end,stress_mode=config['stress'])
    if not config['benchmark']:
        kw.update(exit_signals=data.features,action_dates=list(zip(data.events.stock_id,data.events.event_date)))
    if config.get('reserved'):
        engine=ReservedCapacityReplay(*args,**kw)
    else:
        cls=ResourceBenchmark if config['benchmark'] else ResourceCapacityReplay
        engine=cls(*args,**{k:config[k] for k in FLAGS},**kw)
    try:
        account=engine.run()
        plans=engine.plans if config.get('reserved') else engine.resource_plans
        audit=(audit_reservations(account,plans) if config.get('reserved') else
            audit_resources(account,plans,**{k:config[k] for k in FLAGS}))
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):raise
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)
    if config.get('reserved'):
        old=read(bridge.OUTPUT/'cases'/f"capacity_{config['stress']}_reserved.json")['account']
    elif not any(config[k] for k in FLAGS):
        label='benchmark' if config['benchmark'] else 'capacity'
        old=read(parent.SOURCE/'cases'/f"{label}_{config['stress']}.json")['account']
    else:old=None
    if old is not None and encoded(old)!=encoded(account):raise ValueError('Legacy control changed')
    for trade in account['trades']:
        if trade['stock_id']!='0050' and trade['signal_date']>=trade['date']:raise ValueError('Noncausal trade')
    return dict(completed=True,config=config,account=account,plans=plans,audit=audit,summary=summarize(account),
        legacy_identical=old is not None,live_qualified=False,unseen_validation=False)


def factorial(rows):
    contrasts=[]
    for stress in ('control','combined'):
        for index,flag in enumerate(FLAGS):
            other=[i for i in range(3) if i!=index]
            for fixed in itertools.product((0,1),repeat=2):
                bits=[0,0,0]
                for i,v in zip(other,fixed):bits[i]=v
                a='capacity_'+stress+'_'+''.join(map(str,bits))
                bits[index]=1;b='capacity_'+stress+'_'+''.join(map(str,bits))
                if rows[a]['completed'] and rows[b]['completed']:
                    contrasts.append(dict(stress=stress,factor=flag,from_case=a,to_case=b,
                        net_return_difference=rows[b]['summary']['total_return']-rows[a]['summary']['total_return']))
    return contrasts


def run(output=OUTPUT,offline_replay=False,prepare=False):
    if prepare and offline_replay:raise ValueError('Offline replay cannot prepare')
    started=time.monotonic();source=inventory();identity=output/'identity.json'
    if identity.exists() and read(identity)!=source:raise ValueError('Source/code changed; choose a new output')
    if offline_replay:
        for name,digest in read(output/'manifest.json')['files_sha256'].items():
            if sha(output/name)!=digest:raise ValueError('Evidence changed: '+name)
    else:write(identity,source)
    cache=output/'inputs'
    if not cache.exists():
        if offline_replay:raise ValueError('Offline inputs missing')
        shutil.copytree(bridge.OUTPUT/'inputs',cache)
    data,_=parent.inputs();budget=Budget(output);rows={}
    for name,config in configurations():
        path=output/'cases'/f'{name}.json'
        # Completed checkpoints are immutable; their manifest is checked before reuse.
        if path.exists() and not offline_replay and read(path)['completed']:
            saved=read(output/'checkpoint.json')
            if saved.get(name)!=sha(path):raise ValueError('Case checkpoint changed')
            result=read(path)
        else:
            print('running',name,flush=True)
            result=run_case(data,config,cache,budget,prepare)
            if offline_replay:
                if encoded(result)!=encoded(read(path)):raise ValueError('Offline mismatch: '+name)
            else:
                write(path,result)
                checkpoints=read(output/'checkpoint.json') if (output/'checkpoint.json').exists() else {}
                checkpoints[name]=sha(path);write(output/'checkpoint.json',checkpoints)
        rows[name]={k:v for k,v in result.items() if k not in ('account','plans')}
        rows[name].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
        print(name,round(result['summary']['total_return']*100,2) if result['completed'] else result['reason'],flush=True)
    for name,config in configurations():
        if config['benchmark'] or not rows[name]['completed']:continue
        bits='100' if config.get('reserved') else f"{int(config['opening_cash_only'])}0{int(config['lock_unused'])}"
        benchmark='benchmark_'+config['stress']+'_'+bits
        if rows[benchmark]['completed']:
            rows[name]['benchmark_case']=benchmark
            rows[name]['benchmark']=rows[benchmark]['summary']
            rows[name]['rolling252']=parent.rolling_comparison(read(ROOT/rows[name]['path'])['account'],read(ROOT/rows[benchmark]['path'])['account'])
    if inventory()!=source:raise ValueError('Sources changed during research')
    report=dict(cases=rows,all_completed=all(r['completed'] for r in rows.values()),
        candidate_count=len(data.entries),conditional_contrasts=factorial(rows),
        live_qualified=False,unseen_validation=False,network_calls=budget.calls,database_writes=0,
        elapsed_seconds=round(time.monotonic()-started,3),limitations=[
            'Daily execution order is not verified intraday order timing',
            'Historical odd-lot auctions and complete historical universe remain incomplete',
            'All cases reuse explored history; no optimization or live adoption',
            'Factor contrasts include compounded path and membership changes, not per-trade causality'])
    if offline_replay:
        original=read(output/'summary.json')
        if rows!=original['cases'] or report['conditional_contrasts']!=original['conditional_contrasts']:
            raise ValueError('Offline summary differs')
        write(output/'offline.json',dict(all_cases_identical=True,all_completed=report['all_completed'],
            network_calls=0,elapsed_seconds=report['elapsed_seconds'],manifest_sha256=sha(output/'manifest.json')))
    else:
        write(output/'summary.json',report)
        files=[p for folder in ('inputs','cases') for p in (output/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.parquet')]
        files.extend([identity,output/'summary.json',output/'checkpoint.json'])
        write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p) for p in files}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--offline-replay',action='store_true');p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    with file_lock(ROOT/'.cache/execution-resources.lock',timeout=0):report=run(args.output,args.offline_replay,args.prepare)
    print('complete',sum(r['completed'] for r in report['cases'].values()),'/',len(report['cases']),flush=True)
