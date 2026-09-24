#!/usr/bin/env python3
"""Paired broker persistence studies with matched coverage and sealed accounting."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import requests
from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts import research_priority as parent
from scripts.prepare_broker_persistence import OUTPUT as INPUT, SPEC
from scripts.research_exit_scenarios import read,write,sha,encoded,summarize,TrackedCorporateActions
from scripts.research_cash_risk import attribution
from skills.broker_persistence import BrokerPersistenceReplay
from skills.execution_stress import audit_stress
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

OUTPUT = ROOT/'.cache/broker-persistence-research-20260924'
CODE = [Path(__file__),ROOT/'skills/broker_persistence.py',ROOT/'scripts/prepare_broker_persistence.py',SPEC]


def inventory():
    refs = parent.inventory()
    meta = read(INPUT/'manifest.json')
    for name,digest in meta['files_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Broker evidence changed: '+name)
        refs[name]=digest
    for p in [*CODE,INPUT/'manifest.json',ROOT/'app/finmind.py']:
        refs[str(p.relative_to(ROOT))]=sha(p)
    return refs


class Budget:
    def __init__(self, output, enabled):
        self.path=output/'request-budget.json';self.enabled=enabled
        self.counts=read(self.path) if self.path.exists() else dict(finmind=0,official=0)
    def take(self,kind):
        if not self.enabled:raise ReplayDataUnavailable('Network forbidden in replay')
        if self.counts[kind]>=(20 if kind=='finmind' else 40):raise ReplayDataUnavailable('Execution request ceiling reached')
        self.counts[kind]+=1;write(self.path,self.counts)
    def finmind(self,*args,**kwargs):
        self.take('finmind');kwargs['max_retries']=0;kwargs['requests_per_hour']=5400
        return fetch_dataset(*args,**kwargs)
    def http(self,*args,**kwargs):
        self.take('official');return requests.get(*args,**kwargs)


def run_case(data,signals,config,cache,budget):
    token=load_config().finmind_token if budget.enabled else None
    feeds=ReplayMarketFeeds(cache/'execution-feeds',offline=not budget.enabled,token=token,
                           finmind_fetch=budget.finmind,http_get=budget.http)
    from scripts.research_chip import ADDITIONS
    overrides=read(parent.five.parent.cash.OVERRIDES)['overrides']|read(ADDITIONS)['overrides']
    corp=TrackedCorporateActions(data.events,cache/'dividends',None,offline=True,overrides=overrides)
    entries=data.entries
    known={}
    if config['window']:
        known={k:v[str(config['window'])] for k,v in signals.items() if v[str(config['window'])]['known']}
        entries=[e for e in entries if e['event_id'] in known]
        if config['mode']=='filter':entries=[e for e in entries if known[e['event_id']]['passed']]
    engine=BrokerPersistenceReplay(data.quotes,data.companies,data.days,entries,feeds,corp,
        start=data.start,end=data.end,stress_mode=config['stress'],exit_signals=data.features,
        action_dates=list(zip(data.events.stock_id,data.events.event_date)),
        broker_mode='rank' if config['mode']=='rank' else 'control',broker_signals=known)
    try:
        account=engine.run();checked=audit_stress(account)
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):raise
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)
    for trade in account['trades']:
        if trade['stock_id']!='0050' and pd.Timestamp(trade['signal_date'])>=pd.Timestamp(trade['date']):
            raise ValueError('Broker trade uses same-day or future signal')
    benchmark=read(parent.SOURCE/'cases'/f"benchmark_{config['stress']}.json")
    return dict(completed=True,config=config,candidate_count=len(entries),summary=summarize(account),
                account=account,audit=checked,attribution=attribution(account),
                rolling252=parent.rolling_comparison(account,benchmark['account']),
                benchmark=benchmark['summary'],live_qualified=False,unseen_validation=False,
                priority_decisions=engine.broker_decisions)


def run(output,prepare=False,replay=False):
    tick=time.monotonic();refs=inventory();identity=output/'identity.json'
    if identity.exists() and read(identity)!=refs:raise ValueError('Code/source drift: choose a new output')
    if replay:
        for name,digest in read(output/'manifest.json')['files_sha256'].items():
            if sha(output/name)!=digest:raise ValueError('Execution artifact changed: '+name)
    elif not identity.exists():write(identity,refs)
    data,_=parent.inputs();signals=read(INPUT/'signals.json')
    cache=output/'execution-inputs'
    if not cache.exists():
        if replay:raise ValueError('Missing execution input copy')
        shutil.copytree(parent.SOURCE/'execution-inputs',cache)
    budget=Budget(output,prepare)
    configs=[]
    for stress in ('control','combined'):
        configs.append((f'full_{stress}',dict(window=0,mode='control',stress=stress)))
        for n in (5,20):
            for mode in ('control','rank','filter'):
                configs.append((f'{n}_{mode}_{stress}',dict(window=n,mode=mode,stress=stress)))
    results={}
    for name,config in configs:
        target=output/'cases'/f'{name}.json'
        previous=read(target) if target.exists() else None
        if previous and previous['completed'] and not replay:
            result=previous
        else:
            result=run_case(data,signals,config,cache,budget)
            if prepare and not result['completed'] and result['reason'].startswith('Frozen dividend source missing:'):
                sid=result['reason'].split(': ',1)[1]
                if sid not in {e['members'][0] for e in data.entries}:raise ValueError('Unexpected dividend identity')
                from skills.replay_corporate_actions import START,END
                frame=budget.finmind('TaiwanStockDividend',START,END,data_id=sid,token=load_config().finmind_token,timeout=30)
                frame.to_parquet(cache/'dividends'/f'{sid}.parquet',index=False)
                result=run_case(data,signals,config,cache,budget)
            if replay:
                if encoded(previous)!=encoded(result):raise ValueError('Offline mismatch: '+name)
            else:write(target,result)
        if config['window']==0 and result['completed']:
            baseline=read(parent.SOURCE/'cases'/f"capacity_{config['stress']}.json")
            if encoded(result['account'])!=encoded(baseline['account']):raise ValueError('Full control mismatch')
        results[name]={k:v for k,v in result.items() if k not in ('account','audit','priority_decisions')}
        print(name,'COMPLETE' if result['completed'] else 'BLOCKED '+result['reason'],flush=True)
    if inventory()!=refs:raise ValueError('Research inputs changed while executing')
    coverage={str(n):dict(Counter('known' if r[str(n)]['known'] else r[str(n)]['reason'] for r in signals.values())) for n in (5,20)}
    report=dict(cases=results,coverage=coverage,all_completed=all(r['completed'] for r in results.values()),
                live_qualified=False,unseen_validation=False,execution_requests=budget.counts,
                elapsed_seconds=round(time.monotonic()-tick,3))
    if replay:
        write(output/'verification.json',dict(all_identical=True,all_completed=report['all_completed'],network_requests=0,
              elapsed_seconds=report['elapsed_seconds'],manifest_sha256=sha(output/'manifest.json')))
    else:
        write(output/'summary.json',report)
        files={str(p.relative_to(output)):sha(p) for folder in [output/'cases',cache] for p in folder.rglob('*')
               if p.suffix in ('.json','.parquet')}
        for p in [identity,output/'summary.json']:files[str(p.relative_to(output))]=sha(p)
        write(output/'manifest.json',dict(files_sha256=files))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--offline-replay',action='store_true');a=p.parse_args()
    if a.prepare and a.offline_replay:p.error('Choose preparation or offline replay')
    with file_lock(ROOT/'.cache/broker-persistence-research.lock',timeout=0):r=run(a.output,a.prepare,a.offline_replay)
    print('complete',sum(c['completed'] for c in r['cases'].values()),'/',len(r['cases']))
