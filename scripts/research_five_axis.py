#!/usr/bin/env python3
"""Paired full-account research for all five requested axes, with bounded feeds."""
from dataclasses import replace
from pathlib import Path
import argparse
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import requests
from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts import research_cash_risk as parent
from scripts.prepare_five_axis import OUTPUT, SPEC, AUDIT, verify_rebuild
from scripts.research_exit_scenarios import read,write,sha,encoded,summarize,TrackedCorporateActions
from skills.five_axis_replay import FiveAxisReplay,ARMS
from skills.earnings_expectations import revenue_daily,quality_daily,filter_entries,revenue_months,expected_revenue
from skills.scenario_exit_replay import ExitSignals
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.execution_stress import StressBenchmark,audit_stress
from skills.million_replay import UnresolvedAction

CODE=['scripts/prepare_five_axis.py','scripts/research_five_axis.py',
      'skills/five_axis_replay.py','skills/earnings_expectations.py']


def identity(output):
    files=parent.source_inventory()['files_sha256']
    verify_rebuild(output)
    for folder in (output/'rebuild',output/'financial'):
        meta=read(folder/'manifest.json')
        for name,digest in meta['files_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Prepared source changed: '+str(folder/name))
            files[str((folder/name).relative_to(ROOT))]=digest
        files[str((folder/'manifest.json').relative_to(ROOT))]=sha(folder/'manifest.json')
    for path in [SPEC,AUDIT,*[ROOT/name for name in CODE],
                 ROOT/'.cache/revenue-research/revenue.parquet',
                 ROOT/'.cache/revenue-research/inputs.json']:
        files[str(path.relative_to(ROOT))]=sha(path)
    return files


class Budget:
    def __init__(self,output,enabled):
        self.output,self.enabled=output,enabled
        self.path=output/'execution-budget.json'
        self.counts=read(self.path) if self.path.exists() else dict(finmind=0,official=0)
        self.financial_calls=read(output/'financial/manifest.json')['calls_reserved']
    def take(self,kind):
        if not self.enabled:raise ReplayDataUnavailable('Network forbidden in offline replay')
        limit=600 if kind=='official' else 400-self.financial_calls
        if self.counts[kind]>=limit:raise ReplayDataUnavailable('Research lifetime request ceiling: '+kind)
        self.counts[kind]+=1;write(self.path,self.counts)
    def http(self,*args,**kwargs):
        self.take('official');return requests.get(*args,**kwargs)
    def finmind(self,*args,**kwargs):
        self.take('finmind');kwargs['max_retries']=0
        return fetch_dataset(*args,**kwargs)


def load_data(output):
    original=parent.cash.load_inputs()
    frames={name:pd.read_parquet(output/'rebuild'/f'{name}.parquet').set_index('date')
            for name in ('raw-close','raw-volume','close-official')}
    entries=read(output/'rebuild/signals.json')['entries']
    pool=sorted({'0050'}|{e['members'][0] for e in entries})
    refs=read(parent.cash.INPUT/'manifest.json')['references']
    quotes=pd.read_parquet(ROOT/refs['quotes']['path'],filters=[('stock_id','in',pool)])
    quotes['date']=pd.to_datetime(quotes.date)
    for row in read(AUDIT)['quarantine']:
        quotes=quotes[~(quotes.stock_id.eq(row['stock_id'])&quotes.date.eq(pd.Timestamp(row['date'])))]
    close=frames['close-official'][pool];close.index=pd.to_datetime(close.index)
    clean=replace(original,quotes=quotes,entries=entries,features=ExitSignals(close,original.days))
    revenue=pd.read_parquet(ROOT/'.cache/revenue-research/revenue.parquet')
    financial=pd.concat([pd.read_parquet(p) for p in sorted((output/'financial').glob('*.parquet'))],ignore_index=True)
    assumptions={}
    for lag in (15,30):
        rev=revenue_daily(revenue,clean.days,pool,lag)
        quality=quality_daily(financial,clean.days,pool,120 if lag==15 else 150)
        for mode in ('revenue_covered','surprise','quality_covered','surprise_quality'):
            selected,decisions=filter_entries(entries,rev,quality,mode)
            assumptions[f'{mode}_{lag}']=(selected,decisions)
    return original,clean,assumptions


def case(data,config,cache,budget):
    token=load_config().finmind_token if budget.enabled else None
    feeds=ReplayMarketFeeds(cache/'execution-feeds',offline=not budget.enabled,token=token,
                            http_get=budget.http,finmind_fetch=budget.finmind)
    from scripts.research_chip import ADDITIONS
    overrides=read(parent.cash.OVERRIDES)['overrides']|read(ADDITIONS)['overrides']
    corporate=TrackedCorporateActions(data.events,cache/'dividends',token,offline=True,overrides=overrides)
    args=(data.quotes,data.companies,data.days,data.entries,feeds,corporate)
    kwargs=dict(start=data.start,end=data.end,stress_mode=config['stress'])
    if config.get('benchmark'):
        engine=StressBenchmark(*args,**kwargs)
    else:
        engine=FiveAxisReplay(*args,arm=config.get('arm','control'),exit_signals=data.features,
                              action_dates=list(zip(data.events.stock_id,data.events.event_date)),**kwargs)
    try:
        account=engine.run()
        checked=audit_stress(account)
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):raise
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)
    for trade in account['trades']:
        if trade['stock_id']!='0050' and pd.Timestamp(trade['signal_date'])>=pd.Timestamp(trade['date']):
            raise ValueError('Non-causal trade timestamp')
    return dict(completed=True,config=config,account=account,summary=summarize(account),
        audit=checked,policy_decisions=getattr(engine,'policy_decisions',[]),
        candidate_count=len(data.entries),live_qualified=False,
        publication_timing='assumed_not_verified' if config.get('fundamental') else 'price_only')


def run(output,prepare=False,replay=False):
    tick=time.monotonic();context=identity(output)
    identity_path=output/'execution-identity.json'
    if identity_path.exists() and read(identity_path)!=context:
        raise ValueError('Five-axis code/source changed; choose a new output')
    if replay:
        for name,value in read(output/'execution-manifest.json')['files_sha256'].items():
            if sha(output/name)!=value:raise ValueError('Execution evidence changed: '+name)
    else:write(identity_path,context)
    cache=output/'execution-inputs'
    if not cache.exists():
        if replay:raise ValueError('Missing offline execution inputs')
        shutil.copytree(ROOT/'.cache/observed-risk-20260913/inputs',cache)
    budget=Budget(output,prepare)
    original,clean,assumptions=load_data(output)
    configs=[]
    for stress in ('control','combined'):
        configs += [('original_'+stress,dict(original=True,stress=stress)),
                    ('benchmark_'+stress,dict(benchmark=True,stress=stress))]
        configs += [(arm+'_'+stress,dict(arm=arm,stress=stress)) for arm in ARMS]
        configs += [(name+'_'+stress,dict(fundamental=name,stress=stress)) for name in assumptions]
    results={}
    for name,config in configs:
        target=output/'cases'/f'{name}.json'
        previous=read(target) if target.exists() else None
        data=original if config.get('original') else clean
        if config.get('fundamental'):
            selected,decisions=assumptions[config['fundamental']]
            data=replace(clean,entries=selected)
        if previous and previous['completed'] and not replay:
            result=previous
        else:
            result=case(data,config,cache,budget)
            # Explicit bounded preparation of a missing dividend snapshot. The
            # execution engine itself stays offline for corporate action terms.
            if prepare and not result['completed'] and result['reason'].startswith('Frozen dividend source missing:'):
                sid=result['reason'].split(': ',1)[1]
                if sid not in {e['members'][0] for e in clean.entries}|{'0050'}:
                    raise ValueError('Unexpected missing dividend identity')
                from skills.replay_corporate_actions import START,END
                frame=budget.finmind('TaiwanStockDividend',START,END,data_id=sid,
                                      token=load_config().finmind_token,timeout=30)
                frame.to_parquet(cache/'dividends'/f'{sid}.parquet',index=False)
                result=case(data,config,cache,budget)
            if config.get('fundamental'):
                result['fundamental_decisions']=decisions
            if replay and (previous is None or encoded(previous)!=encoded(result)):
                raise ValueError('Offline account mismatch: '+name)
            if not replay:write(target,result)
        if config.get('original') and result['completed']:
            parent_case=read(parent.OUTPUT/'cases'/f'{config["stress"]}.json')['account']
            if encoded(parent_case)!=encoded(result['account']):raise ValueError('Original cash control mismatch')
        results[name]={k:v for k,v in result.items() if k not in ('account','audit','policy_decisions','fundamental_decisions')}
        print(name,'COMPLETE' if result['completed'] else 'BLOCKED '+result['reason'],flush=True)
    if identity(output)!=context:raise ValueError('Inputs changed during execution')
    summary=dict(cases=results,all_completed=all(r['completed'] for r in results.values()),
        elapsed_seconds=round(time.monotonic()-tick,3),execution_requests=budget.counts,
        financial_requests=budget.financial_calls,live_qualified=False,unseen_validation=False)
    if not replay:
        write(output/'summary.json',summary)
        files={str(p.relative_to(output)):sha(p) for folder in (output/'cases',cache)
               for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.parquet')}
        files['execution-identity.json']=sha(identity_path)
        files['summary.json']=sha(output/'summary.json')
        write(output/'execution-manifest.json',dict(files_sha256=files))
    else:
        if not summary['all_completed']:raise ValueError('Incomplete cases cannot qualify offline replay')
        write(output/'offline-verification.json',dict(all_identical=True,network_calls=0,
            elapsed_seconds=summary['elapsed_seconds'],manifest_sha256=sha(output/'execution-manifest.json'),
            identity_sha256=sha(identity_path)))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--offline-replay',action='store_true')
    args=p.parse_args()
    if args.prepare and args.offline_replay:p.error('Preparation and replay are separate modes')
    with file_lock(args.output/'run.lock',timeout=0):result=run(args.output,args.prepare,args.offline_replay)
    print('all_completed',result['all_completed'],'seconds',result['elapsed_seconds'])
