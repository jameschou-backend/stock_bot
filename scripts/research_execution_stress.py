#!/usr/bin/env python3
"""Freeze execution sensitivities and reproduce all accounts without network."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.config import load_config
from app.file_lock import file_lock
from scripts import research_technical as parent
from scripts import research_chip as chip
from scripts.research_exit_scenarios import read,write,sha,encoded,summarize,TrackedCorporateActions
from skills.execution_stress import MODES,StressReplay,StressBenchmark,audit_stress
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

INPUT=ROOT/'.cache/execution-stress-inputs'
OUTPUT=ROOT/'.cache/execution-stress'
SPEC=ROOT/'docs/prereg_execution_holder_20260911.md'
CODE=('scripts/research_execution_stress.py','skills/execution_stress.py')


def case(data,mode,benchmark,offline):
    tick=time.monotonic()
    token=None if offline else load_config().finmind_token
    feeds=ReplayMarketFeeds(INPUT/'execution-feeds',offline=offline,token=token)
    overrides=read(parent.OVERRIDES)['overrides']|read(chip.ADDITIONS)['overrides']
    corp=TrackedCorporateActions(data.events,INPUT/'dividends',token,offline=offline,overrides=overrides)
    options=dict(start=data.start,end=data.end,stress_mode=mode)
    cls=StressBenchmark if benchmark else StressReplay
    if not benchmark:
        options['technical_signals']=data.features
    engine=cls(data.quotes,data.companies,data.days,data.entries,feeds,corp,**options)
    try:
        account=engine.run()
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        return dict(completed=False,reason=str(exc),mode=mode,benchmark=benchmark)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(completed=False,reason=str(exc),mode=mode,benchmark=benchmark)
    audit=audit_stress(account)
    used=defaultdict(int)
    for trade in account['trades']:
        if trade['channel']=='odd':
            key=(trade['date'],trade['stock_id']);used[key]+=trade['qty']
            if engine.stress_depth:
                opposing=trade['odd_ask_qty' if trade['side']=='buy' else 'odd_bid_qty']
                if opposing is None or used[key]>opposing:
                    raise ValueError('Displayed depth reused or exceeded')
            if engine.stress_quote and trade['reference_price']!=trade['odd_ask' if trade['side']=='buy' else 'odd_bid']:
                raise ValueError('Incorrect opposing execution price')
        if trade['stock_id']!='0050':
            distance=engine.positions[__import__('pandas').Timestamp(trade['date'])]-engine.positions[__import__('pandas').Timestamp(trade['signal_date'])]
            minimum=2 if mode=='combined' or mode==('entry_delay' if trade['side']=='buy' else 'exit_delay') else 1
            if distance<minimum:
                raise ValueError('Execution preceded permitted market session')
    if mode=='control' and encoded(account)!=encoded(data.parent['benchmark' if benchmark else 'strategy']):
        raise ValueError('Control differs from sealed parent')
    summary=summarize(account)
    summary['corporate_assumptions']=[r['action_id'] for r in account['corporate_actions']
        if r.get('stock_id')=='6691' and r.get('date')=='2023-07-17' and r.get('kind')=='stock_dividend']
    summary['order_failures']=dict(Counter(r['failure'] for r in account['orders'] if r.get('failure')))
    summary['odd_trades']=sum(t['channel']=='odd' for t in account['trades'])
    print(mode,'benchmark' if benchmark else 'strategy','complete',round(time.monotonic()-tick,2),flush=True)
    return dict(completed=True,mode=mode,benchmark=benchmark,account=account,summary=summary,audit=audit,
        exit_states=getattr(engine,'exit_states',{}),live_qualified=False)


def verify():
    meta=read(OUTPUT/'manifest.json')
    if meta.get('offline_identical') is not True:
        raise ValueError('Execution study not sealed')
    for name,digest in meta['files_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Execution evidence changed: '+name)
    return meta


def run(replay=False):
    tick=time.monotonic()
    if replay:
        verify()
    else:
        chip.verify()
        INPUT.mkdir(parents=True,exist_ok=True)
        for name in ('execution-feeds','dividends'):
            if not (INPUT/name).exists():
                shutil.copytree(chip.INPUT/name,INPUT/name)
        identity=dict(parent_sha=sha(chip.OUTPUT/'manifest.json'),spec_sha=sha(SPEC),
            code_sha={n:sha(ROOT/n) for n in CODE})
        old=OUTPUT/'identity.json'
        if old.exists() and read(old)!=identity:
            raise ValueError('Code/spec changed; explicitly archive previous run before restarting')
        write(old,identity)
    data=parent.load_inputs()
    results={}
    for mode in MODES:
        for benchmark in (False,True):
            name=mode+('_benchmark' if benchmark else '')
            path=OUTPUT/'cases'/(name+'.json')
            result=read(path) if path.exists() and not replay else case(data,mode,benchmark,replay)
            if replay:
                if encoded(result)!=encoded(read(path)):
                    raise ValueError('Offline differs: '+name)
            else:
                write(path,result)
            results[name]={k:v for k,v in result.items() if k in ('completed','mode','benchmark','summary','reason')}
            if not result['completed']:
                print(name,'BLOCKED',result['reason'],flush=True)
    if replay:
        print('all execution cases identical offline',flush=True)
        return
    write(OUTPUT/'summary.json',dict(cases=results,elapsed_seconds=time.monotonic()-tick,live_qualified=False,unseen_validation=False))
    for mode in MODES:
        for benchmark in (False,True):
            name=mode+('_benchmark' if benchmark else '')
            if encoded(case(data,mode,benchmark,True))!=encoded(read(OUTPUT/'cases'/(name+'.json'))):
                raise ValueError('Offline differs: '+name)
    files=dict(read(chip.OUTPUT/'manifest.json')['files_sha256'])
    for base in (INPUT,OUTPUT):
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.parquet','.csv') and p!=OUTPUT/'manifest.json':
                files[str(p.relative_to(ROOT))]=sha(p)
    for p in (SPEC,*[ROOT/n for n in CODE],chip.OUTPUT/'manifest.json'):
        files[str(p.relative_to(ROOT))]=sha(p)
    write(OUTPUT/'manifest.json',dict(files_sha256=files,offline_identical=True,live_qualified=False,
        total_seconds=time.monotonic()-tick))
    print('execution study sealed',round(time.monotonic()-tick,2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--offline-replay',action='store_true')
    args=parser.parse_args()
    with file_lock(OUTPUT/'run.lock',timeout=1):
        if args.verify:
            verify();print('execution hashes verified')
        else:
            run(args.offline_replay)
