#!/usr/bin/env python3
"""Checkpointed candidate-only aggregate queries, under a lifetime request ceiling."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read,write,sha
from skills.broker_persistence import select_branches,validate_interval,persistence

OUTPUT = ROOT/'.cache/broker-persistence-20260924'
SPEC = ROOT/'docs/prereg_broker_persistence_20260924.md'
SIGNALS = ROOT/'.cache/five-axis-20260913/rebuild/signals.json'
RAW = ROOT/'.cache/chip-inputs/raw/broker'
CALENDAR = ROOT/'.cache/five-axis-20260913/rebuild/close-official.parquet'


def plan():
    entries = read(SIGNALS)['entries']
    days = pd.DatetimeIndex(pd.read_parquet(CALENDAR, columns=['date']).date)
    refs = {str(p.relative_to(ROOT)):sha(p) for p in (SIGNALS,CALENDAR,SPEC)}
    sealed = read(ROOT/'.cache/chip-inputs/manifest.json')['files_sha256']
    events, tasks = {}, {}
    for entry in entries:
        sid, end = entry['members'][0], entry['signal_date']
        path = RAW/f'{sid}_{end}_{end}.parquet'
        key = str(path.relative_to(ROOT))
        if sha(path) != sealed[key]:
            raise ValueError('Frozen branch source changed')
        refs[key] = sha(path)
        top, error = select_branches(pd.read_parquet(path), sid, end)
        start = str(days[days.get_loc(pd.Timestamp(end))-19].date())
        task_keys = []
        for branch in top:
            broker = str(branch['securities_trader_id'])
            name = f'{sid}_{broker}_{start}_{end}'
            tasks[name] = dict(stock_id=sid, broker=broker, start=start, end=end)
            task_keys.append(name)
        events[entry['event_id']] = dict(branches=top, error=error, tasks=task_keys, signal_date=end)
    return dict(source_sha256=refs, events=events, tasks=tasks), days


def prepare(offline=False):
    tick = time.monotonic(); blueprint, days = plan()
    identity = OUTPUT/'plan.json'
    if identity.exists() and read(identity) != blueprint:
        raise ValueError('Plan changed; choose a new research version')
    if offline and not identity.exists():
        raise ValueError('Missing preparation plan')
    if not identity.exists(): write(identity,blueprint)
    ledger = OUTPUT/'requests.json'
    counts = read(ledger) if ledger.exists() else dict(reserved=2, note='includes failed all-branch and successful single-branch probes')
    todo = []
    for name, task in blueprint['tasks'].items():
        meta = OUTPUT/'raw'/f'{name}.json'
        path = meta.with_suffix('.parquet')
        if meta.exists():
            if read(meta)['sha256'] != sha(path): raise ValueError('Broker checkpoint changed')
        else: todo.append((name,task))
    print('planned',len(blueprint['tasks']),'missing',len(todo),flush=True)
    if offline and todo: raise ValueError('Missing offline broker intervals')
    token = load_config().finmind_token if todo else None
    def task(item):
        name, t = item
        x = fetch_dataset('TaiwanStockTradingDailyReportSecIdAgg',date.fromisoformat(t['start']),
            date.fromisoformat(t['end']), data_id=t['stock_id'],securities_trader_id=t['broker'],
            token=token,requests_per_hour=5400,max_retries=0,timeout=30)
        validate_interval(x,t['stock_id'],t['broker'],t['start'],t['end'])
        p=OUTPUT/'raw'/f'{name}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
        x.to_parquet(p,index=False)
        write(p.with_suffix('.json'),dict(**t,rows=len(x),sha256=sha(p),retrieved_at=x.attrs.get('retrieved_at'),cache_hit=x.attrs.get('cache_hit')))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0,len(todo),4):
            batch=todo[i:i+4]
            if counts['reserved']+len(batch)>3000: raise ValueError('Research request ceiling reached')
            counts['reserved']+=len(batch);write(ledger,counts)
            futures=[pool.submit(task,item) for item in batch]
            errors=[]
            for f in futures:
                try: f.result()
                except Exception as exc: errors.append(exc)
            if errors: raise errors[0]
            if i%100==0: print('prepared',min(i+4,len(todo)),'/',len(todo),flush=True)
    values={}
    for event_id,event in blueprint['events'].items():
        frames={}
        for name in event['tasks']:
            t=blueprint['tasks'][name]
            frames[t['broker']]=validate_interval(pd.read_parquet(OUTPUT/'raw'/f'{name}.parquet'),t['stock_id'],t['broker'],t['start'],t['end'])
        values[event_id]={str(n):persistence(event['branches'],frames,days,event['signal_date'],n) for n in (5,20)}
    refs=dict(blueprint['source_sha256'])
    for p in (OUTPUT/'raw').glob('*'):
        if p.suffix in ('.json','.parquet'):refs[str(p.relative_to(ROOT))]=sha(p)
    for p in (Path(__file__),ROOT/'skills/broker_persistence.py',ROOT/'app/finmind.py',identity):refs[str(p.relative_to(ROOT))]=sha(p)
    if offline:
        if values != read(OUTPUT/'signals.json'):raise ValueError('Broker signals changed on offline replay')
    else:
        write(OUTPUT/'signals.json',values)
        refs[str((OUTPUT/'signals.json').relative_to(ROOT))]=sha(OUTPUT/'signals.json')
        write(OUTPUT/'manifest.json',dict(files_sha256=refs,requests=counts,elapsed_seconds=round(time.monotonic()-tick,3)))
    from collections import Counter
    print({n:dict(Counter('known' if r[str(n)]['known'] else r[str(n)]['reason'] for r in values.values())) for n in (5,20)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--offline',action='store_true');a=p.parse_args()
    with file_lock(OUTPUT/'prepare.lock',timeout=0):prepare(a.offline)
