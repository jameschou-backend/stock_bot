#!/usr/bin/env python3
"""Bounded historical tick requests, full accounts and verified offline replay."""
from datetime import date, datetime, timezone
from pathlib import Path
import argparse
import hashlib
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from app.config import load_config
from app.file_lock import file_lock
from app.finmind import fetch_dataset, FinMindError
from scripts import research_five_axis as parent
from scripts.research_exit_scenarios import read,write,sha,encoded,TrackedCorporateActions,summarize
from skills.intraday_limit_replay import (normalize_ticks,IntradayReplay,IntradayBenchmark,audit_intraday)
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

OUTPUT = ROOT/'.cache/intraday-limit-20260914-final'
CODE = ('skills/intraday_limit_replay.py','scripts/research_intraday_limit.py',
        'docs/prereg_intraday_limit_20260914.md','docs/intraday_corporate_additions_20260914.json')


class TickCache:
    def __init__(self,root,online=False,maximum=1500):
        self.root,self.online,self.maximum = Path(root),online,maximum
        self.root.mkdir(parents=True,exist_ok=True)
        self.calls,self.hits,self.memo = 0,0,{}
        self.token = load_config().finmind_token if online else None

    def fetch(self,dataset,sid,day,end=None):
        if not self.online:
            raise ReplayDataUnavailable(f'Offline source missing: {dataset} {sid} {day}')
        budget_path = self.root/'budget.json'
        with file_lock(self.root/'budget.lock'):
            budget = read(budget_path) if budget_path.exists() else {'reserved':0}
            if budget['reserved'] >= self.maximum:
                raise ReplayDataUnavailable('Intraday research request ceiling reached')
            budget['reserved'] += 1
            write(budget_path,budget)
        self.calls += 1
        return fetch_dataset(dataset,date.fromisoformat(day),
            end_date=date.fromisoformat(end) if end else None,data_id=sid,token=self.token,
            max_retries=0,timeout=30)

    def get(self,sid,day,market):
        if not isinstance(sid,str) or len(sid)!=4 or not sid.isdigit() or date.fromisoformat(day).isoformat()!=day:
            raise ValueError('Invalid tick query identity')
        key = (sid,day,market)
        if key in self.memo:
            self.hits += 1
            return self.memo[key]
        path = self.root/f'{sid}-{day}.parquet'
        meta_path = path.with_suffix('.json')
        query = dict(dataset='TaiwanStockPriceTick',data_id=sid,start_date=day)
        with file_lock(path.with_suffix('.lock')):
            if path.exists() and meta_path.exists():
                meta = read(meta_path)
                if meta['query']!=query or meta['raw_sha256']!=sha(path):
                    raise ReplayDataUnavailable('Tick cache identity/hash changed')
                frame = pd.read_parquet(path)
                self.hits += 1
            else:
                frame = self.fetch('TaiwanStockPriceTick',sid,day)
                normalize_ticks(frame,sid,day,market)
                temp = path.with_suffix('.tmp.parquet')
                frame.to_parquet(temp,index=False); temp.replace(path)
                meta = dict(query=query,raw_sha256=sha(path),rows=len(frame),
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    unit='lots_for_twse_tpex',odd_lot_verified=False)
                write(meta_path,meta)
        result = normalize_ticks(frame,sid,day,market),meta['raw_sha256']
        # Bounded memory: disk cache remains shared by every case and rerun.
        if len(self.memo)>=64:
            self.memo.pop(next(iter(self.memo)))
        self.memo[key] = result
        return result


def run(output,online=False,offline_replay=False):
    started = time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    context = parent.identity(parent.OUTPUT)
    context.update({name:sha(ROOT/name) for name in CODE})
    additions = read(ROOT/'docs/intraday_corporate_additions_20260914.json')
    for name,digest in additions['evidence_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Corporate evidence changed: '+name)
        context[name] = digest
    context['parent_execution_manifest'] = sha(parent.OUTPUT/'execution-manifest.json')
    identity_path = output/'identity.json'
    if identity_path.exists() and read(identity_path)!=context:
        raise ValueError('Research code/source changed; use a new output directory')
    write(identity_path,context)
    if offline_replay:
        for name,digest in read(output/'manifest.json')['files_sha256'].items():
            if sha(output/name)!=digest:
                raise ValueError('Frozen execution evidence changed: '+name)
    cache = output/'inputs'
    if not cache.exists():
        if offline_replay:
            raise ValueError('Offline execution input directory missing')
        shutil.copytree(parent.OUTPUT/'execution-inputs',cache)
    ticks = TickCache(output/'ticks',online)
    _,data,_ = parent.load_data(parent.OUTPUT)
    for item in additions['overrides'].values():
        if item.get('accounting_only_after_end') and pd.Timestamp(item['pay_date'])<=pd.Timestamp(data.end):
            raise ValueError('Unconfirmed post-period delivery cannot become available during study')
    from scripts.research_chip import ADDITIONS
    overrides = (read(parent.parent.cash.OVERRIDES)['overrides']|read(ADDITIONS)['overrides']
                 |additions['overrides'])
    results = {}
    for scenario,participation,friction in [('normal',.01,.0045),('stress',.005,.009)]:
        for ranking in ('original','capacity','benchmark'):
            name = ranking+'_'+scenario
            print('running',name,'requests',ticks.calls,flush=True)
            while True:
                feeds = ReplayMarketFeeds(cache/'execution-feeds',offline=True)
                corp = TrackedCorporateActions(data.events,cache/'dividends',None,offline=True,overrides=overrides)
                args = (data.quotes,data.companies,data.days,data.entries,feeds,corp)
                kwargs = dict(start=data.start,end=data.end,ticks=ticks,
                              participation=participation,friction=friction)
                engine = (IntradayBenchmark(*args,**kwargs) if ranking=='benchmark' else
                          IntradayReplay(*args,ranking=ranking,exit_signals=data.features,**kwargs))
                try:
                    account = engine.run()
                    break
                except ValueError as exc:
                    prefix = 'Frozen dividend source missing: '
                    if not str(exc).startswith(prefix):
                        raise
                    sid = str(exc)[len(prefix):]
                    df = ticks.fetch('TaiwanStockDividend',sid,'2022-01-01','2026-09-09')
                    df.to_parquet(cache/'dividends'/f'{sid}.parquet',index=False)
            checked = audit_intraday(account,ticks,engine.markets)
            summary = summarize(account)
            result = dict(completed=True,live_qualified=False,config=dict(ranking=ranking,
                scenario=scenario,participation=participation,friction=friction),
                summary=summary,audit=checked,account=account)
            path = output/'cases'/f'{name}.json'
            if offline_replay:
                if encoded(result)!=encoded(read(path)):
                    raise ValueError('Offline account differs: '+name)
            else:
                write(path,result)
            results[name] = dict(summary=summary,audit=checked,sha256=sha(path))
            print(name,'return',round(summary['total_return']*100,2),
                  'trades',summary['trade_count'],'requests',ticks.calls,flush=True)
    report = dict(completed=True,live_qualified=False,scope='board_lot_diagnostic',
        historical_odd_ticks_verified=False,cases=results,requests_this_run=ticks.calls,
        cache_hits=ticks.hits,elapsed_seconds=round(time.monotonic()-started,3))
    if offline_replay:
        if ticks.calls:
            raise ValueError('Offline replay made network calls')
        write(output/'offline-verification.json',dict(identical=True,requests=0,
            elapsed_seconds=report['elapsed_seconds'],case_sha256={n:r['sha256'] for n,r in results.items()}))
    else:
        write(output/'summary.json',report)
        files = [p for folder in ('ticks','inputs','cases') for p in (output/folder).rglob('*')
                 if p.is_file() and p.suffix in ('.json','.parquet') and p.name!='budget.json']
        write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p) for p in files}))
    print('completed',report['elapsed_seconds'],'seconds',ticks.calls,'requests',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--offline-replay',action='store_true')
    args = parser.parse_args()
    with file_lock(args.output/'run.lock',timeout=1):
        try:
            run(args.output,args.prepare,args.offline_replay)
        except (ReplayDataUnavailable,FinMindError,UnresolvedAction) as exc:
            write(args.output/'blocked.json',dict(completed=False,live_qualified=False,reason=str(exc)))
            raise SystemExit('Historical replay blocked: '+str(exc))


if __name__=='__main__':
    main()
