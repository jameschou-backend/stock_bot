#!/usr/bin/env python3
"""Run the full requested calendar; report the first unresolved causal session."""
from pathlib import Path
from decimal import Decimal
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from scripts import research_five_axis as parent
from scripts.research_exit_scenarios import read,write,sha,encoded
from scripts.replay_contingent_day import load_tape,CODE as DAY_CODE
from scripts.research_intraday_limit import OUTPUT as OLD
from scripts.prepare_contingent_ticks import OUTPUT as NEW
from skills.contingent_portfolio import PortfolioReplay,cents
from skills.replay_market_feeds import ReplayDataUnavailable

OUTPUT=ROOT/'.cache/crossday-contingent-20260914-verified'
TARGET=ROOT/'artifacts/forward_simulation/crossday_contingent_delivery_20260914.json'
CODE=DAY_CODE+['skills/contingent_session.py','skills/contingent_portfolio.py',
    'scripts/research_contingent_portfolio.py','docs/prereg_crossday_contingent_20260914.md']


class FrozenProviders:
    def __init__(self,data,inputs=None):
        self.data=data;self.sources={};self.requests=[]
        self.external=read(inputs) if inputs else dict(actions=[],tapes=[])
        self.base=Path(inputs).resolve().parent if inputs else ROOT
        if inputs:self.sources[str(Path(inputs).resolve())]=sha(Path(inputs))
        self.market=dict(zip(data.companies.stock_id,data.companies.market))
        # The stock-only company catalog excludes ETFs; 0050 is the explicit TWSE benchmark.
        self.market['0050']='TWSE'
        self.close=data.quotes.pivot(index='date',columns='stock_id',values='close').reindex(data.days)
        volume=data.quotes.pivot(index='date',columns='stock_id',values='volume').reindex(data.days)
        self.adv=volume.rolling(20,min_periods=20).mean()
        self.amount=(self.close*volume).rolling(20,min_periods=20).mean()
        self.adjusted=data.features.adjusted_close
        self.old_manifest=read(OLD/'manifest.json')['files_sha256']
        self.new_sources={(r['date'],r['stock_id']):r for r in read(NEW/'summary.json')['rows'] if r['completed']}

    def view(self,day,sids):
        result={};d=pd.Timestamp(day)
        for sid in sorted(sids):
            if sid not in self.close or d not in self.close.index or sid not in self.adjusted:
                raise ReplayDataUnavailable(f'Missing source quote: {sid} {day}')
            raw,adj,adv,amount=(frame.at[d,sid] for frame in (self.close,self.adjusted,self.adv,self.amount))
            if not all(np.isfinite(float(v)) and v>0 for v in (raw,adj,adv,amount)):
                raise ReplayDataUnavailable(f'Incomplete price or rolling liquidity: {sid} {day}')
            price=Decimal(str(raw))*100
            if price!=price.to_integral_value():raise ReplayDataUnavailable('Unknown price precision')
            result[sid]=dict(date=day,raw_cents=int(price),adjusted_close=str(adj),
                adv20_shares=int(adv),amount20_cents=cents(amount*100))
        return result

    def actions(self,day,sids):
        observed=self.data.events[self.data.events.event_date.eq(pd.Timestamp(day))&self.data.events.stock_id.isin(sids)]
        provided=[a for a in self.external['actions'] if a['date']==day and a['stock_id'] in sids]
        for sid in observed.stock_id.unique():
            if not any(a['stock_id']==sid for a in provided):
                raise ReplayDataUnavailable(f'Corporate action needs announcement, delivery and conversion evidence: {sid} {day}')
        return provided

    def tapes(self,day,keys):
        result={}
        for sid,channel in sorted(keys):
            self.requests.append(dict(date=day,stock_id=sid,channel=channel))
            items=[i for i in self.external['tapes'] if i['date']==day and i['stock_id']==sid and i['channel']==channel]
            if len(items)>1:raise ValueError('Duplicate declared external tape')
            item=items[0] if items else None;base=self.base
            if item is None and channel=='board':
                for folder in (NEW,OLD):
                    path=folder/'ticks'/f'{sid}-{day}.parquet';meta=path.with_suffix('.json')
                    if not path.exists() or not meta.exists():continue
                    if folder==OLD:
                        for p in (path,meta):
                            if self.old_manifest.get(str(p.relative_to(OLD)))!=sha(p):raise ValueError('Original tick source changed')
                    else:
                        known=self.new_sources.get((day,sid))
                        if not known or known['sha256']!=sha(path) or known['metadata_sha256']!=sha(meta):
                            raise ValueError('Prepared tick source changed')
                    item=dict(format='finmind_board',path=str(path),sha256=sha(path),metadata_sha256=sha(meta),
                        stock_id=sid,date=day,market=self.market[sid]);base=ROOT;break
            if item is None:continue
            tape=load_tape(item,base)
            if tape.channel!=channel:raise ValueError('Source channel mismatch')
            result[(sid,channel)]=tape
            path=(base/item['path']).resolve();self.sources[str(path)]=sha(path)
            if item['format']=='finmind_board':self.sources[str(path.with_suffix('.json'))]=sha(path.with_suffix('.json'))
        return result


def run(output=OUTPUT,target=TARGET,inputs=None,verify=False):
    started=time.monotonic();output=Path(output).resolve();target=Path(target).resolve()
    identity=parent.identity(parent.OUTPUT)
    _,data,_=parent.load_data(parent.OUTPUT)
    identity.update({p:sha(ROOT/p) for p in CODE})
    identity['external_manifest']=sha(inputs) if inputs else None
    if (output/'identity.json').exists() and read(output/'identity.json')!=identity:
        raise ValueError('Rules or inputs changed; choose new output and publication paths')
    if not verify:write(output/'identity.json',identity)
    cases={};sources={};requests={}
    for name,benchmark in [('strategy',False),('benchmark',True)]:
        providers=FrozenProviders(data,inputs)
        engine=PortfolioReplay([str(d.date()) for d in data.days],data.entries,
            providers.view,providers.actions,providers.tapes,start=str(data.start)[:10],end=str(data.end)[:10],benchmark=benchmark)
        result=engine.run();path=output/f'{name}.json'
        if verify:
            if encoded(result)!=encoded(read(path)):raise ValueError('Cross-day replay differs: '+name)
        else:write(path,result)
        cases[name]=dict(completed=result['completed'],blocked=result['blocked'],
            completed_sessions=len(result['daily']),total_return=result['total_return'],
            path=str(path.relative_to(ROOT)),sha256=sha(path),
            first_plan=result['plans'][-1] if result['blocked'] and result['plans'] else None)
        sources.update(providers.sources);requests[name]=providers.requests
        print(name,'completed sessions',len(result['daily']),'blocked',result['blocked'],flush=True)
    report=dict(scope='crossday_contingent_research',cases=cases,source_requests=requests,
        all_completed=all(c['completed'] for c in cases.values()),live_qualified=False,
        start=str(data.start)[:10],end=str(data.end)[:10],initial_cents=100_000_000,
        code_sha256={p:sha(ROOT/p) for p in CODE},sources_sha256=sources,
        identity_path=str((output/'identity.json').relative_to(ROOT)),identity_sha256=sha(output/'identity.json'),
        network_calls=0,warning='跨日帳本已串接；資料不足的當日不提交帳務，未完成的完整策略報酬留空。')
    if verify:
        if read(target)!=report or target.with_suffix('.sha256').read_text().strip()!=sha(target):
            raise ValueError('Published cross-day evidence differs')
    else:
        if target.exists() and read(target)!=report:raise ValueError('Published report is immutable')
        write(target,report);target.with_suffix('.sha256').write_text(sha(target)+'\n')
    print('seconds',round(time.monotonic()-started,3),'network calls 0','verified',verify,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--publish',type=Path,default=TARGET);p.add_argument('--inputs',type=Path)
    p.add_argument('--verify',action='store_true');a=p.parse_args()
    run(a.output,a.publish,a.inputs,a.verify)
