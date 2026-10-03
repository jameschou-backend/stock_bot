#!/usr/bin/env python3
"""Seal complete raw institutional categories only for pre-signal coordinates.

Existing cache is historical evidence, not a fresh API response. Conflicting
versions are unknown; no latest-wins selection and no DB zero-filled aggregates.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from scripts.research_early_signal_losses import digest
from skills.launch_flows import normalize

BUNDLE=ROOT/'.cache/all-signals-2019-20261002/inputs'
RANKS=ROOT/'.cache/signal-rank-20261003/rank-v2/signal-ranks.parquet'
RANK_SHA='e9c3e1826d012bf5c960f53d9ab1fb7a172d39c7a92e3c24d9fc054abd36752a'


def needed_coordinates():
    if digest(RANKS)!=RANK_SHA:raise ValueError('Frozen rank source changed')
    days=pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(BUNDLE/'close-official.parquet',columns=['date']).date))
    sig=pd.read_parquet(RANKS,columns=['signal_date','stock_id'])
    needed={}
    for r in sig.itertuples(index=False):
        i=days.get_loc(pd.Timestamp(r.signal_date))
        for lag in (1,3):
            for j in range(max(0,i-lag-4),i-lag+1):needed[(str(days[j].date()),r.stock_id)]=True
    return days,needed


def combine_versions(frames):
    f=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['date','stock_id','foreign','trust'])
    f['date']=pd.to_datetime(f.date)
    if f.empty:return f.assign(conflict=pd.Series(dtype=bool))
    prior_conflicts=(f.groupby(['date','stock_id'])['conflict'].any()
                     if 'conflict' in f else None)
    f=f[['date','stock_id','foreign','trust']]
    grouped=f.groupby(['date','stock_id'],sort=True)
    # A missing category in one complete response cannot become zero or disappear.
    distinct=grouped[['foreign','trust']].nunique(dropna=False)
    out=grouped[['foreign','trust']].first()
    out['conflict']=distinct.gt(1).any(axis=1)
    if prior_conflicts is not None:out['conflict'] |= prior_conflicts.reindex(out.index,fill_value=False)
    out.loc[out.conflict,['foreign','trust']]=np.nan
    return out.reset_index()


def run(output, inventory, fetch_missing=False):
    output.mkdir(parents=True,exist_ok=True)
    final=output/'manifest.json'
    if final.exists():raise ValueError('Input manifest already sealed; use a new destination')
    days,needed=needed_coordinates();sources={};frames=[]
    cache_stage=output/'cached-normalized.parquet';source_stage=output/'cache-sources.json'
    if cache_stage.exists():
        meta=json.loads(source_stage.read_text())
        if digest(cache_stage)!=meta['sha256'] or meta['rank_sha256']!=RANK_SHA:raise ValueError('Preparation checkpoint changed')
        frames=[pd.read_parquet(cache_stage)];sources=meta['sources']
    else:
        for k,entry in enumerate(json.loads(inventory.read_text()),1):
            p=Path(entry['path'])
            with gzip.open(p,'rt') as stream:document=json.load(stream)
            rows=[r for r in document['data'] if (r['date'],r['stock_id']) in needed]
            if rows:
                f=normalize(pd.DataFrame(rows))
                frames.append(f[['date','stock_id','foreign','trust']])
                sources[p.name]=dict(sha256=digest(p),retrieved_at=document['retrieved_at'],rows_used=len(rows))
            if k%50==0:print('cache sources',k,flush=True)
        cached=combine_versions(frames);cached.to_parquet(cache_stage,index=False);frames=[cached]
        source_stage.write_text(json.dumps(dict(sha256=digest(cache_stage),rank_sha256=RANK_SHA,sources=sources),indent=2)+'\n')
    cached=combine_versions(frames)
    # Missing coordinates only. Conflicting existing versions stay unknown and
    # are not "fixed" by requesting a more favorable replacement snapshot.
    existing={(str(r.date.date()),r.stock_id) for r in cached.itertuples(index=False)}
    missing=[key for key in needed if key not in existing]
    jobs={}
    for day,sid in missing:jobs.setdefault(sid,[]).append(day)
    jobs=[(sid,min(dates),max(dates)) for sid,dates in sorted(jobs.items())]
    (output/'fetch-plan.json').write_text(json.dumps(dict(maximum_requests=1329,workers=4,retries=0,
        selection='missing pre-signal coordinates, no outcomes used',jobs=jobs),indent=2)+'\n')
    if len(jobs)>1329:raise ValueError('Missing-source request plan exceeds one request per original signal stock')
    print('missing stock request plan',len(jobs),'missing stock-days',len(missing),flush=True)
    if jobs and not fetch_missing:raise ValueError('Prepared cache; rerun with --fetch-missing for explicit bounded source collection')
    config=load_config() if jobs else None
    rawdir=output/'raw';rawdir.mkdir(exist_ok=True)
    def fetch(job):
        sid,lo,hi=job;p=rawdir/(sid+'.parquet');meta=p.with_suffix('.json')
        if meta.exists():
            receipt=json.loads(meta.read_text())
            if receipt['range']!=[lo,hi] or receipt['sha256']!=digest(p):raise ValueError('Supplement checkpoint changed')
            return sid,receipt
        f=fetch_dataset('TaiwanStockInstitutionalInvestorsBuySell',date.fromisoformat(lo),date.fromisoformat(hi),
            token=config.finmind_token,data_id=sid,requests_per_hour=min(6000,config.finmind_requests_per_hour),
            max_retries=0,timeout=45)
        if not f.empty:
            if not f.stock_id.eq(sid).all() or not pd.to_datetime(f.date).between(lo,hi).all():raise ValueError('Provider scope mismatch')
            normalize(f) # fail explicitly for unknown category, duplicate, invalid shares
        f.to_parquet(p,index=False)
        receipt=dict(sha256=digest(p),range=[lo,hi],rows=len(f),retrieved_at=f.attrs.get('retrieved_at'),
                     cache_hit=f.attrs.get('cache_hit',False))
        meta.write_text(json.dumps(receipt,indent=2)+'\n');return sid,receipt
    if jobs:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,(sid,receipt) in enumerate(pool.map(fetch,jobs),1):
                if i%25==0 or i==len(jobs):print('supplements',i,'/',len(jobs),flush=True)
    for sid,_,_ in jobs:
        p=rawdir/(sid+'.parquet');raw=pd.read_parquet(p)
        if not raw.empty:
            raw=raw[[ (d,s) in needed for d,s in zip(raw.date,raw.stock_id) ]]
            if not raw.empty:frames.append(normalize(raw)[['date','stock_id','foreign','trust']])
    combined=combine_versions(frames)
    index=pd.MultiIndex.from_tuples(sorted(needed),names=['date','stock_id'])
    full=pd.DataFrame(index=index).reset_index();full.date=pd.to_datetime(full.date)
    full=full.merge(combined,on=['date','stock_id'],how='left',validate='one_to_one')
    p=output/'flows.parquet';full.to_parquet(p,index=False)
    files={str(p.relative_to(ROOT)):digest(p) for p in output.rglob('*') if p.is_file()}
    known=np.isfinite(full.foreign)&np.isfinite(full.trust)
    report=dict(schema='sequential_raw_flows_v1',created_at=time.time(),rank_sha256=RANK_SHA,
        script_sha256=digest(Path(__file__)),cache_source_count=len(sources),supplement_requests=len(jobs),
        needed_stock_days=len(needed),known_stock_days=int(known.sum()),unknown_stock_days=int((~known).sum()),
        conflict_stock_days=int(full.conflict.eq(True).sum()),files_sha256=files,
        selection='all original signal pre-known coordinates, no outcome selection',
        first_publication_verified=False,revision_conflicts_preserved=True,db_used=False)
    final.write_text(json.dumps(report,indent=2)+'\n');(output/'manifest.sha256').write_text(digest(final)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path,default=ROOT/'.cache/sequential-research-20261003/institutional-cache-inventory.json')
    p.add_argument('--fetch-missing',action='store_true');a=p.parse_args();run(a.output.resolve(),a.inventory.resolve(),a.fetch_missing)
