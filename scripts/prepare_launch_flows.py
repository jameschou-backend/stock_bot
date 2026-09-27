#!/usr/bin/env python3
"""Bounded three-stock supplement; preserve existing institutional evidence."""
from datetime import date
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read,write,sha

OUTPUT=ROOT/'.cache/launch-flow-inputs-20260927'
OLD=ROOT/'.cache/chip-inputs/raw/TaiwanStockInstitutionalInvestorsBuySell'
SPEC=ROOT/'docs/prereg_launch_flows_20260927.md'
SUPPLEMENT=('2308','3026','2303')


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with file_lock(OUTPUT/'prepare.lock',timeout=0):
        paths=sorted(OLD.glob('*.parquet'))
        if len(paths)!=280:raise ValueError('Frozen 280-stock institutional scope changed')
        sources={};requests=0
        for p in paths:
            meta=p.with_suffix('.json')
            if sha(p)!=read(meta)['sha256']:raise ValueError('Old institutional evidence changed')
            sources[str(p.relative_to(ROOT))]=sha(p);sources[str(meta.relative_to(ROOT))]=sha(meta)
        config=load_config()
        for sid in SUPPLEMENT:
            p=OUTPUT/(sid+'.parquet');meta=p.with_suffix('.json')
            if meta.exists():
                if sha(p)!=read(meta)['sha256']:raise ValueError('Supplement changed')
            else:
                frame=fetch_dataset('TaiwanStockInstitutionalInvestorsBuySell',date(2021,1,1),date(2026,9,9),
                    token=config.finmind_token,data_id=sid,requests_per_hour=config.finmind_requests_per_hour,
                    max_retries=0,timeout=45)
                if frame.empty or not frame.stock_id.eq(sid).all() or not pd.to_datetime(frame.date).between('2021-01-01','2026-09-09').all():
                    raise ValueError('Supplement identity, range or coverage invalid')
                frame.to_parquet(p,index=False)
                write(meta,dict(sha256=sha(p),rows=len(frame),retrieved_at=frame.attrs.get('retrieved_at'),
                    cache_hit=frame.attrs.get('cache_hit',False),dataset='TaiwanStockInstitutionalInvestorsBuySell',stock_id=sid))
                requests+=int(not frame.attrs.get('cache_hit',False))
            sources[str(p.relative_to(ROOT))]=sha(p);sources[str(meta.relative_to(ROOT))]=sha(meta)
            print(sid,read(meta)['rows'],'rows sealed',flush=True)
        sources[str(SPEC.relative_to(ROOT))]=sha(SPEC)
        manifest=dict(schema='launch_flows_inputs_v1',sources=sources,
            stock_files=[str(p.relative_to(ROOT)) for p in paths]+[str((OUTPUT/(s+'.parquet')).relative_to(ROOT)) for s in SUPPLEMENT],
            stocks=283,new_requests=sum(not read(OUTPUT/(s+'.json'))['cache_hit'] for s in SUPPLEMENT),
            selection='Prior strategy-selected 280 stocks plus three named stocks; not all-market',
            historical_first_publication_verified=False)
        p=OUTPUT/'manifest.json'
        if p.exists() and read(p)!=manifest:raise ValueError('Cannot replace sealed input manifest')
        if not p.exists():write(p,manifest)
        print('new requests this invocation',requests,flush=True)


if __name__=='__main__':prepare()
