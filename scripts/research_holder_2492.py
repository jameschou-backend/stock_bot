#!/usr/bin/env python3
"""Single-company holder study, separate from executable portfolio returns."""
from pathlib import Path
from datetime import date
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from scripts.research_exit_scenarios import read,write,sha,encoded
from skills.holder_case import aggregate,features,diagnostics

INPUT=ROOT/'.cache/holder-case-2492'
OUTPUT=ROOT/'.cache/holder-analysis-2492'
SPEC=ROOT/'docs/prereg_execution_holder_20260911.md'
SOURCES={'TaiwanStockHoldingSharesPer':('TaiwanStockHoldingSharesPer','2492'),
    'TaiwanStockPrice':('TaiwanStockPrice','2492'),
    'TaiwanStockPriceAdj_2492':('TaiwanStockPriceAdj','2492'),
    'TaiwanStockPriceAdj_0050':('TaiwanStockPriceAdj','0050'),
    'TaiwanStockDividend_2492':('TaiwanStockDividend','2492')}
CODE=('scripts/research_holder_2492.py','skills/holder_case.py','skills/chip_research.py')


def prepare():
    INPUT.mkdir(parents=True,exist_ok=True)
    meta=read(INPUT/'manifest.json') if (INPUT/'manifest.json').exists() else {}
    for name,(dataset,sid) in SOURCES.items():
        path=INPUT/(name+'.parquet')
        if name in meta:
            if not path.exists() or sha(path)!=meta[name]['sha256']:
                raise ValueError('Raw holder evidence changed: '+name)
            continue
        frame=fetch_dataset(dataset,date(2010,1,29),date(2026,9,11),data_id=sid,
            token=load_config().finmind_token,requests_per_hour=5400,max_retries=0)
        if frame.empty or not frame.stock_id.eq(sid).all():
            raise ValueError('Missing or wrong company data: '+name)
        frame.to_parquet(path,index=False)
        meta[name]=dict(rows=len(frame),sha256=sha(path),retrieved_at=frame.attrs.get('retrieved_at'))
        write(INPUT/'manifest.json',meta)


def calculate():
    meta=read(INPUT/'manifest.json')
    for name in SOURCES:
        if sha(INPUT/(name+'.parquet'))!=meta[name]['sha256']:
            raise ValueError('Raw source changed: '+name)
    weekly=aggregate(pd.read_parquet(INPUT/'TaiwanStockHoldingSharesPer.parquet'))
    def prices(name):
        p=pd.read_parquet(INPUT/(name+'.parquet'))
        if p.date.duplicated().any():
            raise ValueError('Duplicate price date')
        return p.assign(date=pd.to_datetime(p.date)).set_index('date').close.sort_index()
    stock,benchmark=prices('TaiwanStockPriceAdj_2492'),prices('TaiwanStockPriceAdj_0050')
    raw=prices('TaiwanStockPrice')
    if raw.index.max()!=pd.Timestamp('2026-09-11') or benchmark.index.max()!=raw.index.max():
        raise ValueError('Unexpected price cutoff')
    view=weekly[weekly.date.between('2026-04-02','2026-09-11')].copy()
    if view.empty or not view.valid.all() or view.date.iloc[0]!=pd.Timestamp('2026-04-02'):
        raise ValueError('Requested holder interval incomplete')
    view['raw_close']=raw.reindex(pd.DatetimeIndex(view.date)).to_numpy()
    view['inventory_change_pct']=view.total_shares.pct_change(fill_method=None)*100
    diagnostic={}
    frames={}
    for lag in (7,14):
        f=features(weekly,stock,benchmark,lag)
        current=f[f.date.between('2026-04-02','2026-09-11')]
        historical=features(weekly,stock.loc[:'2025-12-31'],benchmark.loc[:'2025-12-31'],lag)
        historical=historical[historical.date.between('2022-01-01','2025-12-31')]
        diagnostic[str(lag)]=dict(current=diagnostics(current),historical_2022_2025=diagnostics(historical))
        frames[str(lag)]=f
    first,last=view.iloc[0],view.iloc[-1]
    fields=('date','raw_close','small_pct','large_pct','small_people','large_people','small_shares','large_shares','total_people','total_shares')
    def snap(row):
        return {k:str(row[k].date()) if k=='date' else float(row[k]) for k in fields}
    start,end=pd.Timestamp('2026-04-02'),pd.Timestamp('2026-09-11')
    summary=dict(stock_id='2492',start='2026-04-02',end='2026-09-11',first=snap(first),last=snap(last),
        latest_price=float(raw.loc[end]),raw_price_change=float(raw.loc[end]/raw.loc[start]-1),
        adjusted_price_change=float(stock.loc[end]/stock.loc[start]-1),
        benchmark_adjusted_change=float(benchmark.loc[end]/benchmark.loc[start]-1),
        observed_weeks=len(view),all_history_weeks=len(weekly),invalid_history_weeks=int((~weekly.valid).sum()),
        small_pct_change=float(last.small_pct-first.small_pct),large_pct_change=float(last.large_pct-first.large_pct),
        diagnostics=diagnostic,live_qualified=False,is_account_backtest=False,
        first_publication_verified=False,spec_sha256=sha(SPEC))
    return summary,view,frames


def run():
    tick=time.monotonic()
    summary,view,frames=calculate()
    other,other_view,other_frames=calculate()
    if encoded(summary)!=encoded(other) or not view.equals(other_view) or any(not frames[k].equals(other_frames[k]) for k in frames):
        raise ValueError('Holder diagnostic replay differs')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write(OUTPUT/'summary.json',summary)
    view.to_csv(OUTPUT/'weekly.csv',index=False)
    for lag,frame in frames.items():
        frame.to_csv(OUTPUT/('lag'+lag+'.csv'),index=False)
    paths=[SPEC,*[ROOT/n for n in CODE],INPUT/'manifest.json',*[INPUT/(n+'.parquet') for n in SOURCES],
        OUTPUT/'summary.json',OUTPUT/'weekly.csv',*[OUTPUT/('lag'+k+'.csv') for k in frames]]
    write(OUTPUT/'manifest.json',dict(files_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        offline_identical=True,live_qualified=False,elapsed_seconds=time.monotonic()-tick))
    print(json.dumps({k:v for k,v in summary.items() if k!='diagnostics'},ensure_ascii=False),flush=True)


def verify():
    meta=read(OUTPUT/'manifest.json')
    if meta.get('offline_identical') is not True:
        raise ValueError('Holder analysis is not sealed')
    for name,digest in meta['files_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Holder source changed: '+name)
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:
        verify();print('holder hashes verified')
    else:
        if args.prepare:
            prepare()
        run()
