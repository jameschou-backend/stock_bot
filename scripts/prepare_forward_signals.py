#!/usr/bin/env python3
"""Extend frozen original rule inputs; capture new sources without changing history.

Only creates today's signals. The next official session is an empty calendar row,
never a future price. Existing historical account/source files are read-only.
"""
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import text
from app.config import load_config
from app.db import get_session
from app.finmind import fetch_dataset
from app.file_lock import file_lock
from skills.official_adj_factors import OfficialAdjClient, FETCH_SPECS, events_to_dataframe, validate_events_in_range
from scripts.prepare_million_signals import build_signals

BASE=ROOT/'.cache/million-replay-signals'
OUTPUT=ROOT/'.cache/forward-validation/signals'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,default=str,indent=2))
    temp.replace(path)


def prepare():
    today=datetime.now(ZoneInfo('Asia/Taipei')).date()
    if datetime.now(ZoneInfo('Asia/Taipei')).hour<18: raise ValueError('Wait until 18:00 publication window')
    out=OUTPUT/str(today);out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'.lock'):
        if (out/'manifest.json').exists():
            meta=json.loads((out/'manifest.json').read_text())
            for name,h in meta['sha256'].items():
                if sha(out/name)!=h: raise ValueError('Forward source changed: '+name)
            return out/'signals.json'
        original=json.loads((BASE/'manifest.json').read_text())
        names=('close-official.parquet','close-quality.parquet','raw-close.parquet','raw-volume.parquet','companies.parquet')
        frames={}
        for name in names:
            if sha(BASE/name)!=original['files_sha256'][name]: raise ValueError('Historical source changed: '+name)
            frame=pd.read_parquet(BASE/name)
            if name!='companies.parquet':
                frame=frame.set_index('date');frame.index=pd.to_datetime(frame.index)
            frames[name]=frame
        anchor=frames['raw-close.parquet'].index[-1].date()
        if today<=anchor: raise ValueError('Today is already inside historical research, not a new forward date')
        with get_session() as session:
            calendar=pd.read_sql(text('SELECT trading_date AS date,is_open FROM trading_calendar WHERE trading_date BETWEEN :a AND :b ORDER BY trading_date'),session.get_bind(),params={'a':anchor,'b':today+timedelta(days=15)})
            raw=pd.read_sql(text('SELECT stock_id,trading_date AS date,close,volume FROM raw_prices WHERE trading_date BETWEEN :a AND :b'),session.get_bind(),params={'a':anchor+timedelta(days=1),'b':today})
        days=pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open.astype(bool),'date']))
        if pd.Timestamp(today) not in days: raise ValueError('Today is not an explicit open calendar session')
        future=days[days>pd.Timestamp(today)]
        if future.empty: raise ValueError('Next official market session is missing')
        next_day=future[0]
        new_days=days[(days>pd.Timestamp(anchor)) & (days<=pd.Timestamp(today))]
        if len(new_days)>10: raise ValueError('Forward source gap exceeds 10 sessions; explicit backfill required')
        raw.date=pd.to_datetime(raw.date)
        ids=frames['raw-close.parquet'].columns
        for day in new_days:
            # Coverage against the frozen ordinary-stock cohort, separately by market.
            companies=frames['companies.parquet']
            observed=set(raw.loc[raw.date.eq(day) & raw.close.gt(0) & raw.volume.gt(0),'stock_id'])
            for market in ('TWSE','TPEX'):
                expected=set(companies.loc[companies.market.eq(market),'stock_id'])
                if not expected or len(expected & observed)/len(expected)<.9:
                    raise ValueError('Forward market coverage incomplete: '+str(day.date())+' '+market)
            if '0050' not in observed: raise ValueError('Benchmark quote missing')
        calendar.to_parquet(out/'calendar.parquet',index=False);raw.to_parquet(out/'raw.parquet',index=False)
        for name,field in [('raw-close.parquet','close'),('raw-volume.parquet','volume')]:
            new=raw.pivot(index='date',columns='stock_id',values=field).reindex(index=new_days,columns=ids)
            frames[name]=pd.concat([frames[name],new])
        # Each new session and the bridge anchor use one shared full-market request.
        config=load_config(); adjusted=[]
        for day in [pd.Timestamp(anchor),*new_days]:
            p=out/('adj-'+str(day.date())+'.parquet')
            if p.exists(): f=pd.read_parquet(p)
            else:
                f=fetch_dataset('TaiwanStockPriceAdj',day.date(),day.date(),token=config.finmind_token,
                    requests_per_hour=config.finmind_requests_per_hour,max_retries=0)
                if f.empty or set(f.date)!={str(day.date())}: raise ValueError('Adjusted source date mismatch')
                f.to_parquet(p,index=False)
            adjusted.append(f)
        adj=pd.concat(adjusted);adj.date=pd.to_datetime(adj.date)
        adj=adj.pivot(index='date',columns='stock_id',values='close').reindex(columns=ids)
        quality=frames['close-quality.parquet']
        scale=quality.iloc[-1]/adj.loc[pd.Timestamp(anchor)].where(adj.loc[pd.Timestamp(anchor)].gt(0))
        frames['close-quality.parquet']=pd.concat([quality,adj.reindex(new_days)*scale])
        # Official adjustment events across both exchanges are retained individually.
        client=OfficialAdjClient(max_retries=0,timeout=20)
        events=[]
        for kind,parser,method in FETCH_SPECS:
            p=out/(kind+'.json')
            if p.exists(): payload=json.loads(p.read_text())
            else:
                payload=getattr(client,method)(anchor+timedelta(days=1),today)
                write(p,payload)
            found=parser(payload)
            validate_events_in_range(found,anchor+timedelta(days=1),today,kind)
            events.extend(found)
        ev=events_to_dataframe(events);ev.to_parquet(out/'events.parquet',index=False)
        official=pd.concat([frames['close-official.parquet'],frames['raw-close.parquet'].reindex(new_days)])
        for e in ev.itertuples(index=False):
            if e.stock_id in official.columns:
                official.loc[official.index<pd.Timestamp(e.event_date),e.stock_id]*=float(e.ratio)
        frames['close-official.parquet']=official
        first=str(today.replace(day=1))
        # Full past history is retained so the original monthly fit and warm-up match.
        for name in names[:-1]:
            frames[name]=frames[name].reindex(frames[name].index.union(pd.DatetimeIndex([next_day])))
            frames[name].rename_axis('date').reset_index().to_parquet(out/name,index=False)
        result=build_signals(frames['close-official.parquet'],frames['close-quality.parquet'],
            frames['raw-close.parquet'],frames['raw-volume.parquet'],frames['companies.parquet'],
            start=first,signal_end=str(today))
        result.update(prepared_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
                      next_session=str(next_day.date()),source_kind='original_rule_forward_extension')
        write(out/'signals.json',result)
        manifest=dict(prepared_at=result['prepared_at'],historical_manifest_sha256=sha(BASE/'manifest.json'),
            next_session=result['next_session'],sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.parquet')},
            code_sha256={name:sha(ROOT/name) for name in ('scripts/prepare_forward_signals.py','scripts/prepare_million_signals.py','skills/diffusion_signals.py','skills/regime_state.py','skills/official_adj_factors.py')},
            live_qualified=False,calendar_future_prices=False)
        write(out/'manifest.json',manifest)
        return out/'signals.json'

if __name__=='__main__': print(prepare())
