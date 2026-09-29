#!/usr/bin/env python3
"""Remove future quotes/actions and require identical early stock selections."""
from pathlib import Path
import sys,json
import pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from skills.historical_selector_replay import build_signals
from scripts.prepare_million_signals import official_adjusted
from scripts.research_exit_scenarios import read,write,sha
B=R/'.cache/partial-risk-2019-20260929';I=B/'inputs-final'
companies=pd.read_parquet(I/'companies.parquet');events=pd.read_parquet(I/'events.parquet');full=read(I/'signals.json')['entries']
frames={name:pd.read_parquet(I/(name+'.parquet')).set_index('date') for name in ('raw-close','raw-volume','close-quality')}
mask=pd.read_parquet(I/'eligibility.parquet').set_index('date');out={}
def identity(rows):return [(r['signal_date'],r['entry_date'],r['members'],r['group_members']) for r in rows]
for cutoff,last in [('2020-12-31','2021-01-04'),('2021-12-30','2022-01-03')]:
 f={k:v.loc[:last].copy() for k,v in frames.items()};e=events[pd.to_datetime(events.event_date).le(pd.Timestamp(last))]
 f['close-official']=official_adjusted(f['raw-close'],e)
 result=build_signals(f,companies,mask.loc[:last],start='2019-01-02',signal_end=cutoff)
 expected=[r for r in full if r['signal_date']<=cutoff]
 if identity(result['entries'])!=identity(expected):raise ValueError('Prefix selected stocks changed '+cutoff)
 out[cutoff]=dict(selected_signals=len(expected),same_selected_stocks=True,future_prices_and_events_removed=True)
 print(cutoff,out[cutoff],flush=True)
write(B/'signal-prefix-final.json',dict(checks=out,unseen_validation=False,source_sha256={str(Path(__file__).relative_to(R)):sha(Path(__file__)),str((I/'manifest.json').relative_to(R)):sha(I/'manifest.json')}))
