#!/usr/bin/env python3
"""Build fixed 2019 selector inputs from hash-verified historical sources."""
from pathlib import Path
import sys, json, re
from copy import deepcopy
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from scripts.research_adjustment_continuation import collect_events
from scripts.prepare_million_signals import official_adjusted
from scripts import research_historical_selector_replay as old
from skills.historical_selector_replay import eligibility_matrix,build_signals
from skills.historical_universe_completion import parse_twse_halts,roc_day

B=ROOT/'.cache/partial-risk-2019-20260929'; O=B/'official'
import argparse
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
OUT=Path(args.output).resolve()
if OUT.exists():raise ValueError('Preserve prior input versions')
OUT.mkdir()
refs={}
def ref(p):
 p=Path(p);refs[str(p.relative_to(ROOT))]=sha(p);return p
def official(name):
 p=O/name;m=p.with_suffix('.source.json');v=read(ref(m))
 if v['status']!=200 or sha(ref(p))!=v['sha256']:raise ValueError('Invalid official source '+name)
 return read(p)
ref(Path(__file__));ref(ROOT/'skills/historical_selector_replay.py')
identity=read(ref(old.IDENTITY));identity=deepcopy(identity)
membership=read(ref(B/'membership-discovery.json'))
identity['coverage_start']='2018-01-02'
identity['complete_historical_universe']=False
identity['extension_policy']='2018 observed official membership is a conservative eligibility bound, never an IPO date'
for ep in identity['episodes']:
 if ep['stock_id']=='3054':
  if ep['start']!='2026-09-01':raise ValueError('Unexpected 3054 identity')
  ep.update(start='2002-11-18',start_evidence='official_listing_record',listing_source='.cache/readiness-continuation-20260914/twse-newlisting.json')
ref(ROOT/'.cache/readiness-continuation-20260914/twse-newlisting.json')
from scripts.research_adjustment_continuation import parse_twse_listings
assert any(r['stock_id']=='3054' and r['start']=='2002-11-18' for r in parse_twse_listings(read(ROOT/'.cache/readiness-continuation-20260914/twse-newlisting.json')))
for y in (2018,2019,2020,2021):official(f'tpex-delisted-{y}.json')
for n in ('twse-market-20180102.bin','tpex-market-20180102.bin'):official(n)
ref(ROOT/'.cache/readiness-remediation-20260914/twse-delisted.json')
added=[]
for r in membership['ended']:
 if r['stock_id']=='9157' or r['end']<='2018-07-01':continue
 start=r['listing_date'] if r['listing_date'] and r['listing_date']<r['end'] else r['observed_by']
 if not start:raise ValueError('Unknown historical eligibility lower bound '+r['stock_id'])
 if any(e['stock_id']==r['stock_id'] and e['market']==r['market'] for e in identity['episodes']):raise ValueError('Unexpected overlapping identity')
 ep=dict(stock_id=r['stock_id'],name=r['name'],market=r['market'],start=start,end=r['end'],category='股票',
  start_evidence='official_listing_record' if r['listing_date'] else 'official_observed_membership_bound_not_IPO',
  historical_bound=r)
 identity['episodes'].append(ep);added.append(ep)
tdr={e['stock_id'] for e in identity['episodes'] if e['category']=='臺灣存託憑證(TDR)'}|{'9157'}
halts=[]
for y in (2018,2019,2020):
 name=f'twse-halt-{y}.bin';v=official(name)
 # Official 4551 interval begins and ends at 08:00 on the same date:
 # its half-open interval is empty, so it excludes no trading session.
 zero=[r for r in v['data'] if re.fullmatch('[0-9]{4}',r[1]) and r[3:5]==r[5:7]]
 if zero:
  if zero!=[[928,'4551','智伸科','108/10/01','8:00','108/10/01','8:00']]:raise ValueError('Unreviewed empty interval')
  v=deepcopy(v);v['data']=[r for r in v['data'] if r not in zero]
  for k in ('total','totalCount'):
   if k in v:v[k]=len(v['data'])
 halts+=parse_twse_halts(v,str((O/name).relative_to(ROOT)),identity['coverage_end'],y,tdr)
 opened={}
 name=f'tpex-halt-{y}.bin';v=official(name);t=v['tables'][0]
 if v['date']!=str(y) or t['date']!=str(y) or len(t['data'])!=t['totalCount']:raise ValueError('TPEx halt date/count mismatch')
 expected=['編號','有價證券代號','有價證券名稱','暫停交易日期','暫停交易時間','恢復交易日期','恢復交易時間']
 if t['fields'] != (expected[:1]+['有價證券類別']+expected[1:] if y==2020 else expected):raise ValueError('Unexpected historical TPEx schema')
 # 2018/19 raw rows include an undocumented empty category slot despite seven headings.
 for raw in t['data']:
  if len(raw)!=8 or (y<2020 and raw[1]!=''):raise ValueError('Historical TPEx row shape differs')
  sid=str(raw[2])
  if not re.fullmatch('[0-9]{4}',sid):continue
  if raw[4]!='-':
   if raw[5]!='8:00' or sid in opened:raise ValueError('Unknown/intraday halt')
   opened[sid]=(roc_day(raw[4]),raw)
  if raw[6]!='-':
   if raw[7]!='8:00' or sid not in opened:raise ValueError('Unmatched resume')
   start,start_raw=opened.pop(sid);end=roc_day(raw[6])
   if start>=end:raise ValueError('Invalid suspension interval')
   halts.append(dict(stock_id=sid,market='TPEx',start=start,end=end,kind='trading_suspension',
    source_path=str((O/name).relative_to(ROOT)),start_source_row=start_raw,resume_source_row=raw))
 if opened:raise ValueError('Unresolved open historical halt')
identity['trading_exclusions']+=halts
write(OUT/'identity.json',identity)
base=ROOT/'.cache/historical-selector-replay-20260925/final-v7/combined'
companies=pd.read_parquet(ref(base/'companies.parquet'))
for e in added:
 if e['stock_id'] in set(companies.stock_id):
  companies.loc[companies.stock_id.eq(e['stock_id']),'listed_date']=pd.Timestamp(e['start'])
 else:
  companies=pd.concat([companies,pd.DataFrame([dict(stock_id=e['stock_id'],name=e['name'],market=e['market'],listed_date=pd.Timestamp(e['start']),industry=None)])],ignore_index=True)
companies.to_parquet(OUT/'companies.parquet',index=False)
pieces=[]
for p in (B/'raw-2018-2020.parquet',B/'early-ended-2021-quotes.parquet',ROOT/'.cache/million-replay-inputs/quotes.parquet',old.DIRECTORY/'quotes.parquet',old.PREFIX/'quotes.parquet'):
 f=pd.read_parquet(ref(p));f['date']=pd.to_datetime(f.date)
 if p.parent in (old.DIRECTORY,old.PREFIX):
  report=read(ref(p.parent/'manifest.json'));bad={(r['stock_id'],pd.Timestamp(r['date'])) for r in report['summary']['quarantine']}
  f=f.loc[[(s,d) not in bad for s,d in zip(f.stock_id,f.date)]]
 pieces.append(f)
benchmark=pd.read_parquet(ref(B/'TaiwanStockPrice-0050.parquet')).rename(columns={'max':'high','min':'low','Trading_Volume':'volume'})
benchmark['date']=pd.to_datetime(benchmark.date);pieces.append(benchmark[pieces[0].columns])
quotes=pd.concat(pieces,ignore_index=True)
bad={(r['stock_id'],pd.Timestamp(r['date'])) for r in read(ref(old.sealed.parent.source.five.AUDIT))['quarantine']}
quotes=quotes.loc[[(s,d) not in bad for s,d in zip(quotes.stock_id,quotes.date)]]
dup=quotes[quotes.duplicated(['stock_id','date'],False)]
if not dup.empty and (dup.groupby(['stock_id','date'])[['open','high','low','close','volume']].nunique(dropna=False)>1).any().any():raise ValueError('Overlapping price sources conflict')
quotes=quotes.drop_duplicates(['stock_id','date'])
days=pd.DatetimeIndex(sorted(set(benchmark.date)|set(pd.read_parquet(base/'raw-close.parquet',columns=['date']).date)))
ids=list(companies.stock_id)+['0050']
quotes=quotes[quotes.stock_id.isin(ids)]
mask=eligibility_matrix(identity,companies,days)
mask.rename_axis('date').reset_index().to_parquet(OUT/'eligibility.parquet',index=False)
raw=quotes.pivot(index='date',columns='stock_id',values='close').reindex(index=days,columns=ids)
volume=quotes.pivot(index='date',columns='stock_id',values='volume').reindex_like(raw)
events,meta=collect_events();refs.update(meta['source_sha256']);events.to_parquet(OUT/'events.parquet',index=False)
print('building official adjusted',len(companies),len(days),flush=True)
adjusted=official_adjusted(raw,events)
quality=pd.read_parquet(ref(base/'close-quality.parquet')).set_index('date').reindex(index=days,columns=ids)
early=pd.read_parquet(ref(ROOT/'artifacts/adj_prices/adj_prices_10y.parquet'),columns=['stock_id','trading_date','close']).rename(columns={'trading_date':'date'})
early['date']=pd.to_datetime(early.date);early=early[early.stock_id.isin(ids)]
early=early.pivot(index='date',columns='stock_id',values='close').reindex(index=days,columns=ids)
old_unmasked=pd.read_parquet(ref(old.BASE/'close-quality.parquet')).set_index('date')
quality.loc[old_unmasked.index,'3054']=old_unmasked['3054']
scales={}
for sid in ids:
 common=early[sid].gt(0)&quality[sid].gt(0)
 if common.any():
  ratio=quality.loc[common,sid]/early.loc[common,sid];scale=float(ratio.median());scales[sid]=dict(scale=scale,max_relative_error=float((ratio/scale-1).abs().max()))
  quality.loc[days<'2021-01-01',sid]=early.loc[days<'2021-01-01',sid]*scale
 else:
  if not mask.loc[days<'2021-01-01',sid].any():
   scales[sid]=dict(scale=None,max_relative_error=None,no_eligible_earlier_history=True)
   continue
  # Explicit independent historical series for retired additions, never a raw-price proxy.
  if sid not in {e['stock_id'] for e in added}:raise ValueError('No quality overlap '+sid)
  quality[sid]=early[sid];scales[sid]=dict(scale=1.,max_relative_error=None,new_ended_series=True)
frames={'raw-close':raw,'raw-volume':volume,'close-official':adjusted,'close-quality':quality}
for name,f in frames.items():
 frames[name]=f.where(mask);frames[name].rename_axis('date').reset_index().to_parquet(OUT/(name+'.parquet'),index=False)
quotes.to_parquet(OUT/'quotes-unmasked.parquet',index=False)
write(OUT/'quality-scales.json',scales)
write(OUT/'sources.json',refs)
print('building signals',flush=True)
signals=build_signals(frames,companies,mask,start='2019-01-02',signal_end='2026-09-08')
write(OUT/'signals.json',signals)
write(OUT/'manifest.json',dict(start='2019-01-02',end='2026-09-09',added_episodes=added,source_sha256=refs,
 files_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},performance_ready=False,
 complete_historical_universe=False,actual_fill_verified=False,live_qualified=False))
print('entries',len(signals['entries']),flush=True)
