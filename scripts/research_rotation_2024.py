#!/usr/bin/env python3
"""2024 fixed group-relaxation and rotation experiment; preserve 2019–2023 account."""
from pathlib import Path
import sys,json,time,math,argparse
from datetime import date,datetime,timezone
from copy import deepcopy
from dataclasses import replace
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_midpoint as study
from scripts.research_exit_scenarios import read,write,sha,RunInputs
from app.finmind import fetch_dataset
from app.config import load_config
from skills.replay_market_feeds import parse_odd,_number,ReplayDataUnavailable,URLS
from skills.scenario_exit_replay import ExitSignals
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.partial_risk import PartialRisk
from skills.midpoint_replay import MidpointBenchmark
from scripts.export_midpoint_2025_report import verify_cash
from skills.partial_risk_audit import audit_partial_risk,audit_core_resources
from skills.midpoint_exit_audit import audit_midpoint_exit
from skills.midpoint_audit import audit_midpoint
from skills.strict_tick_inputs import restore_halt_zeroes

OLD=ROOT/'.cache/partial-risk-2019-20260929'; I=OLD/'inputs-final'
B=ROOT/'.cache/rotation-2024-20260929'; C=B/'execution-v1'
from skills.rotation_2024 import Rotation2024
from skills.trial_registry import append_trial_registry
from skills.historical_odd_regime import HistoricalOddEra, normalized_era_account, parse_after_hours
BEFORE='2020-10-26';AFTER={'twse':'https://www.twse.com.tw/rwd/zh/afterTrading/TWT53U','tpex':'https://www.tpex.org.tw/www/zh-tw/afterTrading/odd'}
parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--output',required=True);parser.add_argument('--arms',default='original,relaxed,stagnant,stronger,combined,benchmark');a=parser.parse_args()
output=B/a.output
if output.exists():raise ValueError('Choose new output')
PREP=a.prepare; C.mkdir(exist_ok=True); refs={};token=load_config().finmind_token if PREP else None
ledger=C/'budget.json';budget=read(ledger) if ledger.exists() else dict(attempts=[],maximum=400)
def reserve(key):
 if not PREP:raise ReplayDataUnavailable('Offline input missing '+key)
 if (C/'security-stop.json').exists():raise ReplayDataUnavailable('Official security stop active')
 if len(budget['attempts'])>=budget['maximum'] or key in budget['attempts']:raise ReplayDataUnavailable('Budget or prior attempt requires review '+key)
 budget['attempts'].append(key);write(ledger,budget)
def mark(p):refs[str(p.relative_to(ROOT))]=sha(p)
def finmind(sid,dataset):
 p=C/(sid+'-'+dataset+'.parquet')
 if not p.exists() and (OLD/'execution-v1'/p.name).exists():p=OLD/'execution-v1'/p.name
 meta=p.with_suffix('.json')
 if p.exists():
  m=read(meta)
  if sha(p)!=m['sha256'] or any(m.get(k)!=v for k,v in dict(stock_id=sid,dataset=dataset,start='2018-01-01',end='2026-09-09').items()):raise ValueError('Cached source identity changed')
 else:
  reserve('finmind:'+sid+':'+dataset)
  f=fetch_dataset(dataset,date(2018,1,1),date(2026,9,9),data_id=sid,token=token,requests_per_hour=5400,max_retries=0,timeout=30)
  if not f.empty and (set(f.stock_id)!={sid} or not pd.to_datetime(f.date).between('2018-01-01','2026-09-09').all()):raise ValueError('FinMind source stock/date differs')
  f.to_parquet(p,index=False);write(meta,dict(stock_id=sid,dataset=dataset,start='2018-01-01',end='2026-09-09',sha256=sha(p)))
  print('fetch',dataset,sid,len(f),flush=True)
 mark(p);mark(meta);f=pd.read_parquet(p)
 if not f.empty and (set(f.stock_id)!={sid} or not pd.to_datetime(f.date).between('2018-01-01','2026-09-09').all()):raise ValueError('Cached stock/date differs')
 return f

class Feeds:
 def __init__(self):self.loaded={}
 def get_limits(self,sid):
  if sid not in self.loaded:
   f=finmind(sid,'TaiwanStockPriceLimit');d={}
   for r in f.to_dict('records'):
    if r['date'] in d:raise ValueError('Duplicate limits')
    lo,hi=float(r['limit_down']),float(r['limit_up'])
    if not math.isfinite(lo+hi) or not 0<=lo<=hi or (lo==0)!=(hi==0):raise ValueError('Invalid limits')
    d[r['date']]=dict(lower=lo,upper=hi)
   self.loaded[sid]=d
  return self.loaded[sid]

class Odds:
 def __init__(self,base):self.base=base;self.files={};self.queries=[];self.loaded={};self.last=0
 def get_odd(self,day,sid,market):
  market=market.lower();key=f'odd:{market}:{day}';self.queries.append(dict(stock_id=sid,date=day,market=market.upper()))
  if day>=BEFORE and key in self.base.sources:
   result=self.base.get_odd(day,sid,market);self.files.update(self.base.files);return result
  if key not in self.loaded:
   p=C/(key.replace(':','-')+'.json')
   if not p.exists() and (OLD/'execution-v1'/p.name).exists():p=OLD/'execution-v1'/p.name
   if not p.exists():
    reserve(key);time.sleep(max(0,2-(time.monotonic()-self.last)))
    url=(AFTER if day<BEFORE else URLS)[market]
    params=dict(date=day.replace('-','' if market=='twse' else '/'),response='json')
    if day<BEFORE:params.update({'type':'ALL' if market=='twse' else 'Daily'})
    r=requests.get(url,params=params,timeout=30,allow_redirects=False);self.last=time.monotonic()
    if r.status_code!=200 or b'FOR SECURITY REASONS' in r.content:
     write(C/'security-stop.json',dict(key=key,status=r.status_code));raise ReplayDataUnavailable('Official response blocked '+key)
    record=dict(schema=1,provider=market,day=day,url=url,params=params,http_status=200,retrieved_at=datetime.now(timezone.utc).isoformat(),payload=r.json())
    write(p,record);print('fetch',key,flush=True)
   record=read(p);mark(p);self.files[str(p.relative_to(ROOT))]=sha(p)
   if day>=BEFORE:rows=parse_odd(record,market,day)
   else:rows=parse_after_hours(record,market,day)
   self.loaded[key]=rows
  if sid not in self.loaded[key]:raise ReplayDataUnavailable('Missing after-hours/odd stock row '+key+':'+sid)
  return self.loaded[key][sid]

class Corp(study.old.TrackedCorporateActions):
 def prepare(self,sid):
  if sid not in self.loaded:
   f=finmind(sid,'TaiwanStockDividend');p=self.directory/(sid+'.parquet')
   if not p.exists():f.to_parquet(p,index=False)
   elif not f.equals(pd.read_parquet(p)):raise ValueError('Dividend execution copy differs')
  result=super().prepare(sid)
  mark(self.directory/(sid+'.parquet'))
  return result
 def manifest(self):
  result=super().manifest();result.update(start='2018-01-01',end='2026-09-09');return result

class Era(HistoricalOddEra):
 def cash_move(self, day, kind, change, **extra):
  if kind=='fractional_share_payment' and any(r.get('stock_id')==extra.get('stock_id') and r.get('pay_date')==str(day.date()) and self.corporate.overrides.get(r.get('action_id','').removesuffix('-stock'),{}).get('fractional_rounding')=='floor_ntd' for r in self.receivables):change=math.floor(change)
  return super().cash_move(day,kind,change,**extra)
 def receivable_value(self):
  value=super().receivable_value()
  for r in self.receivables:
   if self.corporate.overrides.get(r.get('action_id','').removesuffix('-stock'),{}).get('fractional_rounding')=='floor_ntd':
    cash=r['fraction']*r['fractional_cash_per_share'];value+=math.floor(cash)-cash
  return value

class Original(Rotation2024,Era,MidpointExitReplay):pass
class Cap(Era,PartialRisk):pass
class Benchmark(Era,MidpointBenchmark):pass

manifest=read(I/'manifest.json')
for name,h in manifest['files_sha256'].items():
 if sha(I/name)!=h:raise ValueError('Input changed '+name)
refs.update(manifest['source_sha256']);refs.update({str((I/n).relative_to(ROOT)):h for n,h in manifest['files_sha256'].items()});mark(I/'manifest.json');mark(Path(__file__))
pub=study.old.load_selector();old,inputs,old_identity,_,extra_refs=study.strict.repaired_data(pub);refs.update(extra_refs);refs.update(pub['source_sha256'])
refs.update(study.old.file_identities([ROOT/'skills/historical_odd_regime.py',ROOT/'skills/partial_risk.py',ROOT/'skills/partial_risk_audit.py',ROOT/'skills/midpoint_exit_replay.py',ROOT/'skills/midpoint_exit_audit.py',ROOT/'docs/prereg_partial_risk_2019_20260929.md',*study.CODE],ROOT))
prepared=read(B/'signals-v1.json');refs.update(prepared['source_sha256']);mark(B/'signals-v1.json')
entries=prepared['entries']['original'];pool=sorted({'0050'}|{e['members'][0] for rows in prepared['entries'].values() for e in rows})
companies=pd.read_parquet(I/'companies.parquet');identity=read(I/'identity.json')
identity['trading_exclusions'] += [r for r in old_identity['trading_exclusions'] if r.get('stock_id')=='0050']
quotes=pd.read_parquet(I/'quotes-unmasked.parquet');quotes=quotes[quotes.stock_id.isin(pool)].copy();quotes['date']=pd.to_datetime(quotes.date)
mask=pd.read_parquet(I/'eligibility.parquet').set_index('date');mask.index=pd.to_datetime(mask.index);days=mask.index
raw=quotes.copy();quotes=quotes.loc[[bool(mask.at[d,s]) for s,d in zip(quotes.stock_id,quotes.date)]]
quotes,repairs=restore_halt_zeroes(quotes,[raw],identity)
split=old.quotes[old.quotes.stock_id.eq('0050')&old.quotes.volume.eq(0)]
quotes=pd.concat([quotes,split],ignore_index=True)
if quotes.duplicated(['date','stock_id']).any():raise ValueError('Duplicate repaired quote')
close=pd.read_parquet(I/'close-official.parquet',columns=['date',*pool]).set_index('date');close.index=pd.to_datetime(close.index)
events=pd.read_parquet(I/'events.parquet');data=RunInputs(quotes,companies,days,entries,events,ExitSignals(close,days),{},start='2019-01-02',end='2024-12-31')
extra,extra_refs=study.load_exit_completion(ROOT);refs.update(extra_refs)
additions=study.old.parent.parent.parent.load_corporate_completion(ROOT)|study.old.load_capital_terms(ROOT)[0]|extra
overrides=(study.old.read(study.old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides']|study.old.read(study.old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides']|study.old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides']|additions)
terms=read(ROOT/'docs/partial_risk_2019_corporate_terms.json')
overrides |= terms['overrides'];refs.update(terms['source_sha256']);mark(ROOT/'docs/partial_risk_2019_corporate_terms.json')

def assert_prefix(account,arm):
 parent=read(OLD/'final-a'/('benchmark.json' if arm=='benchmark' else 'original.json'))['account']
 end='2024-12-31' if arm in ('original','benchmark') else '2023-12-31'
 for key in ('daily','trades','orders','cash_ledger','holdings'):
  if [r for r in account[key] if r['date']<=end]!=[r for r in parent[key] if r['date']<=end]:
   raise ValueError('Frozen original prefix changed: '+arm+':'+key)

def audit_rotation(account,data,arm):
 from skills.rotation_2024 import stagnant,stronger
 c=data.features.adjusted_close; r=c/c.shift(20)-1
 cohorts={row['event_id']:row for row in account['cohorts']}
 indexed={row['event_id']:row for row in account.get('rotation_decisions',[])}
 for row in indexed.values():
  sid=row['stock_id'];day=pd.Timestamp(row['date']);prior=data.days[data.days.get_loc(day)-1]
  cohort=cohorts[row['event_id']];age=data.days.get_loc(day)-data.days.get_loc(pd.Timestamp(cohort['entry_date']))
  own=float(r.at[prior,sid]);relative=own-float(r.at[prior,'0050'])
  if row['signal_date']!=str(prior.date()) or row['age']!=age or row['return20']!=own or abs(row['relative20']-relative)>1e-12:
   raise ValueError('Rotation prior-only context differs')
  if row['reason']=='stagnant20':valid=stagnant(age,own,relative)
  else:
   candidate=row['candidate'];other=candidate['members'][0]
   candidate_relative=float(r.at[prior,other]-r.at[prior,'0050'])
   valid=(len(row['opening_members'])>=3 and other not in row['opening_members']
    and candidate['signal_date']==str(prior.date()) and stronger(age,relative,candidate_relative)
    and abs(candidate['priority']-candidate_relative)<1e-12)
  if not valid:raise ValueError('Rotation predicate differs')
 for trade in account['trades']:
  if trade['reason'] not in ('stagnant20','stronger_signal'):continue
  decision=indexed.get(trade['event_id'])
  if not decision or decision['reason']!=trade['reason'] or decision['signal_date']!=trade['signal_date'] or trade['date']<decision['date']:
   raise ValueError('Rotation fill lacks prior instruction')

cases={}
def replay():
 for arm in a.arms.split(','):
  feeds=Feeds();odds=Odds(study.OddDailyCache(ROOT,inputs/'execution-feeds'));ticks=study.strict.AdditionalTicks()
  corp=Corp(events,C/'dividends',None,offline=True,overrides=overrides)
  cls=Benchmark if arm=='benchmark' else Original
  opts={} if arm=='benchmark' else dict(ordering='original',position_count=3,factor_mask=0,residual_policy='release',identity_report=identity,exit_signals=data.features,action_dates=list(zip(events.stock_id,events.event_date)))
  if arm!='benchmark':opts.update(rotation_arm=arm)
  selected=entries if arm=='benchmark' else prepared['entries'][arm]
  engine=cls(quotes,companies,days,selected,feeds,corp,start=data.start,end=data.end,ticks=ticks,participation=.01,liquidity_identity=identity,odd_feeds=odds,**opts)
  print('start',arm,flush=True)
  try:
   account=engine.run();study.old.validate_completed_account(account,[str(d.date()) for d in days],data.start,data.end)
   view,era_audit=normalized_era_account(account)
   if arm=='benchmark':audit=study.old.audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
   elif arm=='cap40':audit=audit_core_resources(account,engine,quotes)
   else:audit=study.audit_high_return_resources(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,quotes)
   checker=audit_midpoint if arm=='benchmark' else audit_midpoint_exit
   audit.update(checker(view,ticks,odds,study.market_routes([*ticks.queries,*odds.queries]),quotes,days,corp,feeds))
   audit.update(era_audit)
   if arm=='cap40':audit['partial_risk']=audit_partial_risk(account,data,engine)
   value=dict(completed=True,account=account,summary=study.old.summarize(account),audit=audit);verify_cash(value)
   audit_rotation(account,data,arm)
   assert_prefix(account,arm)
   if PREP:value=dict(completed=True,summary=None)
  except (ReplayDataUnavailable,study.old.UnresolvedAction,ValueError) as exc:
   value=dict(completed=False,summary=None,reason=str(exc),completed_sessions=len(engine.daily),last_date=engine.daily[-1]['date'] if engine.daily else None)
  refs.update(odds.files)
  path=output/(arm+'.json');write(path,value);cases[arm]={k:v for k,v in value.items() if k not in ('account','audit')}
  cases[arm].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
  append_trial_registry(dict(timestamp=datetime.now().isoformat(timespec='seconds'),source='rotation_2024_20260929',params=dict(arm=arm,start=data.start,end=data.end),preparation=PREP,completed=value['completed'],result_path=str(path.relative_to(ROOT)),live_qualified=False))
  print(arm, [r for r in value['summary']['annual'] if r['year']=='2024'] if value.get('summary') else cases[arm],flush=True)
if PREP:replay()
else:
 with study.old.offline_only():replay()
if study.old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Run sources changed')
write(output/'report.json',dict(start=data.start,end=data.end,initial_cash=1_000_000,preparation=PREP,cases=cases,source_sha256=refs,all_completed=all(c['completed'] for c in cases.values()),live_qualified=False,actual_fill_verified=False,complete_historical_universe=False,unseen_validation=False,validated=not PREP and all(c['completed'] for c in cases.values())))
