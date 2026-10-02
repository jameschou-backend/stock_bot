#!/usr/bin/env python3
"""Fixed three-black replay with an explicit, source-verified repaired input bundle."""
from pathlib import Path
import sys,json,time,math,argparse
from datetime import date,datetime,timezone
from copy import deepcopy
from unittest.mock import patch
from urllib.parse import urlparse
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_midpoint as study
from scripts.research_exit_scenarios import read,write,sha,RunInputs
from app.finmind import fetch_dataset, FinMindError
from app.config import load_config
from skills.replay_market_feeds import parse_odd,_number,ReplayDataUnavailable,URLS
from skills.scenario_exit_replay import ExitSignals
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.midpoint_replay import MidpointBenchmark
from scripts.export_midpoint_2025_report import verify_cash
from skills.midpoint_exit_audit import audit_midpoint_exit
from skills.midpoint_audit import audit_midpoint
from skills.strict_tick_inputs import restore_halt_zeroes

OLD=ROOT/'.cache/partial-risk-2019-20260929'; I=OLD/'inputs-final'
B=ROOT/'.cache/stock-universe-2019-20260929'; C=B/'execution-v1'
from skills.candidate_quality import CandidateQuality
from skills.three_black_exit import ARMS, ThreeBlackControl as DrawdownControl, ThreeBlackSignals, audit_three_black
from skills.zero_value_share_delivery import ZeroValueShareDelivery
from skills.frozen_dividend_copy import ensure_dividend_copy
from skills.deferred_corporate import DeferredCorporatePreparation
from skills.trial_registry import append_trial_registry
from skills.candidate_corporate import complete_cash_dividends
from skills.candidate_execution_context import load_context
from skills.rescheduled_corporate import CorporateCalendar
from skills.face_value_capital import FaceValueCapitalActions, audit_face_resources
from skills import pending_share_entitlements
from skills.pending_share_entitlements import install_pending_share_rights
from skills.prepared_corporate_settlement import validate_delivery_terms
from skills.historical_odd_regime import HistoricalOddEra, normalized_era_account, parse_after_hours
BEFORE='2020-10-26';AFTER={'twse':'https://www.twse.com.tw/rwd/zh/afterTrading/TWT53U','tpex':'https://www.tpex.org.tw/www/zh-tw/afterTrading/odd'}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepare',action='store_true')
parser.add_argument('--output',required=True)
parser.add_argument('--arms',default='three_black,benchmark')
parser.add_argument('--retry-source',action='append',default=[])
parser.add_argument('--input-bundle',type=Path)
parser.add_argument('--volume-policy', choices=('strict', 'legacy_total_research'))
a=parser.parse_args()
if a.input_bundle is not None:
 I=a.input_bundle.resolve()
 if not I.is_relative_to(ROOT) or not I.is_dir():raise ValueError('Input bundle must be a repository directory')
if a.input_bundle is None and a.volume_policy is not None:
 raise ValueError('The frozen control cannot change its execution policy')
volume_policy=a.volume_policy or ('strict' if a.input_bundle else 'legacy_total_research')

BASE=B;LIQ=ROOT/'.cache/liquidity-universe-20261001'
B=ROOT/'.cache/market-input-repair-20261002';C=B/'execution-v1';SOURCE='three_black_data_repair_20261002'
CACHES=(ROOT/'.cache/three-black-20261001/execution-v1',ROOT/'.cache/drawdown-control-20261001/execution-v1',LIQ/'execution-v1',ROOT/'.cache/waiting-exit-20260930/execution-v1',ROOT/'.cache/entry-filters-20260930/execution-v1',ROOT/'.cache/holding-release-20260929/execution-v1',BASE/'execution-v1',ROOT/'.cache/stock-universe-five-20260929/execution-v1',ROOT/'.cache/liquidity-account-20261001/execution-v1',ROOT/'.cache/allocation-2019-20260929/execution-v1',ROOT/'.cache/candidate-quality-20260929/execution-v1',OLD/'execution-v1',ROOT/'.cache/rotation-2024-20260929/execution-v1')
output=B/a.output
output.resolve().relative_to(B.resolve())
if output.exists():raise ValueError('Choose new output')
if any(arm not in ARMS for arm in a.arms.split(',')):raise ValueError('Unregistered arm')
output.mkdir(parents=True)
source_snapshot=output/'runner_source.py'
source_snapshot.write_bytes(Path(__file__).read_bytes())
PREP=a.prepare; C.mkdir(parents=True,exist_ok=True); refs={};token=load_config().finmind_token if PREP else None
ledger=C/'budget.json';budget=read(ledger) if ledger.exists() else dict(attempts=[],maximum=500)
def reserve(key):
 if not PREP:raise ReplayDataUnavailable('Offline input missing '+key)
 if (C/'security-stop.json').exists():raise ReplayDataUnavailable('Official security stop active')
 if len(budget['attempts'])>=budget['maximum']:raise ReplayDataUnavailable('Request budget exhausted '+key)
 attempt=key
 if key in budget['attempts']:
  if key not in a.retry_source:raise ReplayDataUnavailable('Prior attempt requires explicit retry '+key)
  a.retry_source.remove(key);attempt=key+'#retry-'+str(sum(k.startswith(key+'#retry-') for k in budget['attempts'])+1)
 budget['attempts'].append(attempt);write(ledger,budget)
def merge_refs(values):
 for key,value in values.items():
  if not (ROOT/key).resolve().is_relative_to(ROOT):raise ValueError('Source escapes repository '+key)
  if key in refs and refs[key]!=value:raise ValueError('Conflicting frozen source hash '+key)
  refs[key]=value
def mark(p):merge_refs({str(p.relative_to(ROOT)):sha(p)})
def finmind(sid,dataset):
 p=C/(sid+'-'+dataset+'.parquet')
 for cached in CACHES:
  if not p.exists() and (cached/p.name).exists():p=cached/p.name
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
   self.loaded[sid]=calendar.limits(sid,d)
  return self.loaded[sid]

class Odds:
 def __init__(self,base):self.base=base;self.files={};self.queries=[];self.loaded={};self.last=0
 def get_odd(self,day,sid,market):
  market=market.lower();key=f'odd:{market}:{day}';self.queries.append(dict(stock_id=sid,date=day,market=market.upper()))
  if day>=BEFORE and key in self.base.sources:
   result=self.base.get_odd(day,sid,market);self.files.update(self.base.files);return result
  if key not in self.loaded:
   p=C/(key.replace(':','-')+'.json')
   for cached in CACHES:
    if not p.exists() and (cached/p.name).exists():p=cached/p.name
   if not p.exists():
    url=(AFTER if day<BEFORE else URLS)[market]
    hold=ROOT/'.cache/official-origin-holds'/(urlparse(url).hostname+'.json')
    if hold.exists():raise ReplayDataUnavailable('Official source hold remains active: '+str(hold.relative_to(ROOT)))
    reserve(key);time.sleep(max(0,2-(time.monotonic()-self.last)))
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

class CorpLoader(study.old.TrackedCorporateActions):
 def prepare(self,sid):
  if sid not in self.loaded:
   f=finmind(sid,'TaiwanStockDividend');p=self.directory/(sid+'.parquet')
   ensure_dividend_copy(f,p,prepare=PREP)
  result=super().prepare(sid)
  self.loaded[sid]=complete_cash_dividends(self.loaded[sid],sid,supplement['cash_supplements'])
  self.loaded[sid]=calendar.actions(sid,self.loaded[sid])
  mark(self.directory/(sid+'.parquet'))
  return result
 def on_date(self,sid,day):
  result=super().on_date(sid,day)
  calendar.guard(sid,day,self.loaded[sid])
  return result
 def reference_price(self,sid,day,prior):
  corrected=calendar.opening_reference(sid,day)
  return corrected if corrected is not None else super().reference_price(sid,day,prior)
 def manifest(self):
  result=super().manifest();result.update(start='2018-01-01',end='2026-09-09');return result

class Corp(DeferredCorporatePreparation, CorpLoader):pass

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

class Original(ZeroValueShareDelivery,DrawdownControl,CandidateQuality,Era,MidpointExitReplay):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.corporate=FaceValueCapitalActions(self.corporate.provider,self)
 def _execute_order(self,day,sid,side,qty,reason,event_id,signal_date=None):
  try:return super()._execute_order(day,sid,side,qty,reason,event_id,signal_date)
  except ReplayDataUnavailable as exc:raise ReplayDataUnavailable(f'{sid} {day.date()} {side} {event_id}: {exc}') from exc
class Benchmark(Era,MidpointBenchmark):pass

# New policy classes preserve the original frozen source and its MRO. The era
# adapter must remain before the order implementation for pre-2020 odd lots.
if a.input_bundle is not None:
 from skills.verified_volume_midpoint import VerifiedVolumeMidpointOrders
 from skills.verified_volume_midpoint_audit import audit_verified_volume_midpoint
 from skills.ordinary_volume_bundle import load_ordinary_matrices
 from skills.ordinary_volume_evidence import verify_halts, benchmark_split_halt
 from skills.repaired_execution_context import benchmark_exclusion_with_market, dated_market_resolver
 class RepairedOriginal(ZeroValueShareDelivery,DrawdownControl,CandidateQuality,Era,
                         VerifiedVolumeMidpointOrders,MidpointExitReplay):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.corporate=FaceValueCapitalActions(self.corporate.provider,self)
 class RepairedBenchmark(Era,VerifiedVolumeMidpointOrders,MidpointBenchmark):pass

manifest=read(I/'manifest.json')
for name,h in manifest['files_sha256'].items():
 if sha(I/name)!=h:raise ValueError('Input changed '+name)
merge_refs(manifest['source_sha256']);merge_refs({str((I/n).relative_to(ROOT)):h for n,h in manifest['files_sha256'].items()});mark(I/'manifest.json');mark(source_snapshot)
print('verify frozen execution context',flush=True)
mark(ROOT/'skills/candidate_execution_context.py')
merge_refs(study.old.file_identities([ROOT/'skills/historical_odd_regime.py',ROOT/'skills/partial_risk.py',ROOT/'skills/partial_risk_audit.py',ROOT/'skills/midpoint_exit_replay.py',ROOT/'skills/midpoint_exit_audit.py',ROOT/'docs/prereg_partial_risk_2019_20260929.md',*study.CODE],ROOT))
signal_path=(I/'signals.json') if a.input_bundle is not None else LIQ/'signals-v1.json'
prepared=read(signal_path);merge_refs(prepared['source_sha256']);mark(signal_path)
mark(ROOT/'docs/prereg_stock_universe_2019_20260929.md');mark(ROOT/'skills/stock_universe_2019.py')
mark(ROOT/'skills/deferred_corporate.py')
mark(ROOT/'skills/three_black_exit.py');mark(ROOT/'docs/prereg_three_black_exit_20261001.md');mark(ROOT/'docs/prereg_market_input_repair_20261002.md')
mark(ROOT/'skills/zero_value_share_delivery.py')
mark(ROOT/'skills/frozen_dividend_copy.py')
entries=prepared['entries']['median50m'];pool=sorted({'0050'}|{e['members'][0] for rows in prepared['entries'].values() for e in rows})
companies=pd.read_parquet(I/'companies.parquet');identity=read(I/'identity.json')
quotes=pd.read_parquet(I/'quotes-unmasked.parquet');quotes=quotes[quotes.stock_id.isin(pool)].copy();quotes['date']=pd.to_datetime(quotes.date)
mask=pd.read_parquet(I/'eligibility.parquet').set_index('date');mask.index=pd.to_datetime(mask.index);days=mask.index
completion=read(ROOT/'docs/liquidity_universe_corporate_terms_v2_20261001.json')
drawdown_terms=read(ROOT/'docs/drawdown_control_corporate_terms_20261001.json')
merge_refs(drawdown_terms['source_sha256']);mark(ROOT/'docs/drawdown_control_corporate_terms_20261001.json')
black_terms=read(ROOT/'docs/three_black_corporate_terms_20261001.json')
merge_refs(black_terms['source_sha256']);mark(ROOT/'docs/three_black_corporate_terms_20261001.json')
calendar=CorporateCalendar(days,completion['rescheduled']+drawdown_terms['rescheduled']+black_terms['rescheduled'])
merge_refs(completion['source_sha256']);mark(ROOT/'docs/liquidity_universe_corporate_terms_v2_20261001.json');mark(ROOT/'skills/rescheduled_corporate.py')
inputs,split,benchmark_exclusion,extra_refs=load_context(ROOT,days);merge_refs(extra_refs)
identity['trading_exclusions'].append(benchmark_exclusion_with_market(benchmark_exclusion) if a.input_bundle else benchmark_exclusion)
from skills.verified_halt_observations import add_verified_halt_observations
additional=read(ROOT/'docs/stock_universe_2019_corporate_terms.json')
followup=read(ROOT/'docs/stock_universe_five_corporate_terms.json')
additional['verified_halts'].extend(followup['verified_halts'])
mark(ROOT/'skills/face_value_capital.py');mark(ROOT/'skills/corporate_account_audit.py')
for name in ('pending_share_entitlements','prepared_corporate_settlement','account_source_preflight'):
 mark(ROOT/'skills'/(name+'.py'))
identity['trading_exclusions'].extend(additional['verified_halts'])
mark(ROOT/'skills/verified_halt_observations.py')
raw=quotes.copy();quotes=quotes.loc[[bool(mask.at[d,s]) for s,d in zip(quotes.stock_id,quotes.date)]]
quotes,repairs=restore_halt_zeroes(quotes,[raw],identity)
quotes,verified_halt_evidence=add_verified_halt_observations(quotes,days,additional['verified_halts'],ROOT)
quotes=pd.concat([quotes,split],ignore_index=True)
if quotes.duplicated(['date','stock_id']).any():raise ValueError('Duplicate repaired quote')
close=pd.read_parquet(I/'close-official.parquet',columns=['date',*pool]).set_index('date');close.index=pd.to_datetime(close.index)
black_signals=ThreeBlackSignals(close,quotes,days)
events=pd.read_parquet(I/'events.parquet');data=RunInputs(quotes,companies,days,entries,events,ExitSignals(close,days),{},start='2019-01-02')
extra,extra_refs=study.load_exit_completion(ROOT);merge_refs(extra_refs)
additions=study.old.parent.parent.parent.load_corporate_completion(ROOT)|study.old.load_capital_terms(ROOT)[0]|extra
overrides=(study.old.read(study.old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides']|study.old.read(study.old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides']|study.old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides']|additions)
terms=read(ROOT/'docs/partial_risk_2019_corporate_terms.json')
overrides |= terms['overrides'];merge_refs(terms['source_sha256']);mark(ROOT/'docs/partial_risk_2019_corporate_terms.json')
supplement=read(ROOT/'docs/candidate_quality_corporate_terms.json')
if data.end>supplement['latest_supported_account_end']:raise ValueError('Supplement cannot support post-certificate trading')
overrides |= supplement['overrides'];merge_refs(supplement['source_sha256'])
mark(ROOT/'docs/candidate_quality_corporate_terms.json');mark(ROOT/'skills/candidate_corporate.py')
for row in supplement['cash_supplements']:
 event=events.loc[events.stock_id.eq(row['stock_id']) & pd.to_datetime(events.event_date).eq(row['date'])]
 if len(event)!=1 or abs(json.loads(event.iloc[0].payload_json)['cash_dividend']-row['cash_per_share'])>1e-10:raise ValueError('Cash supplement differs from official ex-date record')

additional=read(ROOT/'docs/stock_universe_2019_corporate_terms.json')
overrides |= additional['overrides'];merge_refs(additional['source_sha256']);mark(ROOT/'docs/stock_universe_2019_corporate_terms.json')
followup=read(ROOT/'docs/stock_universe_five_corporate_terms.json')
# Reuse the reviewed exchange adapter as well as ordinary deliveries.
# Unverified fractional capital cash stays unavailable as a receivable.
overrides.update(followup['overrides'])
merge_refs(followup['source_sha256']);mark(ROOT/'docs/stock_universe_five_corporate_terms.json')
overrides.update(completion['overrides'])
overrides.update(drawdown_terms['overrides']);overrides.update(black_terms['overrides'])
ordinary={}
if a.input_bundle is not None:
 for module in ('verified_volume_midpoint','verified_volume_midpoint_audit','ordinary_volume_bundle','ordinary_volume_evidence','repaired_execution_context'):
  mark(ROOT/'skills'/(module+'.py'))
 dated_market=dated_market_resolver(identity)
 if volume_policy=='strict':
  reviewed_halts=verify_halts(additional['verified_halts']+followup['verified_halts'],ROOT,refs)
  trading_repair=read(ROOT/'docs/historical_trading_repair_20261002.json')
  mark(ROOT/'docs/historical_trading_repair_20261002.json')
  merge_refs(trading_repair['source_sha256'])
  extra_halts=[dict(row,source_sha256=trading_repair['source_sha256'][row['source_path']])
               for row in trading_repair['entries'] if row['kind']=='trading_suspension']
  reviewed_halts.extend(verify_halts(extra_halts,ROOT,refs))
  reviewed_halts.append(benchmark_split_halt('docs/benchmark_split_evidence_20260914.json',ROOT,refs,
   expected_metadata_sha256='b969e443b75cf2cf7ed14a5c8314379982d6aa3fabb9d44e46e4f650c48e4742'))
  ordinary=load_ordinary_matrices(ROOT,
   '.cache/market-input-repair-20261002/quote-evidence/official-sources.json',days,pool,refs,halts=reviewed_halts)
cases={}
def replay():
 for arm in a.arms.split(','):
  feeds=Feeds();odds=Odds(study.OddDailyCache(ROOT,inputs/'execution-feeds'));ticks=study.strict.AdditionalTicks()
  corp=Corp(events,C/'dividends',None,offline=True,overrides=overrides)
  cls=Benchmark if arm=='benchmark' else Original
  if a.input_bundle is not None:cls=RepairedBenchmark if arm=='benchmark' else RepairedOriginal
  opts={} if arm=='benchmark' else dict(ordering='original',position_count=3,factor_mask=0,residual_policy='release',identity_report=identity,exit_signals=data.features,action_dates=[(r.stock_id,calendar.effective(r.stock_id,str(r.event_date))) for r in events.itertuples()])
  if arm!='benchmark':opts.update(candidate_arm='original',drawdown_arm=arm,black_signals=black_signals)
  if a.input_bundle is not None:
   opts.update(ordinary_volumes=ordinary,ordinary_market_resolver=dated_market,volume_policy=volume_policy)
  selected=entries
  engine=cls(quotes,companies,days,selected,feeds,corp,start=data.start,end=data.end,ticks=ticks,participation=.01,liquidity_identity=identity,odd_feeds=odds,**opts)
  corp.preparation_engine=engine
  if arm!='benchmark':install_pending_share_rights(engine,ROOT)
  print('start',arm,flush=True)
  try:
   account=engine.run();study.old.validate_completed_account(account,[str(d.date()) for d in days],data.start,data.end)
   view,era_audit=normalized_era_account(account)
   if arm=='benchmark':audit=study.old.audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
   else:audit=audit_face_resources(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,quotes)
   routes=study.market_routes([*ticks.queries,*odds.queries])
   routes.update({(r['stock_id'],r['date']):dated_market(r['date'],r['stock_id'])
                 for r in account['orders'] if a.input_bundle is not None})
   checker=audit_midpoint if arm=='benchmark' else audit_midpoint_exit
   if a.input_bundle is not None and volume_policy=='strict':
    audit.update(audit_verified_volume_midpoint(view,ticks,odds,routes,quotes,days,corp,feeds,ordinary,dated_market,
                                              verified_halts=engine.full_halts))
   else:audit.update(checker(view,ticks,odds,routes,quotes,days,corp,feeds))
   audit.update(era_audit);audit['verified_halt_observations']=verified_halt_evidence
   if arm!='benchmark':audit['three_black']=audit_three_black(account,black_signals)
   value=dict(completed=True,account=account,summary=study.old.summarize(account),audit=audit);verify_cash(value)
   value['ordinary_capacity_complete']=account.get('ordinary_volume_evidence',{}).get('all_requested_board_capacity_observed',False)
   value['ordinary_evidence_blocked_orders']=account.get('ordinary_volume_evidence',{}).get('blocked_board_children',0)
   if a.input_bundle is None and arm=='benchmark' and account!=read(ROOT/'.cache/drawdown-control-20261001/final-a/benchmark.json')['account']:raise ValueError('Frozen benchmark account changed')
   if a.input_bundle is None and arm=='control' and account!=read(LIQ/'final-a/median50m.json')['account']:raise ValueError('Median liquidity baseline account changed')
   if a.input_bundle is None and arm=='three_black':
    frozen=ROOT/'.cache/three-black-20261001/final-b/three_black.json';mark(frozen)
    if account!=read(frozen)['account']:raise ValueError('Read-only cache fix changed the first complete three-black account')
   if PREP:value=dict(completed=True,summary=None)
  except (ReplayDataUnavailable,study.old.UnresolvedAction,ValueError,requests.RequestException,FinMindError) as exc:
   value=dict(completed=False,summary=None,reason=str(exc),completed_sessions=len(engine.daily),last_date=engine.daily[-1]['date'] if engine.daily else None)
   value['failure_holdings']=deepcopy(engine.holdings)
   value['ordinary_evidence_blocks']=deepcopy(getattr(engine,'volume_evidence_blocks',[]))
   # Preserve the executed prefix for diagnosis, without reporting a partial
   # period as a completed strategy return.
   value['partial_journal']=dict(daily=engine.daily,trades=engine.trades,orders=engine.orders)
  value['deferred_corporate_preparations']=len(getattr(corp,'deferred_preparations',[]))
  merge_refs(odds.files)
  path=output/(arm+'.json');write(path,value);cases[arm]={k:v for k,v in value.items() if k not in ('account','audit','partial_journal','ordinary_evidence_blocks')}
  cases[arm].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
  record=dict(timestamp=datetime.now().isoformat(timespec='seconds'),source=SOURCE,params=dict(arm=arm,start=data.start,end=data.end,input_bundle=str(I.relative_to(ROOT)),execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy=volume_policy,data_revision='repaired' if a.input_bundle else 'frozen_control'),preparation=PREP,completed=value['completed'],result_path=str(path.relative_to(ROOT)),live_qualified=False)
  append_trial_registry(record,registry_path=output/'trials.jsonl');append_trial_registry(record)
  print(arm,{k:value['summary'][k] for k in ('total_return','max_drawdown','final_nav')} if value.get('summary') else cases[arm],flush=True)
with patch.object(pending_share_entitlements,'validate_pending_terms',validate_delivery_terms):
 if PREP:replay()
 else:
  with study.old.offline_only():replay()
if study.old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Run sources changed')
write(output/'report.json',dict(start=data.start,end=data.end,initial_cash=1_000_000,input_bundle=str(I.relative_to(ROOT)),execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy=volume_policy,data_revision='repaired' if a.input_bundle else 'frozen_control',return_recomputed=not PREP and all(c['completed'] for c in cases.values()),preparation=PREP,cases=cases,source_sha256=refs,all_completed=all(c['completed'] for c in cases.values()),live_qualified=False,actual_fill_verified=False,corporate_fractional_cash_date_verified=False,complete_historical_universe=False,unseen_validation=False,accounting_validated=not PREP and all(c['completed'] for c in cases.values()),validated=False))
