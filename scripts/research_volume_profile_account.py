#!/usr/bin/env python3
"""Preregistered 2024 empty-account baseline, adapted from the repaired cash runner.

The frozen runner is never imported or executed as text. Providers below are
local-cache-only; missing inputs stop an arm and preserve its journal.
"""
from pathlib import Path
import sys,json,time,math,argparse
from datetime import datetime
from copy import deepcopy
from unittest.mock import patch
from contextlib import nullcontext
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_midpoint as study
from scripts.research_exit_scenarios import read,write,sha,RunInputs
from skills.replay_market_feeds import parse_odd,ReplayDataUnavailable
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
from skills.three_black_exit import ThreeBlackControl as DrawdownControl, ThreeBlackSignals, audit_three_black
from skills.volume_profile_account_adapter import AccountCandidateHook, AccountReservationAudit
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
BEFORE='2020-10-26'
FROZEN_RUNNER = ROOT/'scripts/replay_repaired_market_inputs.py'
PREREG = ROOT/'docs/prereg_volume_profile_account_baseline_20261003.md'
POC_PREREG = ROOT/'docs/prereg_volume_profile_account_poc_20261003.md'
POC_PREREG_SHA = 'e31cbc7d4c00d69c3a803b0f54c22394e1b6f2f3cc1912180fa0cce269a98832'
ARMS = ('original','benchmark','poc_priority','poc_filter',
        'poc_priority_available','poc_filter_available')
FROZEN_BINDINGS = {
    'scripts/replay_repaired_market_inputs.py': '199e171d5433760039ffdae1152cabe633b427b98c60e88d3c1d72fd0735ce26',
    '.cache/market-input-repair-20261002/inputs-v2/manifest.json': 'f62dbe32d263abb464c9f283879b7d6340b0667b6fa9e99404c6ece7e04663d6',
    '.cache/market-input-repair-20261002/inputs-v2/signals.json': 'f0821897f077d616494bcaafe2b2278df27666c03466758642c80784f7542513',
    'skills/three_black_exit.py': 'b03d2209c3d35c6fefac285d500888efa555a0222a913f01c58687f9dee8afa3',
    'docs/prereg_volume_profile_account_baseline_20261003.md': 'd89cdf12062245d1c195bc8d694b5023ecf2d6df02e9cee17d298d4c02485660',
}


def _run(output, arms=('original', 'benchmark'), profile_provider=None, candidate_selector=None,
        *, execution_fetch=False, limit_overlay=None, odd_fetch=False):
 started=time.monotonic()
 for name,digest in FROZEN_BINDINGS.items():
  if sha(ROOT/name)!=digest:raise ValueError('Preregistered source changed: '+name)
 if not arms or len(set(arms))!=len(arms) or any(x not in ARMS for x in arms):
  raise ValueError('Use only the six preregistered account arms')
 uses_profiles=any(arm.startswith('poc_') for arm in arms)
 if uses_profiles and (profile_provider is None or candidate_selector is None or sha(POC_PREREG)!=POC_PREREG_SHA):
  raise ValueError('POC arms require their frozen preregistration, provider and selector')
 output=Path(output).resolve()
 output.relative_to(ROOT/'.cache/volume-profile-account-20261003')
 if output.exists():raise ValueError('Choose a new output directory')
 output.mkdir(parents=True)
 source_snapshot=output/'runner_source.py'
 source_snapshot.write_bytes(Path(__file__).read_bytes())
 I=ROOT/'.cache/market-input-repair-20261002/inputs-v2'
 volume_policy='legacy_total_research'
 PREP=False
 refs=dict(FROZEN_BINDINGS)
 BASE=ROOT/'.cache/stock-universe-2019-20260929';LIQ=ROOT/'.cache/liquidity-universe-20261001'
 B=ROOT/'.cache/market-input-repair-20261002';C=B/'execution-v1';SOURCE='volume_profile_account_20261003'
 CACHES=(ROOT/'.cache/three-black-20261001/execution-v1',ROOT/'.cache/drawdown-control-20261001/execution-v1',LIQ/'execution-v1',ROOT/'.cache/waiting-exit-20260930/execution-v1',ROOT/'.cache/entry-filters-20260930/execution-v1',ROOT/'.cache/holding-release-20260929/execution-v1',BASE/'execution-v1',ROOT/'.cache/stock-universe-five-20260929/execution-v1',ROOT/'.cache/liquidity-account-20261001/execution-v1',ROOT/'.cache/allocation-2019-20260929/execution-v1',ROOT/'.cache/candidate-quality-20260929/execution-v1',OLD/'execution-v1',ROOT/'.cache/rotation-2024-20260929/execution-v1')
 def merge_refs(values):
  for key,value in values.items():
   if not (ROOT/key).resolve().is_relative_to(ROOT):raise ValueError('Source escapes repository '+key)
   if key in refs and refs[key]!=value:raise ValueError('Conflicting frozen source hash '+key)
   refs[key]=value
 def mark(p):merge_refs({str(p.relative_to(ROOT)):sha(p)})
 execution_attempts=[]
 odd_attempts=[];odd_acquisition=None
 if odd_fetch:
  from scripts.prepare_volume_profile_odd import OddCompletion
  odd_acquisition=OddCompletion(ROOT)
  for name in ('scripts/prepare_volume_profile_odd.py','tests/test_prepare_volume_profile_odd.py',
               'docs/prereg_volume_profile_odd_completion_20261003.md'):
   mark(ROOT/name)
 if uses_profiles:
  mark(POC_PREREG)
  for name in ('volume_profile_data','volume_profile_selection','volume_profile'):
   mark(ROOT/'skills'/(name+'.py'))
  for name in ('test_volume_profile_data','test_volume_profile_selection'):
   mark(ROOT/'tests'/(name+'.py'))
 if execution_fetch:
  mark(ROOT/'scripts/prepare_volume_profile_execution.py')
  mark(ROOT/'docs/prereg_volume_profile_execution_preparation_20261003.md')
  mark(ROOT/'tests/test_prepare_volume_profile_execution.py')
 if limit_overlay is not None:
  merge_refs(limit_overlay.refs)
  mark(ROOT/'skills/benchmark_limit_overlay.py')
 def finmind(sid,dataset):
  p=C/(sid+'-'+dataset+'.parquet')
  for cached in CACHES:
   if not p.exists() and (cached/p.name).exists():p=cached/p.name
  meta=p.with_suffix('.json')
  if p.exists():
   m=read(meta)
   if sha(p)!=m['sha256'] or any(m.get(k)!=v for k,v in dict(stock_id=sid,dataset=dataset,start='2018-01-01',end='2026-09-09').items()):raise ValueError('Cached source identity changed')
  else:
   if not execution_fetch:
    raise ReplayDataUnavailable('Offline FinMind cache missing: '+sid+' '+dataset)
   from scripts.prepare_volume_profile_execution import prepare, BASE as EXECUTION_PREP
   receipt=prepare(sid,dataset)
   execution_attempts.append(receipt)
   mark(EXECUTION_PREP/'attempts'/(sid+'-'+dataset+'.json'))
   m=read(meta)
   if sha(p)!=m['sha256']:raise ValueError('New execution-source hash mismatch')
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
    if limit_overlay is not None:
     self.loaded[sid]=limit_overlay.apply(sid,self.loaded[sid])
   return self.loaded[sid]

 class Odds:
  def __init__(self,base):self.base=base;self.files={};self.queries=[];self.loaded={};self.last=0;self.engine=None
  def get_odd(self,day,sid,market):
   market=market.lower();key=f'odd:{market}:{day}';self.queries.append(dict(stock_id=sid,date=day,market=market.upper()))
   if day>=BEFORE and key in self.base.sources:
    result=self.base.get_odd(day,sid,market);self.files.update(self.base.files);return result
   if key not in self.loaded:
    p=C/(key.replace(':','-')+'.json')
    for cached in CACHES:
     if not p.exists() and (cached/p.name).exists():p=cached/p.name
    if not p.exists():
     reason='Offline official odd cache missing (holds retained): '+key
     if odd_acquisition is None:raise ReplayDataUnavailable(reason)
     if self.engine is None:raise ValueError('Odd acquisition has no current execution context')
     demand=output/'odd-demands'/(market+'-'+day+'-'+sid+'.json')
     if not demand.exists():
      write(demand,dict(completed=False,summary=None,reason=reason,
       partial_journal=dict(day_plans=deepcopy(list(self.engine.day_plans.values())))))
     receipt=odd_acquisition.fetch(market.upper(),day,sid,demand)
     odd_attempts.append(receipt);merge_refs(odd_acquisition.refs)
     if not receipt['accepted']:raise ReplayDataUnavailable('Official odd acquisition stopped: '+key+' '+receipt['status'])
     if not p.exists():raise ReplayDataUnavailable('Verified odd acquisition did not publish execution evidence')
    record=read(p);mark(p);self.files[str(p.relative_to(ROOT))]=sha(p)
    if 'source_receipt' in record:
     # Newly collected odd evidence has a complete receipt/attempt/probe chain.
     # Verify it even on an offline cache hit, including probes made before this run.
     from scripts.prepare_volume_profile_odd import OddCompletion, BASE as ODD_PREPARATION
     if record['source_receipt'].startswith(ODD_PREPARATION+'/receipts/'):
      verifier=odd_acquisition if odd_acquisition is not None else OddCompletion(ROOT)
      verified=verifier.verify_cached(p)
      if verified['record']!=record:raise ValueError('Verified odd wrapper changed during read')
      merge_refs(verified['source_sha256'])
      mark(ROOT/'scripts/prepare_volume_profile_odd.py')
      mark(ROOT/'tests/test_prepare_volume_profile_odd.py')
     for field in ('source_receipt','source_raw'):
      linked=(ROOT/record[field]).resolve()
      if not linked.is_relative_to(ROOT) or sha(linked)!=record[field+'_sha256']:
       raise ValueError('Linked official odd evidence changed: '+field)
      mark(linked)
     if read(ROOT/record['source_raw'])!=record['payload']:
      raise ValueError('Official odd response wrapper differs from its raw bytes')
    if day>=BEFORE:rows=parse_odd(record,market,day)
    else:rows=parse_after_hours(record,market,day)
    self.loaded[key]=rows
   if sid not in self.loaded[key]:raise ReplayDataUnavailable('Missing after-hours/odd stock row '+key+':'+sid)
   return self.loaded[key][sid]

 class CorpLoader(study.old.TrackedCorporateActions):
  def prepare(self,sid):
   if sid not in self.loaded:
    f=finmind(sid,'TaiwanStockDividend');p=self.directory/(sid+'.parquet')
    ensure_dividend_copy(f,p,prepare=execution_fetch)
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

 from skills.verified_volume_midpoint import VerifiedVolumeMidpointOrders
 from skills.repaired_execution_context import benchmark_exclusion_with_market, dated_market_resolver
 class RepairedOriginal(AccountReservationAudit,ZeroValueShareDelivery,DrawdownControl,CandidateQuality,Era,
                        VerifiedVolumeMidpointOrders,MidpointExitReplay,AccountCandidateHook):
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
 signal_path=I/'signals.json'
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
 identity['trading_exclusions'].append(benchmark_exclusion_with_market(benchmark_exclusion))
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
 events=pd.read_parquet(I/'events.parquet');data=RunInputs(quotes,companies,days,entries,events,ExitSignals(close,days),{},start='2024-01-02')
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
 for module in ('verified_volume_midpoint','repaired_execution_context','volume_profile_account_adapter'):
  mark(ROOT/'skills'/(module+'.py'))
 dated_market=dated_market_resolver(identity)
 mark(PREREG);mark(FROZEN_RUNNER)
 mark(ROOT/'scripts/research_volume_profile_account.py')
 mark(ROOT/'tests/test_volume_profile_account_adapter.py')
 source_snapshots={}
 for source in list(refs):
  if any(token in source for token in ('volume_profile_account','volume_profile_selection',
                                       'volume_profile_data','volume_profile_execution','volume_profile_odd','benchmark_limit_overlay')):
   original=ROOT/source
   snapshot=output/'source-snapshots'/source
   snapshot.parent.mkdir(parents=True,exist_ok=True)
   snapshot.write_bytes(original.read_bytes())
   if sha(snapshot)!=refs[source]:raise ValueError('Source changed before execution snapshot: '+source)
   source_snapshots[source]=dict(path=str(snapshot.relative_to(ROOT)),sha256=sha(snapshot))
   mark(snapshot)
 cases={}
 def replay():
  for arm in arms:
   feeds=Feeds();odds=Odds(study.OddDailyCache(ROOT,inputs/'execution-feeds'));ticks=study.strict.AdditionalTicks()
   corp=Corp(events,C/'dividends',None,offline=True,overrides=overrides)
   cls=RepairedBenchmark if arm=='benchmark' else RepairedOriginal
   opts={} if arm=='benchmark' else dict(ordering='original',position_count=3,factor_mask=0,residual_policy='release',identity_report=identity,exit_signals=data.features,action_dates=[(r.stock_id,calendar.effective(r.stock_id,str(r.event_date))) for r in events.itertuples()])
   profile_queries=[]
   def observed_profile(event):
    profile_queries.append(dict(event_id=event['event_id'],stock_id=event['members'][0],signal_date=event['signal_date']))
    return profile_provider(event)
   if arm!='benchmark':opts.update(candidate_arm='original',drawdown_arm='three_black',black_signals=black_signals,selection_arm=arm,profile_provider=observed_profile if profile_provider else None,candidate_selector=candidate_selector)
   opts.update(ordinary_volumes=ordinary,ordinary_market_resolver=dated_market,volume_policy=volume_policy)
   selected=entries
   engine=cls(quotes,companies,days,selected,feeds,corp,start=data.start,end=data.end,initial_cash=1_000_000,ticks=ticks,participation=.01,liquidity_identity=identity,odd_feeds=odds,**opts)
   corp.preparation_engine=engine
   odds.engine=engine
   if arm!='benchmark':install_pending_share_rights(engine,ROOT)
   print('start',arm,flush=True)
   try:
    account=engine.run();study.old.validate_completed_account(account,[str(d.date()) for d in days],data.start,data.end)
    view,era_audit=normalized_era_account(account)
    if arm=='benchmark':audit=study.old.audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
    else:audit=audit_face_resources(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,quotes)
    routes=study.market_routes([*ticks.queries,*odds.queries])
    routes.update({(r['stock_id'],r['date']):dated_market(r['date'],r['stock_id'])
                  for r in account['orders']})
    checker=audit_midpoint if arm=='benchmark' else audit_midpoint_exit
    audit.update(checker(view,ticks,odds,routes,quotes,days,corp,feeds))
    audit.update(era_audit);audit['verified_halt_observations']=verified_halt_evidence
    if arm!='benchmark':audit['three_black']=audit_three_black(account,black_signals)
    value=dict(completed=True,account=account,summary=study.old.summarize(account),audit=audit);verify_cash(value)
    value['ordinary_capacity_complete']=account.get('ordinary_volume_evidence',{}).get('all_requested_board_capacity_observed',False)
    value['ordinary_evidence_blocked_orders']=account.get('ordinary_volume_evidence',{}).get('blocked_board_children',0)
    account['resource_plans']=deepcopy(engine.resource_plans)
    if arm!='benchmark':
     account['slot_decisions']=deepcopy(engine.slot_decisions)
     account['residual_days']=deepcopy(engine.residual_days)
     account['board_decisions']=deepcopy(engine.board_decisions)
   except (ReplayDataUnavailable,study.old.UnresolvedAction,ValueError,RuntimeError) as exc:
    value=dict(completed=False,summary=None,reason=str(exc),completed_sessions=len(engine.daily),last_date=engine.daily[-1]['date'] if engine.daily else None)
    value['failure_holdings']=deepcopy(engine.holdings)
    value['ordinary_evidence_blocks']=deepcopy(getattr(engine,'volume_evidence_blocks',[]))
    # Preserve the executed prefix for diagnosis, without reporting a partial
    # period as a completed strategy return.
    value['partial_journal']=dict(daily=engine.daily,trades=engine.trades,orders=engine.orders,
     cash_ledger=engine.cash_ledger,corporate_actions=engine.actions,holdings=engine.holding_rows,
     cohorts=engine.cohorts,receivables=engine.receivables,resource_plans=engine.resource_plans,
     selection_decisions=getattr(engine,'selection_decisions',[]),tick_plans=engine.tick_plans,
     day_plans=list(engine.day_plans.values()))
   value['deferred_corporate_preparations']=len(getattr(corp,'deferred_preparations',[]))
   value['profile_queries']=profile_queries
   merge_refs(odds.files)
   path=output/(arm+'.json');write(path,value);cases[arm]={k:v for k,v in value.items() if k not in ('account','audit','partial_journal','ordinary_evidence_blocks')}
   cases[arm].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
   record=dict(timestamp=datetime.now().isoformat(timespec='seconds'),source=SOURCE,params=dict(arm=arm,start=data.start,end=data.end,input_bundle=str(I.relative_to(ROOT)),execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy=volume_policy,data_revision='repaired',execution_fetch=execution_fetch,profile_fetch=bool(profile_provider and profile_provider.online),odd_fetch=odd_fetch,benchmark_limit_overlay=limit_overlay is not None),preparation=PREP,completed=value['completed'],result_path=str(path.relative_to(ROOT)),live_qualified=False)
   append_trial_registry(record,registry_path=output/'trials.jsonl');append_trial_registry(record)
   print(arm,{k:value['summary'][k] for k in ('total_return','max_drawdown','final_nav')} if value.get('summary') else cases[arm],flush=True)
 with patch.object(pending_share_entitlements,'validate_pending_terms',validate_delivery_terms):
  # The old inputs above are independently cache-only. Only the explicitly
  # authorized profile/preparation adapters can call the shared FinMind client.
  network_enabled=execution_fetch or odd_fetch or bool(profile_provider and profile_provider.online)
  with nullcontext() if network_enabled else study.old.offline_only():replay()
 profile_data=None
 if profile_provider is not None:
  profile_data=profile_provider.snapshot(output/'profile-data')
  merge_refs(profile_provider.refs)
  for p in (output/'profile-data').glob('*.json'):mark(p)
 if limit_overlay is not None:merge_refs(limit_overlay.refs)
 if study.old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Run sources changed')
 report=dict(start=data.start,end=data.end,initial_cash=1_000_000,input_bundle=str(I.relative_to(ROOT)),execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy=volume_policy,data_revision='repaired',return_recomputed=not PREP and all(c['completed'] for c in cases.values()),preparation=PREP,cases=cases,source_sha256=refs,all_completed=all(c['completed'] for c in cases.values()),live_qualified=False,actual_fill_verified=False,corporate_fractional_cash_date_verified=False,complete_historical_universe=False,unseen_validation=False,accounting_validated=not PREP and all(c['completed'] for c in cases.values()),validated=False,finmind_requests=None if network_enabled else 0,network_requests=None if network_enabled else 0,
  execution_preparation_attempts=execution_attempts,odd_acquisition_attempts=odd_attempts,
  profile_data=profile_data,source_snapshots=source_snapshots,
  benchmark_limit_overlay=getattr(limit_overlay,'audit_rows',[]),
  candidate_count=sum(data.start<=e['entry_date']<=data.end for e in entries),
  source_signal_count=len(entries),boundary_pre_2024_signals=sum(e['signal_date']<'2024-01-01' and data.start<=e['entry_date']<=data.end for e in entries),
  preregistration=dict(path=str(PREREG.relative_to(ROOT)),sha256=sha(PREREG)),
  elapsed_seconds=round(time.monotonic()-started,3))
 write(output/'report.json',report)
 (output/'report.sha256').write_text(sha(output/'report.json')+'\n')
 return report


def run(output, arms=('original', 'benchmark'), profile_provider=None, candidate_selector=None,
        *, execution_fetch=False, limit_overlay=None, odd_fetch=False):
 from app.file_lock import file_lock
 lock=(file_lock(profile_provider.directory/'.run.lock',timeout=0)
       if profile_provider is not None else nullcontext())
 with lock:
  return _run(output,arms,profile_provider,candidate_selector,
              execution_fetch=execution_fetch,limit_overlay=limit_overlay,odd_fetch=odd_fetch)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--arms',default='original,benchmark')
 parser.add_argument('--profile-fetch',action='store_true',help='Enable only the bounded profile FinMind provider')
 parser.add_argument('--execution-fetch',action='store_true',help='Prepare missing exact execution queries, at most 100 persistent attempts')
 parser.add_argument('--benchmark-limit-overlay',action='store_true',help='Apply the separately evidenced one-date 0050 legal-limit correction')
 parser.add_argument('--odd-fetch',action='store_true',help='Complete necessary odd-lot days using prior successful endpoint probes; at most 50 attempts')
 args=parser.parse_args()
 output_existed=args.output.exists()
 try:
  arms=tuple(args.arms.split(','))
  provider=selector=overlay=None
  if any(arm.startswith('poc_') for arm in arms):
   from skills.volume_profile_data import AccountProfileData
   from skills.volume_profile_selection import select_candidates
   provider=AccountProfileData(ROOT/'.cache/market-input-repair-20261002/inputs-v2',online=args.profile_fetch)
   selector=select_candidates
  elif args.profile_fetch:
   parser.error('--profile-fetch requires a POC arm')
  if args.benchmark_limit_overlay:
   from skills.benchmark_limit_overlay import load_overlay
   overlay=load_overlay(ROOT)
  report=run(args.output,arms,provider,selector,execution_fetch=args.execution_fetch,limit_overlay=overlay,odd_fetch=args.odd_fetch)
 except Exception as exc:
  # A setup/publication failure is visible and counted, without a fake return.
  failure=dict(completed=False,stage='setup_or_publication',reason=str(exc),
               source='volume_profile_account_baseline_20261003',live_qualified=False)
  target=args.output.resolve()
  try:
   target.relative_to(ROOT/'.cache/volume-profile-account-20261003')
   if not output_existed and target.is_dir() and not (target/'report.json').exists():
    write(target/'failure.json',failure)
    append_trial_registry(failure,registry_path=target/'trials.jsonl')
    append_trial_registry(failure)
  except ValueError:
   pass
  raise
 return 0 if report['all_completed'] else 2


if __name__=='__main__':
 raise SystemExit(main())
