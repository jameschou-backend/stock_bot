#!/usr/bin/env python3
"""Explicit one-order data-conflict exclusion from the sealed executable study.

This additive financial body preserves the sealed account rules. The observed
prints support an execution estimate, never proof of our own queue or fills.
No daily HL2 execution method is used. Missing evidence preserves an unfinished
journal without presenting a partial period as a completed return.
"""
from pathlib import Path
import sys,json,time,math,argparse,hashlib
from datetime import datetime
from copy import deepcopy
from unittest.mock import patch
from contextlib import nullcontext
from types import SimpleNamespace
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import research_midpoint as study
from scripts.research_exit_scenarios import read,write,sha,RunInputs
from skills.replay_market_feeds import parse_odd,ReplayDataUnavailable
from skills.scenario_exit_replay import ExitSignals
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.midpoint_replay import MidpointBenchmark
from scripts.export_midpoint_2025_report import verify_cash
from skills.strict_tick_inputs import restore_halt_zeroes

OLD=ROOT/'.cache/partial-risk-2019-20260929'; I=OLD/'inputs-final'
B=ROOT/'.cache/stock-universe-2019-20260929'; C=B/'execution-v1'
from skills.candidate_quality import CandidateQuality
from skills.three_black_exit import ThreeBlackControl as DrawdownControl, ThreeBlackSignals, audit_three_black
from skills.volume_profile_account_adapter import AccountCandidateHook, AccountReservationAudit
from skills.candle_volume_rules import CandleVolumeSignals, CandleVolumeExit, audit_candle_volume, audit_entry_gate
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
from skills.historical_odd_regime import HistoricalOddEra

RUN_ROOT=ROOT/'.cache/poc-executable-skip-20261004'
INPUTS=ROOT/'.cache/poc-latest-20261003/inputs-v1'
START='2024-01-02';END='2026-10-02'
FROZEN_RUNNER=ROOT/'scripts/replay_repaired_market_inputs.py'
ARMS=('poc_red_executable','benchmark_executable')
ARM_RULES={'poc_red_executable':(True,'none'),'benchmark_executable':(False,'none')}
PREREG=ROOT/'docs/prereg_poc_executable_skip_20261004.md'
STRICT_CASE=ROOT/'.cache/poc-executable-20261004/online-v2/poc_red_executable.json'
STRICT_CASE_SHA='fc45bd29bc55ba25621bce078c246562c0a212845fc5e3e109f5a99c92508b14'
SKIP_BINDINGS={
 'scripts/research_poc_executable_account.py':'35b72fdb754bffd65221341556277a0777c7673b37b76b7be9071f2e882fadaf',
 'skills/poc_executable_replay.py':'52f81fca7b02ae997bc7ec66f079aa6aead0f885020291ae3710be08f5335d01',
 '.cache/poc-executable-20261004/online-v2/poc_red_executable.json':STRICT_CASE_SHA,
 '.cache/poc-executable-20261004/online-v2/benchmark_executable.json':'6c67a34efd0fda23189469e7c352b599d5f91fb9e785a6d02b862b1f0b9d7b08',
 '.cache/poc-executable-20261004/board-v1/receipts/3230-2024-10-16.json':'b62ffd027fcf5daf61f5f2e92ab9b96726d7d046603d0cf57fbb9db002b53712',
 '.cache/poc-executable-20261004/board-v1/raw/3230-2024-10-16.parquet':'54be8276f2530a2c5b7ea2d915353acfb8388fb1295205b10379d95bc25541d7',
 '.cache/board-tape-reconciliation-20260925/official/tpex-no1430-2024-10-16.json':'d6d1b7a8d5558e2677aaa80a4b08e00c3a32686b373b4596422648ae50b71aa9',
}


def bind_skip_sources(provider):
 """Reuse the same bounded acquisition provider; add an experiment registration."""
 for name,expected in SKIP_BINDINGS.items():provider._bind(ROOT/name,expected)
 provider.prereg_path=PREREG
 provider.prereg_sha256=sha(PREREG)
 provider._bind(PREREG,provider.prereg_sha256)
 return provider

BASE_REPORT=ROOT/'.cache/poc-latest-20261003/run-v3/report.json'
BASE_REPORT_SHA='f82dbcf728199ead57bc41eeea7463acc6c2c78f26cfa490deef4a9735a3623c'
BASE_CASE_SHA='7657f95c7633a322582fd53cc3aeddb6edb9ab07c06aae142ae4a8cd18375973'
from scripts.research_poc_latest_account import (
 canonical_sha,merge_source_refs,validate_source_paths,RunSourceCache,
 load_candidate_bundle,verify_entry_prefix,merge_overrides,merge_supplements,
 FROZEN_BINDINGS as PARENT_BINDINGS)
FROZEN_BINDINGS=dict(PARENT_BINDINGS, **{
 'scripts/research_poc_latest_account.py':'97371cb0930f42d81903e79bee0e50afd668e4a6f59c661c70ca609cfeafbaec'})


def load_signal_reference(root=ROOT):
 """Bind old signals/gate evidence, not old cash paths as a new fill oracle."""
 root=Path(root).resolve();cache=RunSourceCache(root)
 report=cache.read_json(BASE_REPORT,BASE_REPORT_SHA)
 if not report.get('all_completed') or (report.get('start'),report.get('end'),report.get('initial_cash'))!=(START,END,1_000_000):
  raise ValueError('Sealed signal reference scope differs')
 item=report['cases']['poc_red'];path=(root/item['path']).resolve()
 if not item['completed'] or item['sha256']!=BASE_CASE_SHA:
  raise ValueError('Sealed POC/red signal reference changed')
 case=cache.read_json(path,item['sha256'])
 if not case.get('completed'):raise ValueError('Incomplete POC/red signal reference')
 for name,expected in report['source_sha256'].items():cache.verify(root/name,expected)
 refs=dict(report['source_sha256'])
 refs[str(BASE_REPORT.relative_to(root))]=BASE_REPORT_SHA;refs[item['path']]=item['sha256']
 return case,refs


def select_arm_entries(arm,entries,signals,start,end):
 if arm not in ARMS:raise ValueError('Unregistered executable account arm')
 if arm=='benchmark_executable':return entries,[]
 scoped=[deepcopy(e) for e in entries if start<=e['entry_date']<=end]
 expected=deepcopy(scoped)
 kept,decisions=signals.filter_entries(scoped);ids={e['event_id'] for e in kept}
 if len(ids)!=len(kept) or kept!=[e for e in expected if e['event_id'] in ids]:
  raise ValueError('Red gate changed candidate identity, data or order')
 return [e for e in entries if not start<=e['entry_date']<=end or e['event_id'] in ids],decisions


def compare_signal_gate(decisions,reference):
 if decisions!=reference['entry_gate_decisions']:
  raise ValueError('Executable case changed the sealed red-candle gate')
 return dict(all_exact=True,through=END,entry_gate_sha256=canonical_sha(decisions),
             compared='all original red-candle decisions; candidate bundle is hash-bound',
             account_equality_required=False,reason='execution policy intentionally changed')


def preserve_failure(exc,engine,stage='arm_execution'):
 """Keep diagnostics and unknown distinct from a zero or partial return."""
 out=dict(completed=False,summary=None,reason=str(exc),stage=stage,
          completed_sessions=len(engine.daily) if engine is not None else 0,
          last_date=engine.daily[-1]['date'] if engine is not None and engine.daily else None)
 if engine is not None:
  out['failure_holdings']=deepcopy(engine.holdings)
  out['partial_journal']=deepcopy(dict(daily=engine.daily,trades=engine.trades,orders=engine.orders,
      cash_ledger=engine.cash_ledger,corporate_actions=engine.actions,holdings=engine.holding_rows,
      cohorts=engine.cohorts,receivables=engine.receivables,resource_plans=engine.resource_plans,
      selection_decisions=getattr(engine,'selection_decisions',[]),tick_plans=engine.tick_plans,
      day_plans=list(engine.day_plans.values()),
      explicit_exclusions=getattr(engine,'explicit_exclusions',[]),
      slot_decisions=getattr(engine,'slot_decisions',[]),
      board_decisions=getattr(engine,'board_decisions',[]),
      residual_days=getattr(engine,'residual_days',[])))
 return out


def odd_plan_context(engine,snapshots,day,sid,market):
 """Audit an old request with its original plan, never the engine's last day."""
 key=(day,sid,market.upper())
 if key not in snapshots:
  values=getattr(engine,'day_plans',None)
  if not isinstance(values,dict):raise ValueError('Odd request needs an existing planned order')
  matching={k:deepcopy(p) for k,p in values.items()
            if p.get('date')==day and p.get('stock_id')==sid
            and p.get('planned_qty',0)>0 and p.get('odd_qty',0)>0}
  if not matching:raise ValueError('Odd request lacks a precommitted positive odd quantity')
  snapshots[key]=matching
 return SimpleNamespace(day_plans=deepcopy(snapshots[key]))


def engine_types(execution_orders,era):
 """Place the new planner/executor ahead of every dormant HL2 ancestor."""
 from skills.verified_volume_midpoint import VerifiedVolumeMidpointOrders
 class RepairedOriginal(execution_orders,AccountReservationAudit,ZeroValueShareDelivery,
                        DrawdownControl,CandidateQuality,era,VerifiedVolumeMidpointOrders,
                        MidpointExitReplay,AccountCandidateHook):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.corporate=FaceValueCapitalActions(self.corporate.provider,self)
 class RepairedBenchmark(execution_orders,era,VerifiedVolumeMidpointOrders,MidpointBenchmark):pass
 class CandleVolumeAccount(CandleVolumeExit,RepairedOriginal):pass
 return CandleVolumeAccount,RepairedBenchmark


# Financial body statically adapted from the frozen latest runner. Only the
# execution adapter, corresponding audit and experiment publication differ.
def _run(output, arms, data_provider, *, limit_overlay=None):
 started=time.monotonic()
 if not arms or len(set(arms))!=len(arms) or any(x not in ARMS for x in arms):
  raise ValueError('Only the two preregistered executable arms are allowed')
 output=Path(output).resolve();output.relative_to(RUN_ROOT)
 if output.exists():raise ValueError('Choose a new output directory')
 if Path(data_provider.bundle).resolve()!=INPUTS:raise ValueError('Unexpected extension bundle')
 data_provider.verify_sources()
 prereg=Path(data_provider.prereg_path).resolve()
 prereg.relative_to(ROOT)
 if prereg!=PREREG or sha(prereg)!=data_provider.prereg_sha256:raise ValueError('Latest preregistration changed')
 for name,expected in (FROZEN_BINDINGS | SKIP_BINDINGS).items():
  if sha(ROOT/name)!=expected:raise ValueError('Changed frozen financial source: '+name)
 reference,reference_refs=load_signal_reference(ROOT)
 output.mkdir(parents=True)
 source_snapshot=output/'runner_source.py';source_snapshot.write_bytes(Path(__file__).read_bytes())
 I=INPUTS;C=RUN_ROOT/'execution-v1';SOURCE='poc_executable_skip_20261004'
 # Only initializes the dormant midpoint ancestor. ExecutableOrders never calls
 # its matcher; current-day capacity comes exclusively from the bound tape.
 volume_policy='legacy_total_research';PREP=False
 refs=dict(FROZEN_BINDINGS)
 def merge_refs(values):
  merge_source_refs(refs,values,ROOT)
 def mark(p):merge_refs({str(p.relative_to(ROOT)):sha(p)})
 merge_refs(data_provider.refs);merge_refs(SKIP_BINDINGS);mark(prereg)
 strict_partial=read(STRICT_CASE)
 merge_refs(reference_refs)
 if limit_overlay is not None:merge_refs(limit_overlay.refs);mark(ROOT/'skills/benchmark_limit_overlay.py')
 from skills.volume_profile_selection import select_candidates
 candidate_selector=select_candidates
 def finmind(sid,dataset):
  result=data_provider.finmind(sid,dataset)
  merge_refs(data_provider.refs)
  if not result.empty:
   if set(result.stock_id)!={sid} or not pd.to_datetime(result.date).between('2018-01-01',END).all():
    raise ValueError('Latest execution source stock/date differs')
  return result

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
  def __init__(self):self.files={};self.queries=[];self.engine=None;self.plan_snapshots={}
  def get_odd(self,day,sid,market):
   self.queries.append(dict(stock_id=sid,date=day,market=market.upper()))
   context=odd_plan_context(self.engine,self.plan_snapshots,day,sid,market)
   result=data_provider.get_odd(day,sid,market,engine=context)
   merge_refs(data_provider.refs);self.files.update(data_provider.refs)
   return result

 class CorpLoader(study.old.TrackedCorporateActions):
  def prepare(self,sid):
   if sid not in self.loaded:
    f=finmind(sid,'TaiwanStockDividend');p=self.directory/(sid+'.parquet')
    ensure_dividend_copy(f,p,prepare=False)
   result=super().prepare(sid)
   self.loaded[sid]=complete_cash_dividends(self.loaded[sid],sid,cash_supplements)
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
   result=super().manifest();result.update(start='2018-01-01',end=END);return result

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

 from skills.poc_executable_replay import ExecutableOrders,audit_executable
 from skills.poc_executable_skip import ExplicitConflictSkipOrders,audit_executable_skip,verify_strict_prefix,audit_explicit_exclusion
 from skills.repaired_execution_context import benchmark_exclusion_with_market, dated_market_resolver
 CandleVolumeAccount,_=engine_types(ExplicitConflictSkipOrders,Era)
 _,RepairedBenchmark=engine_types(ExecutableOrders,Era)

 manifest=read(I/'manifest.json')
 for name,h in manifest['files_sha256'].items():
  if sha(I/name)!=h:raise ValueError('Input changed '+name)
 merge_refs(manifest['source_sha256']);merge_refs({str((I/n).relative_to(ROOT)):h for n,h in manifest['files_sha256'].items()});mark(I/'manifest.json');mark(source_snapshot)
 print('verify frozen execution context',flush=True)
 mark(ROOT/'skills/candidate_execution_context.py')
 merge_refs(study.old.file_identities([ROOT/'skills/historical_odd_regime.py',ROOT/'skills/partial_risk.py',ROOT/'skills/partial_risk_audit.py',ROOT/'skills/midpoint_exit_replay.py',ROOT/'skills/midpoint_exit_audit.py',ROOT/'docs/prereg_partial_risk_2019_20260929.md',*study.CODE],ROOT))
 signal_path=I/'signals.json'
 prepared=read(signal_path);merge_refs(prepared.get('source_sha256',{}));mark(signal_path)
 mark(ROOT/'docs/prereg_stock_universe_2019_20260929.md');mark(ROOT/'skills/stock_universe_2019.py')
 mark(ROOT/'skills/deferred_corporate.py')
 mark(ROOT/'skills/three_black_exit.py');mark(ROOT/'docs/prereg_three_black_exit_20261001.md');mark(ROOT/'docs/prereg_market_input_repair_20261002.md')
 mark(ROOT/'skills/zero_value_share_delivery.py')
 mark(ROOT/'skills/frozen_dividend_copy.py')
 entries,pending_entries=load_candidate_bundle(I,END)
 verify_entry_prefix(entries)
 pool=sorted({'0050'}|{e['members'][0] for e in entries})
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
 events=pd.read_parquet(I/'events.parquet');data=RunInputs(quotes,companies,days,entries,events,ExitSignals(close,days),{},start=START,end=END)
 action_dates=[(r.stock_id,calendar.effective(r.stock_id,str(r.event_date))) for r in events.itertuples()]
 candle_signals=CandleVolumeSignals(black_signals,entries,action_dates)
 extra,extra_refs=study.load_exit_completion(ROOT);merge_refs(extra_refs)
 additions=study.old.parent.parent.parent.load_corporate_completion(ROOT)|study.old.load_capital_terms(ROOT)[0]|extra
 overrides=(study.old.read(study.old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides']|study.old.read(study.old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides']|study.old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides']|additions)
 terms=read(ROOT/'docs/partial_risk_2019_corporate_terms.json')
 overrides |= terms['overrides'];merge_refs(terms['source_sha256']);mark(ROOT/'docs/partial_risk_2019_corporate_terms.json')
 supplement=read(ROOT/'docs/candidate_quality_corporate_terms.json')
 if data.end>data_provider.latest_supported_account_end:raise ValueError('Latest corporate evidence does not cover account end')
 cash_supplements=merge_supplements(supplement['cash_supplements'],data_provider.cash_supplements)
 overrides |= supplement['overrides'];merge_refs(supplement['source_sha256'])
 mark(ROOT/'docs/candidate_quality_corporate_terms.json');mark(ROOT/'skills/candidate_corporate.py')
 for row in cash_supplements:
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
 overrides=merge_overrides(overrides,data_provider.corporate_overrides)
 ordinary={}
 for module in ('verified_volume_midpoint','repaired_execution_context','volume_profile_account_adapter'):
  mark(ROOT/'skills'/(module+'.py'))
 dated_market=dated_market_resolver(identity)
 mark(prereg);mark(FROZEN_RUNNER)
 mark(ROOT/'scripts/research_volume_profile_account.py')
 mark(ROOT/'tests/test_volume_profile_account_adapter.py')
 for name in ('scripts/research_poc_executable_skip.py','skills/poc_executable_skip.py',
              'tests/test_poc_executable_skip.py','tests/test_poc_executable_skip_account.py',
              'scripts/research_poc_executable_account.py','tests/test_poc_executable_account.py',
              'skills/poc_executable_replay.py','skills/poc_executable_data.py',
              'skills/poc_executable_odd.py','tests/test_poc_executable_replay.py',
              'tests/test_poc_executable_data.py','tests/test_poc_executable_odd.py',
              'scripts/research_poc_latest_account.py','tests/test_poc_latest_account.py',
              'scripts/research_candle_volume_account.py','tests/test_candle_volume_account.py',
              'skills/candle_volume_rules.py','tests/test_candle_volume_rules.py'):
  mark(ROOT/name)
 source_snapshots={}
 for source in list(refs):
  if any(token in source for token in ('volume_profile_account','volume_profile_selection',
                                       'volume_profile_data','volume_profile_execution','volume_profile_odd','benchmark_limit_overlay',
                                       'candle_volume_account','candle_volume_rules','poc_latest','poc_executable')):
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
   feeds=Feeds();odds=Odds();ticks=data_provider.board_ticks
   corp=Corp(events,Path(data_provider.dividend_directory),None,offline=True,overrides=overrides)
   cls=RepairedBenchmark if arm=='benchmark_executable' else CandleVolumeAccount
   red_gate,volume_mode=ARM_RULES[arm]
   opts={} if arm=='benchmark_executable' else dict(ordering='original',position_count=3,factor_mask=0,residual_policy='release',identity_report=identity,exit_signals=data.features,action_dates=action_dates)
   profile_queries=[]
   def observed_profile(event):
    profile_queries.append(dict(event_id=event['event_id'],stock_id=event['members'][0],signal_date=event['signal_date']))
    return data_provider.profile(arm,event)
   if arm!='benchmark_executable':opts.update(candidate_arm='original',drawdown_arm='three_black',black_signals=black_signals,selection_arm='poc_priority_available' if arm.startswith('poc_') else 'original',profile_provider=observed_profile if arm.startswith('poc_') else None,candidate_selector=candidate_selector,candle_volume_signals=candle_signals,volume_exit_mode=volume_mode)
   opts.update(ordinary_volumes=ordinary,ordinary_market_resolver=dated_market,volume_policy=volume_policy)
   engine=None;entry_gate_decisions=[];signal_parity=None
   try:
    selected,entry_gate_decisions=select_arm_entries(arm,entries,candle_signals,data.start,data.end)
    if red_gate:signal_parity=compare_signal_gate(entry_gate_decisions,reference)
    engine=cls(quotes,companies,days,selected,feeds,corp,start=data.start,end=data.end,initial_cash=1_000_000,ticks=ticks,participation=.01,liquidity_identity=identity,odd_feeds=odds,**opts)
    corp.preparation_engine=engine
    odds.engine=engine
    if arm!='benchmark_executable':install_pending_share_rights(engine,ROOT)
    print('start',arm,flush=True)
    account=engine.run();study.old.validate_completed_account(account,[str(d.date()) for d in days],data.start,data.end)
    if arm=='benchmark_executable':audit=study.old.audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
    else:audit=audit_face_resources(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,quotes)
    routes=study.market_routes([*ticks.queries,*odds.queries])
    routes.update({(r['stock_id'],r['date']):dated_market(r['date'],r['stock_id'])
                  for r in account['orders']})
    account['resource_plans']=deepcopy(engine.resource_plans)
    if arm!='benchmark_executable':
     account['slot_decisions']=deepcopy(engine.slot_decisions)
     account['board_decisions']=deepcopy(engine.board_decisions)
     account['residual_days']=deepcopy(engine.residual_days)
     audit['strict_prefix']=verify_strict_prefix(account,strict_partial)
    execution_auditor=audit_executable if arm=='benchmark_executable' else audit_executable_skip
    exclusion_audit_opts={} if arm=='benchmark_executable' else dict(strict_partial=strict_partial)
    audit.update(execution_auditor(account,ticks,odds,routes,quotes,days,corp,feeds,verified_halts=identity['trading_exclusions'],**exclusion_audit_opts))
    audit['verified_halt_observations']=verified_halt_evidence
    if arm!='benchmark_executable':audit['three_black']=audit_three_black(account,black_signals)
    policy_audit=audit_candle_volume(account,candle_signals,volume_mode) if volume_mode!='none' else None
    gate_audit=audit_entry_gate(account,candle_signals) if red_gate else None
    account.pop('volume_exit_log',[])
    account['settings'].pop('volume_exit_mode',None)
    value=dict(completed=True,account=account,summary=study.old.summarize(account),audit=audit,
     rule_audit=dict(entry_gate=gate_audit,volume_exit=policy_audit));verify_cash(value)
    account['resource_plans']=deepcopy(engine.resource_plans)
    if arm!='benchmark_executable':
     account['slot_decisions']=deepcopy(engine.slot_decisions)
     account['residual_days']=deepcopy(engine.residual_days)
     account['board_decisions']=deepcopy(engine.board_decisions)
   except (ReplayDataUnavailable,study.old.UnresolvedAction,ValueError,RuntimeError) as exc:
    entry_gate_decisions=getattr(exc,'decisions',entry_gate_decisions)
    value=preserve_failure(exc,engine,'arm_initialization' if engine is None else 'arm_execution')
   if arm!='benchmark_executable':
    value['explicit_exclusions']=deepcopy(getattr(engine,'explicit_exclusions',[]))
    value['posthoc_data_exclusion']=True
    if not value['completed'] and value.get('last_date','') and value['last_date']>='2024-10-15':
     value['strict_prefix']=verify_strict_prefix(value['partial_journal'],strict_partial)
     if value['last_date']>='2024-10-16':
      value['exclusion_audit']=audit_explicit_exclusion(value['partial_journal'],strict_partial)
   value['deferred_corporate_preparations']=len(getattr(corp,'deferred_preparations',[]))
   value['profile_queries']=profile_queries
   value['family_rules']=dict(red_gate=red_gate,volume_exit_mode=volume_mode,
    anchor='poc_priority_available' if arm.startswith('poc_') else arm)
   value['entry_gate_decisions']=entry_gate_decisions
   value['volume_exit_log']=deepcopy(getattr(engine,'volume_exit_log',[]))
   value['signal_parity']=signal_parity
   value.update(live_qualified=False,actual_fill_verified=False,unseen_validation=False)
   merge_refs(odds.files)
   path=output/(arm+'.json');write(path,value);cases[arm]={k:v for k,v in value.items() if k not in ('account','audit','partial_journal','ordinary_evidence_blocks','entry_gate_decisions','volume_exit_log')}
   cases[arm].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
   record=dict(timestamp=datetime.now().isoformat(timespec='seconds'),source=SOURCE,params=dict(arm=arm,start=data.start,end=data.end,input_bundle=str(I.relative_to(ROOT)),execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy='observed_provider_tape_1pct_prior_total_adv20',data_revision='explicit_3230_20241016_buy_exclusion',fetch=bool(data_provider.online),benchmark_limit_overlay=limit_overlay is not None),preparation=PREP,completed=value['completed'],status='completed' if value['completed'] else 'incomplete',result_path=str(path.relative_to(ROOT)),live_qualified=False,actual_fill_verified=False,unseen_validation=False)
   append_trial_registry(record,registry_path=output/'trials.jsonl');append_trial_registry(record)
   print(arm,{k:value['summary'][k] for k in ('total_return','max_drawdown','final_nav')} if value.get('summary') else {k:cases[arm].get(k) for k in ('completed','reason','last_date','completed_sessions')},flush=True)
 with patch.object(pending_share_entitlements,'validate_pending_terms',validate_delivery_terms):
  with nullcontext() if data_provider.online else study.old.offline_only():replay()
 profile_data=data_provider.profile_snapshot(output/'profile-data')
 merge_refs(data_provider.refs)
 for p in (output/'profile-data').rglob('*.json'):mark(p)
 if limit_overlay is not None:merge_refs(limit_overlay.refs)
 data_provider.verify_sources()
 validate_source_paths(refs,ROOT)
 if study.old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Run sources changed')
 report=dict(start=data.start,end=data.end,initial_cash=1_000_000,input_bundle=str(I.relative_to(ROOT)),
  execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy='observed_provider_tape_1pct_prior_total_adv20',
  data_revision='explicit_3230_20241016_buy_exclusion',return_recomputed=all(c['completed'] for c in cases.values()),
  preparation=False,cases=cases,source_sha256=refs,all_completed=all(c['completed'] for c in cases.values()),
  live_qualified=False,actual_fill_verified=False,corporate_fractional_cash_date_verified=False,
  complete_historical_universe=False,unseen_validation=False,
  posthoc_data_exclusion=True,original_strategy_fully_verified=False,
  exclusion_scope=dict(stock_id='3230',date='2024-10-16',side='buy',event_id='liquid_universe-2024-10-15-3230',replacement_buy_allowed=False),
  accounting_validated=all(c['completed'] for c in cases.values()),validated=False,
  network_requests=getattr(data_provider,'network_calls',None) if data_provider.online else 0,
  finmind_requests=getattr(data_provider,'finmind_calls',None) if data_provider.online else 0,
  profile_data=profile_data,source_snapshots=source_snapshots,
  signal_validation=dict(through=END,required=True,compared='sealed candidates and red-candle gate',account_equality_required=False),
  execution_model=dict(ordinary='chronological_strict_trade_through_participation',odd='after_hours_single_auction_estimate',
      complete_exchange_tape_verified=False,actual_queue_verified=False,actual_broker_fills_verified=False),
  registered_arms=list(ARMS),requested_arms=list(arms),
  all_registered_completed=set(cases)==set(ARMS) and all(c['completed'] for c in cases.values()),
  sealed_hl2_reference=dict(path=str(BASE_REPORT.relative_to(ROOT)),sha256=BASE_REPORT_SHA,
      used_for='signal/gate parity only; not an executable return'),
  benchmark_limit_overlay=getattr(limit_overlay,'audit_rows',[]),
  candidate_count=sum(data.start<=e['entry_date']<=data.end for e in entries),
  source_signal_count=len(entries)+len(pending_entries),pending_terminal_signals=pending_entries,
  boundary_pre_2024_signals=sum(e['signal_date']<START and data.start<=e['entry_date']<=data.end for e in entries),
  preregistration=dict(path=str(prereg.relative_to(ROOT)),sha256=sha(prereg)),
  elapsed_seconds=round(time.monotonic()-started,3))
 write(output/'report.json',report)
 (output/'report.sha256').write_text(sha(output/'report.json')+'\n')
 return report


def run(output,arms,data_provider,*,limit_overlay=None):
 from app.file_lock import file_lock
 with file_lock(RUN_ROOT/'.run.lock',timeout=0):
  return _run(output,arms,data_provider,limit_overlay=limit_overlay)


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--arms',default=','.join(ARMS))
 parser.add_argument('--fetch',action='store_true',help='Allow only the bounded executable-evidence adapter')
 parser.add_argument('--benchmark-limit-overlay',action='store_true')
 args=parser.parse_args(argv)
 output_existed=args.output.exists()
 try:
  from skills.poc_executable_data import ExecutableAccountData
  provider=bind_skip_sources(ExecutableAccountData(ROOT,online=args.fetch))
  overlay=None
  if args.benchmark_limit_overlay:
   from skills.benchmark_limit_overlay import load_overlay
   overlay=load_overlay(ROOT)
  report=run(args.output,tuple(args.arms.split(',')),provider,limit_overlay=overlay)
 except Exception as exc:
  failure=dict(completed=False,summary=None,stage='setup_or_publication',reason=str(exc),
               source='poc_executable_skip_20261004',arms=args.arms.split(','),live_qualified=False)
  target=args.output.resolve()
  try:
   target.relative_to(RUN_ROOT)
   if not output_existed and not (target/'report.json').exists():
    write(target/'failure.json',failure)
    append_trial_registry(failure,registry_path=target/'trials.jsonl');append_trial_registry(failure)
  except ValueError:pass
  raise
 return 0 if report['all_completed'] else 2


if __name__=='__main__':
 raise SystemExit(main())
