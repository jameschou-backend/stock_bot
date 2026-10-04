#!/usr/bin/env python3
"""Additive broker-entry gates on the sealed three-stock POC/red account.

Financial execution is statically adapted from research_poc_latest_account.py.
No frozen source is rewritten or executed as text. Missing execution evidence
stops the affected arm and preserves its journal; no partial-return promotion.
"""
from pathlib import Path
import sys,json,time,math,argparse,hashlib
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
from skills.historical_odd_regime import HistoricalOddEra, normalized_era_account, parse_after_hours

RUN_ROOT=ROOT/'.cache/poc-broker-account-20261004'
INPUTS=ROOT/'.cache/poc-latest-20261003/inputs-v1'
START='2024-01-02';END='2026-10-02';BOUNDARY='2026-09-09'
FROZEN_RUNNER=ROOT/'scripts/replay_repaired_market_inputs.py'
ARMS=('poc_red','poc_persist_guard','poc_combined_guard','poc_known5_control',
      'poc_known5_filter','poc_known5_combined')
ARM_RULES={a:(True,'none') for a in ARMS}
BASE_REPORT=ROOT/'.cache/poc-latest-20261003/run-v3/report.json'
BASE_REPORT_SHA='f82dbcf728199ead57bc41eeea7463acc6c2c78f26cfa490deef4a9735a3623c'
BASE_CASE_SHA='7657f95c7633a322582fd53cc3aeddb6edb9ab07c06aae142ae4a8cd18375973'
PREREG=ROOT/'docs/prereg_poc_broker_account_20261004.md'
PREFIX_PARENTS={
 'anchors-v2':'4d670409c617d2d562f60630dcafaef40c775598878a2b920415ba10c0a2aef2',
 'full-v1':'2b33236460533746c4fab4f2e65a9dd3df04efb2ab0bd5031b61a1c4d8e37a55'}
FROZEN_BINDINGS={
 'scripts/research_poc_latest_account.py':'97371cb0930f42d81903e79bee0e50afd668e4a6f59c661c70ca609cfeafbaec',
 'scripts/research_candle_volume_account.py':'43be2f40492c474475aa2ece1b0dcef4064fa1190e682edb20e87e42a0a00b89',
 'scripts/research_volume_profile_account.py':'576668c79215f97fb82e3ed2aaaaead44b8cb6f08a15b6f70eae60c6fa94ed23',
 'scripts/replay_repaired_market_inputs.py':'199e171d5433760039ffdae1152cabe633b427b98c60e88d3c1d72fd0735ce26',
 'skills/candle_volume_rules.py':'470f17c0643faf7e436ca740a21b01b428ce6aa97704d2e5f46717dd9d8d7600',
 'skills/three_black_exit.py':'b03d2209c3d35c6fefac285d500888efa555a0222a913f01c58687f9dee8afa3',
 '.cache/market-input-repair-20261002/inputs-v2/manifest.json':'f62dbe32d263abb464c9f283879b7d6340b0667b6fa9e99404c6ece7e04663d6',
 '.cache/market-input-repair-20261002/inputs-v2/signals.json':'f0821897f077d616494bcaafe2b2278df27666c03466758642c80784f7542513',
}
JOURNALS=('daily','trades','orders','cash_ledger','corporate_actions','holdings',
          'resource_plans','slot_decisions','residual_days','board_decisions',
          'selection_decisions','tick_plans','base_tick_plans','black_log')


def canonical_sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def merge_source_refs(refs,values,root=ROOT):
 """Resolve each new source once; retain hash-conflict checks on every merge."""
 for key,value in values.items():
  if key in refs:
   if refs[key]!=value:raise ValueError('Conflicting frozen source hash '+key)
   continue
  if not (root/key).resolve().is_relative_to(root):raise ValueError('Source escapes repository '+key)
  refs[key]=value


def validate_source_paths(refs,root=ROOT):
 # Recheck every path at publication, including a symlink changed after binding.
 for key in refs:
  if not (root/key).resolve().is_relative_to(root):raise ValueError('Source escapes repository '+key)


class RunSourceCache:
 """One invocation's verified bytes, invalidated by any filesystem metadata change."""
 def __init__(self,root):
  self.root=Path(root).resolve();self.verified={};self.json_values={}

 @staticmethod
 def fingerprint(path):
  stat=path.stat()
  return stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns

 def verify(self,path,expected):
  path=Path(path).resolve();path.relative_to(self.root)
  before=self.fingerprint(path);prior=self.verified.get(path)
  if prior is not None and prior[1]!=expected:
   raise ValueError('Conflicting sealed source hash: '+str(path))
  if prior is not None and prior[0]==before:return expected
  actual=sha(path)
  if self.fingerprint(path)!=before:
   raise ValueError('Sealed source changed during verification: '+str(path))
  if actual!=expected:raise ValueError('Sealed source hash changed: '+str(path))
  self.verified[path]=(before,actual)
  self.json_values.pop(path,None)
  return actual

 def read_json(self,path,expected):
  path=Path(path).resolve();self.verify(path,expected)
  if path not in self.json_values:
   value=read(path)
   if self.fingerprint(path)!=self.verified[path][0]:
    raise ValueError('Sealed source changed during JSON read: '+str(path))
   self.json_values[path]=value
  return deepcopy(self.json_values[path])


def executable_entries(prepared,end=END):
 rows=prepared['entries']
 if isinstance(rows,dict):rows=rows['median50m']
 seen=set();active=[];pending=[]
 for row in rows:
  eid=row['event_id']
  if eid in seen:raise ValueError('Duplicate candidate identity')
  seen.add(eid)
  if row.get('entry_date') is None:
   if row['signal_date']!=end:raise ValueError('Only the last signal session may await entry')
   pending.append(deepcopy(row))
  else:
   if not row['signal_date']<row['entry_date']<=end:raise ValueError('Invalid executable candidate dates')
   active.append(deepcopy(row))
 return active,pending


def load_candidate_bundle(bundle,end=END):
 """Read both sealed partitions; the last close must never disappear silently."""
 bundle=Path(bundle)
 manifest=read(bundle/'manifest.json')
 if manifest.get('end')!=end:raise ValueError('Candidate bundle end changed')
 for name in ('signals.json','pending-signals.json'):
  expected=manifest.get('files_sha256',{}).get(name)
  if not expected or sha(bundle/name)!=expected:raise ValueError('Candidate partition missing or changed: '+name)
 prepared=read(bundle/'signals.json')
 if prepared.get('pending_signal_file')!='pending-signals.json':
  raise ValueError('Candidate bundle must declare its terminal partition')
 terminal=read(bundle/'pending-signals.json')
 if (terminal.get('schema')!='pending_last_close_signals_v1'
     or terminal.get('signal_date')!=end or terminal.get('next_session_observed') is not False
     or terminal.get('execution_inferred') is not False or terminal.get('live_qualified') is not False):
  raise ValueError('Terminal partition must await an unobserved next session')
 active,unexpected=executable_entries(prepared,end)
 if unexpected:raise ValueError('Executable partition contains terminal signals')
 unexpected,pending=executable_entries(terminal,end)
 if unexpected:raise ValueError('Terminal partition contains executable signals')
 # Check duplicate identities across the two files as well as within each file.
 executable_entries({'entries':active+pending},end)
 if (manifest.get('executable_candidate_count')!=len(active)
     or manifest.get('pending_candidate_count')!=len(pending)):
  raise ValueError('Candidate partition counts differ from manifest')
 return active,pending


def verify_entry_prefix(entries,root=ROOT):
 path=root/'.cache/market-input-repair-20261002/inputs-v2/signals.json'
 if sha(path)!=FROZEN_BINDINGS[str(path.relative_to(root))]:raise ValueError('Old candidate source changed')
 old=read(path)['entries']['median50m']
 if [e for e in entries if e['entry_date']<=BOUNDARY]!=old:
  raise ValueError('Extension changes the original candidate prefix')


def merge_overrides(original,additions):
 result=deepcopy(original)
 for key,value in additions.items():
  if key in result and result[key]!=value:raise ValueError('Latest corporate terms change old action: '+key)
  result[key]=deepcopy(value)
 return result


def merge_supplements(original,additions):
 result=deepcopy(original)
 index={(r['stock_id'],r['date']):r for r in result}
 for row in additions:
  key=row['stock_id'],row['date']
  if key in index and index[key]!=row:raise ValueError('Latest cash supplement changes old action')
  if key not in index:result.append(deepcopy(row));index[key]=row
 return result


def select_arm_entries(arm,entries,signals,start,end):
 if arm not in ARM_RULES:raise ValueError('Unregistered latest account arm')
 if not ARM_RULES[arm][0]:return entries,[]
 scoped=[e for e in entries if start<=e['entry_date']<=end]
 kept,decisions=signals.filter_entries(scoped);identities={e['event_id'] for e in kept}
 if len(identities)!=len(kept) or kept!=[e for e in scoped if e['event_id'] in identities]:
  raise ValueError('Entry gate changed candidate identity, data or order')
 return [e for e in entries if not start<=e['entry_date']<=end or e['event_id'] in identities],decisions


def select_account_entries(arm,entries,signals,evidence,start,end,*,gate_fn=None):
 """Apply the completed-T red gate, then a frozen, order-preserving broker gate.

 The broker ledger is separate from the candle ledger. Events outside the
 account's execution window remain unchanged; pending terminal signals are
 already in their separate sealed partition and never arrive here.
 """
 red,red_decisions=select_arm_entries(arm,entries,signals,start,end)
 scoped=[e for e in red if start<=e['entry_date']<=end]
 if gate_fn is None:
  from skills.poc_broker_data import apply_broker_gate
  gate_fn=apply_broker_gate
 try:
  selected,broker_decisions=gate_fn(arm,deepcopy(scoped),evidence)
  ids=[e['event_id'] for e in selected]
  if (len(ids)!=len(set(ids)) or selected!=[e for e in scoped if e['event_id'] in set(ids)]):
   raise ValueError('Broker gate changed original candidate order, identity or fields')
  if arm=='poc_red' and selected!=scoped:
   raise ValueError('Baseline broker gate must preserve every red candidate')
 except Exception as exc:
  exc.entry_gate_decisions=deepcopy(red_decisions)
  exc.broker_gate_decisions=deepcopy(getattr(exc,'decisions',[]))
  raise
 kept=set(ids)
 return ([e for e in red if not start<=e['entry_date']<=end or e['event_id'] in kept],
         red_decisions,broker_decisions)


def load_account_reference(root=ROOT,*,verified_cache=None):
 """Bind one unchanged full-period POC/red baseline and the sealed 0050 account."""
 cache=verified_cache or RunSourceCache(root)
 path=root/BASE_REPORT.relative_to(ROOT)
 cache.verify(path,BASE_REPORT_SHA)
 report=cache.read_json(path,BASE_REPORT_SHA)
 if (report.get('all_completed') is not True or report.get('start')!=START
     or report.get('end')!=END or report.get('initial_cash')!=1_000_000
     or report.get('volume_policy')!='legacy_total_research'):
  raise ValueError('Frozen latest account reference scope differs')
 refs=dict(report['source_sha256'])
 refs[str(path.relative_to(root))]=BASE_REPORT_SHA
 for name,digest in refs.items():cache.verify(root/name,digest)
 values={}
 for arm in ('poc_red','benchmark'):
  info=report['cases'][arm]
  if not info.get('completed'):raise ValueError('Frozen reference case is incomplete')
  if arm=='poc_red' and info['sha256']!=BASE_CASE_SHA:
   raise ValueError('Frozen baseline case identity differs')
  p=root/info['path'];value=cache.read_json(p,info['sha256'])
  if (not value.get('completed') or value['summary']!=info['summary']
      or value['summary']['start']!=START or value['summary']['end']!=END
      or value['summary']['initial_cash']!=1_000_000):
   raise ValueError('Frozen reference account or summary differs')
  refs[info['path']]=info['sha256'];values[arm]=value
 benchmark=dict(path=report['cases']['benchmark']['path'],
  sha256=report['cases']['benchmark']['sha256'],summary=values['benchmark']['summary'],
  reused_sealed_account=True,recomputed_this_study=False)
 return values['poc_red'],benchmark,refs


def compare_account_baseline(actual,expected):
 """The neutral arm must reproduce every economic field, not just final NAV."""
 if not actual.get('completed') or not expected.get('completed'):
  raise ValueError('Complete accounts are required for baseline equality')
 checks={}
 for key in ('account','summary','profile_queries','entry_gate_decisions'):
  if actual[key]!=expected[key]:raise ValueError('Full POC/red baseline differs: '+key)
  checks[key]=canonical_sha(actual[key])
 return dict(required=True,all_exact=True,through=END,reference_sha256=BASE_CASE_SHA,
             field_sha256=checks)


class BoundaryCapture:
 """Observe the previous close before the first new session changes anything."""
 def __init__(self,*args,**kwargs):
  self.prefix_boundary_state=None
  super().__init__(*args,**kwargs)
 def corporate_day(self,day):
  if str(day.date())>BOUNDARY and self.prefix_boundary_state is None:
   if not self.daily or self.daily[-1]['date']!=BOUNDARY:
    raise ValueError('Cannot capture an uncompleted 9/9 boundary')
   self.prefix_boundary_state=deepcopy(dict(cash=self.cash,nav=self.previous_nav,
       receivables=self.receivables,cohorts=self.cohorts,
       positions=[r for r in self.holding_rows if r['date']==BOUNDARY]))
  return super().corporate_day(day)


# Static financial runner body below is adapted from the hash-bound candle/volume source.
def _run(output, arms, data_provider, *, limit_overlay=None):
 started=time.monotonic()
 if not arms or len(set(arms))!=len(arms) or any(x not in ARMS for x in arms):
  raise ValueError('Only the six fixed broker-account arms are allowed')
 output=Path(output).resolve();output.relative_to(RUN_ROOT)
 if output.exists():raise ValueError('Choose a new output directory')
 if Path(data_provider.bundle).resolve()!=INPUTS:raise ValueError('Unexpected extension bundle')
 data_provider.verify_sources()
 prereg=Path(data_provider.prereg_path).resolve()
 prereg.relative_to(ROOT)
 if prereg!=PREREG:raise ValueError('Broker account needs its own frozen preregistration')
 if sha(prereg)!=data_provider.prereg_sha256:raise ValueError('Latest preregistration changed')
 for name,expected in FROZEN_BINDINGS.items():
  if sha(ROOT/name)!=expected:raise ValueError('Changed frozen financial source: '+name)
 prefix_cache=RunSourceCache(ROOT)
 baseline,benchmark,reference_refs=load_account_reference(verified_cache=prefix_cache)
 output.mkdir(parents=True)
 source_snapshot=output/'runner_source.py';source_snapshot.write_bytes(Path(__file__).read_bytes())
 I=INPUTS;C=RUN_ROOT/'execution-v1';SOURCE='poc_broker_account_20261004'
 volume_policy='legacy_total_research';PREP=False
 refs=dict(FROZEN_BINDINGS)
 def merge_refs(values):
  merge_source_refs(refs,values,ROOT)
 def mark(p):merge_refs({str(p.relative_to(ROOT)):sha(p)})
 merge_refs(data_provider.refs);mark(prereg)
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
  def __init__(self):self.files={};self.queries=[];self.engine=None
  def get_odd(self,day,sid,market):
   self.queries.append(dict(stock_id=sid,date=day,market=market.upper()))
   result=data_provider.get_odd(day,sid,market,engine=self.engine)
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

 from skills.verified_volume_midpoint import VerifiedVolumeMidpointOrders
 from skills.repaired_execution_context import benchmark_exclusion_with_market, dated_market_resolver
 class RepairedOriginal(AccountReservationAudit,ZeroValueShareDelivery,DrawdownControl,CandidateQuality,Era,
                        VerifiedVolumeMidpointOrders,MidpointExitReplay,AccountCandidateHook):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.corporate=FaceValueCapitalActions(self.corporate.provider,self)
 class RepairedBenchmark(BoundaryCapture,Era,VerifiedVolumeMidpointOrders,MidpointBenchmark):pass
 class CandleVolumeAccount(BoundaryCapture,CandleVolumeExit,RepairedOriginal):pass

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
 for name in ('scripts/research_poc_broker_account.py','skills/poc_broker_data.py',
              'skills/poc_broker_odd.py','tests/test_poc_broker_data.py',
              'skills/poc_broker_odd_acquisition.py','tests/test_poc_broker_odd_acquisition.py',
              'tests/test_poc_broker_odd.py','tests/test_poc_broker_account.py'):
  mark(ROOT/name)
 mark(ROOT/'scripts/research_volume_profile_account.py')
 mark(ROOT/'tests/test_volume_profile_account_adapter.py')
 for name in ('scripts/research_poc_latest_account.py','tests/test_poc_latest_account.py',
              'scripts/research_candle_volume_account.py','tests/test_candle_volume_account.py',
              'skills/candle_volume_rules.py','tests/test_candle_volume_rules.py'):
  mark(ROOT/name)
 source_snapshots={}
 for source in list(refs):
  if any(token in source for token in ('volume_profile_account','volume_profile_selection',
                                       'volume_profile_data','volume_profile_execution','volume_profile_odd','benchmark_limit_overlay',
                                       'candle_volume_account','candle_volume_rules','poc_latest','poc_broker')):
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
   feeds=Feeds();odds=Odds();ticks=study.strict.AdditionalTicks()
   corp=Corp(events,Path(data_provider.dividend_directory),None,offline=True,overrides=overrides)
   cls=RepairedBenchmark if arm=='benchmark' else CandleVolumeAccount
   red_gate,volume_mode=ARM_RULES[arm]
   opts={} if arm=='benchmark' else dict(ordering='original',position_count=3,factor_mask=0,residual_policy='release',identity_report=identity,exit_signals=data.features,action_dates=action_dates)
   profile_queries=[]
   def observed_profile(event):
    profile_queries.append(dict(event_id=event['event_id'],stock_id=event['members'][0],signal_date=event['signal_date']))
    return data_provider.profile(arm,event)
   if arm!='benchmark':opts.update(candidate_arm='original',drawdown_arm='three_black',black_signals=black_signals,selection_arm='poc_priority_available' if arm.startswith('poc_') else 'original',profile_provider=observed_profile if arm.startswith('poc_') else None,candidate_selector=candidate_selector,candle_volume_signals=candle_signals,volume_exit_mode=volume_mode)
   opts.update(ordinary_volumes=ordinary,ordinary_market_resolver=dated_market,volume_policy=volume_policy)
   engine=None;entry_gate_decisions=[];broker_gate_decisions=[]
   try:
    selected,entry_gate_decisions,broker_gate_decisions=select_account_entries(
     arm,entries,candle_signals,data_provider.broker_evidence,data.start,data.end)
    engine=cls(quotes,companies,days,selected,feeds,corp,start=data.start,end=data.end,initial_cash=1_000_000,ticks=ticks,participation=.01,liquidity_identity=identity,odd_feeds=odds,**opts)
    corp.preparation_engine=engine
    odds.engine=engine
    if arm!='benchmark':install_pending_share_rights(engine,ROOT)
    print('start',arm,flush=True)
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
    policy_audit=audit_candle_volume(account,candle_signals,volume_mode) if volume_mode!='none' else None
    gate_audit=audit_entry_gate(account,candle_signals) if red_gate else None
    account.pop('volume_exit_log',[])
    account['settings'].pop('volume_exit_mode',None)
    value=dict(completed=True,account=account,summary=study.old.summarize(account),audit=audit,
     rule_audit=dict(entry_gate=gate_audit,volume_exit=policy_audit));verify_cash(value)
    value['ordinary_capacity_complete']=account.get('ordinary_volume_evidence',{}).get('all_requested_board_capacity_observed',False)
    value['ordinary_evidence_blocked_orders']=account.get('ordinary_volume_evidence',{}).get('blocked_board_children',0)
    account['resource_plans']=deepcopy(engine.resource_plans)
    if arm!='benchmark':
     account['slot_decisions']=deepcopy(engine.slot_decisions)
     account['residual_days']=deepcopy(engine.residual_days)
     account['board_decisions']=deepcopy(engine.board_decisions)
   except (ReplayDataUnavailable,study.old.UnresolvedAction,ValueError,RuntimeError) as exc:
    entry_gate_decisions=getattr(exc,'entry_gate_decisions',getattr(exc,'decisions',entry_gate_decisions))
    broker_gate_decisions=getattr(exc,'broker_gate_decisions',broker_gate_decisions)
    if engine is None:
     value=dict(completed=False,summary=None,reason=str(exc),stage='arm_initialization',completed_sessions=0,last_date=None)
    else:
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
   value['family_rules']=dict(red_gate=red_gate,volume_exit_mode=volume_mode,
    anchor='poc_priority_available' if arm.startswith('poc_') else arm)
   value['entry_gate_decisions']=entry_gate_decisions
   value['broker_gate_decisions']=broker_gate_decisions
   value['volume_exit_log']=deepcopy(getattr(engine,'volume_exit_log',[]))
   value['prefix_boundary_state']=deepcopy(getattr(engine,'prefix_boundary_state',None))
   if value['completed']:
    try:
     value['baseline_parity']=compare_account_baseline(value,baseline) if arm=='poc_red' else dict(
      required=False,reason='preregistered_broker_gate_changes_account_path')
    except ValueError as exc:
     value.update(completed=False,reason=str(exc),observed_summary=value['summary'],summary=None)
   merge_refs(odds.files)
   path=output/(arm+'.json');write(path,value);cases[arm]={k:v for k,v in value.items() if k not in ('account','audit','partial_journal','ordinary_evidence_blocks','entry_gate_decisions','broker_gate_decisions','volume_exit_log')}
   cases[arm].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
   record=dict(timestamp=datetime.now().isoformat(timespec='seconds'),source=SOURCE,params=dict(arm=arm,start=data.start,end=data.end,input_bundle=str(I.relative_to(ROOT)),execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy=volume_policy,data_revision='sealed_latest_inputs_broker_gate_study',fetch=bool(data_provider.online),benchmark_limit_overlay=limit_overlay is not None),preparation=PREP,completed=value['completed'],result_path=str(path.relative_to(ROOT)),live_qualified=False)
   append_trial_registry(record,registry_path=output/'trials.jsonl');append_trial_registry(record)
   print(arm,{k:value['summary'][k] for k in ('total_return','max_drawdown','final_nav')} if value.get('summary') else cases[arm],flush=True)
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
  execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy=volume_policy,
  data_revision='sealed_latest_inputs_broker_gate_study',return_recomputed=all(c['completed'] for c in cases.values()),
  preparation=False,cases=cases,source_sha256=refs,all_completed=all(c['completed'] for c in cases.values()),
  live_qualified=False,actual_fill_verified=False,corporate_fractional_cash_date_verified=False,
  complete_historical_universe=False,unseen_validation=False,
  accounting_validated=all(c['completed'] for c in cases.values()),validated=False,
  network_requests=None if data_provider.online else 0,finmind_requests=None if data_provider.online else 0,
  profile_data=profile_data,source_snapshots=source_snapshots,
  baseline_validation=dict(through=END,required_arm='poc_red',
   compared='complete account, summary, red gate and POC query sequence',
   variants_preserve_baseline_prefix=False),benchmark=benchmark,
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
 parser.add_argument('--fetch',action='store_true',help='Allow only the bounded, preregistered broker-account data adapter')
 parser.add_argument('--benchmark-limit-overlay',action='store_true')
 args=parser.parse_args(argv)
 output_existed=args.output.exists()
 try:
  from skills.poc_broker_data import BranchAccountData
  provider=BranchAccountData(ROOT,online=args.fetch)
  overlay=None
  if args.benchmark_limit_overlay:
   from skills.benchmark_limit_overlay import load_overlay
   overlay=load_overlay(ROOT)
  report=run(args.output,tuple(args.arms.split(',')),provider,limit_overlay=overlay)
 except Exception as exc:
  failure=dict(completed=False,summary=None,stage='setup_or_publication',reason=str(exc),
               source='poc_broker_account_20261004',arms=args.arms.split(','),live_qualified=False)
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
