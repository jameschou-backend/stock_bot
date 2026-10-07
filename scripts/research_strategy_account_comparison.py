#!/usr/bin/env python3
"""Controlled red/POC/RSI account comparison on the frozen financial engine.

All arms use buy range70 / sell range30 proxies, one million NTD, and
three funded stocks with idle cash. Prices are post-session proxies, not fills.
"""
from pathlib import Path
import sys,json,time,math,argparse,hashlib
from datetime import datetime
from copy import deepcopy
from unittest.mock import patch
from contextlib import nullcontext
from types import SimpleNamespace
RUNNER_SOURCE=Path(__file__).resolve()
IMPORTED_RUNNER_SHA=hashlib.sha256(RUNNER_SOURCE.read_bytes()).hexdigest()
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
from skills.volume_profile_account_adapter import AccountCandidateHook, AccountReservationAudit, QUALITY_REASONS
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

RUN_ROOT=ROOT/'.cache/strategy-account-comparison-20261007'
CANDIDATE_BUNDLE=RUN_ROOT/'inputs-v1'
INPUTS=ROOT/'.cache/poc-latest-20261003/inputs-v1'
START='2024-01-02';END='2026-10-02'
FROZEN_RUNNER=ROOT/'scripts/replay_repaired_market_inputs.py'
ARM_RULES={
 'red_original':(True,'none',.7,.3,False),
 'poc_priority':(True,'none',.7,.3,False),
 'poc_filter':(True,'none',.7,.3,False),
 'rsi_shared_exit':(False,'none',.7,.3,False),
 'rsi_time20':(False,'none',.7,.3,False),
 'benchmark':(False,'none',.7,.3,False),
 'red_known':(True,'none',.7,.3,False),
 'poc_priority_known':(True,'none',.7,.3,False),
}
ARMS=tuple(ARM_RULES)
COMMON_KNOWN_ARMS=frozenset({'red_known','poc_priority_known','poc_filter'})
RSI_ARMS=frozenset({'rsi_shared_exit','rsi_time20'})
SELECTION_ARMS=dict(red_original='original',poc_priority='poc_priority_available',
 poc_filter='poc_filter',rsi_shared_exit='original',rsi_time20='original',
 red_known='poc_filter',poc_priority_known='poc_priority')


def verify_runner_source():
 if sha(RUNNER_SOURCE)!=IMPORTED_RUNNER_SHA:
  raise ValueError('Running comparison source changed since import')
 return IMPORTED_RUNNER_SHA


def is_benchmark(arm):
 if arm not in ARM_RULES:raise ValueError('Unregistered strategy-comparison arm')
 return arm=='benchmark'
PREREG=ROOT/'docs/prereg_strategy_account_comparison_20261007.md'
RANGE_BINDINGS={
 'scripts/research_poc_gap_account.py':'708221d8660751cf8f36eee4897e25edf5cfe7e9dc95ef36a45d32317c7fb5fe',
 'skills/poc_gap_execution.py':'1e15cdbfe106bfd307444bc8beed65ee86d519a4a75649c3e74163bd05da78dc',
 'skills/poc_gap_audit.py':'c97d5646f631e12e41cf7fca629ed1b06d541bddc4a8f5d2cc97602ab13e94cf',
 'skills/poc_gap_selection.py':'1f38f25eda16ae56334a11016ee378369fc38d7671dccafdc44989236e675027',
 'scripts/research_poc_executable_account.py':'35b72fdb754bffd65221341556277a0777c7673b37b76b7be9071f2e882fadaf',
 'skills/poc_executable_replay.py':'52f81fca7b02ae997bc7ec66f079aa6aead0f885020291ae3710be08f5335d01',
}
RANGE_BINDINGS.update({'scripts/research_poc_intraday_account.py': '0bd5c4cc38028beda609e9a7461e46ef9b5c649b0638b1ef2f8cbe960b942b27', 'skills/poc_intraday_execution.py': 'ca23bcc4d95a109c042f5ada5cbea2cbb14cc524cde8b7ed22504a5a3524266a', 'skills/poc_intraday_audit.py': '130b3c63e14e0db7539916b708db2ef5558b99af29e078a1b939476fd82e836e', 'skills/poc_intraday_data.py': 'ca07a823d404829f5f982d45784cf3402885be952f6cad99f76ab0acfaa6ae5b'})


def bind_comparison_sources(provider):
 """Reuse persistent evidence budgets and bind this disclosed missing-data policy."""
 for name,expected in RANGE_BINDINGS.items():provider._bind(ROOT/name,expected)
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


def load_range_corporate_terms(root=ROOT):
 """Bind settlement repairs without changing selection or spending unknown cash."""
 from datetime import date
 root=Path(root).resolve();path=root/'docs/poc_range_corporate_terms_20261005.json'
 value=read(path)
 if (value.get('schema')!='poc_range_settlement_repair_v1'
     or value.get('strategy_parameters_changed') is not False
     or value.get('cash_supplements')!=[]
     or set(value.get('overrides',{}))!={'2515-2025-11-10','5876-2026-07-28'}):
  raise ValueError('Unexpected range-study corporate repair scope')
 refs=value.get('source_sha256',{})
 if not refs:raise ValueError('Corporate repair lacks source evidence')
 for name,expected in refs.items():
  file=root/name
  if (Path(name).is_absolute() or '..' in Path(name).parts or Path(name).as_posix()!=name
      or not file.resolve().is_relative_to(root) or file.is_symlink()
      or sha(file)!=expected):raise ValueError('Corporate repair source changed: '+name)
 for key,row in value['overrides'].items():
  ex=date.fromisoformat(key[5:]);pay=date.fromisoformat(row['pay_date'])
  if (row.get('use_scope')!='account_settlement_only_not_selection'
      or date.fromisoformat(row['entitlement_announcement_date'])>ex
      or date.fromisoformat(row['delivery_announcement_date'])>pay or pay<ex
      or not row.get('evidence_files') or any(p not in refs for p in row['evidence_files'])):
   raise ValueError('Corporate settlement chronology or provenance invalid: '+key)
  for field in ('shares_per_share','fractional_cash_per_share'):
   n=row.get(field)
   if type(n) not in (int,float) or not math.isfinite(n) or n<0:
    raise ValueError('Corporate settlement amount invalid: '+key)
  if row['shares_per_share']<=0:raise ValueError('Stock distribution must be positive')
 bank=value['overrides']['5876-2026-07-28']
 if (bank.get('certificate_trading_modeled') is not True
     or bank.get('same_code_fungible_trading_verified') is not True
     or bank.get('certificate_stock_id')!='5876'
     or bank.get('certificate_delivery_date')!=bank['pay_date']
     or not date.fromisoformat(bank['pay_date'])<date.fromisoformat(bank['ordinary_conversion_date'])
     or bank.get('fractional_cash_rounding')!='floor_ntd'
     or bank.get('fractional_cash_pay_date') is not None):
  raise ValueError('Bank certificate trading needs same-code listing proof; fractional cash remains unavailable')
 return deepcopy(value['overrides']),dict(refs,**{str(path.relative_to(root)):sha(path)})


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


def select_arm_entries(arm,entries,signals,start,end,*,rsi_entries=()):
 if arm not in ARMS:raise ValueError('Unregistered range-study arm')
 if is_benchmark(arm):return entries,[],[]
 if arm in RSI_ARMS:return deepcopy(list(rsi_entries)),[],[]
 scoped=[deepcopy(e) for e in entries if start<=e['entry_date']<=end]
 expected=deepcopy(scoped)
 kept,decisions=signals.filter_entries(scoped);ids={e['event_id'] for e in kept}
 if len(ids)!=len(kept) or kept!=[e for e in expected if e['event_id'] in ids]:
  raise ValueError('Red gate changed candidate identity, data or order')
 return [e for e in entries if not start<=e['entry_date']<=end or e['event_id'] in ids],decisions,[]


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
      data_gap_exclusions=getattr(engine,'data_gap_exclusions',[]),
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




LEGACY_REPORT=ROOT/'.cache/poc-range-first-20261005/online-v2/report.json'
LEGACY_REPORT_SHA='637071e14cf08bf06123d919bcc97b43b929f03bd12fece22f655fc9acc35594'
LEGACY_ANCHORS={
 'poc_priority':('poc_range70_30_all','f6046594148c0ab6660b36b204d3d878ae716ab0fcb5c6f564ffece0981cb771'),
 'benchmark':('benchmark_range70_30','96c6833c1f3b40d2601dbbfceddfbf199587554b235eac12356ffb912a0e6d14'),
}


def load_legacy_anchors(arms):
 needed={a for a in arms if a in LEGACY_ANCHORS}
 if not needed:return {},{}
 if sha(LEGACY_REPORT)!=LEGACY_REPORT_SHA:raise ValueError('Legacy range comparison report changed')
 report=read(LEGACY_REPORT)
 if not report.get('all_completed') or (report.get('start'),report.get('end'))!=(START,END):
  raise ValueError('Legacy range comparison is incomplete or period differs')
 refs={str(LEGACY_REPORT.relative_to(ROOT)):LEGACY_REPORT_SHA};cases={}
 for arm in sorted(needed):
  name,digest=LEGACY_ANCHORS[arm];item=report['cases'][name]
  path=(ROOT/item['path']).resolve();path.relative_to(ROOT)
  if not item.get('completed') or item['sha256']!=digest or sha(path)!=digest:
   raise ValueError('Legacy range anchor case changed: '+arm)
  case=read(path)
  if not case.get('completed'):raise ValueError('Legacy range anchor is incomplete: '+arm)
  cases[arm]=case;refs[str(path.relative_to(ROOT))]=digest
 return cases,refs


def compare_legacy_anchor(account,reference):
 """Compare every daily/ledger row, retaining explicit first differences.

 A new source revision may legitimately alter a path; this is then visibly NOT
 an exact reproduction. Never discard mismatched dates or substitute old returns.
 """
 if not reference.get('completed'):raise ValueError('Legacy anchor must be complete')
 expected=reference['account'];fields={}
 for key in ('daily','trades','cash_ledger','corporate_actions','cohorts','holdings',
             'receivables','resource_plans','tick_plans','selection_decisions'):
  if key=='selection_decisions' and key not in account and key not in expected:continue
  actual=account.get(key);old=expected.get(key)
  if actual is None or old is None:raise ValueError('Anchor comparison lacks ledger '+key)
  details=dict(equal=actual==old,actual_sha256=canonical_sha(actual),reference_sha256=canonical_sha(old),
               actual_rows=len(actual),reference_rows=len(old))
  if actual!=old:
   index=next((i for i,(a,b) in enumerate(zip(actual,old)) if a!=b),min(len(actual),len(old)))
   details.update(first_difference_index=index,
    actual=actual[index] if index<len(actual) else None,
    reference=old[index] if index<len(old) else None)
  fields[key]=details
 return dict(all_exact=all(row['equal'] for row in fields.values()),fields=fields,
  compared='all daily assets, cash, fills, corporate rights, holdings and precommitted plans',
  requires_difference_diagnosis=not all(row['equal'] for row in fields.values()))


def load_rsi_candidates(directory):
 """Read the prepared, hash-bound RSI family; never regenerate candidates here."""
 from scripts.prepare_strategy_account_comparison import load_candidates
 directory=Path(directory).resolve()
 directory.relative_to(ROOT)
 payload,calendar,manifest=load_candidates(directory,verify_sources=True)
 if (payload.get('schema')!='strategy_account_comparison_entries_v1'
     or payload.get('strategy_id')!='rsi14_reclaim30'
     or (payload.get('start'),payload.get('end'))!=(START,END)):
  raise ValueError('RSI candidate bundle scope differs')
 refs=dict(manifest.get('source_sha256',{}))
 merge_source_refs(refs,payload.get('source_sha256',{}),ROOT)
 for name,expected in manifest['files_sha256'].items():
  file=directory/name
  if sha(file)!=expected:raise ValueError('RSI candidate file changed: '+name)
  merge_source_refs(refs,{str(file.relative_to(ROOT)):expected},ROOT)
 for name in ('manifest.json','manifest.sha256'):
  file=directory/name;merge_source_refs(refs,{str(file.relative_to(ROOT)):sha(file)},ROOT)
 entries=deepcopy(payload['entries'])
 validate_candidate_calendar(entries,pd.DatetimeIndex(calendar))
 for event in entries:
  if not START<=event['entry_date']<=END:
   raise ValueError('RSI funded candidate is outside comparison period')
 # Replay breaks priority ties by event_id. Verify that this fixed order agrees
 # with the preregistered stock-id tiebreak instead of silently changing it.
 groups={}
 for event in entries:groups.setdefault(event['entry_date'],[]).append(event)
 for rows in groups.values():
  by_stock=sorted(rows,key=lambda e:(-e['priority'],e['members'][0]))
  by_id=sorted(rows,key=lambda e:(-e['priority'],e['event_id']))
  if by_stock!=by_id:raise ValueError('RSI event IDs violate the stock-code priority tiebreak')
 out=deepcopy(payload);out['_calendar']=list(calendar)
 return entries,out,refs


def merge_candidate_families(original,rsi):
 result=deepcopy(list(original))+deepcopy(list(rsi))
 ids=[e['event_id'] for e in result]
 if len(ids)!=len(set(ids)):raise ValueError('Candidate families share or duplicate an event ID')
 return result


def validate_candidate_calendar(entries,days):
 days=pd.DatetimeIndex(days)
 if (not days.is_unique or not days.is_monotonic_increasing or days.tz is not None
     or not days.equals(days.normalize())):
  raise ValueError('Unique ordered market calendar required')
 positions={str(day.date()):i for i,day in enumerate(days)}
 seen=set()
 for event in entries:
  eid=event.get('event_id');members=event.get('members',[])
  if (not isinstance(eid,str) or not eid or eid in seen or len(members)!=1
      or not isinstance(members[0],str) or len(members[0])!=4
      or not members[0].isdigit() or members[0].startswith('0')):
   raise ValueError('Unique ordinary-stock candidate identity required')
  seen.add(eid)
  priority=event.get('priority')
  if isinstance(priority,bool) or not isinstance(priority,(int,float)) or not math.isfinite(priority):
   raise ValueError('Finite candidate priority required')
  index=positions.get(event.get('signal_date'))
  if index is None or index+1>=len(days) or event.get('entry_date')!=str(days[index+1].date()):
   raise ValueError('Candidate must execute T+1 on the unchanged market calendar')


class MemoizedProfiles:
 """One deterministic profile result per original event across comparison arms.

 A missing acquisition must raise at the data adapter; it is never cached as a
 false flag. Explicit quality-unavailable results are retained with their reason.
 """
 def __init__(self,provider):self.provider=provider;self.values={};self.events={}
 def __call__(self,event):
  eid=event['event_id']
  if eid in self.events and self.events[eid]!=event:
   raise ValueError('Profile event identity changed between arms')
  if eid not in self.values:
   value=self.provider('poc_red_executable',deepcopy(event))
   if not isinstance(value,dict) or type(value.get('available')) is not bool:
    raise ValueError('Profile needs an explicit availability bool')
   if value['available'] and type(value.get('poc_up')) is not bool:
    raise ValueError('Known profile requires a Boolean POC flag')
   if not value['available'] and (not isinstance(value.get('reason'),str) or not value['reason']):
    raise ValueError('Unavailable profile requires an explicit reason')
   if not value['available'] and value['reason'] not in QUALITY_REASONS:
    raise RuntimeError('Necessary profile acquisition is incomplete: '+eid+': '+value['reason'])
   self.events[eid]=deepcopy(event);self.values[eid]=deepcopy(value)
  return deepcopy(self.values[eid])


def profile_for_arm(arm,shared,event):
 value=shared(event)
 if arm=='red_known' and value['available']:value['poc_up']=True
 return value


def select_known_profiles(events,*,arm,provider,planner,planner_factory=None):
 """Lazy shared availability policy; unknown quality never restores original rows.

 The provider raises on missing acquisition/budget; only the sealed selector's
 structured unknown outcome is excluded. Each retry uses a pristine private
 planner so failed provisional reservations do not alter the account.
 """
 from skills.volume_profile_account_adapter import ReservationPlanner
 from skills.volume_profile_selection import _events,select_candidates,UnresolvedProfile
 rows=_events(events);initial=deepcopy(planner.snapshot())
 factory=planner_factory or (lambda source:ReservationPlanner(source.engine,source.day))
 remaining=list(rows);profiles={};excluded={};attempts=[]
 def profile(event):
  eid=event['event_id']
  if eid not in profiles:profiles[eid]=deepcopy(provider(deepcopy(event)))
  return deepcopy(profiles[eid])
 while True:
  private=factory(planner)
  if private is planner or any(private is p for p in attempts) or private.snapshot()!=initial:
   raise ValueError('Common-known selector requires a new pristine reservation planner')
  attempts.append(private)
  try:result=select_candidates(remaining,arm=arm,provider=profile,planner=private)
  except UnresolvedProfile as exc:
   if planner.snapshot()!=initial:raise ValueError('Unknown POC mutated live reservation inputs')
   row=next((e for e in remaining if e['event_id']==exc.event_id),None)
   decision=next((d for d in exc.certificate.get('decisions',[]) if d['event_id']==exc.event_id),None)
   value=profiles.get(exc.event_id)
   if (row is None or value is None or value.get('available') is not False
       or decision is None or decision.get('profile_status')!='unknown'
       or decision.get('selection_status')!='unresolved'
       or exc.certificate.get('initial_state')!=initial):raise
   if exc.reason not in QUALITY_REASONS:
    raise RuntimeError('Necessary profile acquisition is incomplete: '+exc.event_id+': '+exc.reason) from exc
   excluded[exc.event_id]=dict(event_id=exc.event_id,stock_id=row['members'][0],
    signal_date=row['signal_date'],available=False,poc_up=None,reason=exc.reason,
    recoverable=exc.recoverable,applied=True,policy='shared_known_profile_no_fallback',
    profile_status='unknown',selection_status='profile_data_excluded')
   remaining=[e for e in remaining if e['event_id']!=exc.event_id]
   continue
  if planner.snapshot()!=initial:raise ValueError('POC selection changed live reservation inputs')
  cert=result['certificate'];decisions={d['event_id']:d for d in cert['decisions']}
  decisions.update(excluded)
  cert.update(original_ordered_ids=[e['event_id'] for e in rows],
   retained_ordered_ids=[e['event_id'] for e in remaining],
   decisions=[dict(decisions[e['event_id']],original_rank=i+1) for i,e in enumerate(rows)],
   profile_data_exclusions=list(excluded.values()),
   profile_data_exclusions_applied=bool(excluded),
   profile_provider_event_ids=list(profiles),selection_passes=len(attempts),
   shared_availability_policy='quality_unknown_excluded_acquisition_missing_is_fatal',
   profile_ranking_complete=cert['profile_ranking_complete'] and not excluded,
   source_planner_unchanged=True)
  return result


def audit_profile_admissions(account,profiles,*,hard_filter):
 checked=0
 for cohort in account['cohorts']:
  value=profiles.get(cohort['event_id'])
  if value is None or value.get('available') is not True or type(value.get('poc_up')) is not bool:
   raise ValueError('Common-known arm funded an unknown POC candidate')
  if hard_filter and value['poc_up'] is not True:
   raise ValueError('POC hard filter funded a non-upward profile')
  checked+=1
 if any(d.get('fallback_to_original') for d in account.get('selection_decisions',[])):
  raise ValueError('Common-known arm restored unknown original candidates')
 return dict(funded_events_checked=checked,poc_up_required=hard_filter,fallback_allowed=False)


def audit_candidate_identity(account,entries):
 by_id={e['event_id']:e for e in entries}
 for cohort in account['cohorts']:
  event=by_id.get(cohort['event_id'])
  if (event is None or event['members']!=[cohort['stock_id']]
      or event['signal_date']!=cohort['signal_date'] or event['entry_date']!=cohort['entry_date']):
   raise ValueError('Funded RSI entry differs from the registered T+1 candidate')
 return dict(funded_events_checked=len(account['cohorts']),candidate_count=len(entries),
             signal_clock='T_close_to_T_plus_1',no_red_gate_added=True)


from skills.scenario_exit_replay import ScenarioExitReplay


class NativeTime20Scenario(ScenarioExitReplay):
 """Override only exit scheduling before the inherited financial corporate day.

 Entry day is held day 1; after its twentieth close, execute at entry_index+20.
 No price trigger, lower stop or three-black exit is consulted. Inherited order
 routing still reads these latched states and carries late share deliveries.
 """
 def corporate_day(self,day):
  index=self.positions[day]
  active={h['event_id'] for sid,h in self.holdings.items() if sid!='0050'}
  active|={r['event_id'] for r in self.receivables if r.get('qty',0)>0}
  for cohort in self.cohorts:
   eid,sid=cohort['event_id'],cohort['stock_id']
   if eid not in active:continue
   if eid not in self.exit_states:
    entry_index=self.positions[pd.Timestamp(cohort['entry_date'])]
    price=self.exit_signals.price(entry_index,sid)
    self.exit_states[eid]=dict(stock_id=sid,event_id=eid,entry_index=entry_index,
     entry_price=price,peak_price=price,trigger_reason=None,signal_date=None,
     target_date=None,target_index=None)
   state=self.exit_states[eid];age=index-state['entry_index']
   if age<1:raise ValueError('Native20 exit check requires a completed holding session')
   if state['trigger_reason'] is None and age>=20:
    state.update(trigger_reason='time20',signal_date=str(self.days[index-1].date()),
     target_date=str(day.date()),target_index=index)
   triggered=state['trigger_reason'] is not None
   self.exit_decisions.append(dict(date=str(day.date()),signal_date=str(self.days[index-1].date()),
    stock_id=sid,event_id=eid,mode='time20_next_session',held_sessions=age,
    exit=triggered,reason='time20' if triggered else None,
    first_signal_date=state['signal_date'],target_date=state['target_date'],
    phase='exiting' if triggered else 'holding',extend=False,
    reads_execution_day_price=False))
   if sid in self.holdings and self.holdings[sid]['event_id']==eid and triggered:
    self.holdings[sid]['due_index']=state['target_index']
  # Continue strictly AFTER the original ScenarioExitReplay decision method;
  # higher-level planning, identity, rights, and resource mixins still execute.
  income=super(ScenarioExitReplay,self).corporate_day(day)
  for holding in self.holdings.values():
   state=self.exit_states.get(holding['event_id'])
   if state and state['target_index'] is not None:holding['due_index']=state['target_index']
  return income

 def run(self):
  result=super().run()
  result['exit_decisions']=deepcopy(self.exit_decisions)
  result['settings']['exit_policy']='time20_next_session'
  result['settings']['exit_policy_holding_sessions']=20
  result['settings']['inherited_cohort_horizon_metadata']=63
  return result


def audit_time20(account,days):
 """Independent scalar oracle checks first fill, trigger timing and retries."""
 positions={str(day.date()):i for i,day in enumerate(pd.DatetimeIndex(days))}
 cohorts={c['event_id']:c for c in account['cohorts']};first_buy={}
 for trade in account['trades']:
  if trade['side']=='buy':first_buy.setdefault(trade['event_id'],trade['date'])
 if any(first_buy.get(eid)!=c['entry_date'] for eid,c in cohorts.items()):
  raise ValueError('Native20 holding clock is not the first positive fill')
 seen=set();triggers={}
 for row in account.get('exit_decisions',[]):
  eid=row['event_id'];cohort=cohorts[eid];i=positions[row['date']]
  age=i-positions[cohort['entry_date']];key=(eid,i)
  if (key in seen or age<1 or row['stock_id']!=cohort['stock_id']
      or row['held_sessions']!=age or row['signal_date']!=str(pd.Timestamp(days[i-1]).date())
      or row['exit']!=(age>=20) or row['reason']!=('time20' if age>=20 else None)):
   raise ValueError('Native20 decision clock or reason differs')
  seen.add(key)
  if age>=20:
   if eid not in triggers:
    if age!=20:raise ValueError('Native20 first trigger was late')
    triggers[eid]=(row['signal_date'],row['date'])
   if (row['first_signal_date'],row['target_date'])!=triggers[eid]:
    raise ValueError('Native20 exit instruction changed during retry')
 for trade in account['trades']:
  if trade['side']!='sell':continue
  eid=trade['event_id'];cohort=cohorts[eid]
  if (trade['reason']!='time20' or eid not in triggers
      or positions[trade['date']]-positions[cohort['entry_date']]<20
      or trade.get('signal_date')!=triggers[eid][0]):
   raise ValueError('Native20 sale preceded the completed holding clock')
 if account.get('black_log') or any(t.get('reason') in ('loss12','three_black','time63') for t in account['trades']):
  raise ValueError('Native20 mixed a shared early-exit rule')
 return dict(decisions_checked=len(seen),first_triggers=len(triggers),holding_sessions=20,
             entry_day_is_one=True,sell_on_following_session=True)


def engine_types(execution_orders,era,*,native_time20=False):
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
 if native_time20:
  class NativeAccount(CandleVolumeAccount,NativeTime20Scenario):pass
  return NativeAccount,RepairedBenchmark
 return CandleVolumeAccount,RepairedBenchmark


# Financial body statically adapted from the frozen latest runner. Only the
# execution adapter, first-signal gate, audit and publication differ.
def _run(output, arms, data_provider, *, limit_overlay=None, candidate_bundle=None):
 verify_runner_source()
 started=time.monotonic()
 if not arms or len(set(arms))!=len(arms) or any(x not in ARMS for x in arms):
  raise ValueError('Only registered strategy-comparison arms are allowed')
 output=Path(output).resolve();output.relative_to(RUN_ROOT)
 if output.exists():raise ValueError('Choose a new output directory')
 if Path(data_provider.bundle).resolve()!=INPUTS:raise ValueError('Unexpected extension bundle')
 data_provider.verify_sources()
 prereg=Path(data_provider.prereg_path).resolve()
 prereg.relative_to(ROOT)
 if prereg!=PREREG or sha(prereg)!=data_provider.prereg_sha256:raise ValueError('Latest preregistration changed')
 for name,expected in (FROZEN_BINDINGS | RANGE_BINDINGS).items():
  if sha(ROOT/name)!=expected:raise ValueError('Changed frozen financial source: '+name)
 reference,reference_refs=load_signal_reference(ROOT)
 legacy_anchors,legacy_refs=load_legacy_anchors(arms)
 output.mkdir(parents=True)
 source_bytes=RUNNER_SOURCE.read_bytes()
 if hashlib.sha256(source_bytes).hexdigest()!=IMPORTED_RUNNER_SHA:
  raise ValueError('Comparison source changed before snapshot')
 source_snapshot=output/'runner_source.py';source_snapshot.write_bytes(source_bytes)
 if sha(source_snapshot)!=IMPORTED_RUNNER_SHA:raise ValueError('Comparison source snapshot differs')
 I=INPUTS;C=RUN_ROOT/'execution-v1';SOURCE='strategy_account_comparison_20261007'
 # Initializes the inherited account only. The new executor uses ordinary
 # daily-range proxies on independently verified ordinary and odd data.
 volume_policy='legacy_total_research';PREP=False
 refs=dict(FROZEN_BINDINGS)
 refs[str(RUNNER_SOURCE.relative_to(ROOT))]=IMPORTED_RUNNER_SHA
 def merge_refs(values):
  merge_source_refs(refs,values,ROOT)
 def mark(p):merge_refs({str(p.relative_to(ROOT)):sha(p)})
 merge_refs(data_provider.refs);merge_refs(RANGE_BINDINGS);mark(prereg)
 merge_refs(reference_refs);merge_refs(legacy_refs)
 if limit_overlay is not None:merge_refs(limit_overlay.refs);mark(ROOT/'skills/benchmark_limit_overlay.py')
 from skills.poc_gap_selection import select_with_profile_gaps
 candidate_selector=select_with_profile_gaps
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
   terms=self.overrides.get(f'{sid}-{day}',{})
   if terms.get('same_code_fungible_trading_verified') is True:
    for row in result:
     if row['kind']=='stock_dividend':
      row.update(delivered_instrument='same_code_new_share_rights_certificate',
       certificate_stock_id=terms['certificate_stock_id'],
       ordinary_conversion_date=terms['ordinary_conversion_date'],
       ordinary_conversion_adds_shares=False)
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

 from skills.poc_range_execution import RangeGapOrders
 from skills.poc_range_audit import audit_range_execution
 from skills.poc_first_signal import audit_first_signal
 from skills.repaired_execution_context import benchmark_exclusion_with_market, dated_market_resolver
 CandleVolumeAccount,RepairedBenchmark=engine_types(RangeGapOrders,Era)
 NativeAccount,_=engine_types(RangeGapOrders,Era,native_time20=True)

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
 rsi_entries=[];rsi_payload=None
 if any(arm in RSI_ARMS for arm in arms):
  rsi_entries,rsi_payload,rsi_refs=load_rsi_candidates(candidate_bundle or CANDIDATE_BUNDLE)
  merge_refs(rsi_refs)
 all_entries=merge_candidate_families(entries,rsi_entries)
 pool=sorted({'0050'}|{e['members'][0] for e in all_entries})
 companies=pd.read_parquet(I/'companies.parquet');identity=read(I/'identity.json')
 quotes=pd.read_parquet(I/'quotes-unmasked.parquet');quotes=quotes[quotes.stock_id.isin(pool)].copy();quotes['date']=pd.to_datetime(quotes.date)
 mask=pd.read_parquet(I/'eligibility.parquet').set_index('date');mask.index=pd.to_datetime(mask.index);days=mask.index
 validate_candidate_calendar(all_entries,days)
 if rsi_payload is not None and rsi_payload['_calendar']!=[str(d.date()) for d in days]:
  raise ValueError('RSI and financial input calendars differ')
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
 candle_signals=CandleVolumeSignals(black_signals,all_entries,action_dates)
 shared_profiles=MemoizedProfiles(data_provider.profile)
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
 range_terms,range_refs=load_range_corporate_terms(ROOT)
 overrides=merge_overrides(overrides,range_terms);merge_refs(range_refs)
 ordinary={}
 for module in ('verified_volume_midpoint','repaired_execution_context','volume_profile_account_adapter'):
  mark(ROOT/'skills'/(module+'.py'))
 dated_market=dated_market_resolver(identity)
 mark(prereg);mark(FROZEN_RUNNER)
 mark(ROOT/'scripts/research_volume_profile_account.py')
 mark(ROOT/'tests/test_volume_profile_account_adapter.py')
 for name in ('scripts/research_poc_range_first_account.py','skills/poc_range_execution.py',
              'skills/poc_range_audit.py','skills/poc_range_data.py','skills/poc_first_signal.py',
              'tests/test_poc_range_first_account.py','tests/test_poc_range_execution.py',
              'tests/test_poc_range_audit.py','tests/test_poc_range_data.py','tests/test_poc_first_signal.py',
              'scripts/research_poc_intraday_account.py','skills/poc_intraday_execution.py',
              'skills/poc_intraday_audit.py','skills/poc_intraday_data.py',
              'tests/test_poc_intraday_account.py','tests/test_poc_intraday_execution.py',
              'tests/test_poc_intraday_audit.py','tests/test_poc_intraday_data.py',
              'scripts/research_poc_gap_account.py','skills/poc_gap_execution.py',
              'skills/poc_gap_audit.py','skills/poc_gap_selection.py',
              'tests/test_poc_gap_account.py','tests/test_poc_gap_execution.py',
              'tests/test_poc_gap_audit.py','tests/test_poc_gap_selection.py',
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
                                       'strategy_account_comparison','strategy_comparison_data','strategy_comparison_corporate','candle_volume_account','candle_volume_rules','poc_latest','poc_executable','poc_gap','poc_intraday','poc_range','poc_first_signal')):
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
   cls=RepairedBenchmark if is_benchmark(arm) else NativeAccount if arm=='rsi_time20' else CandleVolumeAccount
   red_gate,volume_mode,buy_fraction,sell_fraction,first_only=ARM_RULES[arm]
   opts={} if is_benchmark(arm) else dict(ordering='original',position_count=3,factor_mask=0,residual_policy='release',identity_report=identity,exit_signals=data.features,action_dates=action_dates)
   profile_queries=[]
   def observed_profile(event):
    profile_queries.append(dict(event_id=event['event_id'],stock_id=event['members'][0],signal_date=event['signal_date']))
    return profile_for_arm(arm,shared_profiles,event)
   if not is_benchmark(arm):opts.update(candidate_arm='original',drawdown_arm='control' if arm=='rsi_time20' else 'three_black',black_signals=black_signals,selection_arm=SELECTION_ARMS[arm],profile_provider=observed_profile if SELECTION_ARMS[arm]!='original' else None,candidate_selector=select_known_profiles if arm in COMMON_KNOWN_ARMS else candidate_selector,candle_volume_signals=candle_signals,volume_exit_mode=volume_mode)
   opts.update(ordinary_volumes=ordinary,ordinary_market_resolver=dated_market,volume_policy=volume_policy)
   engine=None;entry_gate_decisions=[];first_gate_decisions=[];signal_parity=None
   try:
    selected,entry_gate_decisions,first_gate_decisions=select_arm_entries(arm,entries,candle_signals,data.start,data.end,rsi_entries=rsi_entries)
    if red_gate:signal_parity=compare_signal_gate(entry_gate_decisions,reference)
    engine=cls(quotes,companies,days,selected,feeds,corp,start=data.start,end=data.end,initial_cash=1_000_000,ticks=ticks,participation=.01,liquidity_identity=identity,odd_feeds=odds,buy_fraction=buy_fraction,sell_fraction=sell_fraction,**opts)
    corp.preparation_engine=engine
    odds.engine=engine
    if not is_benchmark(arm):install_pending_share_rights(engine,ROOT)
    print('start',arm,flush=True)
    account=engine.run();study.old.validate_completed_account(account,[str(d.date()) for d in days],data.start,data.end)
    if is_benchmark(arm):audit=study.old.audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
    else:audit=audit_face_resources(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,quotes)
    routes=study.market_routes([*ticks.queries,*odds.queries])
    routes.update({(r['stock_id'],r['date']):dated_market(r['date'],r['stock_id'])
                  for r in account['orders']})
    account['resource_plans']=deepcopy(engine.resource_plans)
    if not is_benchmark(arm):
     account['slot_decisions']=deepcopy(engine.slot_decisions)
     account['board_decisions']=deepcopy(engine.board_decisions)
     account['residual_days']=deepcopy(engine.residual_days)
    audit.update(audit_range_execution(account,ticks,odds,routes,quotes,days,corp,feeds,
      verified_halts=identity['trading_exclusions'],identity_report=identity,buy_fraction=buy_fraction,sell_fraction=sell_fraction))
    audit['verified_halt_observations']=verified_halt_evidence
    if arm=='rsi_time20':audit['time20']=audit_time20(account,days)
    elif not is_benchmark(arm):audit['three_black']=audit_three_black(account,black_signals)
    if arm in COMMON_KNOWN_ARMS:audit['common_known']=audit_profile_admissions(account,shared_profiles.values,hard_filter=arm=='poc_filter')
    if arm in RSI_ARMS:audit['rsi_candidates']=audit_candidate_identity(account,rsi_entries)
    policy_audit=audit_candle_volume(account,candle_signals,volume_mode) if volume_mode!='none' else None
    gate_audit=audit_entry_gate(account,candle_signals) if red_gate else None
    first_audit=(audit_first_signal(account,first_gate_decisions,entries=entries,
      candle_signals=candle_signals,start=data.start,end=data.end) if first_only else None)
    account.pop('volume_exit_log',[])
    account['settings'].pop('volume_exit_mode',None)
    value=dict(completed=True,account=account,summary=study.old.summarize(account),audit=audit,
     rule_audit=dict(entry_gate=gate_audit,volume_exit=policy_audit,first_signal=first_audit));verify_cash(value)
    account['resource_plans']=deepcopy(engine.resource_plans)
    if not is_benchmark(arm):
     account['slot_decisions']=deepcopy(engine.slot_decisions)
     account['residual_days']=deepcopy(engine.residual_days)
     account['board_decisions']=deepcopy(engine.board_decisions)
   except (ReplayDataUnavailable,study.old.UnresolvedAction,ValueError,RuntimeError) as exc:
    entry_gate_decisions=getattr(exc,'decisions',entry_gate_decisions)
    value=preserve_failure(exc,engine,'arm_initialization' if engine is None else 'arm_execution')
   value['legacy_anchor_parity']=(compare_legacy_anchor(value['account'],legacy_anchors[arm])
      if value.get('completed') and arm in legacy_anchors else None)
   value['data_gap_exclusions']=deepcopy(getattr(engine,'data_gap_exclusions',[]))
   value['profile_data_exclusions']=[deepcopy(row) for item in getattr(engine,'selection_decisions',[])
      for row in item.get('certificate',{}).get('profile_data_exclusions',[])]
   value['selection_fallback_days']=sum(bool(r.get('fallback_to_original')) for r in getattr(engine,'selection_decisions',[]))
   value['posthoc_data_exclusion']=True
   value['daily_range_proxy']=True
   value['intraday_odd_daily_proxy']=True
   value['intraday_odd_sequence_verified']=False
   value['deferred_corporate_preparations']=len(getattr(corp,'deferred_preparations',[]))
   value['profile_queries']=profile_queries
   value['family_rules']=dict(red_gate=red_gate,volume_exit_mode=volume_mode,
    buy_fraction=buy_fraction,sell_fraction=sell_fraction,first_signal_only=first_only,
    anchor=SELECTION_ARMS.get(arm,arm),candidate_family='rsi14_reclaim30' if arm in RSI_ARMS else 'original_red' if red_gate else '0050',
    exit_policy='time20_next_session' if arm=='rsi_time20' else 'buy_and_hold' if is_benchmark(arm) else 'loss12_time63_post_entry_three_black')
   value['entry_gate_decisions']=entry_gate_decisions
   value['first_gate_decisions']=first_gate_decisions
   value['volume_exit_log']=deepcopy(getattr(engine,'volume_exit_log',[]))
   value['signal_parity']=signal_parity
   value.update(live_qualified=False,actual_fill_verified=False,unseen_validation=False,historical_period_already_researched=True)
   merge_refs(odds.files)
   path=output/(arm+'.json');write(path,value);cases[arm]={k:v for k,v in value.items() if k not in ('account','audit','partial_journal','ordinary_evidence_blocks','entry_gate_decisions','first_gate_decisions','volume_exit_log')}
   cases[arm].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
   record=dict(timestamp=datetime.now().isoformat(timespec='seconds'),source=SOURCE,params=dict(arm=arm,start=data.start,end=data.end,input_bundle=str(I.relative_to(ROOT)),execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy='regular_full_session_daily_1pct_prior_total_adv20',data_revision='daily_range_proxy_with_explicit_gaps',buy_fraction=buy_fraction,sell_fraction=sell_fraction,first_signal_only=first_only,fetch=bool(data_provider.online),benchmark_limit_overlay=limit_overlay is not None),preparation=PREP,completed=value['completed'],status='completed' if value['completed'] else 'incomplete',result_path=str(path.relative_to(ROOT)),live_qualified=False,actual_fill_verified=False,unseen_validation=False)
   append_trial_registry(record,registry_path=output/'trials.jsonl');append_trial_registry(record)
   print(arm,{k:value['summary'][k] for k in ('total_return','max_drawdown','final_nav')} if value.get('summary') else {k:cases[arm].get(k) for k in ('completed','reason','last_date','completed_sessions')},flush=True)
 with patch.object(pending_share_entitlements,'validate_pending_terms',validate_delivery_terms):
  with nullcontext() if data_provider.online else study.old.offline_only():replay()
 profile_data=data_provider.profile_snapshot(output/'profile-data')
 merge_refs(data_provider.refs)
 for p in (output/'profile-data').rglob('*.json'):mark(p)
 if limit_overlay is not None:merge_refs(limit_overlay.refs)
 verify_runner_source()
 data_provider.verify_sources()
 validate_source_paths(refs,ROOT)
 if study.old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Run sources changed')
 report=dict(start=data.start,end=data.end,initial_cash=1_000_000,input_bundle=str(I.relative_to(ROOT)),
  execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy='regular_full_session_daily_1pct_prior_total_adv20',
  data_revision='daily_range_proxy_with_explicit_gaps',return_recomputed=all(c['completed'] for c in cases.values()),
  preparation=False,cases=cases,source_sha256=refs,all_completed=all(c['completed'] for c in cases.values()),
  live_qualified=False,actual_fill_verified=False,corporate_fractional_cash_date_verified=False,
  complete_historical_universe=False,unseen_validation=False,
  posthoc_data_exclusion=True,original_strategy_fully_verified=False,
  daily_range_proxy=True,intraday_odd_daily_proxy=True,intraday_odd_sequence_verified=False,
  exclusion_scope=dict(buys='zero_fill_keep_daily_resource_lock',sells='zero_fill_keep_holdings_retry_next_session',
    unknown_profiles=dict(poc_priority='sealed_available_fallback',poc_filter='shared_known_only_no_fallback',
      red_known='same_shared_known_only',poc_priority_known='same_shared_known_only')),
  accounting_validated=all(c['completed'] for c in cases.values()),validated=False,
  network_requests=getattr(data_provider,'network_calls',None) if data_provider.online else 0,
  finmind_requests=getattr(data_provider,'finmind_calls',None) if data_provider.online else 0,
  profile_data=profile_data,source_snapshots=source_snapshots,
  signal_validation=dict(through=END,required=True,compared='sealed candidates and red-candle gate',account_equality_required=False),
  execution_model=dict(ordinary='regular_session_daily_range_proxy',odd='intraday_odd_daily_range_proxy',intraday_odd_sequence_verified=False,
      complete_exchange_tape_verified=False,actual_queue_verified=False,actual_broker_fills_verified=False),
  registered_arms=list(ARMS),requested_arms=list(arms),
  rsi_candidate_count=len(rsi_entries),shared_profiles_evaluated=len(shared_profiles.values),
  shared_profile_results=shared_profiles.values,historical_period_already_researched=True,
  rsi_source=rsi_payload.get('parameters') if rsi_payload else None,
  candidate_stock_pool=pool,
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


def run(output,arms,data_provider,*,limit_overlay=None,candidate_bundle=None):
 from app.file_lock import file_lock
 with file_lock(RUN_ROOT/'.run.lock',timeout=0):
  return _run(output,arms,data_provider,limit_overlay=limit_overlay,candidate_bundle=candidate_bundle)


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--arms',default=','.join(ARMS))
 parser.add_argument('--candidate-bundle',type=Path,default=CANDIDATE_BUNDLE)
 parser.add_argument('--fetch',action='store_true',help='Allow only the bounded executable-evidence adapter')
 parser.add_argument('--benchmark-limit-overlay',action='store_true')
 args=parser.parse_args(argv)
 output_existed=args.output.exists()
 try:
  from skills.strategy_comparison_data import StrategyComparisonData
  provider=bind_comparison_sources(StrategyComparisonData(ROOT,online=args.fetch))
  overlay=None
  if args.benchmark_limit_overlay:
   from skills.benchmark_limit_overlay import load_overlay
   overlay=load_overlay(ROOT)
  report=run(args.output,tuple(args.arms.split(',')),provider,limit_overlay=overlay,candidate_bundle=args.candidate_bundle)
 except Exception as exc:
  failure=dict(completed=False,summary=None,stage='setup_or_publication',reason=str(exc),
               source='strategy_account_comparison_20261007',arms=args.arms.split(','),live_qualified=False)
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
