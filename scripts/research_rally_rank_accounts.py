#!/usr/bin/env python3
"""Fixed common-known, first-only candidate ranking on the original cash engine.

Statically adapted from research_strategy_account_comparison._run. Only the
candidate registry, ranking and 3/5 position count change. Historical range70/30
execution, integer shares, fees, limits, corporate rights and gap gates remain.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_strategy_account_comparison import *
from scripts import research_strategy_account_comparison as parent
import numpy as np
from skills.rally_ranking import build_rankings, RANKING_VARIANTS
from skills.strategy_scanner.data import load_bundle

RUNNER_SOURCE = Path(__file__).resolve()
IMPORTED_RUNNER_SHA = sha(RUNNER_SOURCE)
RUN_ROOT = ROOT / '.cache/rally-optimization-20261007'
PREREG = ROOT / 'docs/prereg_rally_optimization_20261007.md'
CONTEXT_REPORT = ROOT / '.cache/rally-context-20261007/run-v2/report.json'
CONTEXT_REPORT_SHA = 'fc302ce2e30e057071b14bce4ef1659e1ad617a77b2359e4e0a6b78688c5ebfc'
COHORTS = ('original_red', 'legacy_course_breakout')
ARM_CONFIG = {f'{cohort}__{ranking}__{slots}':dict(cohort=cohort,ranking=ranking,slots=slots)
              for cohort in COHORTS for ranking in RANKING_VARIANTS for slots in (3,5)}
ARMS = tuple(ARM_CONFIG)
ARM_RULES = {arm:(config['cohort']=='original_red','none',.7,.3,False)
             for arm,config in ARM_CONFIG.items()}
SELECTION_ARMS = {arm:'original' for arm in ARMS}
COMMON_KNOWN_ARMS = frozenset()
RSI_ARMS = frozenset()


def is_benchmark(arm):
 if arm not in ARMS:raise ValueError('Unregistered ranking account arm')
 return False


from skills.poc_range_execution import RangeGapOrders


class StrictRankRangeOrders(RangeGapOrders):
 """Stop on inherited source gaps; normal zero-capacity orders stay valid.

 The historical ancestor records a source gap and returns zero. Preserve that
 diagnostic journal, but reject the whole account before another date/order can
 run. Thus a missing source can never masquerade as an unsuccessful trade.
 """
 def _execute_order(self,*args,**kwargs):
  before=len(self.data_gap_exclusions)
  result=super()._execute_order(*args,**kwargs)
  added=self.data_gap_exclusions[before:]
  if added:
   row=added[0]
   raise ReplayDataUnavailable('Strict ranking account missing execution evidence: '
       +str(row.get('failure_reason'))+' ['+str(row.get('date'))+' '
       +str(row.get('stock_id'))+' '+str(row.get('side'))+']')
  return result

 def run(self):
  account=super().run()
  if self.data_gap_exclusions:
   raise ReplayDataUnavailable('Ranking account cannot complete with source-gap exclusions')
  account['settings'].update(data_gap_policy='halt_account_on_missing_execution_evidence',
      missing_ordinary_policy='halt_account_without_partial_return',
      missing_odd_policy='halt_account_without_partial_return',posthoc_data_exclusion=False)
  return account


def verify_runner_source():
 if sha(RUNNER_SOURCE)!=IMPORTED_RUNNER_SHA:raise ValueError('Ranking runner changed during execution')
 return IMPORTED_RUNNER_SHA


def verify_financial_prefix(old_quotes,new_quotes,old_close,new_close,old_eligible,new_eligible):
 """Exact source equality on the full old calendar, including missingness.

 A different adjusted scale, revised OHLCV or historical identity is a blocker,
 not an excuse to combine a new signal with another financial-price version.
 """
 raw=['open','high','low','close','volume'];keys=['date','stock_id']
 for frame in (old_quotes,new_quotes):
  if frame.duplicated(keys).any():raise ValueError('Duplicate raw source coordinate')
 left=old_quotes.set_index(keys)[raw].sort_index()
 right=new_quotes.set_index(keys)[raw].reindex(left.index)
 if not np.isclose(left.to_numpy(),right.to_numpy(),rtol=0,atol=0,equal_nan=True).all():
  raise ValueError('Financial/scanner raw OHLCV prefix differs or is missing')
 first,last=old_close.index[0],old_close.index[-1]
 extras=new_quotes.loc[new_quotes.date.between(first,last)].set_index(keys).index.difference(left.index)
 if len(extras):raise ValueError('Scanner raw source contains extra historical coordinates')
 for name,left,right in [('adjusted_close',old_close,new_close),('eligibility',old_eligible,new_eligible)]:
  if not left.index.is_unique or not left.index.is_monotonic_increasing or not left.columns.is_unique:
   raise ValueError('Invalid financial matrix axes')
  if not left.index.isin(right.index).all() or not left.columns.isin(right.columns).all():
   raise ValueError('Financial/scanner '+name+' axes missing')
  if not left.equals(right.reindex(index=left.index,columns=left.columns)):
   raise ValueError('Financial/scanner '+name+' prefix differs')
 return dict(raw_coordinates=len(old_quotes),matrix_sessions=len(old_close),
             matrix_stocks=len(old_close.columns),raw_exact=True,adjusted_exact=True,eligibility_exact=True)


def map_ranked_entries(ranked,calendar,*,start=START,end=END):
 """Use precomputed causal ranks, never re-rank after account/data outcomes."""
 days=pd.DatetimeIndex(calendar);positions={str(d.date()):i for i,d in enumerate(days)}
 if not days.is_unique or not days.is_monotonic_increasing or start not in positions or end not in positions:
  raise ValueError('Invalid financial candidate calendar')
 if set(ranked.cohort)-set(COHORTS):raise ValueError('Unexpected candidate cohort')
 if ranked.duplicated(['cohort','signal_date','stock_id']).any():raise ValueError('Duplicate signal coordinate')
 output={arm:[] for arm in ARMS};outside=0;unknown=0
 for row in ranked.to_dict('records'):
  if not row['ranking_known']:unknown+=1;continue
  signal=row['signal_date'];i=positions.get(signal)
  if i is None and start<=signal<=end:raise ValueError('Candidate signal is not an observed financial session')
  if i is None or i+1>=len(days) or not start<=str(days[i+1].date())<=end:
   outside+=1;continue
  sid=row['stock_id']
  if not isinstance(sid,str) or len(sid)!=4 or not sid.isdigit() or sid.startswith('0'):
   raise ValueError('Only ordinary four-digit candidate stocks permitted')
  cutoff=days[days<pd.Timestamp(signal[:7]+'-01')]
  for variant in RANKING_VARIANTS:
   rank=row['rank_'+variant]
   if isinstance(rank,bool) or not math.isfinite(float(rank)) or rank<1 or int(rank)!=rank:
    raise ValueError('Known candidate requires a positive integer frozen rank')
   event=dict(event_id=row['event_id'],signal_date=signal,entry_date=str(days[i+1].date()),
      members=[sid],priority=-int(rank),group_id=row['cohort']+'-'+signal[:7],
      group_members=[sid],group_cutoff_date=str(cutoff[-1].date()) if len(cutoff) else None,
      selection_reason='frozen common-known first occurrence; '+variant+' ranking',
      leader_evidence=dict(cohort=row['cohort'],ranking=variant,rank=int(rank),
        score=float(row['score_'+variant]),information_cutoff=signal,uses_forward_outcome=False))
   for slots in (3,5):output[f"{row['cohort']}__{variant}__{slots}"].append(deepcopy(event))
 for arm,events in output.items():
  events.sort(key=lambda e:(e['entry_date'],-e['priority'],e['event_id']))
  validate_candidate_calendar(events,days)
 for cohort in COHORTS:
  sets=[{e['event_id'] for e in output[a]} for a,c in ARM_CONFIG.items() if c['cohort']==cohort]
  if any(ids!=sets[0] for ids in sets):raise ValueError('Ranking arms changed common candidate set')
 return output,dict(input_events=len(ranked),unknown_evidence=unknown,outside_financial_entry_scope=outside,
   candidate_counts={arm:len(events) for arm,events in output.items()},first_only_source=True,
   account_start_signal_boundary='start is signal-date boundary from fixed study; no pre2024 events added',
   rankings_computed_before_financial_scope_filter=True)


def audit_ranked_account(account,entries):
 result=audit_candidate_identity(account,entries)
 expected={}
 for e in entries:expected.setdefault(e['entry_date'],[]).append(e['event_id'])
 for row in account['selection_decisions']:
  if row['original_event_ids']!=expected.get(row['date'],[]):
   raise ValueError('Financial reservation candidate order differs from frozen ranking')
  if row['selected_event_ids']!=row['original_event_ids'] or row.get('fallback_to_original'):
   raise ValueError('Ranking account unexpectedly filtered or restored candidates')
 if any(t['side']=='buy' and t['stock_id']=='0050' for t in account['trades']):
  raise ValueError('Idle 0050 purchase in stock ranking account')
 result.pop('no_red_gate_added',None)
 result.update(ranking_before_reservations=True,common_known_only=True,
    candidate_gate_unchanged=True,red_candle_revalidated=bool(entries and entries[0]['leader_evidence']['cohort']=='original_red'))
 return result


def load_ranked_payload():
 if sha(CONTEXT_REPORT)!=CONTEXT_REPORT_SHA:raise ValueError('Frozen context report changed')
 report=read(CONTEXT_REPORT);refs=dict(report['source_sha256'])
 refs[str(CONTEXT_REPORT.relative_to(ROOT))]=CONTEXT_REPORT_SHA
 feature=report['features'];path=ROOT/feature['path']
 if sha(path)!=feature['sha256']:raise ValueError('Frozen signal features changed')
 refs[feature['path']]=feature['sha256']
 for name,expected in refs.items():
  if sha(ROOT/name)!=expected:raise ValueError('Context source changed: '+name)
 source=Path(report['source_provenance']['bundle'])
 scanner=load_bundle(source,START,'2026-10-05')
 if scanner['provenance']!=report['source_provenance']:raise ValueError('Scanner provenance changed')
 manifest=read(INPUTS/'manifest.json')
 for name,expected in manifest['files_sha256'].items():
  if sha(INPUTS/name)!=expected:raise ValueError('Frozen financial input changed: '+name)
  refs[str((INPUTS/name).relative_to(ROOT))]=expected
 refs[str((INPUTS/'manifest.json').relative_to(ROOT))]=sha(INPUTS/'manifest.json')
 def matrix(folder,name):return pd.read_parquet(folder/name).set_index('date')
 close=matrix(INPUTS,'close-official.parquet');eligible=matrix(INPUTS,'eligibility.parquet')
 parity=verify_financial_prefix(pd.read_parquet(INPUTS/'quotes-unmasked.parquet'),
    pd.read_parquet(source/'quotes-unmasked.parquet'),close,matrix(source,'close-official.parquet'),
    eligible,matrix(source,'eligibility.parquet'))
 ranked=build_rankings(pd.read_parquet(path));entries,audit=map_ranked_entries(ranked,close.index)
 audit['financial_scanner_parity']=parity
 all_entries=[e for cohort in COHORTS for e in entries[cohort+'__rs__3']]
 all_entries.sort(key=lambda e:(e['entry_date'],-e['priority'],e['event_id']))
 for file in (Path(parent.__file__),ROOT/'skills/rally_ranking.py',ROOT/'tests/test_rally_rank_accounts.py',RUNNER_SOURCE,PREREG):
  refs[str(file.relative_to(ROOT))]=sha(file)
 return dict(source_sha256=refs,entries_by_arm=entries,all_entries=all_entries,audit=audit,
    calendar=[str(d.date()) for d in close.index],ranked=ranked)


def _run(output, arms, data_provider, *, ranked_payload, limit_overlay=None):
 verify_runner_source()
 if data_provider.online:raise ValueError('Ranking accounts must remain offline')
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
 reference_refs=ranked_payload['source_sha256'];legacy_refs={}
 output.mkdir(parents=True)
 ranked_payload['ranked'].to_parquet(output/'ranked-candidates.parquet',index=False)
 write(output/'candidate-preparation.json',ranked_payload['audit'])
 source_bytes=RUNNER_SOURCE.read_bytes()
 if hashlib.sha256(source_bytes).hexdigest()!=IMPORTED_RUNNER_SHA:
  raise ValueError('Comparison source changed before snapshot')
 source_snapshot=output/'runner_source.py';source_snapshot.write_bytes(source_bytes)
 if sha(source_snapshot)!=IMPORTED_RUNNER_SHA:raise ValueError('Comparison source snapshot differs')
 I=INPUTS;C=RUN_ROOT/'execution-v1';SOURCE='rally_rank_accounts_20261007'
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
 CandleVolumeAccount,RepairedBenchmark=engine_types(StrictRankRangeOrders,Era)
 NativeAccount,_=engine_types(StrictRankRangeOrders,Era,native_time20=True)

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
 entries=ranked_payload['all_entries'];pending_entries=[]
 all_entries=entries
 pool=sorted({'0050'}|{e['members'][0] for e in all_entries})
 companies=pd.read_parquet(I/'companies.parquet');identity=read(I/'identity.json')
 quotes=pd.read_parquet(I/'quotes-unmasked.parquet');quotes=quotes[quotes.stock_id.isin(pool)].copy();quotes['date']=pd.to_datetime(quotes.date)
 mask=pd.read_parquet(I/'eligibility.parquet').set_index('date');mask.index=pd.to_datetime(mask.index);days=mask.index
 validate_candidate_calendar(all_entries,days)
 if ranked_payload['calendar']!=[str(d.date()) for d in days]:
  raise ValueError('Ranking and financial input calendars differ')
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
                                       'rally_rank_accounts','rally_ranking','strategy_account_comparison','strategy_comparison_data','strategy_comparison_corporate','candle_volume_account','candle_volume_rules','poc_latest','poc_executable','poc_gap','poc_intraday','poc_range','poc_first_signal')):
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
   cls=CandleVolumeAccount
   red_gate,volume_mode,buy_fraction,sell_fraction,first_only=ARM_RULES[arm]
   opts={} if is_benchmark(arm) else dict(ordering='original',position_count=ARM_CONFIG[arm]['slots'],factor_mask=0,residual_policy='release',identity_report=identity,exit_signals=data.features,action_dates=action_dates)
   profile_queries=[]
   def observed_profile(event):
    profile_queries.append(dict(event_id=event['event_id'],stock_id=event['members'][0],signal_date=event['signal_date']))
    return profile_for_arm(arm,shared_profiles,event)
   if not is_benchmark(arm):opts.update(candidate_arm='original',drawdown_arm='control' if arm=='rsi_time20' else 'three_black',black_signals=black_signals,selection_arm=SELECTION_ARMS[arm],profile_provider=observed_profile if SELECTION_ARMS[arm]!='original' else None,candidate_selector=select_known_profiles if arm in COMMON_KNOWN_ARMS else candidate_selector,candle_volume_signals=candle_signals,volume_exit_mode=volume_mode)
   opts.update(ordinary_volumes=ordinary,ordinary_market_resolver=dated_market,volume_policy=volume_policy)
   engine=None;entry_gate_decisions=[];first_gate_decisions=[];signal_parity=None
   try:
    selected=deepcopy(ranked_payload['entries_by_arm'][arm])
    validate_candidate_calendar(selected,days)
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
    audit['ranking_candidates']=audit_ranked_account(account,selected)
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
   value['legacy_anchor_parity']=None
   value['data_gap_exclusions']=deepcopy(getattr(engine,'data_gap_exclusions',[]))
   value['profile_data_exclusions']=[deepcopy(row) for item in getattr(engine,'selection_decisions',[])
      for row in item.get('certificate',{}).get('profile_data_exclusions',[])]
   value['selection_fallback_days']=sum(bool(r.get('fallback_to_original')) for r in getattr(engine,'selection_decisions',[]))
   value['posthoc_data_exclusion']=False
   value['daily_range_proxy']=True
   value['intraday_odd_daily_proxy']=True
   value['intraday_odd_sequence_verified']=False
   value['deferred_corporate_preparations']=len(getattr(corp,'deferred_preparations',[]))
   value['profile_queries']=profile_queries
   value['family_rules']=dict(red_gate=red_gate,volume_exit_mode=volume_mode,
    buy_fraction=buy_fraction,sell_fraction=sell_fraction,first_signal_only=True,
    anchor=SELECTION_ARMS.get(arm,arm),candidate_family=ARM_CONFIG[arm]['cohort'],source_first_only=True,
    exit_policy='time20_next_session' if arm=='rsi_time20' else 'buy_and_hold' if is_benchmark(arm) else 'loss12_time63_post_entry_three_black')
   value['entry_gate_decisions']=entry_gate_decisions
   value['first_gate_decisions']=first_gate_decisions
   value['volume_exit_log']=deepcopy(getattr(engine,'volume_exit_log',[]))
   value['signal_parity']=signal_parity
   value.update(live_qualified=False,actual_fill_verified=False,unseen_validation=False,historical_period_already_researched=True)
   merge_refs(odds.files)
   path=output/(arm+'.json');write(path,value);cases[arm]={k:v for k,v in value.items() if k not in ('account','audit','partial_journal','ordinary_evidence_blocks','entry_gate_decisions','first_gate_decisions','volume_exit_log')}
   cases[arm].update(path=str(path.relative_to(ROOT)),sha256=sha(path))
   record=dict(timestamp=datetime.now().isoformat(timespec='seconds'),source=SOURCE,params=dict(arm=arm,ranking=ARM_CONFIG[arm]['ranking'],position_count=ARM_CONFIG[arm]['slots'],start=data.start,end=data.end,input_bundle=str(I.relative_to(ROOT)),execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy='regular_full_session_daily_1pct_prior_total_adv20',data_revision='daily_range_proxy_with_explicit_gaps',buy_fraction=buy_fraction,sell_fraction=sell_fraction,first_signal_only=True,fetch=bool(data_provider.online),benchmark_limit_overlay=limit_overlay is not None),preparation=PREP,completed=value['completed'],status='completed' if value['completed'] else 'incomplete',result_path=str(path.relative_to(ROOT)),live_qualified=False,actual_fill_verified=False,unseen_validation=False)
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
 report=dict(schema='rally_rank_accounts_v1',strict_missing_data_halt=True,
  start=data.start,end=data.end,initial_cash=1_000_000,input_bundle=str(I.relative_to(ROOT)),
  execution_source_snapshot=str(source_snapshot.relative_to(ROOT)),volume_policy='regular_full_session_daily_1pct_prior_total_adv20',
  data_revision='daily_range_proxy_with_explicit_gaps',return_recomputed=all(c['completed'] for c in cases.values()),
  preparation=False,cases=cases,source_sha256=refs,all_completed=all(c['completed'] for c in cases.values()),
  live_qualified=False,actual_fill_verified=False,corporate_fractional_cash_date_verified=False,
  complete_historical_universe=False,unseen_validation=False,
  posthoc_data_exclusion=False,original_strategy_fully_verified=False,
  daily_range_proxy=True,intraday_odd_daily_proxy=True,intraday_odd_sequence_verified=False,
  exclusion_scope=dict(missing_execution_data='halt_account_without_partial_return',
    observed_capacity_or_limit_zerofill='keep_resource_lock_or_retry_sales',
    unknown_ranking_features='fixed_common_known_candidate_pool_before_all_outcomes'),
  accounting_validated=all(c['completed'] for c in cases.values()),validated=False,
  network_requests=getattr(data_provider,'network_calls',None) if data_provider.online else 0,
  finmind_requests=getattr(data_provider,'finmind_calls',None) if data_provider.online else 0,
  profile_data=profile_data,source_snapshots=source_snapshots,
  signal_validation=dict(through=END,required=True,compared='frozen first-only common-known ranks; exact scanner/financial prefix',account_equality_required=False),
  execution_model=dict(ordinary='regular_session_daily_range_proxy',odd='intraday_odd_daily_range_proxy',intraday_odd_sequence_verified=False,
      complete_exchange_tape_verified=False,actual_queue_verified=False,actual_broker_fills_verified=False),
  registered_arms=list(ARMS),requested_arms=list(arms),
  shared_profiles_evaluated=len(shared_profiles.values),
  shared_profile_results=shared_profiles.values,historical_period_already_researched=True,
  ranking_preparation=ranked_payload['audit'],
  candidate_stock_pool=pool,
  all_registered_completed=set(cases)==set(ARMS) and all(c['completed'] for c in cases.values()),
  old_account_return_reused=False,
  benchmark_limit_overlay=getattr(limit_overlay,'audit_rows',[]),
  candidate_count=sum(data.start<=e['entry_date']<=data.end for e in entries),
  source_signal_count=len(entries)+len(pending_entries),pending_terminal_signals=pending_entries,
  boundary_pre_2024_signals=sum(e['signal_date']<START and data.start<=e['entry_date']<=data.end for e in entries),
  preregistration=dict(path=str(prereg.relative_to(ROOT)),sha256=sha(prereg)),
  elapsed_seconds=round(time.monotonic()-started,3))
 write(output/'report.json',report)
 (output/'report.sha256').write_text(sha(output/'report.json')+'\n')
 return report



def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args(argv)
 target=args.output.resolve();target.relative_to(RUN_ROOT)
 if target.exists():raise ValueError('Choose a new output directory')
 from app.file_lock import file_lock
 from skills.strategy_comparison_data import StrategyComparisonData
 with file_lock(RUN_ROOT/'.account-run.lock',timeout=0):
  with study.old.offline_only():
   payload=load_ranked_payload()
   provider=parent.bind_comparison_sources(StrategyComparisonData(ROOT,online=False))
   provider.prereg_path=PREREG;provider.prereg_sha256=sha(PREREG)
   provider._bind(PREREG,provider.prereg_sha256)
   report=_run(target,ARMS,provider,ranked_payload=payload)
 return 0 if report['all_completed'] else 2


if __name__=='__main__':raise SystemExit(main())
