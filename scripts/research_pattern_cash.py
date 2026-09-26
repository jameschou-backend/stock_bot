#!/usr/bin/env python3
"""Test the fixed contraction breakout filter, reusing verified control accounts."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.file_lock import file_lock
from app.residual_slots_ui import load, REPORT
from scripts import research_residual_slots as parent
from scripts.research_exit_scenarios import read,write,sha,encoded,summarize,TrackedCorporateActions
from scripts.research_volatility_budget import metrics
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.pattern_cash_replay import PatternCashReplay, audit_pattern_entries
from skills.support_risk_audit import audit_support_risk
from skills.technical_signals import TechnicalSignals
from skills.exit_corporate_completion import load_exit_completion, DOCUMENT
from skills.residual_slot_replay import audit_residual_slots
from skills.execution_factorial import flags,stock_pnl,load_capital_terms
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

OUTPUT=ROOT/'.cache/pattern-cash-20260927'
ARMS={'control':('control',False),'pattern':('control',True),
      'support_risk2':('support_risk2',False),'support_risk2_pattern':('support_risk2',True)}
PARENT=ROOT/'artifacts/forward_simulation/support_risk_20260927.json'
CODE=[Path(__file__),ROOT/'skills/pattern_cash_replay.py',ROOT/'skills/trial_registry.py',
      ROOT/'docs/prereg_pattern_cash_20260927.md']


def case(data,inputs,identity,additions,mask,mode,technical):
    config=dict(pattern_filter=True,factor_mask=mask,factors=flags(mask),technical_mode=mode,
                benchmark=False,board_only=True,position_count=5)
    feeds=ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    overrides=(read(parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        read(parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp=TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
    engine=PatternCashReplay(data.quotes,data.companies,data.days,data.entries,feeds,corp,
        start=data.start,end=data.end,identity_report=identity,factor_mask=mask,technical_mode=mode,
        pattern_filter=True,technical_signals=technical,exit_signals=data.features,action_dates=list(zip(data.events.stock_id,data.events.event_date)))
    try:
        account=engine.run()
        validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        audit=audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
            engine.board_decisions,engine.residual_days,data.quotes)
        audit.update(audit_support_risk(account,engine.technical_entries,engine.support_decisions,
            engine.exit_decisions,engine.exit_states,engine.slot_decisions,technical,mode,mask,engine.prior))
        audit.update(audit_pattern_entries(account,engine.pattern_entries,engine.slot_decisions,technical,data.entries,mask))
        return dict(completed=True,config=config,account=account,summary=summarize(account),audit=audit,
            stock_pnl=stock_pnl(account,engine.marks),residual_days=engine.residual_days,
            exit_decisions=engine.exit_decisions,exit_states=engine.exit_states,
            technical_entries=engine.technical_entries,support_decisions=engine.support_decisions,
            pattern_entries=engine.pattern_entries,
            resource_plans=engine.resource_plans,slot_decisions=engine.slot_decisions,
            board_decisions=engine.board_decisions,identity_decisions=engine.identity_decisions,
            live_qualified=False,unseen_validation=False)
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        return dict(completed=False,config=config,reason=str(exc),partial_account=dict(
            daily=engine.daily,trades=engine.trades,orders=engine.orders),live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):raise
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)


def screen(rows):
    result={}
    for arm in ARMS:
        cases=[rows[f'{arm}_{m}'] for m in range(8)]
        complete=all(c['completed'] for c in cases)
        if not complete:
            result[arm]=dict(all_completed=False,historical_candidate=False)
            continue
        wins=sum(c['metrics']['excess_return']>0 for c in cases)
        risk=min(c['summary']['max_drawdown'] for c in cases)>=-.5
        rolling=all(cases[m]['metrics']['rolling252_win_rate']>=.6 for m in (0,7))
        years=[sum(a['excess']>0 for a in cases[m]['metrics']['annual_excess']
                   if a['year'] in ('2022','2023','2024','2025')) for m in (0,7)]
        result[arm]=dict(all_completed=True,winning_stresses=wins,drawdown_pass=risk,
            rolling_pass=rolling,positive_full_years=years,
            historical_candidate=wins==8 and risk and rolling and min(years)>=3)
    return result


def run(output):
    tick=time.monotonic();output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT:
        raise ValueError('Choose a new immutable pattern study directory')
    original,selector=load(),parent.load_selector()
    baseline=read(PARENT)
    if (sha(PARENT)!=PARENT.with_suffix('.sha256').read_text().strip()
            or baseline.get('schema')!='support_risk_publication_v1'
            or baseline.get('live_qualified') is not False or baseline.get('adopted') is not False):
        raise ValueError('Invalid sealed control publication')
    refs=read(ROOT/baseline['source_identity']['path'])
    refs.update(file_identities([PARENT,PARENT.with_suffix('.sha256'),*CODE],ROOT))
    proof=read(ROOT/baseline['offline_verification']['path'])
    if (sha(ROOT/baseline['offline_verification']['path'])!=baseline['offline_verification']['sha256']
            or proof.get('passed') is not True or proof.get('all_completed') is not True
            or proof.get('compared_cases')!=32 or len(proof.get('runs',[]))!=2
            or baseline['run_manifest'] not in proof['runs']):
        raise ValueError('Controls lack their two-run reproduction proof')
    parent_folder=(ROOT/baseline['run_manifest']['path']).parent
    if read(parent_folder/'report.json')['cases']!=baseline['cases']:
        raise ValueError('Parent publication differs from its source report')
    refs.update(proof['source_binding_code_sha256'])
    refs.update(file_identities([ROOT/baseline['offline_verification']['path']],ROOT))
    for run_ref in proof['runs']:
        manifest_path=ROOT/run_ref['path'];manifest=read(manifest_path)
        refs[run_ref['path']]=run_ref['sha256']
        for name,digest in manifest['files_sha256'].items():
            path=(manifest_path.parent/name).resolve()
            if not path.is_relative_to(manifest_path.parent):raise ValueError('Invalid parent artifact path')
            refs[str(path.relative_to(ROOT))]=digest
    supplement,primary=load_exit_completion(ROOT);refs.update(primary)
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Frozen pattern sources changed')
    write(output/'identity.json',refs)
    rows,trials={},[]
    with offline_only():
        data,inputs,identity=parent.parent.load_data(selector)
        additions=parent.parent.parent.load_corporate_completion(ROOT)|load_capital_terms(ROOT)[0]|supplement
        technical=TechnicalSignals(data.features.adjusted_close,data.quotes,data.days)
        for arm,(mode,enabled) in ARMS.items():
            for mask in range(8):
                name=f'{arm}_{mask}';value=None;status='failed';error=None
                print('running',name,flush=True)
                try:
                    value=(case(data,inputs,identity,additions,mask,mode,technical) if enabled
                           else read(ROOT/baseline['cases'][f'{mode}_{mask}']['result']['path']))
                    write(output/'cases'/(name+'.json'),value)
                    if not enabled:
                        sealed=read(ROOT/baseline['cases'][f'{mode}_{mask}']['result']['path'])
                        if not value['completed'] or encoded(value['account'])!=encoded(sealed['account']):
                            raise ValueError('Neutral account does not reproduce '+name)
                    status='completed' if value['completed'] else 'blocked'
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}'
                    write(output/'errors'/(name+'.json'),dict(case=name,error=error))
                    raise
                finally:
                    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='pattern_cash_20260927',
                        command=' '.join(sys.argv),case=name,technical_mode=mode,pattern_filter=enabled,factor_mask=mask,
                        result_path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),status=status,error=error)
                    if enabled:
                        append_trial_registry(record);trials.append(record);write(output/'trials.json',trials)
                row=dict(cache_hit=not enabled,reused_case=None if enabled else baseline['cases'][f'{mode}_{mask}']['result'],completed=value['completed'],config=value['config'],summary=value.get('summary'),
                    reason=value.get('reason'),result=dict(path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),
                                                        sha256=sha(output/'cases'/(name+'.json'))))
                if value['completed']:
                    benchmark=read(ROOT/original['cases']['benchmark_combined' if mask&1 else 'benchmark_control']['result']['path'])
                    row['metrics']=metrics(dict(value,volatility_decisions=[]),benchmark)
                    for key in ('reduced_budget_decisions','missing_volatility_decisions'):row['metrics'].pop(key)
                    row['exit_reasons']=value['audit']['trigger_reasons']
                rows[name]=row
                print(name,value.get('summary',{}).get('total_return',value.get('reason')),flush=True)
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Source changed during technical study')
    report=dict(schema='pattern_cash_research_v1',start=data.start,end=data.end,initial_cash=1_000_000,
        candidate_count=len(data.entries),all_completed=all(r['completed'] for r in rows.values()),
        cases=rows,screen=screen(rows),elapsed_seconds=round(time.monotonic()-tick,3),
        trial_count=len(trials),cache_hits=sum(c['cache_hit'] for c in rows.values()),network_calls=0,database_writes=0,live_qualified=False,unseen_validation=False)
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p)
        for p in output.rglob('*.json') if p.name!='manifest.json'}))
    return report


def verify(left,right,output):
    left,right,output=(Path(p).resolve() for p in (left,right,output))
    if left==right or output.exists():raise ValueError('Use separate runs and a new proof file')
    reports=[]
    for folder in (left,right):
        for name,digest in read(folder/'manifest.json')['files_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Artifact changed '+name)
        refs=read(folder/'identity.json')
        if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Source changed')
        reports.append(read(folder/'report.json'))
    if read(left/'identity.json')!=read(right/'identity.json'):raise ValueError('Source identities differ')
    expected={f'{a}_{m}' for a in ARMS for m in range(8)}
    if any(set(r['cases'])!=expected for r in reports):raise ValueError('All 32 cases are required')
    for name in expected:
        if read(left/'cases'/(name+'.json'))!=read(right/'cases'/(name+'.json')):
            raise ValueError('Full case did not reproduce '+name)
    if any(r['screen']!=screen(r['cases']) for r in reports):raise ValueError('Screen differs from accounts')
    proof=dict(schema='pattern_cash_reproduction_v1',passed=True,compared_cases=32,
        all_completed=all(r['all_completed'] for r in reports),network_calls=0,
        newly_executed_cases=sum(r['trial_count'] for r in reports),cache_reuses=sum(r['cache_hits'] for r in reports),
        runs=[dict(path=str((f/'manifest.json').relative_to(ROOT)),sha256=sha(f/'manifest.json')) for f in (left,right)],
        live_qualified=False)
    write(output,proof);output.with_suffix('.sha256').write_text(sha(output)+'\n')
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path,nargs=2)
    args=parser.parse_args()
    with file_lock(OUTPUT/'.run.lock',timeout=0):
        report=verify(*args.compare,args.output) if args.compare else run(args.output)
    print('complete',report['all_completed'],flush=True)
