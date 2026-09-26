#!/usr/bin/env python3
"""Compare seven fixed exits on fully audited cash accounts, offline only."""
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
from skills.exit_mechanism_replay import ExitMechanismReplay,audit_exits
from skills.residual_slot_replay import audit_residual_slots
from skills.execution_factorial import flags,stock_pnl,load_capital_terms
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

OUTPUT=ROOT/'.cache/exit-mechanisms-20260927'
ARMS=('loss12','fixed63','trail20_12','weak20','market_weak','trend126','adaptive')
CODE=[Path(__file__),ROOT/'skills/exit_mechanism_replay.py',ROOT/'skills/trial_registry.py',
      ROOT/'scripts/research_volatility_budget.py',ROOT/'skills/volatility_budget_replay.py',
      ROOT/'skills/strategy_validation_gate.py',ROOT/'docs/prereg_exit_mechanisms_20260927.md']


def case(data,inputs,identity,additions,mask,mode):
    config=dict(factor_mask=mask,factors=flags(mask),exit_mode=mode,
                benchmark=False,board_only=True,position_count=5)
    feeds=ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    overrides=(read(parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        read(parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp=TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
    engine=ExitMechanismReplay(data.quotes,data.companies,data.days,data.entries,feeds,corp,
        start=data.start,end=data.end,identity_report=identity,factor_mask=mask,exit_mode=mode,
        exit_signals=data.features,action_dates=list(zip(data.events.stock_id,data.events.event_date)))
    try:
        account=engine.run()
        validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        audit=audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
            engine.board_decisions,engine.residual_days,data.quotes)
        audit.update(audit_exits(account,engine.exit_decisions,engine.exit_states,
                                data.features.adjusted_close,data.days,mode,mask))
        return dict(completed=True,config=config,account=account,summary=summarize(account),audit=audit,
            stock_pnl=stock_pnl(account,engine.marks),residual_days=engine.residual_days,
            exit_decisions=engine.exit_decisions,exit_states=engine.exit_states,
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
        raise ValueError('Choose a new immutable exit study directory')
    original,selector=load(),parent.load_selector()
    refs=dict(original['source_sha256'])
    refs.update(file_identities([REPORT,REPORT.with_suffix('.sha256'),*CODE],ROOT))
    write(output/'identity.json',refs)
    rows,trials={},[]
    with offline_only():
        data,inputs,identity=parent.parent.load_data(selector)
        additions=parent.parent.parent.load_corporate_completion(ROOT)|load_capital_terms(ROOT)[0]
        for arm in ARMS:
            for mask in range(8):
                name=f'{arm}_{mask}';value=None;status='failed';error=None
                print('running',name,flush=True)
                try:
                    value=case(data,inputs,identity,additions,mask,arm)
                    write(output/'cases'/(name+'.json'),value)
                    if arm=='loss12':
                        sealed=read(ROOT/original['cases']['release_'+str(mask)]['result']['path'])
                        if not value['completed'] or encoded(value['account'])!=encoded(sealed['account']):
                            raise ValueError('Neutral account does not reproduce '+name)
                    status='completed' if value['completed'] else 'blocked'
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}'
                    write(output/'errors'/(name+'.json'),dict(case=name,error=error))
                    raise
                finally:
                    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='exit_mechanisms_20260927',
                        command=' '.join(sys.argv),case=name,exit_mode=arm,factor_mask=mask,
                        result_path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),status=status,error=error)
                    append_trial_registry(record);trials.append(record);write(output/'trials.json',trials)
                row=dict(completed=value['completed'],config=value['config'],summary=value.get('summary'),
                    reason=value.get('reason'),result=dict(path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),
                                                        sha256=sha(output/'cases'/(name+'.json'))))
                if value['completed']:
                    benchmark=read(ROOT/original['cases']['benchmark_combined' if mask&1 else 'benchmark_control']['result']['path'])
                    row['metrics']=metrics(dict(value,volatility_decisions=[]),benchmark)
                    for key in ('reduced_budget_decisions','missing_volatility_decisions'):row['metrics'].pop(key)
                    row['exit_reasons']=value['audit']['trigger_reasons']
                rows[name]=row
                print(name,value.get('summary',{}).get('total_return',value.get('reason')),flush=True)
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Source changed during exit study')
    report=dict(schema='exit_mechanisms_research_v1',start=data.start,end=data.end,initial_cash=1_000_000,
        candidate_count=len(data.entries),all_completed=all(r['completed'] for r in rows.values()),
        cases=rows,screen=screen(rows),elapsed_seconds=round(time.monotonic()-tick,3),
        trial_count=len(trials),network_calls=0,database_writes=0,live_qualified=False,unseen_validation=False)
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
    if any(set(r['cases'])!=expected for r in reports):raise ValueError('All 56 cases are required')
    for name in expected:
        if read(left/'cases'/(name+'.json'))!=read(right/'cases'/(name+'.json')):
            raise ValueError('Full case did not reproduce '+name)
    if any(r['screen']!=screen(r['cases']) for r in reports):raise ValueError('Screen differs from accounts')
    proof=dict(schema='exit_mechanisms_reproduction_v1',passed=True,compared_cases=56,
        all_completed=all(r['all_completed'] for r in reports),network_calls=0,
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
