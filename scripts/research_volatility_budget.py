#!/usr/bin/env python3
"""Fixed three-point risk-cap study with full accounts and explicit validation gates."""
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
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.volatility_budget_replay import VolatilityBudgetReplay,audit_volatility_budget
from skills.residual_slot_replay import audit_residual_slots
from skills.execution_factorial import flags,stock_pnl,load_capital_terms
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.trial_registry import append_trial_registry
from skills.strategy_validation_gate import assess_family
from skills.verified_backtest_tool import offline_only

OUTPUT=ROOT/'.cache/volatility-budget-20260927'
ARMS={'equal':None,'vol30':.30,'vol40':.40,'vol50':.50}
CODE=[Path(__file__),ROOT/'skills/volatility_budget_replay.py',ROOT/'skills/strategy_validation_gate.py',
      ROOT/'skills/trial_registry.py',ROOT/'docs/prereg_volatility_budget_20260927.md']


def case(data,inputs,identity,additions,mask,target):
    config=dict(factor_mask=mask,factors=flags(mask),volatility_target=target,
                benchmark=False,board_only=True,position_count=5)
    feeds=ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    overrides=(read(parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        read(parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp=TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
    engine=VolatilityBudgetReplay(data.quotes,data.companies,data.days,data.entries,feeds,corp,
        start=data.start,end=data.end,identity_report=identity,factor_mask=mask,volatility_target=target,
        exit_signals=data.features,action_dates=list(zip(data.events.stock_id,data.events.event_date)))
    try:
        account=engine.run()
        validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        audit=audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
            engine.board_decisions,engine.residual_days,data.quotes)
        if target is not None:
            audit.update(audit_volatility_budget(account,engine.volatility_decisions,engine.residual_days,
                data.features.adjusted_close,data.days,target))
        return dict(completed=True,config=config,account=account,summary=summarize(account),audit=audit,
            stock_pnl=stock_pnl(account,engine.marks),residual_days=engine.residual_days,
            volatility_decisions=engine.volatility_decisions,resource_plans=engine.resource_plans,
            slot_decisions=engine.slot_decisions,board_decisions=engine.board_decisions,
            identity_decisions=engine.identity_decisions,live_qualified=False,unseen_validation=False)
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        return dict(completed=False,config=config,reason=str(exc),partial_account=dict(
            daily=engine.daily,trades=engine.trades,orders=engine.orders),live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(completed=False,config=config,reason=str(exc),live_qualified=False)


def metrics(value,benchmark):
    a,b=value['account']['daily'],benchmark['account']['daily']
    if [r['date'] for r in a]!=[r['date'] for r in b]:
        raise ValueError('Benchmark calendar differs')
    rolling=[a[i]['nav']/a[i-251]['opening_nav']-b[i]['nav']/b[i-251]['opening_nav']
             for i in range(251,len(a))]
    decisions=value['volatility_decisions']
    profits=sorted((x['profit'] for x in value['stock_pnl'].values() if x['profit']>0),reverse=True)
    return dict(benchmark_return=benchmark['summary']['total_return'],
        excess_return=value['summary']['total_return']-benchmark['summary']['total_return'],
        rolling252_count=len(rolling),rolling252_win_rate=sum(x>0 for x in rolling)/len(rolling),
        rolling252_worst_excess=min(rolling),
        annual_excess=[dict(year=x['year'],excess=x['total_return']-y['total_return'])
                      for x,y in zip(value['summary']['annual'],benchmark['summary']['annual'])],
        traded_notional=sum(t['gross'] for t in value['account']['trades']),
        top_two_positive_profit_share=sum(profits[:2])/sum(profits) if profits else None,
        reduced_budget_decisions=sum(r['reason']=='scaled' for r in decisions),
        missing_volatility_decisions=sum(r['reason']=='missing_prior_volatility' for r in decisions))


def run(output):
    tick=time.monotonic();output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT:
        raise ValueError('Choose a new immutable volatility study directory')
    original,selector=load(),parent.load_selector()
    refs=dict(original['source_sha256'])
    refs.update(file_identities([REPORT,REPORT.with_suffix('.sha256'),*CODE],ROOT))
    write(output/'identity.json',refs)
    rows,trials={},[]
    with offline_only():
        data,inputs,identity=parent.parent.load_data(selector)
        additions=parent.parent.parent.load_corporate_completion(ROOT)|load_capital_terms(ROOT)[0]
        for arm,target in ARMS.items():
            for mask in range(8):
                name=f'{arm}_{mask}';value=None;status='failed';error=None
                print('running',name,flush=True)
                try:
                    value=case(data,inputs,identity,additions,mask,target)
                    write(output/'cases'/(name+'.json'),value)
                    if target is None:
                        sealed=read(ROOT/original['cases']['release_'+str(mask)]['result']['path'])
                        if not value['completed'] or encoded(value['account'])!=encoded(sealed['account']):
                            raise ValueError('Neutral account does not reproduce '+name)
                    status='completed' if value['completed'] else 'blocked'
                except Exception as exc:
                    error=f'{type(exc).__name__}: {exc}'
                    write(output/'errors'/(name+'.json'),dict(case=name,error=error))
                    raise
                finally:
                    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='volatility_budget_20260927',
                        command=' '.join(sys.argv),case=name,volatility_target=target,factor_mask=mask,
                        result_path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),status=status,error=error)
                    append_trial_registry(record);trials.append(record);write(output/'trials.json',trials)
                row=dict(completed=value['completed'],config=value['config'],summary=value.get('summary'),
                    reason=value.get('reason'),result=dict(path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),
                                                        sha256=sha(output/'cases'/(name+'.json'))))
                if value['completed']:
                    benchmark=read(ROOT/original['cases']['benchmark_combined' if mask&1 else 'benchmark_control']['result']['path'])
                    row['metrics']=metrics(value,benchmark)
                rows[name]=row
                print(name,value.get('summary',{}).get('total_return',value.get('reason')),flush=True)
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:
        raise ValueError('Source changed during risk-budget study')
    report=dict(schema='volatility_budget_research_v1',start=data.start,end=data.end,initial_cash=1_000_000,
        candidate_count=len(data.entries),all_completed=all(r['completed'] for r in rows.values()),
        cases=rows,validation=assess_family(rows),elapsed_seconds=round(time.monotonic()-tick,3),
        trial_count=len(trials),network_calls=0,database_writes=0,live_qualified=False,unseen_validation=False)
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p)
        for p in output.rglob('*.json') if p.name!='manifest.json'}))
    return report


def verify(left,right,output):
    left,right,output=(Path(p).resolve() for p in (left,right,output))
    if left==right or output.exists():
        raise ValueError('Use separate runs and a new proof file')
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
    if any(r['validation']!=assess_family(r['cases']) for r in reports):
        raise ValueError('Validation differs from accounts')
    proof=dict(schema='volatility_budget_reproduction_v1',passed=True,compared_cases=32,
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
