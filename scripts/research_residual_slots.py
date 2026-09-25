#!/usr/bin/env python3
"""Fixed remnant-slot policy: eight stresses, sealed controls, offline evidence."""
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.file_lock import file_lock
from app.execution_factorial_ui import load as load_publication, REPORT as PUBLICATION
from app.historical_selector_ui import load as load_selector
from scripts import research_execution_factorial as parent
from scripts.research_exit_scenarios import read,write,sha,encoded,summarize,TrackedCorporateActions
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.residual_slot_replay import ResidualSlotReplay,audit_residual_slots
from skills.execution_factorial import flags,stock_pnl,load_capital_terms
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.verified_backtest_tool import offline_only

OUTPUT=ROOT/'.cache/residual-slots-20260926'
CODE=[Path(__file__),ROOT/'skills/residual_slot_replay.py',ROOT/'skills/residual_account_loop.py',
      ROOT/'docs/prereg_residual_slots_20260926.md']


def case(data,inputs,identity,additions,mask,policy):
    feeds=ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    overrides=(read(parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        read(parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp=TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
    engine=ResidualSlotReplay(data.quotes,data.companies,data.days,data.entries,feeds,corp,
        start=data.start,end=data.end,identity_report=identity,factor_mask=mask,residual_policy=policy,
        exit_signals=data.features,action_dates=list(zip(data.events.stock_id,data.events.event_date)))
    config=dict(factor_mask=mask,factors=flags(mask),residual_policy=policy,benchmark=False,board_only=True,position_count=5)
    try:
        account=engine.run()
        validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        audit=(audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,data.quotes)
               if policy=='release' else parent.parent.sealed.audit_account(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,False))
        return dict(completed=True,config=config,account=account,summary=summarize(account),audit=audit,
            stock_pnl=stock_pnl(account,engine.marks),residual_days=engine.residual_days,
            resource_plans=engine.resource_plans,slot_decisions=engine.slot_decisions,
            board_decisions=engine.board_decisions,identity_decisions=engine.identity_decisions,
            live_qualified=False,unseen_validation=False)
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        result=parent.parent.sealed.parent.blocked(config,str(exc),engine)
        result['residual_days']=engine.residual_days
        return result
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return parent.parent.sealed.parent.blocked(config,str(exc),engine)


def comparison(cases,original):
    result={}
    for mask in range(8):
        name='release_'+str(mask);row=cases[name]
        if not row['completed']:
            result[name]=dict(completed=False,reason=row['reason']);continue
        benchmark=cases['benchmark_combined' if mask&1 else 'benchmark_control']
        left,right=row['account']['daily'],benchmark['account']['daily']
        if [r['date'] for r in left] != [r['date'] for r in right]:
            raise ValueError('Strategy and benchmark calendars differ')
        rolling=[]
        for i in range(251,len(left)):
            s=left[i]['nav']/left[i-251]['opening_nav']-1
            b=right[i]['nav']/right[i-251]['opening_nav']-1
            rolling.append(dict(start=left[i-251]['date'],end=left[i]['date'],strategy=s,benchmark=b,excess=s-b))
        profits=sorted((r['profit'] for r in row['stock_pnl'].values() if r['profit']>0),reverse=True)
        residual=row['residual_days']
        result[name]=dict(completed=True,total_return=row['summary']['total_return'],
            original_total_return=original['cases']['factor_'+str(mask)]['summary']['total_return'],
            benchmark_return=benchmark['summary']['total_return'],
            excess_return=row['summary']['total_return']-benchmark['summary']['total_return'],
            rolling252_count=len(rolling),rolling252_win_rate=sum(r['excess']>0 for r in rolling)/len(rolling),
            rolling252_worst_excess=min(r['excess'] for r in rolling),
            annual_excess=[dict(year=a['year'],excess=a['total_return']-b['total_return'])
                for a,b in zip(row['summary']['annual'],benchmark['summary']['annual'])],
            top_two_positive_profit_share=sum(profits[:2])/sum(profits) if profits else None,
            max_residual_nav_ratio=max(r['residual_value']/r['opening_nav'] for r in residual),
            residual_cap_block_days=sum(r['block_new_buys'] for r in residual),
            max_total_holdings=max(r['holdings'] for r in left),
            max_active_opening=max(len(r['opening_active']) for r in residual))
    return result


def run(output):
    started=time.monotonic();output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT:
        raise ValueError('Preserve existing results; choose a new residual study directory')
    original=load_publication();selector=load_selector()
    refs=dict(original['source_sha256'])
    refs.update(file_identities([PUBLICATION,PUBLICATION.with_suffix('.sha256'),*CODE],ROOT))
    write(output/'identity.json',refs)
    cases={}
    with offline_only():
        data,inputs,identity=parent.load_data(selector)
        additions=parent.parent.load_corporate_completion(ROOT) | load_capital_terms(ROOT)[0]
        for mask in (0,7):
            name='keep_'+str(mask)
            print('running',name,flush=True)
            value=case(data,inputs,identity,additions,mask,'keep')
            expected=read(ROOT/original['cases']['factor_'+str(mask)]['result']['path'])
            if not value['completed'] or encoded(value['account'])!=encoded(expected['account']):
                raise ValueError('Neutral loop no longer reproduces the sealed account')
            cases[name]=value;write(output/'cases'/(name+'.json'),value)
        for mask in (0,7,1,2,4,3,5,6):
            name='release_'+str(mask);print('running',name,flush=True)
            value=case(data,inputs,identity,additions,mask,'release')
            cases[name]=value;write(output/'cases'/(name+'.json'),value)
            print(name,value.get('summary',{}).get('total_return',value.get('reason')),flush=True)
        for mode in ('control','combined'):
            name='benchmark_'+mode
            value=parent.parent.case(data,dict(stress=mode,benchmark=True,board_only=True,position_count=0),inputs,additions,None)
            expected=read(ROOT/original['cases'][name]['result']['path'])
            if not value['completed'] or encoded(value['account'])!=encoded(expected['account']):
                raise ValueError('Benchmark did not reproduce')
            cases[name]=value;write(output/'cases'/(name+'.json'),value)
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:
        raise ValueError('Source changed during residual research')
    report=dict(schema='residual_slot_research_v1',start=data.start,end=data.end,candidate_count=len(data.entries),
        all_completed=all(r['completed'] for r in cases.values()),comparison=comparison(cases,original),
        cases={name:dict(completed=r['completed'],config=r['config'],summary=r.get('summary'),reason=r.get('reason'),
            result=dict(path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),sha256=sha(output/'cases'/(name+'.json'))))
            for name,r in cases.items()},network_calls=0,database_writes=0,
        elapsed_seconds=round(time.monotonic()-started,3),live_qualified=False,unseen_validation=False,
        strict_data_ready=False,initial_cash=1_000_000)
    write(output/'report.json',report)
    write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p)
        for p in output.rglob('*.json') if p.name!='manifest.json'}))
    return report


def verify(left,right,output):
    left,right,output=(Path(p).resolve() for p in (left,right,output))
    if left==right or output.exists():
        raise ValueError('Compare separate runs and preserve previous evidence')
    refs={};reports=[]
    for folder in (left,right):
        report=read(folder/'report.json')
        if report.get('all_completed') is not True:
            raise ValueError('Cannot verify an incomplete residual-slot run')
        for name,digest in read(folder/'manifest.json')['files_sha256'].items():
            if sha(folder/name)!=digest:
                raise ValueError('Residual study artifact changed: '+name)
            refs[str((folder/name).relative_to(ROOT))]=digest
        refs.update(read(folder/'identity.json'))
        refs[str((folder/'manifest.json').relative_to(ROOT))]=sha(folder/'manifest.json')
        reports.append(report)
    if read(left/'identity.json')!=read(right/'identity.json'):
        raise ValueError('Source identities differ')
    expected={'keep_0','keep_7','benchmark_control','benchmark_combined'}|{'release_'+str(i) for i in range(8)}
    if any(set(r['cases'])!=expected for r in reports):
        raise ValueError('Both runs must contain twelve cases')
    for name in expected:
        if read(left/'cases'/(name+'.json'))!=read(right/'cases'/(name+'.json')):
            raise ValueError('Full account did not reproduce: '+name)
    if reports[0]['comparison']!=reports[1]['comparison']:
        raise ValueError('Comparison changed between offline runs')
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:
        raise ValueError('Source changed before offline verification')
    proof=dict(schema='residual_slots_offline_v1',passed=True,compared_cases=12,all_completed=True,
               source_sha256=refs,network_calls=0)
    write(output,proof);output.with_suffix('.sha256').write_text(sha(output)+'\n')
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',type=Path,nargs=2)
    args=parser.parse_args()
    with file_lock(OUTPUT/'.run.lock',timeout=0):
        result=verify(*args.compare,args.output) if args.compare else run(args.output)
    print('completed',result.get('passed',result['all_completed']),flush=True)
