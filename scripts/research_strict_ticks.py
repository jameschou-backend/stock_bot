#!/usr/bin/env python3
"""Re-run fixed signals with evidenced halt zeroes and fail-closed inputs."""
from dataclasses import replace
from pathlib import Path
import argparse
import sys
import time
from datetime import date

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts import research_residual_ticks as old
from skills.strict_tick_inputs import restore_halt_zeroes,StrictResidualReplay,StrictResidualBenchmark

SPEC=ROOT/'docs/prereg_strict_ticks_20260928.md'
OUTPUT=ROOT/'.cache/strict-ticks-20260928'
CODE=[Path(__file__),ROOT/'skills/strict_tick_inputs.py',SPEC]
PREP=ROOT/'.cache/strict-ticks-prepared-20260928'


class AdditionalTicks(old.TickCache):
    """One durable 300-request ceiling shared by correction and fixed policies."""
    def __init__(self,prepare=False):
        super().__init__(PREP/'ticks',online=prepare,maximum=300)
        self.donors=[old.TickCache(p,online=False) for p in (old.TICK_SOURCE/'ticks',old.PREP/'ticks')]
        self.files,self.queries={},[]

    def fetch(self,dataset,sid,day,end=None):
        if not self.online:return super().fetch(dataset,sid,day,end)
        with old.file_lock(self.root/'budget.lock'):
            path=self.root/'budget.json'
            budget=old.read(path) if path.exists() else dict(reserved=0)
            if budget['reserved']>=self.maximum:raise old.ReplayDataUnavailable('Shared 300-request preparation ceiling reached')
            budget['reserved']+=1;old.write(path,budget)
        self.calls+=1
        try:
            return old.fetch_dataset(dataset,date.fromisoformat(day),data_id=sid,token=self.token,
                requests_per_hour=5400,max_retries=0,timeout=30)
        except old.FinMindError as exc:
            raise old.ReplayDataUnavailable(f'Tick provider failed: {sid} {day} {type(exc).__name__}') from exc

    def get(self,sid,day,market):
        path=self.root/f'{sid}-{day}.parquet'
        self.queries.append(dict(stock_id=sid,date=day,market=market))
        for donor in self.donors:
            cached=donor.root/path.name
            if cached.exists() or cached.with_suffix('.json').exists():
                value=donor.get(sid,day,market);path=cached;break
        else:value=super().get(sid,day,market)
        self.files.update(old.file_identities([path,path.with_suffix('.json')],ROOT))
        return value


def repaired_data(publication):
    data,inputs,identity=old.parent.parent.load_data(publication)
    split=old.read(ROOT/'docs/benchmark_split_evidence_20260914.json')
    schedule=split['schedule_source'];terms=split['verified_terms']
    notice=ROOT/schedule['local_path']
    if old.sha(notice)!=schedule['sha256'] or schedule['conservative_known_by']>=terms['suspension_start']:
        raise ValueError('Split suspension source is unverified')
    identity=dict(identity,trading_exclusions=[*identity['trading_exclusions'],dict(stock_id='0050',
        kind='trading_suspension',start=terms['suspension_start'],end=terms['new_units_listing_date'],
        source_path=schedule['local_path'])])
    source=old.parent.parent.parent
    paths=[ROOT/'.cache/million-replay-inputs/quotes.parquet',
           source.DIRECTORY/'quotes.parquet',source.PREFIX/'quotes.parquet']
    pool=set(data.quotes.stock_id)
    raw=[]
    for path in paths:
        frame=pd.read_parquet(path)
        raw.append(frame.loc[frame.stock_id.isin(pool)])
    quotes,repairs=restore_halt_zeroes(data.quotes,raw,identity)
    # The official pre-announced split notice proves zero trades on these
    # sessions. Keep zero prices non-executable; do not invent a closing price.
    split_rows=[]
    for day in data.days:
        if not terms['suspension_start']<=str(day.date())<terms['new_units_listing_date']:continue
        if any(f.stock_id.eq('0050').where(pd.to_datetime(f.date).eq(day),False).any() for f in raw):
            raise ValueError('Unexpected split-suspension raw row; review before applying known zero')
        split_rows.append(dict(stock_id='0050',date=day,open=0.,high=0.,low=0.,close=0.,volume=0.))
        repairs.append(dict(stock_id='0050',date=str(day.date()),kind='official_split_no_trades',
            source_path=schedule['local_path'],volume=0.,price_policy='zero_placeholder_not_executable'))
    quotes=pd.concat([quotes,pd.DataFrame(split_rows)],ignore_index=True)
    paths.extend([notice,ROOT/'docs/benchmark_split_evidence_20260914.json'])
    paths.extend(ROOT/r['source_path'] for r in repairs)
    return replace(data,quotes=quotes),inputs,identity,repairs,old.file_identities(paths,ROOT)


def case(data,inputs,identity,additions,*,benchmark,stress,ticks=None,engine_factory=None,audit_ticks=None):
    ticks=ticks or AdditionalTicks()
    feeds=old.ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    overrides=(old.read(old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        old.read(old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp=old.TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
    args=(data.quotes,data.companies,data.days,data.entries,feeds,corp)
    common=dict(start=data.start,end=data.end,ticks=ticks,participation=.005 if stress else .01,
                liquidity_identity=identity)
    factory=engine_factory or (lambda cls:cls)
    engine=(factory(StrictResidualBenchmark)(*args,**common,stress_mode='slip90' if stress else 'control')
        if benchmark else factory(StrictResidualReplay)(*args,**common,factor_mask=1 if stress else 0,
            residual_policy='release',identity_report=identity,exit_signals=data.features,
            action_dates=list(zip(data.events.stock_id,data.events.event_date))))
    try:
        account=engine.run()
        old.validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        if benchmark:
            audit=old.audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
            audit.update(old.audit_verified_board_only(account,engine.board_decisions,engine.resource_plans))
        else:
            audit=old.audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
                engine.board_decisions,engine.residual_days,data.quotes)
        markets={(r['stock_id'],r['date']):r['market'] for r in ticks.queries}
        audit.update((audit_ticks or old.audit_tick_plans)(account,ticks,markets,data.quotes,data.days,corp))
        audit['unknown_liquidity_rejected']=True
        result=dict(completed=True,summary=old.summarize(account),account=account,audit=audit)
    except (old.ReplayDataUnavailable,old.UnresolvedAction) as exc:
        result=dict(completed=False,summary=None,reason=str(exc),partial_diagnostics=dict(
            plans=engine.tick_plans,orders=engine.orders,trades=engine.trades,completed_sessions=len(engine.daily)))
    return dict(result,source_sha256=ticks.files,tick_queries=ticks.queries,network_calls=ticks.calls,
                live_qualified=False,unseen_validation=False)


def run(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT:
        raise ValueError('Choose a new strict-ticks output directory')
    started=time.monotonic()
    publication=old.load_selector()
    refs=dict(publication['source_sha256'])
    refs.update(old.file_identities([*CODE,*old.CODE],ROOT))
    old.write(output/'identity.json',refs)
    cases={}
    with old.offline_only():
        data,inputs,identity,repairs,repair_refs=repaired_data(publication)
        if len(data.entries)!=454:raise ValueError('Frozen signals changed')
        refs.update(repair_refs)
        additions=old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]
        for name in old.CASES:
            old.record('strict_'+name,output,'started')
            try:
                result=case(data,inputs,identity,additions,benchmark=name.startswith('benchmark'),stress=name.endswith('stress'))
            except Exception as exc:
                old.record('strict_'+name,output,'error',error=str(exc));raise
            old.record('strict_'+name,output,'completed' if result['completed'] else 'blocked',
                       summary=result['summary'],reason=result.get('reason'))
            path=output/'cases'/f'{name}.json'
            old.write(path,result);refs.update(result['source_sha256'])
            cases[name]=dict(completed=result['completed'],summary=result['summary'],reason=result.get('reason'),
                            result=dict(path=str(path.relative_to(ROOT)),sha256=old.sha(path)))
            print(name,result['summary'] or result.get('reason'),flush=True)
    if old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Source changed during run')
    report=dict(schema='strict_ticks_v1',cases=cases,repairs=repairs,source_sha256=refs,
        start=data.start,end=data.end,candidate_count=len(data.entries),all_completed=all(r['completed'] for r in cases.values()),
        network_calls=0,database_writes=0,live_qualified=False,unseen_validation=False,
        elapsed_seconds=round(time.monotonic()-started,3))
    old.write(output/'report.json',report)
    return report


def prepare():
    data,inputs,identity,repairs,refs=repaired_data(old.load_selector())
    additions=old.parent.parent.parent.load_corporate_completion(ROOT)|old.load_capital_terms(ROOT)[0]
    statuses=[]
    for name in old.CASES:
        old.record('strict_'+name,PREP,'preparation_started')
        ticks=AdditionalTicks(prepare=True)
        try:value=case(data,inputs,identity,additions,benchmark=name.startswith('benchmark'),stress=name.endswith('stress'),ticks=ticks)
        except Exception as exc:
            old.record('strict_'+name,PREP,'preparation_error',error=str(exc));raise
        receipt=dict(completed=value['completed'],reason=value.get('reason'),network_calls=ticks.calls,
            source_sha256=ticks.files,repairs=repairs,performance_report=False)
        old.write(PREP/(name+'.json'),receipt);statuses.append(value['completed'])
        old.record('strict_'+name,PREP,'preparation_completed' if value['completed'] else 'preparation_blocked',reason=value.get('reason'))
        print(name,receipt['completed'],receipt['reason'],'requests',ticks.calls,flush=True)
    return dict(all_completed=all(statuses))


def verify(left,right,output):
    left,right,output=map(lambda p:Path(p).resolve(),(left,right,output))
    if left==right or output.exists():raise ValueError('Preserve independent runs and publications')
    reports=[old.read(p/'report.json') for p in (left,right)]
    refs={}
    for folder,report in zip((left,right),reports):
        if set(report['cases'])!=set(old.CASES):raise ValueError('Missing fixed cases')
        refs.update(report['source_sha256'])
        refs[str((folder/'report.json').relative_to(ROOT))]=old.sha(folder/'report.json')
        for row in report['cases'].values():refs[row['result']['path']]=row['result']['sha256']
    if reports[0]['source_sha256']!=reports[1]['source_sha256'] or reports[0]['repairs']!=reports[1]['repairs']:
        raise ValueError('Independent input versions differ')
    if old.file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Source changed before publication')
    for name in old.CASES:
        if old.read(ROOT/reports[0]['cases'][name]['result']['path'])!=old.read(ROOT/reports[1]['cases'][name]['result']['path']):
            raise ValueError('Independent account differs: '+name)
    value=dict(reports[0],source_sha256=refs,offline_identical=True,compared_cases=4)
    old.write(output,value);output.with_suffix('.sha256').write_text(old.sha(output)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--compare',nargs=2,type=Path)
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare and (args.output or args.compare):p.error('Preparation cannot publish')
    if not args.prepare and not args.output:p.error('--output required')
    result=prepare() if args.prepare else verify(*args.compare,args.output) if args.compare else run(args.output)
    print('all_completed',result['all_completed'],flush=True)
