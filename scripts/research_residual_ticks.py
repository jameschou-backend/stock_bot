#!/usr/bin/env python3
"""Offline, fixed-rule tick contrast for the current five-slot stock account."""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.historical_selector_ui import load as load_selector
from app.file_lock import file_lock
from app.finmind import fetch_dataset, FinMindError
from app.residual_slots_ui import load as load_baseline, REPORT as BASELINE
from scripts import research_residual_slots as parent
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from scripts.research_intraday_limit import TickCache, OUTPUT as TICK_SOURCE
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.execution_factorial import load_capital_terms
from skills.execution_resources import audit_resources
from skills.board_only_verified_replay import audit_verified_board_only
from skills.residual_slot_replay import audit_residual_slots
from skills.residual_tick_replay import ResidualTickReplay, ResidualTickBenchmark, audit_tick_plans
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.verified_backtest_tool import offline_only
from skills.trial_registry import append_trial_registry

OUTPUT = ROOT / '.cache/residual-ticks-20260928'
SPEC = ROOT / 'docs/prereg_residual_ticks_20260928.md'
CODE = [Path(__file__), ROOT/'skills/residual_tick_replay.py', SPEC,
        ROOT/'skills/intraday_limit_replay.py',ROOT/'scripts/research_intraday_limit.py']
PREP = ROOT / '.cache/residual-ticks-prepared-20260928'
CASES = ('strategy_normal','benchmark_normal','strategy_stress','benchmark_stress')


class RecordedTicks(TickCache):
    def __init__(self, prepare=False):
        super().__init__(PREP/'ticks', online=prepare, maximum=180)
        self.donor = TickCache(TICK_SOURCE/'ticks',online=False)
        self.files, self.queries = {}, []

    def fetch(self,dataset,sid,day,end=None):
        if not self.online:
            return super().fetch(dataset,sid,day,end)
        with file_lock(self.root/'budget.lock'):
            path=self.root/'budget.json'
            budget=read(path) if path.exists() else dict(reserved=0)
            if budget['reserved']>=self.maximum:
                raise ReplayDataUnavailable('Tick preparation 180-request ceiling reached')
            budget['reserved']+=1
            write(path,budget)
        self.calls+=1
        try:
            return fetch_dataset(dataset,date.fromisoformat(day),data_id=sid,token=self.token,
                                 requests_per_hour=5400,max_retries=0,timeout=30)
        except FinMindError as exc:
            raise ReplayDataUnavailable(f'Tick provider failed for {sid} {day}: {type(exc).__name__}') from exc

    def get(self, sid, day, market):
        path = self.root/f'{sid}-{day}.parquet'
        self.queries.append(dict(stock_id=sid,date=day,market=market))
        donor=self.donor.root/path.name
        if donor.exists() or donor.with_suffix('.json').exists():
            path=donor
            value=self.donor.get(sid,day,market)
        else:
            value = super().get(sid,day,market)
        self.files.update(file_identities([path,path.with_suffix('.json')],ROOT))
        return value


def case(data, inputs, identity, additions, *, benchmark, stress, ticks=None):
    ticks = ticks or RecordedTicks()
    feeds = ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
    overrides = (read(parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        read(parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp = TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
    args = (data.quotes,data.companies,data.days,data.entries,feeds,corp)
    common = dict(start=data.start,end=data.end,ticks=ticks,participation=.005 if stress else .01)
    engine = (ResidualTickBenchmark(*args,**common,stress_mode='slip90' if stress else 'control')
        if benchmark else ResidualTickReplay(*args,**common,factor_mask=1 if stress else 0,
            residual_policy='release',identity_report=identity,exit_signals=data.features,
            action_dates=list(zip(data.events.stock_id,data.events.event_date))))
    config = dict(benchmark=benchmark,stress=stress,additional_entry_delay=0,additional_exit_delay=0)
    try:
        account = engine.run()
        validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        if benchmark:
            audit = audit_resources(account,engine.resource_plans,opening_cash_only=True,lock_slots=False,lock_unused=True)
            audit.update(audit_verified_board_only(account,engine.board_decisions,engine.resource_plans))
        else:
            audit = audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
                engine.board_decisions,engine.residual_days,data.quotes)
        markets = {(r['date'],r['stock_id']):r['market'] for r in ticks.queries}
        markets = {(sid,day):m for (day,sid),m in markets.items()}
        audit.update(audit_tick_plans(account,ticks,markets,data.quotes,data.days,engine.corporate))
        result = dict(completed=True,summary=summarize(account),account=account,audit=audit)
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        result = dict(completed=False,summary=None,reason=str(exc),
            partial_diagnostics=dict(plans=engine.tick_plans,orders=engine.orders,trades=engine.trades,
                                     completed_sessions=len(engine.daily)))
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        result = dict(completed=False,summary=None,reason=str(exc),
            partial_diagnostics=dict(plans=engine.tick_plans,orders=engine.orders,trades=engine.trades,
                                     completed_sessions=len(engine.daily)))
    return dict(result,config=config,source_sha256=ticks.files,
                tick_queries=ticks.queries,network_calls=ticks.calls,live_qualified=False,unseen_validation=False)


def record(name,output,status,**extra):
    append_trial_registry(dict(source='residual_tick_execution',case=name,
        run=str(Path(output).relative_to(ROOT)),timestamp=datetime.now(timezone.utc).isoformat(),
        command=' '.join(sys.argv),preregistered=True,unseen_validation=False,status=status,**extra))


def prepare():
    data,inputs,identity=parent.parent.load_data(load_selector())
    additions=parent.parent.parent.load_corporate_completion(ROOT) | load_capital_terms(ROOT)[0]
    for name in ('strategy_normal','strategy_stress'):
        record(name,PREP,'preparation_started')
        ticks=RecordedTicks(prepare=True)
        try:
            value=case(data,inputs,identity,additions,benchmark=False,stress=name.endswith('stress'),ticks=ticks)
        except Exception as exc:
            record(name,PREP,'preparation_error',error=str(exc));raise
        # Preparation discovers dependencies; no performance is published.
        receipt=dict(completed=value['completed'],reason=value.get('reason'),network_calls=ticks.calls,
                     source_sha256=ticks.files,queries=ticks.queries,performance_report=False)
        write(PREP/(name+'.json'),receipt)
        record(name,PREP,'preparation_completed' if value['completed'] else 'preparation_blocked',reason=value.get('reason'))
        print(name,receipt['completed'],receipt['reason'],'requests',ticks.calls,flush=True)
    return dict(all_completed=all(read(PREP/(name+'.json'))['completed'] for name in ('strategy_normal','strategy_stress')))


def run(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT) or output == OUTPUT:
        raise ValueError('Use a new residual-ticks output path; preserve every attempt')
    started = time.monotonic()
    baseline = load_baseline()
    refs = dict(baseline['source_sha256'])
    refs.update(file_identities([BASELINE,BASELINE.with_suffix('.sha256'),*CODE],ROOT))
    write(output/'identity.json',refs)  # Rules fixed before account outcomes.
    cases = {}
    with offline_only():
        data,inputs,identity = parent.parent.load_data(load_selector())
        additions = parent.parent.parent.load_corporate_completion(ROOT) | load_capital_terms(ROOT)[0]
        neutral = parent.case(data,inputs,identity,additions,0,'release')
        expected = read(ROOT/baseline['cases']['release_0']['result']['path'])
        if not neutral['completed'] or encoded(neutral['account']) != encoded(expected['account']):
            raise ValueError('Current five-slot baseline did not reproduce exactly')
        for name in CASES:
            print('running',name,flush=True)
            record(name,output,'started')
            try:
                result = case(data,inputs,identity,additions,
                    benchmark=name.startswith('benchmark'),stress=name.endswith('stress'))
            except Exception as exc:
                record(name,output,'error',error=str(exc));raise
            record(name,output,'completed' if result['completed'] else 'blocked',summary=result['summary'],reason=result.get('reason'))
            cases[name] = result
            write(output/'cases'/f'{name}.json',result)
            print(name,(result.get('summary') or {}).get('total_return',result.get('reason')),flush=True)
            refs.update(result['source_sha256'])
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:
        raise ValueError('Research source changed during execution')
    write(output/'identity.json',refs)
    comparisons = {}
    for mode in ('normal','stress'):
        stock,benchmark = (cases[f'{kind}_{mode}'] for kind in ('strategy','benchmark'))
        complete = stock['completed'] and benchmark['completed']
        comparisons[mode] = dict(completed=complete,excess_return=None)
        if complete:
            if [r['date'] for r in stock['account']['daily']] != [r['date'] for r in benchmark['account']['daily']]:
                raise ValueError('Comparison calendars differ')
            comparisons[mode]['excess_return'] = stock['summary']['total_return']-benchmark['summary']['total_return']
    report = dict(schema='residual_ticks_v1',start=data.start,end=data.end,candidate_count=len(data.entries),
        baseline_reproduced=True,baseline_summary=neutral['summary'],comparison=comparisons,
        all_completed=all(r['completed'] for r in cases.values()),source_sha256=refs,
        cases={k:dict(completed=v['completed'],summary=v['summary'],reason=v.get('reason'),
            result=dict(path=str((output/'cases'/f'{k}.json').relative_to(ROOT)),sha256=sha(output/'cases'/f'{k}.json')))
            for k,v in cases.items()},elapsed_seconds=round(time.monotonic()-started,3),
        network_calls=0,database_writes=0,live_qualified=False,unseen_validation=False)
    write(output/'report.json',report)
    return report


def verify(left,right,output):
    left,right,output = map(lambda p:Path(p).resolve(),(left,right,output))
    if left==right or output.exists():
        raise ValueError('Use separate replays and a new verification path')
    reports = [read(p/'report.json') for p in (left,right)]
    refs = {}
    for folder,report in zip((left,right),reports):
        if set(report['cases'])!=set(CASES) or report['baseline_reproduced'] is not True:
            raise ValueError('Incomplete experiment inventory')
        refs.update(report['source_sha256'])
        refs[str((folder/'report.json').relative_to(ROOT))]=sha(folder/'report.json')
        for row in report['cases'].values():
            refs[row['result']['path']] = row['result']['sha256']
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:
        raise ValueError('Offline evidence changed')
    if reports[0]['source_sha256']!=reports[1]['source_sha256']:
        raise ValueError('Offline source versions differ')
    for name in CASES:
        if read(ROOT/reports[0]['cases'][name]['result']['path']) != read(ROOT/reports[1]['cases'][name]['result']['path']):
            raise ValueError('Full result differs: '+name)
    result = dict(reports[0],offline_identical=True,compared_cases=4,source_sha256=refs,
                  reproducible_includes_blocked=True)
    write(output,result)
    output.with_suffix('.sha256').write_text(sha(output)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--compare',type=Path,nargs=2)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare and (args.output or args.compare):
        parser.error('Preparation and offline publication are separate commands')
    if not args.prepare and not args.output:
        parser.error('--output is required for offline replay')
    result=prepare() if args.prepare else verify(*args.compare,args.output) if args.compare else run(args.output)
    print('all_completed',result['all_completed'],flush=True)
