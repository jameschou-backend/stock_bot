#!/usr/bin/env python3
"""Fixed sixteen-account incremental comparison, source gate and offline repeat."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.config import load_config
from app.finmind import fetch_dataset
from app.file_lock import file_lock
from app.residual_slots_ui import load as load_baseline, REPORT as BASELINE
from app.historical_selector_ui import load as load_selector
from scripts import research_residual_slots as old
from scripts.prepare_sector_account_sources import RequestBudget, initialize
from scripts.prepare_leader_chips import OUTPUT as CHIPS, SPEC
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from skills.account_source_preflight import source_identity, gate, CachedPreparationFeeds
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.execution_factorial import stock_pnl, load_capital_terms, path_comparison
from skills.leader_chip_replay import LeaderChipReplay, features, causality, audit_order, LAGS
from skills.million_replay import UnresolvedAction
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.residual_slot_replay import audit_residual_slots
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

PREP = ROOT / '.cache/leader-chip-prepared-20260928'
OUTPUT = ROOT / '.cache/leader-chip-accounts-20260928'
CODE = [Path(__file__), ROOT/'skills/leader_chip_replay.py', ROOT/'scripts/prepare_leader_chips.py',
        ROOT/'skills/theme_chips.py', ROOT/'skills/launch_flows.py', SPEC,
        ROOT/'docs/leader_chip_corporate_20260928.json',ROOT/'docs/evidence_leader_chip_20260928.json']


def configurations():
    return [(f'baseline_{m}', 'baseline', 'main', m) for m in (0,7)] + [
        (f'{arm}_{timing}_{m}', arm, timing, m) for arm in ('rank','filter','coverage')
        for timing in LAGS for m in (0,7)]


class ExecutionBudget(RequestBudget):
    def finmind(self, dataset, start, end, **kwargs):
        if dataset not in ('TaiwanStockPriceLimit','TaiwanStockDividend'):
            raise ValueError('Unexpected execution dataset')
        kwargs.update(max_retries=0, requests_per_hour=5400, timeout=45)
        identity=f'{dataset}:{kwargs.get("data_id")}:{start}:{end}'
        return self.call('finmind', identity, fetch_dataset, dataset, start, end, **kwargs)


def load_inputs():
    baseline = load_baseline(); selector = load_selector()
    manifest = read(CHIPS/'manifest.json')
    for p, digest in manifest['sources'].items():
        if sha(ROOT/p) != digest:
            raise ValueError('Chip source changed: '+p)
    data, source, identity = old.parent.load_data(selector)
    if data.entries != read(ROOT/manifest['signals'])['entries'] or len(data.entries)!=454:
        raise ValueError('Signal cohort changed')
    flows, weekly = (pd.read_parquet(CHIPS/name) for name in ('flows.parquet','weekly.parquet'))
    scores = {timing:features(data.entries,data.days,flows,weekly,timing) for timing in LAGS}
    additions = old.parent.parent.load_corporate_completion(ROOT) | load_capital_terms(ROOT)[0]
    extra=read(ROOT/'docs/leader_chip_corporate_20260928.json')
    if file_identities([ROOT/p for p in extra['evidence_sha256']],ROOT)!=extra['evidence_sha256']:
        raise ValueError('Supplemental corporate evidence changed')
    if set(extra['overrides']) & set(additions):
        raise ValueError('Supplement must not change old corporate terms')
    additions.update(extra['overrides'])
    return baseline,data,source,identity,scores,additions,flows,weekly


def case(data, inputs, identity, additions, arm, timing, mask, scores, *, budget=None):
    online = budget is not None
    token = load_config().finmind_token if online else None
    feeds = CachedPreparationFeeds(inputs/'execution-feeds',offline=not online,token=token,
        **(dict(finmind_fetch=budget.finmind,http_get=budget.official) if online else {}))
    overrides = (read(old.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        read(old.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp = TrackedCorporateActions(data.events,inputs/'dividends',token,offline=not online,overrides=overrides)
    engine = LeaderChipReplay(data.quotes,data.companies,data.days,data.entries,feeds,corp,
        start=data.start,end=data.end,identity_report=identity,factor_mask=mask,residual_policy='release',
        exit_signals=data.features,action_dates=list(zip(data.events.stock_id,data.events.event_date)),
        chip_arm=arm,chip_scores=scores)
    config = dict(arm=arm,timing=timing,factor_mask=mask,board_only=True,position_count=5,benchmark=False)
    try:
        account=engine.run()
        validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
        audit=audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
            engine.board_decisions,engine.residual_days,data.quotes)
        chip_audit=audit_order(engine.chip_orders,scores,arm)
        # Reuse the engine's issuer-specific whole-NTD adapter. Unknown net
        # fractional cash remains an unavailable gross receivable, not cash.
        for action in account['corporate_actions']:
            if action['kind']=='stock_dividend' and action.get('action_id') in (
                    '2887-2024-08-06-stock','2887-2026-07-21-stock'):
                terms=action['fractional_settlement']
                if terms['rounding']!='floor_ntd' or terms['payment_date'] is not None:
                    raise ValueError('Unknown fractional cash cannot become spendable')
    except (ReplayDataUnavailable,UnresolvedAction) as exc:
        return dict(completed=False,config=config,reason=str(exc))
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(completed=False,config=config,reason=str(exc))
    return dict(completed=True,config=config,account=account,summary=summarize(account),audit=audit,
        chip_audit=chip_audit,chip_orders=engine.chip_orders,stock_pnl=stock_pnl(account,engine.marks),
        resource_plans=engine.resource_plans,slot_decisions=engine.slot_decisions,
        board_decisions=engine.board_decisions,identity_decisions=engine.identity_decisions,
        residual_days=engine.residual_days,live_qualified=False,unseen_validation=False)


def fingerprint(baseline):
    refs=dict(baseline['source_sha256'])
    refs.update(file_identities([BASELINE,BASELINE.with_suffix('.sha256'),CHIPS/'manifest.json',*CODE],ROOT))
    refs.update(read(CHIPS/'manifest.json')['sources'])
    refs.update({str((PREP/'inputs'/p).relative_to(ROOT)):digest
        for p,digest in source_identity(PREP/'inputs').items()})
    return refs


def prepare():
    baseline,data,source,identity,scores,additions,_,_=load_inputs()
    inputs=initialize(PREP,source)
    if (PREP/'preparation.json').exists():
        archive=PREP/'history'/('preparation-'+sha(PREP/'preparation.json')+'.json')
        if not archive.exists():write(archive,read(PREP/'preparation.json'))
    budget=ExecutionBudget(PREP/'budget.json',maximum={'finmind':150,'official':0})
    cases={}
    for name,arm,timing,mask in configurations():
        print('preparing',name,flush=True)
        with patch('skills.replay_corporate_actions.fetch_dataset',budget.finmind):
            result=case(data,inputs,identity,additions,arm,timing,mask,scores[timing],budget=budget)
        if arm=='baseline' and result['completed']:
            expected=read(ROOT/baseline['cases']['release_'+str(mask)]['result']['path'])
            if encoded(result['account'])!=encoded(expected['account']):
                raise ValueError('Preparation baseline does not reproduce sealed account')
        cases[name]=dict(completed=result['completed'],reason=result.get('reason'))
        write(PREP/'progress.json',dict(cases=cases,performance_report=False))
        print(name,cases[name],flush=True)
    for mode in ('control','combined'):
        cases['benchmark_'+mode]=dict(completed=True,reason=None,verified_reuse=True)
    receipt=dict(cases=cases,identity=fingerprint(baseline),performance_report=False,
        budget=budget.state,live_qualified=False)
    write(PREP/'preparation.json',receipt)
    print('all paths ready',all(c['completed'] for c in cases.values()),flush=True)


def analyze(cases):
    result={}
    for name, row in cases.items():
        if name.startswith('benchmark_'):continue
        mask=row['config']['factor_mask']
        benchmark=cases['benchmark_control' if mask==0 else 'benchmark_combined']
        baseline=cases['baseline_'+str(mask)]
        left,right=row['account']['daily'],benchmark['account']['daily']
        if [r['date'] for r in left]!=[r['date'] for r in right]:
            raise ValueError('Account and benchmark calendars differ')
        rolling=[dict(start=left[i-251]['date'],end=left[i]['date'],
            excess=(left[i]['nav']/left[i-251]['opening_nav'] - right[i]['nav']/right[i-251]['opening_nav']))
            for i in range(251,len(left))]
        profits=sorted([p['profit'] for p in row['stock_pnl'].values() if p['profit']>0],reverse=True)
        comparison=dict(total_return=row['summary']['total_return'],max_drawdown=row['summary']['max_drawdown'],
            versus_baseline=row['summary']['total_return']-baseline['summary']['total_return'],
            versus_0050=row['summary']['total_return']-benchmark['summary']['total_return'],
            annual_excess=[dict(year=a['year'],excess=a['total_return']-b['total_return'])
                for a,b in zip(row['summary']['annual'],benchmark['summary']['annual'])],
            rolling252=rolling,rolling252_win_rate=sum(r['excess']>0 for r in rolling)/len(rolling),
            rolling252_worst_excess=min(r['excess'] for r in rolling),
            top_two_positive_profit_share=sum(profits[:2])/sum(profits) if profits else None,
            entry_paths=path_comparison(baseline['account'],row['account']))
        if row['config']['arm']=='filter':
            control=cases[f'coverage_{row["config"]["timing"]}_{mask}']
            comparison['versus_coverage']=row['summary']['total_return']-control['summary']['total_return']
        result[name]=comparison
    retained={}
    for arm in ('rank','filter'):
        rows=[result[f'{arm}_{t}_{m}'] for t in LAGS for m in (0,7)]
        retained[arm]=all(r['versus_baseline']>0 and r['max_drawdown']>=-.5 for r in rows) and all(
            result[f'{arm}_main_{m}']['versus_0050']>0 for m in (0,7))
    return dict(cases=result,preregistered_screen=retained,adopted=False,live_qualified=False)


def run(output):
    output=Path(output).resolve();started=time.monotonic()
    if output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT:
        raise ValueError('Choose a new immutable run directory')
    with offline_only():
        baseline,data,_,identity,scores,additions,flows,weekly=load_inputs()
        refs=fingerprint(baseline)
        names=[r[0] for r in configurations()]+['benchmark_control','benchmark_combined']
        status=gate(read(PREP/'preparation.json'),refs,names)
        if not status['ready']:
            raise ValueError('Preparation gate blocked: '+encoded(status))
        write(output/'identity.json',refs);write(output/'features.json',scores)
        write(output/'causality.json',causality(data.entries,data.days,flows,weekly))
        cases={}
        for name,arm,timing,mask in configurations():
            print('running',name,flush=True)
            record=dict(source='leader_chip_increment',case=name,command=' '.join(sys.argv),
                run=str(output.relative_to(ROOT)),preregistered=True,
                timestamp=datetime.now(timezone.utc).isoformat(),unseen_validation=False)
            append_trial_registry(dict(record,status='started'))
            try:
                result=case(data,PREP/'inputs',identity,additions,arm,timing,mask,scores[timing])
                if not result['completed']:raise ValueError(result['reason'])
                if arm=='baseline':
                    expected=read(ROOT/baseline['cases']['release_'+str(mask)]['result']['path'])
                    if encoded(result['account'])!=encoded(expected['account']):
                        raise ValueError('Baseline full-account reproduction failed')
            except Exception as exc:
                append_trial_registry(dict(record,status='error',error=str(exc)));raise
            append_trial_registry(dict(record,status='completed',summary=result['summary']))
            cases[name]=result;write(output/'cases'/(name+'.json'),result)
            print(name,'complete',flush=True)
        for mode in ('control','combined'):
            name='benchmark_'+mode
            value=read(ROOT/baseline['cases'][name]['result']['path'])
            cases[name]=value;write(output/'cases'/(name+'.json'),value)
        analysis=analyze(cases);write(output/'analysis.json',analysis)
        if fingerprint(baseline)!=refs:raise ValueError('Source changed during replay')
        report=dict(schema='leader_chip_accounts_v1',all_completed=True,case_count=len(cases),candidate_count=454,
            start=data.start,end=data.end,initial_cash=1_000_000,network_calls=0,database_writes=0,
            source_gate=status,baseline_full_account_reproduced=True,
            coverage={t:dict(events=len(s),known=sum(r['known'] for r in s.values()),
                passed=sum(r['passed'] is True for r in s.values()),
                reasons=dict(Counter(r['reason'] for r in s.values()))) for t,s in scores.items()},
            cases={name:dict(summary=r['summary'],config=r['config'],result=dict(
                path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),sha256=sha(output/'cases'/(name+'.json'))))
                for name,r in cases.items()},elapsed_seconds=round(time.monotonic()-started,3),
            live_qualified=False,strict_data_ready=False,unseen_validation=False,adopted=False)
        write(output/'report.json',report)
        write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p)
            for p in output.rglob('*.json') if p.name!='manifest.json'}))
        return report


def verify(left,right,output):
    left,right,output=map(lambda p:Path(p).resolve(),(left,right,output))
    if left==right or output.exists():raise ValueError('Independent immutable runs required')
    reports=[];refs={}
    for folder in (left,right):
        report=read(folder/'report.json');reports.append(report)
        if not report['all_completed'] or report['case_count']!=16:raise ValueError('Incomplete study')
        for name,digest in read(folder/'manifest.json')['files_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('Run artifact changed')
            refs[str((folder/name).relative_to(ROOT))]=digest
        refs[str((folder/'manifest.json').relative_to(ROOT))]=sha(folder/'manifest.json')
    for name in ['identity.json','features.json','analysis.json','causality.json']+[
            'cases/'+c+'.json' for c in reports[0]['cases']]:
        if read(left/name)!=read(right/name):raise ValueError('Offline replay differs: '+name)
    refs.update(read(left/'identity.json'))
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Sources changed before verification')
    proof=dict(schema='leader_chip_offline_v1',passed=True,compared_cases=16,network_calls=0,
        source_sha256=refs,runs=[str(p.relative_to(ROOT)) for p in (left,right)])
    write(output,proof);output.with_suffix('.sha256').write_text(sha(output)+'\n')
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--compare',nargs=2,type=Path);args=parser.parse_args()
    with file_lock(ROOT/'.cache/leader-chip-accounts.lock',timeout=0):
        if args.prepare:prepare()
        elif args.compare:
            if not args.output:parser.error('--output required')
            print(verify(*args.compare,args.output)['passed'])
        else:
            if not args.output:parser.error('--output required')
            print('completed',run(args.output)['case_count'])
