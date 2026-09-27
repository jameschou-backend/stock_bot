#!/usr/bin/env python3
"""Offline concentration/actor cross-study using sealed broad-cohort sources."""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import argparse
import copy
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from app.theme_chips_ui import load as load_chips, PUBLICATION as CHIP_PUBLICATION
from scripts.research_first_bar import inputs
from scripts.research_absorption import PUBLICATION as ABS_PUBLICATION
from scripts.research_exit_scenarios import read,write,sha
from scripts.research_surge_anatomy import clean
from scripts.research_theme_catalyst import source_closure,CACHE
from scripts.audit_current_causality_20260925 import verify_hashes
from skills.holder_flow import ARMS,ACCOUNT_ARMS,combine,orders,summaries,matched_pairs,comparisons
from skills.theme_chips import changes
from skills.sector_account_replay import run_case
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

SPEC=ROOT/'docs/prereg_holder_flow_20260927.md'
PUBLICATION=ROOT/'artifacts/forward_simulation/holder_flow_20260927.json'


def trial_definitions():
    """One explicit registry entry for every evaluated arm/time/window variant."""
    return [dict(arm=arm,flow_lag=flow_lag,chip_lag=chip_lag,horizon=horizon)
            for arm in ARMS for flow_lag in (1,3) for chip_lag in (8,15) for horizon in (20,60)]


def inherited(expected):
    if sha(ABS_PUBLICATION)!=ABS_PUBLICATION.with_suffix('.sha256').read_text().strip():
        raise ValueError('Absorption publication changed')
    pub=read(ABS_PUBLICATION); report_path=ROOT/pub['report']['path'];report=read(report_path)
    proof=pub['reproducibility']
    if (pub['schema']!='absorption_publication_v1' or report['schema']!='absorption_v1'
            or proof['passed'] is not True or len(proof['runs'])!=2
            or len({r['path'] for r in proof['runs']})!=2 or report['completed'] is not True
            or sha(report_path)!=pub['report']['sha256']):
        raise ValueError('Unbound inherited research')
    if any(report[k] is not False for k in ('live_qualified','adopted','unseen_validation')):
        raise ValueError('Unexpected inherited qualification')
    expected.update({ROOT/p:h for p,h in report['source_sha256'].items()})
    expected[ABS_PUBLICATION]=sha(ABS_PUBLICATION)
    expected[ABS_PUBLICATION.with_suffix('.sha256')]=sha(ABS_PUBLICATION.with_suffix('.sha256'))
    bound=False
    for descriptor in proof['runs']:
        p=ROOT/descriptor['path'];m=read(p);expected[p]=descriptor['sha256']
        if m['source_sha256']!=report['source_sha256'] or {k:h for k,h in m['files_sha256'].items() if k!='report.json'}!=proof['files_sha256']:
            raise ValueError('Inherited runs differ')
        expected.update({p.parent/name:h for name,h in m['files_sha256'].items()})
        if p.parent==report_path.parent:
            bound=m['files_sha256']['report.json']==pub['report']['sha256']
    if not bound:raise ValueError('Report is not in the proof')
    verify_hashes(expected)
    base=pd.read_csv(report_path.parent/'candidates.csv',dtype={'stock_id':str},low_memory=False)
    labels=pd.read_csv(report_path.parent/'outcomes.csv',dtype={'stock_id':str},low_memory=False)
    columns=['event_id','flow_lag','horizon','endpoint','entry_date','phase','year','label_reason',
        'forward_return','benchmark_return','excess_return','surge','fee_return','fee_excess','stress_fee_return','stress_fee_excess']
    return base,labels[columns],report['causality_checks']


def causality(base,weekly,table,days):
    checks=[]
    for lag in (8,15):
        for date in ('2023-12-29','2024-12-31','2026-03-31'):
            cutoff=pd.Timestamp(date); query=base[base.signal_date.le(date)]
            baseline=table[table.chip_lag.eq(lag)&table.signal_date.le(date)].reset_index(drop=True)
            # Also alter observations before T which were not yet publishable at T.
            unavailable=pd.to_datetime(weekly.date)+pd.Timedelta(days=lag)>cutoff
            for mode in ('truncate','mutate'):
                altered=weekly.loc[~unavailable].copy() if mode=='truncate' else weekly.copy()
                if mode=='mutate':
                    altered.loc[unavailable,['large_pct','small_pct']]=.9
                    altered.loc[unavailable,['large_units','total_units']]*=7
                    altered.loc[unavailable,'valid']=False
                actual=combine(query,changes(altered),lag).reset_index(drop=True)
                pd.testing.assert_frame_equal(baseline,actual)
                previous=orders(baseline,days); changed=orders(actual,days)
                for arm in ACCOUNT_ARMS:
                    visible=lambda rows:[r for r in rows if r['entry_date']<=date]
                    if visible(previous[arm])!=visible(changed[arm]):raise ValueError('Unavailable data changed an order')
                checks.append(dict(chip_lag=lag,cutoff=date,mode=mode,passed=True,rows=len(actual)))
        print('holder causality lag',lag,'passed',flush=True)
    return checks


def run(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache'):raise ValueError('Use a new immutable cache directory')
    began=time.perf_counter()
    with file_lock(ROOT/'.cache/holder-flow.lock',timeout=0),offline_only():
        frames,_,weekly,data,expected=inputs()
        closure,overrides=source_closure(CACHE);expected.update(closure)
        base,labels,inherited_checks=inherited(expected)
        chip=load_chips();p=ROOT/CHIP_PUBLICATION
        for path in (p,p.with_suffix('.sha256')):expected[path]=sha(path)
        expected.update({ROOT/a['path']:a['sha256'] for a in chip['artifacts'].values()})
        for name in ('scripts/research_holder_flow.py','skills/holder_flow.py','tests/test_holder_flow.py','docs/prereg_holder_flow_20260927.md'):
            expected[ROOT/name]=sha(ROOT/name)
        verify_hashes(expected)
        for definition in trial_definitions():
            append_trial_registry(dict(source='holder_flow',status='started',**definition,
                timestamp=datetime.now(timezone.utc).isoformat(),output=str(output.relative_to(ROOT)),
                preregistration_sha256=sha(SPEC)))
        recomputed=changes(weekly)
        pd.testing.assert_frame_equal(weekly.reset_index(drop=True),recomputed.reset_index(drop=True),check_dtype=False)
        table=pd.concat([combine(base,recomputed,lag) for lag in (8,15)],ignore_index=True)
        entries=orders(table,frames[0].index);pairs=matched_pairs(table)
        print('observations',len(table),'orders',{k:len(v) for k,v in entries.items()},flush=True)
        checks=causality(base,weekly,table,frames[0].index)
        future=table.merge(labels,on=['event_id','flow_lag'],validate='many_to_many')
        if future.duplicated(['event_id','flow_lag','chip_lag','horizon']).any():raise ValueError('Duplicate joined result')
        paired,comparison=comparisons(future,pairs)
        stats=summaries(future);annual=summaries(future,annual=True);cells=summaries(future,cells=True)
        output.mkdir(parents=True)
        for name,f in [('candidates',table),('outcomes',future),('statistics',stats),('annual',annual),
                       ('cells',cells),('pairs',paired),('comparisons',comparison)]:
            f.to_csv(output/(name+'.csv'),index=False,encoding='utf-8-sig')
        write(output/'entries.json',entries)
        cases={}
        for stress in ('control','combined'):
            for arm in ('benchmark',*ACCOUNT_ARMS):
                selected=replace(data,entries_by_arm={'relative_strength':entries.get(arm,[])})
                config=dict(arm='benchmark' if arm=='benchmark' else 'relative_strength',stress=stress,
                    benchmark=arm=='benchmark',board_only=False,position_count=0 if arm=='benchmark' else 5)
                append_trial_registry(dict(source='holder_flow_account',status='started',arm=arm,stress=stress,
                    timestamp=datetime.now(timezone.utc).isoformat(),output=str(output.relative_to(ROOT))))
                result=run_case(selected,config,CACHE,overrides)
                key=arm+'_'+stress;path=output/(key+'.json');write(path,clean(result))
                cases[key]=dict(completed=result['completed'],reason=result.get('reason'),summary=result.get('summary'),
                    trade_count=len(result.get('account',result.get('partial_account',{})).get('trades',[])),
                    path=str(path.relative_to(ROOT)),sha256=sha(path))
                append_trial_registry(dict(source='holder_flow_account',status='completed' if result['completed'] else 'incomplete',
                    arm=arm,stress=stress,reason=result.get('reason'),timestamp=datetime.now(timezone.utc).isoformat(),
                    output=str(output.relative_to(ROOT))))
                print(key,'completed' if result['completed'] else result.get('reason'),flush=True)
        for key,row in cases.items():
            bm=cases['benchmark_'+key.rsplit('_',1)[1]]
            row['excess_return']=(row['summary']['total_return']-bm['summary']['total_return']
                                 if row['completed'] and bm['completed'] else None)
        verify_hashes(expected)
        report=clean(dict(schema='holder_flow_v1',completed=True,live_qualified=False,adopted=False,
            unseen_validation=False,historical_first_publication_verified=False,historical_universe_verified=False,
            fee_results_are_account_returns=False,theme_factor_tested=False,
            start='2022-01-03',end='2026-09-09',candidate_rows=len(table),anchors=table.signal_date.nunique(),
            main_candidate_rows=int((~table.named_case).sum()),signal_counts={a:len(v) for a,v in entries.items()},
            statistics=stats.to_dict('records'),comparisons=comparison.to_dict('records'),cases=cases,
            all_accounts_completed=all(v['completed'] for v in cases.values()),
            collection_new_requests=0,research_network_calls=0,database_writes=0,
            causality_checks=checks,inherited_price_flow_causality=inherited_checks,
            source_sha256={str(p.relative_to(ROOT)):h for p,h in expected.items()},
            elapsed_seconds=round(time.perf_counter()-began,3),
            limitations=['Previously studied history, current company and industry snapshots; no untouched validation.',
                'Holder-size buckets are not identity, intention or observed trades; foreign/trust signs include tiny net values.',
                'Fixed anchors miss launches between observations; missing data is unknown.',
                'Publication lags are assumptions; original first-publication versions remain unverified.',
                'Date-block intervals are descriptive, not corrected for repeated research or overlapping horizons.',
                'Matched controls allow replacement, and unmatched observations cannot prove general effects.',
                'Event diagnostics are not finite-capital account returns; failed accounts do not publish partial returns.']))
        write(output/'report.json',report)
        write(output/'manifest.json',dict(schema='holder_flow_manifest_v1',source_sha256=report['source_sha256'],
            files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
        for definition in trial_definitions():
            append_trial_registry(dict(source='holder_flow',status='completed_study',**definition,
                timestamp=datetime.now(timezone.utc).isoformat(),output=str(output.relative_to(ROOT))))
    print({'elapsed_seconds':report['elapsed_seconds']},flush=True)


def publish(first,second):
    folders=[Path(p).resolve() for p in (first,second)]
    if folders[0]==folders[1] or any(not p.is_relative_to(ROOT/'.cache') for p in folders):raise ValueError('Two independent runs required')
    reports=[];manifests=[]
    for folder in folders:
        r=read(folder/'report.json');m=read(folder/'manifest.json')
        if (r['schema']!='holder_flow_v1' or m['schema']!='holder_flow_manifest_v1' or r['completed'] is not True
                or m['source_sha256']!=r['source_sha256'] or m['files_sha256'].get('report.json')!=sha(folder/'report.json')
                or any(r[k] is not False for k in ('live_qualified','adopted','unseen_validation',
                    'historical_first_publication_verified','historical_universe_verified','fee_results_are_account_returns','theme_factor_tested'))):
            raise ValueError('Unbound or unsupported report')
        verify_hashes({ROOT/p:h for p,h in m['source_sha256'].items()})
        for name,h in m['files_sha256'].items():
            if Path(name).name!=name or sha(folder/name)!=h:raise ValueError('Study output changed')
        value=copy.deepcopy(r);value.pop('elapsed_seconds')
        for row in value['cases'].values():row['path']=Path(row['path']).name
        reports.append(value);manifests.append(m)
    files=lambda m:{k:v for k,v in m['files_sha256'].items() if k!='report.json'}
    if reports[0]!=reports[1] or files(manifests[0])!=files(manifests[1]):raise ValueError('Independent studies differ')
    write(PUBLICATION,dict(schema='holder_flow_publication_v1',live_qualified=False,adopted=False,
        report=dict(path=str((folders[0]/'report.json').relative_to(ROOT)),sha256=sha(folders[0]/'report.json')),
        reproducibility=dict(passed=True,files_sha256=files(manifests[0]),
            runs=[dict(path=str((p/'manifest.json').relative_to(ROOT)),sha256=sha(p/'manifest.json')) for p in folders])))
    PUBLICATION.with_suffix('.sha256').write_text(sha(PUBLICATION)+'\n')
    print('Published identical holder/flow studies')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path);group.add_argument('--publish',nargs=2,type=Path)
    args=parser.parse_args()
    if args.publish:publish(*args.publish)
    else:run(args.output)
