#!/usr/bin/env python3
"""Frozen broad-cohort absorption study, six rules, two lags and strict accounts."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import copy
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from scripts.prepare_absorption import OUTPUT as INPUT, SPEC
from scripts.research_first_bar import inputs
from scripts.research_exit_scenarios import read, write, sha
from scripts.research_surge_anatomy import clean
from scripts.research_theme_catalyst import source_closure, CACHE
from scripts.audit_current_causality_20260925 import verify_hashes
from skills.absorption import ARMS, candidates, orders, outcomes, statistics, matched_pairs, comparisons
from skills.sector_account_replay import run_case
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

PUBLICATION = ROOT/'artifacts/forward_simulation/absorption_20260927.json'


def causality(frames, companies, normalized, baseline, entries):
    close, _, raw, volume = frames; checks=[]
    for date in ('2023-12-29','2024-12-31','2026-03-31'):
        stop=pd.Timestamp(date)
        for mode in ('truncate','mutate'):
            changed=[]
            for frame in (close,raw,volume):
                f=frame.loc[:stop].copy() if mode=='truncate' else frame.copy()
                if mode=='mutate': f.loc[f.index>stop]*=2.7
                changed.append(f)
            inst=normalized[normalized.date<=stop].copy() if mode=='truncate' else normalized.copy()
            if mode=='mutate': inst.loc[inst.date>stop,['foreign','trust','dealer']]*=-11
            actual=candidates(*changed,companies,inst)
            past=lambda f:f[f.signal_date.le(date)].reset_index(drop=True)
            pd.testing.assert_frame_equal(past(baseline),past(actual))
            actual_entries=orders(actual,changed[0].index)
            for arm in ARMS:
                visible=lambda r:[e for e in r if e['entry_date']<=date]
                if visible(entries[arm])!=visible(actual_entries[arm]): raise ValueError('Future changed earlier orders')
            checks.append(dict(cutoff=date,mode=mode,passed=True,rows=len(past(actual))))
            print('causality',date,mode,flush=True)
    return checks


def run(output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache'): raise ValueError('Use a new immutable cache directory')
    began=time.perf_counter()
    with file_lock(ROOT/'.cache/absorption.lock',timeout=0),offline_only():
        manifest=read(INPUT/'manifest.json')
        expected={ROOT/p:h for p,h in manifest['source_sha256'].items()}
        expected[INPUT/'manifest.json']=sha(INPUT/'manifest.json')
        frames,_,_,data,inherited=inputs(); expected.update(inherited)
        closure,overrides=source_closure(CACHE); expected.update(closure)
        for name in ('scripts/research_absorption.py','skills/absorption.py','tests/test_absorption.py',
                     'scripts/prepare_absorption.py','skills/launch_flows.py','skills/launch_warning.py',
                     'skills/early_strength.py','skills/surge_anatomy.py','scripts/research_first_bar.py',
                     'scripts/research_theme_catalyst.py'):
            expected[ROOT/name]=sha(ROOT/name)
        verify_hashes(expected)
        for arm in ARMS:
            append_trial_registry(dict(source='absorption',status='started',arm=arm,
                timestamp=datetime.now(timezone.utc).isoformat(),output=str(output.relative_to(ROOT)),
                preregistration_sha256=sha(SPEC),flow_lags=[1,3],horizons=[20,60]))
        normalized=pd.read_parquet(INPUT/'normalized.parquet'); close,quality,raw,volume=frames
        table=candidates(close,raw,volume,data.companies,normalized); entries=orders(table,close.index)
        pairs=matched_pairs(table)
        print('candidate observations',len(table),'orders',{a:len(r) for a,r in entries.items()},flush=True)
        checks=causality(frames,data.companies,normalized,table,entries)
        future=outcomes(table,close,quality); paired,comparison=comparisons(future,pairs)
        stats=statistics(future); annual=statistics(future,annual=True)
        output.mkdir(parents=True)
        for name,f in [('candidates',table),('outcomes',future),('statistics',stats),('annual',annual),
                       ('pairs',paired),('comparisons',comparison)]:
            f.to_csv(output/(name+'.csv'),index=False,encoding='utf-8-sig')
        write(output/'entries.json',entries)
        cases={}
        for stress in ('control','combined'):
            for arm in ('benchmark',*ARMS):
                selected=replace(data,entries_by_arm={'relative_strength':entries.get(arm,[])})
                config=dict(arm='benchmark' if arm=='benchmark' else 'relative_strength',stress=stress,
                    benchmark=arm=='benchmark',board_only=False,position_count=0 if arm=='benchmark' else 5)
                result=run_case(selected,config,CACHE,overrides)
                key=arm+'_'+stress; p=output/(key+'.json'); write(p,clean(result))
                row=dict(completed=result['completed'],reason=result.get('reason'),summary=result.get('summary'),
                    trade_count=len(result.get('account',result.get('partial_account',{})).get('trades',[])),
                    path=str(p.relative_to(ROOT)),sha256=sha(p))
                cases[key]=row; print(key,'completed' if row['completed'] else row['reason'],flush=True)
        for key,row in cases.items():
            stress=key.rsplit('_',1)[1]; bm=cases['benchmark_'+stress]
            row['excess_return']=(row['summary']['total_return']-bm['summary']['total_return']
                if row['completed'] and bm['completed'] else None)
        verify_hashes(expected)
        report=clean(dict(schema='absorption_v1',completed=True,live_qualified=False,adopted=False,unseen_validation=False,
            historical_first_publication_verified=False,historical_universe_verified=False,
            fee_results_are_account_returns=False,start='2022-01-03',end='2026-09-09',
            candidate_rows=len(table),main_candidate_rows=int((~table.named_case).sum()),anchors=table.signal_date.nunique(),
            signal_counts={a:len(e) for a,e in entries.items()},statistics=stats.to_dict('records'),
            comparisons=comparison.to_dict('records'),cases=cases,
            all_accounts_completed=all(r['completed'] for r in cases.values()),
            collection_new_requests=manifest['new_requests'],collection_seconds=manifest['elapsed_seconds'],
            prior_source_comparison=manifest['prior_source_comparison'],
            research_network_calls=0,database_writes=0,causality_checks=checks,
            source_sha256={str(p.relative_to(ROOT)):h for p,h in expected.items()},
            elapsed_seconds=round(time.perf_counter()-began,3),
            limitations=['Known history and present-day survivor/company cohort; no untouched validation.',
                'Institutional publication lags are assumptions; original first versions not verified.',
                'Net-selling resistance is a hypothesis, not identified buyers, ownership concentration or causation.',
                'Fixed 21-session anchors do not enumerate every rally onset.',
                'Unknown features/results remain unknown; diagnostic fee estimates are not finite-capital accounts.',
                'Matched controls allow replacement; overlapping long windows retain temporal dependence.',
                'Strict account failures keep full-period returns unavailable; do not replace with event averages.']))
        write(output/'report.json',report)
        write(output/'manifest.json',dict(schema='absorption_manifest_v1',source_sha256=report['source_sha256'],
            files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
        for arm in ARMS:
            append_trial_registry(dict(source='absorption',status='completed_study',arm=arm,
                timestamp=datetime.now(timezone.utc).isoformat(),output=str(output.relative_to(ROOT)),
                preregistration_sha256=sha(SPEC)))
    print({'elapsed_seconds':report['elapsed_seconds'],'candidate_rows':len(table)},flush=True)
    return report


def publish(first,second):
    folders=[Path(p).resolve() for p in (first,second)]
    if folders[0]==folders[1] or any(not p.is_relative_to(ROOT/'.cache') for p in folders): raise ValueError('Two distinct cache runs required')
    reports=[]; manifests=[]
    for folder in folders:
        r=read(folder/'report.json'); m=read(folder/'manifest.json')
        if (r['schema']!='absorption_v1' or m['schema']!='absorption_manifest_v1' or r['completed'] is not True
                or m['source_sha256']!=r['source_sha256'] or m['files_sha256'].get('report.json')!=sha(folder/'report.json')
                or any(r[k] is not False for k in ('live_qualified','adopted','unseen_validation',
                    'historical_first_publication_verified','historical_universe_verified','fee_results_are_account_returns'))):
            raise ValueError('Unbound or unsupported research report')
        verify_hashes({ROOT/p:h for p,h in m['source_sha256'].items()})
        for name,h in m['files_sha256'].items():
            if Path(name).name!=name or sha(folder/name)!=h: raise ValueError('Study output changed')
        normalized=copy.deepcopy(r); normalized.pop('elapsed_seconds')
        for row in normalized['cases'].values(): row['path']=Path(row['path']).name
        reports.append(normalized); manifests.append(m)
    files=lambda m:{k:v for k,v in m['files_sha256'].items() if k!='report.json'}
    if reports[0]!=reports[1] or files(manifests[0])!=files(manifests[1]): raise ValueError('Independent studies differ')
    write(PUBLICATION,dict(schema='absorption_publication_v1',live_qualified=False,adopted=False,
        report=dict(path=str((folders[0]/'report.json').relative_to(ROOT)),sha256=sha(folders[0]/'report.json')),
        reproducibility=dict(passed=True,files_sha256=files(manifests[0]),
            runs=[dict(path=str((p/'manifest.json').relative_to(ROOT)),sha256=sha(p/'manifest.json')) for p in folders])))
    PUBLICATION.with_suffix('.sha256').write_text(sha(PUBLICATION)+'\n')
    print('Published identical absorption studies')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path); group.add_argument('--publish',nargs=2,type=Path)
    args=p.parse_args()
    if args.publish: publish(*args.publish)
    else: run(args.output)
