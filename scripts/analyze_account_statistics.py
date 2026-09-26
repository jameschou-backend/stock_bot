#!/usr/bin/env python3
"""Attach descriptive benchmark uncertainty to every recorded research case."""
from collections import Counter
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.residual_slots_ui import load,REPORT
from skills.account_statistics import describe_account
from skills.backtest_case_cache import file_identities
from skills.trial_registry import TRIAL_REGISTRY_PATH,HISTORICAL_TRIALS_BASE
from scripts.research_exit_scenarios import read,write,sha


def run(folders,output):
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'.cache/account-statistics-20260927'):
        raise ValueError('Choose a new statistics output directory')
    baseline=load();refs=dict(baseline['source_sha256']);studies={}
    paths=[REPORT,REPORT.with_suffix('.sha256'),Path(__file__),
           ROOT/'skills/account_statistics.py',ROOT/'skills/statistics.py',ROOT/'skills/trial_registry.py']
    refs.update(file_identities(paths,ROOT))
    benchmarks={mask:read(ROOT/baseline['cases']['benchmark_combined' if mask else 'benchmark_control']['result']['path'])['account']
                for mask in (0,1)}
    calendar=[r['date'] for r in benchmarks[0]['daily']]
    for folder in folders:
        folder=Path(folder).resolve()
        if not folder.is_relative_to(ROOT/'.cache'):
            raise ValueError('Research input must be a cache directory with unique identity')
        key=str(folder.relative_to(ROOT))
        if key in studies:raise ValueError('Duplicate study input')
        manifest=read(folder/'manifest.json');identity=read(folder/'identity.json')
        refs[str((folder/'manifest.json').relative_to(ROOT))]=sha(folder/'manifest.json')
        for name,digest in manifest['files_sha256'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder) or sha(path)!=digest:raise ValueError('Research artifact changed')
            relative=str(path.relative_to(ROOT))
            if relative in refs and refs[relative]!=digest:raise ValueError('Research sources disagree')
            refs[relative]=digest
        for name,digest in identity.items():
            if name in refs and refs[name]!=digest:raise ValueError('Research sources disagree')
            refs[name]=digest
        report=read(folder/'report.json');cases={}
        for name,row in report['cases'].items():
            path=ROOT/row['result']['path']
            if (not path.resolve().is_relative_to(folder) or sha(path)!=row['result']['sha256']
                    or manifest['files_sha256'].get(str(path.relative_to(folder)))!=row['result']['sha256']):
                raise ValueError('Case is not bound to the study manifest')
            case=read(path)
            if row['completed']!=case['completed'] or row['config']!=case['config']:
                raise ValueError('Case completion or configuration differs from report')
            if not case['completed']:
                cases[name]=dict(available=False,reason=case['reason'],live_qualified=False)
                continue
            result=describe_account(case['account'],benchmarks[case['config']['factor_mask']&1],calendar)
            cases[name]=dict(available=True,**result)
        studies[key]=dict(cases=cases,all_cases_complete=all(r['available'] for r in cases.values()))
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:raise ValueError('Source changed during statistics')
    output.mkdir(parents=True)
    registry=TRIAL_REGISTRY_PATH.read_bytes()
    snapshot=output/'trial_registry_snapshot.jsonl';snapshot.write_bytes(registry)
    import json
    records=[json.loads(line) for line in registry.splitlines() if line.strip()]
    report=dict(schema='account_statistics_v1',studies=studies,source_sha256=refs,
        recorded_executions=len(records),recorded_sources=dict(Counter(r.get('source','unspecified') for r in records)),
        historical_trial_base_assumption=HISTORICAL_TRIALS_BASE,complete_trial_coverage_verified=False,
        registry_snapshot=dict(path=str(snapshot.relative_to(ROOT)),sha256=sha(snapshot)),
        selection_adjusted_statistics_verified=False,unseen_validation=False,live_qualified=False,
        note='Bootstrap intervals describe these recorded accounts; they do not correct the full strategy selection process')
    write(output/'report.json',report)
    (output/'report.sha256').write_text(sha(output/'report.json')+'\n')
    return {name:dict(completed=sum(c['available'] for c in s['cases'].values()),
                     positive_lower_bounds=sum(c.get('excess_interval_above_zero',False) for c in s['cases'].values()))
            for name,s in studies.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(run(args.runs,args.output))
