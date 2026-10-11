#!/usr/bin/env python3
"""Consolidate explicit later evidence replays under one unchanged account model.

Later attempted cases always replace earlier cases, including failures. Returns
never decide which version is retained. Any changed candidate file must have a
corresponding replay under the final candidate manifest.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from skills.all_entry_accounts import BASE,digest,registry,write
from scripts.summarize_poc_broker_account import Sources

MODEL_FIELDS=('start','end','initial_cash','input_bundle','volume_policy','data_revision','execution_model',
              'zero_buy_reentry_policy')


def combine(report_paths,output):
    if len(report_paths)<2:raise ValueError('An explicit initial run and a later evidence replay are required')
    output=Path(output).resolve();output.relative_to(BASE)
    if output.exists():raise ValueError('Use a new consolidated report path')
    sources=Sources(ROOT);reports=[];refs={};snapshots={};origins={};cases={};baseline=None;model_sha=None
    for path in map(lambda p:Path(p).resolve(),report_paths):
        path.relative_to(BASE)
        if digest(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Replay report checksum differs')
        report=json.loads(path.read_text())
        if report.get('schema')!='all_entry_accounts_v1':raise ValueError('Unexpected replay schema')
        source_sha=digest(ROOT/report['execution_source_snapshot'])
        if baseline is None:baseline=report;model_sha=source_sha
        elif source_sha!=model_sha or any(report.get(k)!=baseline.get(k) for k in MODEL_FIELDS):
            raise ValueError('Cannot consolidate different account models or periods')
        sources.closure(report,str(path.relative_to(ROOT)))
        candidate=report['candidate_manifest'];manifest=json.loads(sources.bind(candidate['path'],candidate['sha256']).read_text())
        if set(report['cases'])!=set(report['requested_arms']):raise ValueError('Attempted cases differ from requested arms')
        for arm,case in report['cases'].items():
            if arm!='benchmark' and case.get('candidate_coverage')!=manifest['counts'][arm]:
                raise ValueError('Case coverage differs from its own candidate version')
            sources.bind(case['path'],case['sha256'])
            cases[arm]=deepcopy(case);origins[arm]=dict(report_path=str(path.relative_to(ROOT)),
                report_sha256=digest(path),candidate_file_sha256=manifest['files_sha256'].get(arm+'.json'))
        for key,value in report['source_sha256'].items():
            if key in refs and refs[key]!=value:raise ValueError('Shared evidence changed across replays: '+key)
            refs[key]=value
        refs[str(path.relative_to(ROOT))]=digest(path)
        for key,value in report.get('source_snapshots',{}).items():snapshots.setdefault(key,value)
        reports.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),requested_arms=report['requested_arms'],
            all_completed=report['all_completed'],network_requests=report.get('network_requests'),
            finmind_requests=report.get('finmind_requests')))
    expected={r['id'] for r in registry()}|{'benchmark'}
    if set(cases)!=expected:raise ValueError('All 32 account attempts are required')
    # `manifest` is deliberately the final supplied run's candidate version.
    for arm in expected-{'benchmark'}:
        if origins[arm]['candidate_file_sha256']!=manifest['files_sha256'][arm+'.json']:
            raise ValueError('Changed candidates require a new account replay: '+arm)
    result=deepcopy(baseline)
    result.update(cases=cases,source_sha256=refs,source_snapshots=snapshots,
        candidate_manifest=report['candidate_manifest'],candidate_coverage=manifest['counts'],
        consolidated_runs=reports,case_origins=origins,
        consolidation_policy='latest_attempt_for_every_case_including_failure_no_return_selection',
        requested_arms=list(result['registered_arms']))
    completed=all(c['completed'] for c in cases.values())
    for key in ('all_completed','all_registered_completed','return_recomputed','accounting_validated'):result[key]=completed
    for key in ('network_requests','finmind_requests'):
        result[key]=sum(r[key] or 0 for r in reports)
    result['external_candidate_count']=sum(r['executable'] for r in manifest['counts'].values())
    result['elapsed_seconds']=None
    refs[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    for name,sha in refs.items():sources.bind(name,sha)
    sha=write(output,result);output.with_suffix('.sha256').write_text(sha+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',action='append',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=combine(args.report,args.output)
    print(json.dumps(dict(all_completed=r['all_completed'],cases=len(r['cases']))))
