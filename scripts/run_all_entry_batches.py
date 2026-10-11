#!/usr/bin/env python3
"""Run one sealed account per process so replay memory is released between arms.

All attempts, including data failures, remain in the final comparison. The
batch stops on infrastructure errors rather than inventing an account result.
Publication is a separate explicit step after evidence repairs and review.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from skills.all_entry_accounts import BASE,registry,digest,write
from scripts.combine_all_entry_accounts import combine


def progress_write(path,value):
    """Progress is replaceable operational state; financial evidence is sealed."""
    pending=path.with_suffix('.tmp')
    pending.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    pending.replace(path)


def run(output,*,candidate_bundle,terms,completion=None,fetch=False,resume=False):
    output=Path(output).resolve();output.relative_to(BASE)
    if output.exists() and not resume:raise ValueError('Use a new batch directory or explicit --resume')
    if resume and not output.exists():raise ValueError('Cannot resume an absent batch')
    output.mkdir(parents=True,exist_ok=resume)
    candidate_bundle=Path(candidate_bundle).resolve();terms=Path(terms).resolve()
    # Put the repaired failure early, then keep the fixed registry order.
    priority=['original_breakout','benchmark','bollinger_reclaim','poc_up_red','near_high','donchian55']
    arms=priority+[r['id'] for r in registry() if r['id'] not in priority]
    manifest=dict(schema='all_entry_process_batches_v1',arms=arms,
        candidate_manifest_sha256=digest(candidate_bundle/'manifest.json'),
        terms_sha256=digest(terms),completion_sha256=digest(completion) if completion else None,
        fetch=fetch,one_process_per_account=True,failed_accounts_retained=True)
    write(output/'batch.json',manifest)
    reports=[];progress=[];started=time.monotonic()
    for index,arm in enumerate(arms,1):
        command=[sys.executable,'-u',str(ROOT/'scripts/research_all_entry_accounts_audited.py'),
            '--arms',arm,'--output',str(output/arm),'--candidate-bundle',str(candidate_bundle),
            '--terms',str(terms)]
        if completion:command+=['--settlement-completion',str(Path(completion).resolve())]
        if fetch:command.append('--fetch')
        report_path=output/arm/'report.json'
        if not (resume and report_path.exists()):
            if (output/arm).exists():
                raise ValueError('Partial worker directory requires a separate reviewed rerun: '+arm)
            print(f'[{index}/{len(arms)}] start {arm}',flush=True)
            with (output/(arm+'.log')).open('w') as log:
                worker=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if worker.returncode:
                progress_write(output/'infrastructure-error.json',dict(arm=arm,returncode=worker.returncode,
                    log=str((output/(arm+'.log')).relative_to(ROOT))))
                raise RuntimeError('Account worker failed; inspect '+str(output/(arm+'.log')))
        if digest(report_path)!=report_path.with_suffix('.sha256').read_text().strip():
            raise ValueError('Worker report checksum differs')
        report=json.loads(report_path.read_text())
        if report['requested_arms']!=[arm] or set(report['cases'])!={arm}:
            raise ValueError('Worker attempted a different account')
        if report['candidate_manifest']['sha256']!=manifest['candidate_manifest_sha256']:
            raise ValueError('Worker candidate manifest differs from batch')
        case=report['cases'][arm]
        progress.append(dict(id=arm,completed=case['completed'],reason=case.get('reason'),
            elapsed_seconds=report.get('elapsed_seconds'),report=str(report_path.relative_to(ROOT))))
        progress_write(output/'progress.json',dict(attempted=len(progress),expected=len(arms),cases=progress))
        reports.append(report_path)
        print(json.dumps(progress[-1],ensure_ascii=False),flush=True)
    if (output/'report.json').exists():raise ValueError('This batch has already been consolidated')
    result=combine(reports,output/'report.json')
    print(json.dumps(dict(attempted=len(arms),all_completed=result['all_completed'],
        elapsed_seconds=round(time.monotonic()-started,2)),ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate-bundle',type=Path,default=BASE/'inputs-v3')
    parser.add_argument('--terms',type=Path,required=True)
    parser.add_argument('--settlement-completion',type=Path)
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Reuse only checksum-verified completed worker reports')
    args=parser.parse_args()
    run(args.output,candidate_bundle=args.candidate_bundle,terms=args.terms,
        completion=args.settlement_completion,fetch=args.fetch,resume=args.resume)
