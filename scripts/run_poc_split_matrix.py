#!/usr/bin/env python3
"""Run independent accounts in fresh interpreters using data-only donors.

Each account verifies all donor sources before its first financial action.
Native data clients and locks are constructed locally, never copied by fork.
"""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_poc_split_accounts as single
from skills.top3_validation import MODES
from skills.all_entry_accounts import digest, write
from skills.poc_risk_evidence import assemble, evidence_type


def account_output(base, variant, mode, arm):
    # The frozen runner locks output.parent. Each arm needs its own parent,
    # even when variant and costs match another concurrent account.
    return Path(base)/variant/mode/arm/'account'


def child(package, package_sha, output, arm, mode, variant, donor):
    log = output.parent/(arm+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('w', buffering=1)
    sys.stdout = sys.stderr = stream
    Prepared = evidence_type(package, package_sha, Path(__file__))
    print('fresh data readers; source checks remain mandatory', flush=True)
    run = FunctionType(single.run.__code__, dict(single.run.__globals__, ReusedFollowupData=Prepared),
                       single.run.__name__, single.run.__defaults__, single.run.__closure__)
    run.__kwdefaults__ = single.run.__kwdefaults__
    try:
        result = run(output, arm, mode, variant, donor_manifest=donor, fetch=True)
        print(json.dumps(dict(completed=result['all_completed'], output=str(output))), flush=True)
        if not result['all_completed']:
            raise SystemExit(2)
    except Exception as exc:
        write(output/'setup-failure.json', dict(completed=False, reason=str(exc), variant=variant, arm=arm, mode=mode))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude', action='append', default=[])
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.workers not in (1, 2):
        raise ValueError('At most two isolated new accounts')
    configs = [(a, m, v) for m in MODES for a in ('poc_up_red', 'poc_priority') for v in ('split50',)
               if f'{a}:{v}:{m}' not in args.exclude]
    pending = []
    for a, m, v in configs:
        out = account_output(args.output, v, m, a)
        report = out/'report.json'
        if report.exists():
            value = json.loads(report.read_text())
            if digest(report) != report.with_suffix('.sha256').read_text().strip() or value.get('all_completed') is not True:
                raise ValueError('Prior incomplete or changed attempt; choose a separately recorded output version')
            print('reuse completed', a, v, m, flush=True)
        elif out.exists():
            raise ValueError('Prior partial attempt retained: '+str(out))
        else:
            pending.append((out, a, m, v))
    if not pending:
        return
    print('assemble data-only donors', len(pending), 'new accounts', flush=True)
    package = args.output/'prepared-donors.json'
    package_sha = assemble(args.donors, package)
    print('donor assembly ready; each account verifies full source closure', flush=True)
    context = mp.get_context('spawn')
    active, finished = [], []
    while pending or active:
        while pending and len(active) < args.workers:
            job = pending.pop(0)
            proc = context.Process(target=child, args=(package, package_sha, *job, args.donors))
            proc.start()
            active.append((proc, job))
            print('started', job[1:], proc.pid, flush=True)
        for proc, job in list(active):
            proc.join(timeout=1)
            if not proc.is_alive():
                finished.append(dict(arm=job[1], mode=job[2], variant=job[3], exit_code=proc.exitcode, output=str(job[0])))
                active.remove((proc, job))
                print('finished', finished[-1], flush=True)
                write(args.output/'status'/f'{len(finished):02d}.json', dict(updated_at=datetime.now(timezone.utc).isoformat(),
                    finished=finished, active=[dict(pid=x.pid, arm=j[1], mode=j[2], variant=j[3]) for x,j in active], pending=len(pending)))
                if proc.exitcode and not (job[0]/'setup-failure.json').exists():
                    write(job[0]/'setup-failure.json', dict(completed=False, arm=job[1], mode=job[2], variant=job[3],
                        reason='Worker exited before completing report', exit_code=proc.exitcode))
    if any(x['exit_code'] for x in finished):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
