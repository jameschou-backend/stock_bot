#!/usr/bin/env python3
"""Replay only an incomplete split account with the exact recorded data repair."""
import argparse
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_poc_split_accounts as single
from skills.all_entry_accounts import digest
from skills.poc_risk_evidence import evidence_type
from skills.poc_split_profile_completion import DOCUMENT, DIRECTORY, KEY, ExactCompletion, acquire


def failed_configuration(path):
    path = Path(path).resolve()
    path.relative_to(single.BASE)
    sha = path.with_suffix('.sha256').read_text().strip()
    if digest(path) != sha: raise ValueError('Failed report changed')
    value = json.loads(path.read_text())
    meta = value['split_accounts']
    arm, mode = meta['arm'], meta['mode']
    if (value['all_completed'] is not False or set(value['cases']) != {arm}
            or value['cases'][arm]['completed'] is not False
            or value['cases'][arm].get('reason') != 'Necessary profiles tick unavailable: '+KEY+' provider_error'):
        raise ValueError('Only the registered incomplete provider-error case may be replayed')
    return arm, mode, sha


def run(failed, output, package, donor):
    arm, mode, failed_sha = failed_configuration(failed)
    if Path(output).exists(): raise ValueError('Retain any earlier repair attempt; use a new output')
    Prepared = evidence_type(package, digest(package), Path(__file__))
    class Completed(Prepared):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.profiles.raw_store = ExactCompletion(self.profiles.raw_store)
            for path in (DOCUMENT, ROOT/'skills/poc_split_profile_completion.py', Path(__file__),
                         DIRECTORY/'receipts'/(KEY+'.json')):
                self._bind(path)
            self._bind(failed, failed_sha)
    execute = FunctionType(single.run.__code__, dict(single.run.__globals__, ReusedFollowupData=Completed),
                           single.run.__name__, single.run.__defaults__, single.run.__closure__)
    execute.__kwdefaults__ = single.run.__kwdefaults__
    return execute(output, arm, mode, 'split50', donor_manifest=donor, fetch=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acquire', action='store_true')
    p.add_argument('--failed-report', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--package', type=Path)
    p.add_argument('--donors', type=Path)
    a = p.parse_args()
    if a.acquire:
        item = acquire()
        print(json.dumps({k:item.get(k) for k in ('status','rows','raw_sha256')}))
    else:
        if any(v is None for v in (a.failed_report,a.output,a.package,a.donors)):
            p.error('Repair replay requires failed report, output, package and donors')
        report = run(a.failed_report,a.output,a.package,a.donors)
        print(json.dumps(dict(all_completed=report['all_completed'],output=str(a.output))))
        raise SystemExit(0 if report['all_completed'] else 2)
