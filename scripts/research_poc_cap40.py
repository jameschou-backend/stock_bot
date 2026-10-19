#!/usr/bin/env python3
"""Run only missing POC cap40 accounts; original financial runners stay sealed."""
import argparse
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_poc_risk_controls as frozen
from skills.all_entry_accounts import digest, write
from skills.poc_risk_evidence import evidence_type
from skills.poc_cap40 import VARIANT, install
from skills.top3_validation import MODES

BASE = ROOT/'.cache/poc-cap40-20261019'
SPEC = ROOT/'docs/prereg_poc_cap40_20261019.json'
PACKAGE = ROOT/'.cache/poc-split-accounts-20261015/run-v1/prepared-donors.json'


def run(mode, output, *, fetch=False):
    output = Path(output).resolve()
    output.relative_to(BASE)
    if output.exists():
        report = output/'report.json'
        if report.exists() and digest(report) == report.with_suffix('.sha256').read_text().strip():
            old = json.loads(report.read_text())
            meta = old.get('risk_controls', {})
            if (old.get('all_completed') is True and meta.get('variant') == VARIANT
                    and meta.get('mode') == mode and meta.get('preregistration', {}).get('sha256') == digest(SPEC)):
                print('reuse completed', mode, flush=True)
                return old
        raise ValueError('Preserve prior incomplete/changed attempt; choose a new output version')
    for path in BASE.glob('run-v*/*/account/report.json'):
        old = json.loads(path.read_text())
        if old.get('all_completed') is True and old['risk_controls']['mode'] == mode:
            raise ValueError('Completed configuration must be reused: '+str(path))
    package = json.loads(PACKAGE.read_text())
    donor = ROOT/package['manifest']['path']
    Prepared = evidence_type(PACKAGE, digest(PACKAGE), Path(__file__))
    class Evidence(Prepared):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            from skills.poc_trailing_profile_completion import ExactCompletion, DOCUMENT
            self.profiles.raw_store = ExactCompletion(self.profiles.raw_store)
            for path in (SPEC, ROOT/'skills/poc_cap40.py', ROOT/'skills/partial_risk.py', DOCUMENT, ROOT/'skills/poc_trailing_profile_completion.py', Path(__file__)):
                self._bind(path)
    execute = FunctionType(frozen.run.__code__, dict(frozen.run.__globals__,
        BASE=BASE, SPEC=SPEC, VARIANTS=(VARIANT,), install_controls=install,
        ReusedFollowupData=Evidence), frozen.run.__name__, frozen.run.__defaults__, frozen.run.__closure__)
    execute.__kwdefaults__ = frozen.run.__kwdefaults__
    try:
        return execute(output, 'poc_up_red', mode, VARIANT, donor_manifest=donor, fetch=fetch)
    except Exception as exc:
        if not (output/'report.json').exists():
            write(output/'setup-failure.json', dict(completed=False, mode=mode, variant=VARIANT, reason=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    result = run(args.mode, args.output, fetch=args.fetch)
    print(json.dumps(dict(all_completed=result['all_completed'], output=str(args.output))))
    raise SystemExit(0 if result['all_completed'] else 2)
