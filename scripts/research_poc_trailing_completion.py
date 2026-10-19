#!/usr/bin/env python3
"""Complete one failed data coordinate without changing the trailing strategy."""
import argparse
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_poc_trailing as original
from skills.all_entry_accounts import digest
from skills.poc_risk_evidence import evidence_type
from skills.poc_trailing_profile_completion import DOCUMENT,DIRECTORY,KEY,ExactCompletion,acquire
from skills.top3_validation import MODES


def run(mode,output):
    # Preserve every attempt, and reject replay of a finished configuration.
    for path in original.BASE.glob('run-v*/*/account/report.json'):
        report=json.loads(path.read_text())
        if (report.get('all_completed') is True and report['risk_controls']['mode']==mode):
            raise ValueError('This configuration is complete; reuse '+str(path))
    if Path(output).exists():raise ValueError('Preserve the earlier attempt; select a fresh output')
    def prepared(*args,**kwargs):
        Base=evidence_type(*args,**kwargs)
        class Completed(Base):
            def __init__(self,**settings):
                super().__init__(**settings)
                self.profiles.raw_store=ExactCompletion(self.profiles.raw_store)
                for path in (DOCUMENT,ROOT/'skills/poc_trailing_profile_completion.py',Path(__file__),
                             DIRECTORY/'receipts'/(KEY+'.json')):
                    self._bind(path)
        return Completed
    execute=FunctionType(original.run.__code__,dict(original.run.__globals__,evidence_type=prepared),
                         original.run.__name__,original.run.__defaults__,original.run.__closure__)
    execute.__kwdefaults__=original.run.__kwdefaults__
    return execute(mode,output,fetch=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acquire',action='store_true');p.add_argument('--mode',choices=MODES);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.acquire:
        item=acquire();print(json.dumps({k:item.get(k) for k in ('status','rows','raw_sha256')}))
    else:
        if a.mode is None or a.output is None:p.error('--mode and --output required')
        r=run(a.mode,a.output);print(json.dumps(dict(all_completed=r['all_completed'])))
        raise SystemExit(0 if r['all_completed'] else 2)
