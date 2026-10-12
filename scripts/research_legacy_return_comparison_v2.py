#!/usr/bin/env python3
"""Disambiguate overlapping legacy candidate IDs without altering their order."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from skills.all_entry_accounts import digest, write
from skills.legacy_return_comparison import ARMS
from skills.top3_validation import MODES


def namespace_events(rows):
    result=deepcopy(rows)
    for row in result:
        row['legacy_event_id']=row['event_id']
        row['event_id']='legacy949-'+row['event_id']
    return result


def run(output,arms,mode,*,fetch=False):
    source=ROOT/'scripts/research_legacy_return_comparison.py'
    spec=importlib.util.spec_from_file_location('_legacy_v1_private',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.configured_runner
    def configured(bundle,requested):
        runner,refs=original(bundle,requested)
        load=runner.load_rsi_candidates
        def candidates(directory):
            rows,payload,bindings=load(directory)
            payload=deepcopy(payload)
            payload['parameters']['event_namespace']='legacy949-prefix-only-original-priority-and-relative-order-retained'
            return namespace_events(rows),payload,bindings
        runner.load_rsi_candidates=candidates
        refs[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
        return runner,refs
    module.configured_runner=configured
    return module.run(output,arms,mode,fetch=fetch)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arms',default=','.join(ARMS))
    p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--fetch',action='store_true')
    a=p.parse_args();existed=a.output.exists()
    try:r=run(a.output,a.arms.split(','),a.mode,fetch=a.fetch)
    except Exception as exc:
        from skills.trial_registry import append_trial_registry
        failure=dict(completed=False,source='legacy_return_comparison_20261012',legacy_mode=a.mode,
            arms=a.arms.split(','),stage='setup_or_publication',reason=str(exc),live_qualified=False)
        if not existed and not (a.output/'report.json').exists():
            write(a.output/'failure.json',failure)
            append_trial_registry(failure,registry_path=a.output/'trials.jsonl');append_trial_registry(failure)
        raise
    print(json.dumps(dict(completed=r['all_completed'],output=str(a.output))))
    raise SystemExit(0 if r['all_completed'] else 2)
