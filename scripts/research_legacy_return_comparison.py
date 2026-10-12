#!/usr/bin/env python3
"""Replay the two identified legacy strategies through the current cash engine."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_all_entry_accounts import FROZEN, FROZEN_SHA
from scripts.research_top3_validation_v3 import TERMS
from skills.all_entry_accounts import digest, verify_refs, write
from skills.all_entry_account_data import AllEntryData
from skills.legacy_return_comparison import BASE, SPEC, ARMS, NAMES, apply_original_exit, audit_original_exit
from skills.top3_validation import MODES, install
from skills.top3_cost_audit import install as install_cost, BINDINGS
from skills.top3_gap_audit import install as install_gap, SOURCE as GAP_SOURCE, SOURCE_SHA as GAP_SHA


def configured_runner(bundle, arms):
    if not arms or len(set(arms))!=len(arms) or not set(arms)<=set(ARMS):
        raise ValueError('Only the two identified strategies and 0050 may run')
    if digest(FROZEN)!=FROZEN_SHA:
        raise ValueError('Frozen financial runner changed')
    spec=importlib.util.spec_from_file_location('_legacy_comparison_runner',FROZEN)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    path=Path(bundle)/'candidates.json'
    expected=path.with_suffix('.sha256').read_text().strip()
    if digest(path)!=expected: raise ValueError('Sealed legacy candidates changed')
    payload=json.loads(path.read_text());refs=dict(payload['source_sha256'])
    refs[str(path.relative_to(ROOT))]=expected
    verify_refs(refs)
    runner.validate_candidate_calendar(payload['entries'],payload['calendar'])
    runner.RUN_ROOT,runner.PREREG=BASE,SPEC
    runner.RUNNER_SOURCE=Path(__file__).resolve();runner.IMPORTED_RUNNER_SHA=digest(runner.RUNNER_SOURCE)
    runner.ARMS=ARMS
    runner.ARM_RULES={arm:(arm=='poc_priority','none',.5,.3,False) for arm in ARMS}
    runner.RSI_ARMS=frozenset({'liquid_universe'})
    runner.COMMON_KNOWN_ARMS=frozenset()
    runner.SELECTION_ARMS=dict(liquid_universe='original',poc_priority='poc_priority_available')
    runner.load_legacy_anchors=lambda requested:({}, {})
    runner.load_rsi_candidates=lambda directory:(deepcopy(payload['entries']),
        dict(_calendar=payload['calendar'],parameters=dict(ranking=payload['ranking'])),refs)
    original_select=runner.select_arm_entries
    active={}
    def select(arm,*args,**kwargs):
        active['arm']=arm
        return original_select(arm,*args,**kwargs)
    runner.select_arm_entries=select
    original_engines=runner.engine_types
    class OriginalExit:
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            apply_original_exit(self,active['arm'])
        def run(self):
            result=super().run()
            if active['arm']=='liquid_universe': audit_original_exit(result)
            return result
    def engines(*args,**kwargs):
        stock,benchmark=original_engines(*args,**kwargs)
        return type('LegacyExit'+stock.__name__,(OriginalExit,stock),{}),benchmark
    runner.engine_types=engines
    original_write=runner.write
    def writer(path,value):
        if 'family_rules' in value:
            arm=Path(path).stem
            value['family_rules'].update(candidate_family=arm,name=NAMES[arm],
                exit_policy='loss12_time63' if arm=='liquid_universe' else value['family_rules']['exit_policy'])
        return original_write(path,value)
    runner.write=writer
    refs[str(FROZEN.relative_to(ROOT))]=FROZEN_SHA
    return runner,refs


def run(output,arms,mode,*,fetch=False):
    source=ROOT/'scripts/research_all_entry_accounts_tcfhc_completion.py'
    spec=importlib.util.spec_from_file_location('_legacy_settlement_entry',source)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    class Evidence(AllEntryData):
        def __init__(self,*,online=False):
            super().__init__(online=online)
            self.prereg_path,self.prereg_sha256=SPEC,digest(SPEC)
            self._bind(SPEC)
    entry.AllEntryData=Evidence
    def configured(bundle,requested):
        runner,refs=configured_runner(bundle,requested)
        frozen_run,writer,record=runner.run,runner.write,runner.append_trial_registry
        metadata=dict(mode=mode,commission=MODES[mode][0],slippage=MODES[mode][1],
            buy_fraction=.5,sell_fraction=.3,preregistration=dict(path=str(SPEC.relative_to(ROOT)),sha256=digest(SPEC)),
            price_availability='after_full_session_only',paper_enabled=False)
        def execute(*args,**kwargs):
            install(runner,mode);install_cost(runner,mode);install_gap(runner)
            return frozen_run(*args,**kwargs)
        def save(path,value):
            if 'family_rules' in value or Path(path).name=='report.json':
                value['legacy_comparison']=deepcopy(metadata)
            return writer(path,value)
        def trial(value,**kwargs):
            value=deepcopy(value)
            value.update(source='legacy_return_comparison_20261012',legacy_mode=mode,spec_sha256=digest(SPEC))
            return record(value,**kwargs)
        runner.run,runner.write,runner.append_trial_registry=execute,save,trial
        for path in (source,Path(__file__),SPEC,ROOT/'skills/legacy_return_comparison.py',
                ROOT/'skills/top3_validation.py',ROOT/'skills/top3_cost_audit.py',ROOT/'skills/top3_gap_audit.py',
                ROOT/'scripts/research_top3_validation_v3.py'):
            refs[str(path.relative_to(ROOT))]=digest(path)
        refs.update(BINDINGS);refs[str(GAP_SOURCE.relative_to(ROOT))]=GAP_SHA
        return runner,refs
    entry.configured_runner=configured
    return entry.run(output,arms,candidate_bundle=BASE/'candidates-v1',fetch=fetch,
        **{key:ROOT/'docs'/(name+'.json') for key,name in TERMS.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arms',default=','.join(ARMS))
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--fetch',action='store_true')
    a=p.parse_args()
    existed=a.output.exists()
    try:
        result=run(a.output,a.arms.split(','),a.mode,fetch=a.fetch)
    except Exception as exc:
        from skills.trial_registry import append_trial_registry
        failure=dict(completed=False,source='legacy_return_comparison_20261012',legacy_mode=a.mode,
            arms=a.arms.split(','),stage='setup_or_publication',reason=str(exc),live_qualified=False)
        if not existed and not (a.output/'report.json').exists():
            write(a.output/'failure.json',failure)
            append_trial_registry(failure,registry_path=a.output/'trials.jsonl');append_trial_registry(failure)
        raise
    print(json.dumps(dict(completed=result['all_completed'],output=str(a.output))))
    raise SystemExit(0 if result['all_completed'] else 2)
