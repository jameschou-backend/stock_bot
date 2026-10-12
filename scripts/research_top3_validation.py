#!/usr/bin/env python3
"""Versioned HL2/cost replay using all existing verified settlement adapters."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_all_entry_accounts import configured_runner as base_runner
from skills.all_entry_accounts import digest
from skills.top3_validation import ARMS, MODES, install

BASE = ROOT/'.cache/top3-validation-20261012'
SPEC = ROOT/'docs/prereg_top3_validation_20261012.md'
TERMS = {
    'document': 'all_entry_corporate_terms_v3_20261011',
    'completion': 'all_entry_6669_delivery_20261011',
    'capital': 'all_entry_2352_capital_20261011',
    'official': 'all_entry_official_execution_v3_20261011',
    'distributions': 'all_entry_distribution_completion_20261011',
    'albatron': 'all_entry_5386_delivery_20261011',
    'certificates': 'all_entry_certificate_completion_20261011',
    'late_certificate': 'all_entry_9941_certificate_20261011',
    'bank_distributions': 'all_entry_bank_distributions_20261011',
    'issuer_distributions': 'all_entry_issuer_distributions_20261011',
    'northstar': 'all_entry_8927_delivery_20261011',
    'context_distributions': 'all_entry_context_distributions_20261011',
    'poya': 'all_entry_5904_certificate_20261012',
    'merger_halt': 'all_entry_2888_halt_20261012',
    'yageo_split': 'all_entry_2327_split_20261012',
    'tcfhc': 'all_entry_5880_certificate_20261012',
}


def run(output, arms, mode, *, fetch=False):
    if not arms or not set(arms) <= set(ARMS):
        raise ValueError('Fixed top three and 0050 only')
    source = ROOT/'scripts/research_all_entry_accounts_tcfhc_completion.py'
    spec = importlib.util.spec_from_file_location('_top3_settlement_entry', source)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    def configured(bundle, requested):
        runner, refs = base_runner(bundle, requested)
        runner.RUN_ROOT = BASE
        runner.ARM_RULES = {arm: (False, 'none', .5, .3, False) for arm in runner.ARMS}
        frozen_run, writer, record = runner.run, runner.write, runner.append_trial_registry

        def execute(*args, **kwargs):
            install(runner, mode)
            return frozen_run(*args, **kwargs)

        def write(path, value):
            if 'family_rules' in value or Path(path).name == 'report.json':
                value['top3_validation'] = dict(mode=mode, commission=MODES[mode][0],
                    slippage=MODES[mode][1], buy_fraction=.5, sell_fraction=.3,
                    preregistration=dict(path=str(SPEC.relative_to(ROOT)), sha256=digest(SPEC)),
                    source_selection='all_entry_accounts_20261011',
                    price_availability='after_full_session_only', paper_enabled=False)
            return writer(path, value)

        def trial(value, **kwargs):
            value = dict(value, top3_mode=mode, top3_spec_sha256=digest(SPEC))
            return record(value, **kwargs)
        runner.run, runner.write, runner.append_trial_registry = execute, write, trial
        for path in (source, Path(__file__), SPEC, ROOT/'skills/top3_validation.py'):
            refs[str(path.relative_to(ROOT))] = digest(path)
        return runner, refs
    entry.configured_runner = configured
    return entry.run(output, arms, candidate_bundle=ROOT/'.cache/all-entry-accounts-20261011/inputs-v3',
        fetch=fetch, **{key: ROOT/'docs'/(name+'.json') for key, name in TERMS.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    result = run(args.output, args.arms.split(','), args.mode, fetch=args.fetch)
    print(json.dumps({'completed': result['all_completed'], 'output': str(args.output)}))
    raise SystemExit(0 if result['all_completed'] else 2)
