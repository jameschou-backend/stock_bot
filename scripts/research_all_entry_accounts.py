#!/usr/bin/env python3
"""Run the fixed 31 entry rules through the existing audited cash-account engine.

A private module instance supplies a narrow candidate/metadata adapter. The
frozen runner and its financial classes remain byte-for-byte unchanged; no
global import, old research result or production strategy is monkey-patched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.all_entry_accounts import BASE, PREREG, START, END, digest, load_candidates, registry

FROZEN = ROOT/'scripts/research_strategy_account_comparison.py'
FROZEN_SHA = '9981ed28a997fb027bc799cc5162c08243a7960b1b818d009fd232bc5865ed32'


def configured_runner(candidate_bundle, requested_arms):
    if digest(FROZEN) != FROZEN_SHA:
        raise ValueError('Frozen financial runner changed; adapter needs a reviewed version')
    specification = importlib.util.spec_from_file_location('_all_entry_frozen_runner', FROZEN)
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    definitions = registry()
    ids = [x['id'] for x in definitions]
    if not requested_arms or len(set(requested_arms)) != len(requested_arms) or not set(requested_arms) <= set(ids+['benchmark']):
        raise ValueError('Only the 31 registered rules and benchmark may run')
    by_arm, calendar, manifest, refs = load_candidates(candidate_bundle)
    combined = [e for arm in requested_arms if arm != 'benchmark' for e in by_arm[arm]]
    runner.validate_candidate_calendar(combined, calendar)
    runner.RUN_ROOT, runner.PREREG = BASE, PREREG
    runner.RUNNER_SOURCE = Path(__file__).resolve()
    runner.IMPORTED_RUNNER_SHA = digest(runner.RUNNER_SOURCE)
    runner.CANDIDATE_BUNDLE = Path(candidate_bundle)
    runner.ARM_RULES = {arm: (False, 'none', .7, .3, False) for arm in ids+['benchmark']}
    runner.ARMS = tuple(runner.ARM_RULES)
    runner.RSI_ARMS = frozenset(ids)  # frozen runner's generic external-candidate seam
    runner.COMMON_KNOWN_ARMS = frozenset({'poc_up_red'})
    runner.SELECTION_ARMS = {arm: 'poc_filter' if arm == 'poc_up_red' else 'original' for arm in ids}
    runner.load_legacy_anchors = lambda arms: ({}, {})
    runner.load_signal_reference = lambda root: ({}, {str(FROZEN.relative_to(ROOT)): FROZEN_SHA})
    runner.load_rsi_candidates = lambda directory: (
        combined, dict(_calendar=calendar, parameters=manifest['ranking']), refs)

    def selected(arm, entries, signals, start, end, *, rsi_entries=()):
        if (start, end) != (START, END):
            raise ValueError('Financial account period differs from sealed candidates')
        return (entries if arm == 'benchmark' else by_arm[arm]), [], []
    runner.select_arm_entries = selected
    frozen_profile_audit = runner.audit_profile_admissions
    runner.audit_profile_admissions = lambda account, profiles, **kw: frozen_profile_audit(account, profiles, hard_filter=True)
    frozen_identity_audit = runner.audit_candidate_identity

    def identity_audit(account, entries):
        cohorts = account['cohorts']
        funded = {c['event_id'] for c in cohorts}
        families = {e['strategy_id'] for e in entries if e['event_id'] in funded}
        if len(families) > 1:
            raise ValueError('One account funded more than one entry strategy')
        result = frozen_identity_audit(account, entries)
        result['strategy_ids'] = sorted(families)
        return result
    runner.audit_candidate_identity = identity_audit
    frozen_write, frozen_registry = runner.write, runner.append_trial_registry
    names = {x['id']: x['name'] for x in definitions}

    def write_result(path, value):
        path = Path(path)
        if path.stem in runner.ARMS and 'family_rules' in value:
            arm = path.stem
            value['family_rules'].update(candidate_family=arm, name=names.get(arm, '0050 含息基準'),
                signal_policy='known_first_parent' if arm.startswith('entry_') else 'daily_matched',
                candidate_priority='signal_amount20_desc_stock_id_asc')
            value['candidate_coverage'] = manifest['counts'].get(arm)
        elif path.name == 'report.json' and 'registered_arms' in value:
            value['schema'] = 'all_entry_accounts_v1'
            value['definitions'] = definitions
            value['candidate_manifest'] = dict(path=str((Path(candidate_bundle)/'manifest.json').relative_to(ROOT)),
                                               sha256=digest(Path(candidate_bundle)/'manifest.json'))
            value['signal_validation'] = dict(through=END, required=True,
                compared='sealed scanner rules, T-known context features and registered daily candidates',
                account_equality_required=False)
            value['candidate_coverage'] = manifest['counts']
            value['external_candidate_count'] = value.pop('rsi_candidate_count')
            value['external_candidate_parameters'] = value.pop('rsi_source')
            value.pop('sealed_hl2_reference', None)
            value['exclusion_scope']['unknown_profiles'] = dict(poc_up_red='quality_unknown_excluded_acquisition_missing_is_fatal')
            value['best_rule_definition'] = 'completed_full_period_net_return_descending; risk and annual results separately disclosed'
            value['preregistration'] = dict(path=str(PREREG.relative_to(ROOT)), sha256=digest(PREREG))
        return frozen_write(path, value)

    def record(value, **kwargs):
        value = deepcopy(value)
        value['source'] = 'all_entry_accounts_20261011'
        value['params']['candidate_bundle'] = str(Path(candidate_bundle).relative_to(ROOT))
        return frozen_registry(value, **kwargs)
    runner.write, runner.append_trial_registry = write_result, record
    return runner, refs


def run(output, arms, *, candidate_bundle=BASE/'inputs-v2', fetch=False):
    from skills.all_entry_account_data import AllEntryData
    runner, refs = configured_runner(candidate_bundle, arms)
    provider = AllEntryData(online=fetch)
    for name, sha in refs.items():
        provider._bind(ROOT/name, sha)
    provider._bind(FROZEN, FROZEN_SHA)
    provider._bind(Path(__file__))
    report = runner.run(output, tuple(arms), provider, candidate_bundle=candidate_bundle)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', default=','.join([x['id'] for x in registry()]+['benchmark']))
    parser.add_argument('--candidate-bundle', type=Path, default=BASE/'inputs-v2')
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    report = run(args.output, args.arms.split(','), candidate_bundle=args.candidate_bundle, fetch=args.fetch)
    print(json.dumps(dict(output=str(args.output), all_completed=report['all_completed'],
                         elapsed_seconds=report['elapsed_seconds']), ensure_ascii=False))
