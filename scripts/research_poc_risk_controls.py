#!/usr/bin/env python3
"""Two independent fixed risk controls on the preserved original POC strategies."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_all_entry_accounts import configured_runner as hard_runner
from scripts.research_legacy_return_comparison import configured_runner as priority_runner
from scripts.research_legacy_return_comparison_v4 import stock_only
from scripts.research_top3_validation_v3 import TERMS
from skills.all_entry_accounts import digest, write, load_candidates
from skills.all_entry_account_data import AllEntryData
from skills.top3_validation import MODES, install
from skills.top3_cost_audit import install as install_cost, BINDINGS
from skills.top3_gap_audit import install as install_gap, SOURCE, SOURCE_SHA
from skills.poc_followup_merger_black import install as install_merger
from skills.legacy_merger_settlement import load_terms
from skills.poc_risk_controls import VARIANTS, install as install_controls
from skills.poc_followup_data_completion import CompletedProfileData as ReusedFollowupData
from skills.poc_followup_snapshots import install as install_snapshots, POLICY as SNAPSHOT_POLICY
from skills.trial_registry import append_trial_registry

BASE = ROOT/'.cache/poc-risk-controls-20261014'
SPEC = ROOT/'docs/prereg_poc_risk_controls_20261014.json'


def run(output, arm, mode, variant, *, donor_manifest, fetch=False):
    class FollowupData(ReusedFollowupData):
        def __init__(self, *, online=False):
            super().__init__(online=online, donor_manifest=donor_manifest)
    if variant not in VARIANTS:
        raise ValueError("Unregistered follow-up variant")
    if arm not in ('poc_up_red', 'poc_priority') or mode not in MODES:
        raise ValueError('Unregistered arm or costs')
    source = ROOT/'scripts/research_all_entry_accounts_tcfhc_completion.py'
    spec = importlib.util.spec_from_file_location('_poc_optimization_settlement', source)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    priority = arm == 'poc_priority'
    bundle = ROOT/('.cache/legacy-return-comparison-20261012/candidates-v1' if priority
        else '.cache/all-entry-accounts-20261011/inputs-v3')
    calendar = (json.loads((bundle/'candidates.json').read_text())['calendar'] if priority
        else load_candidates(bundle)[1])
    from skills.legacy_return_comparison import SPEC as LEGACY_SPEC
    if priority:
        class Evidence(FollowupData):
            def __init__(self, *, online=False):
                super().__init__(online=online)
                self.prereg_path, self.prereg_sha256 = LEGACY_SPEC, digest(LEGACY_SPEC)
                self._bind(LEGACY_SPEC)
        entry.AllEntryData = Evidence
    else:
        entry.AllEntryData = FollowupData

    def configured(candidate_bundle, requested):
        runner, refs = (priority_runner if priority else hard_runner)(candidate_bundle, requested)
        Path(output).resolve().relative_to(BASE)
        runner.RUN_ROOT = Path(output).resolve().parent
        if not priority:
            runner.ARM_RULES = {a: (False, 'none', .5, .3, False) for a in runner.ARMS}
        metadata = dict(variant=variant, arm=arm, mode=mode,
            preregistration=dict(path=str(SPEC.relative_to(ROOT)), sha256=digest(SPEC)),
            buy_fraction=.5, sell_fraction=.3, paper_enabled=False,
            actual_fill_verified=False, unseen_validation=False, live_qualified=False)
        frozen, record = runner.run, runner.append_trial_registry

        def execute(*args, **kwargs):
            install(runner, mode)
            install_cost(runner, mode)
            install_gap(runner)
            install_merger(runner, mode, variant)
            install_snapshots(runner)
            install_controls(runner, variant, metadata)
            return frozen(*args, **kwargs)

        def trial(value, **kwargs):
            value = dict(value, risk_controls=metadata)
            return record(value, **kwargs)
        writer = runner.write
        def write_result(path, value):
            if Path(path).name == 'report.json':
                value['snapshot_storage_policy'] = SNAPSHOT_POLICY
                value['risk_controls'] = metadata
            return writer(path, value)
        runner.write = write_result
        runner.run, runner.append_trial_registry = execute, trial
        refs.update(BINDINGS)
        refs[str(SOURCE.relative_to(ROOT))] = SOURCE_SHA
        refs.update(load_terms()[1])
        dependencies = [source, Path(__file__), SPEC, ROOT/'skills/poc_risk_controls.py', ROOT/'scripts/research_poc_followup_snapshots.py', ROOT/'skills/poc_followup_correlation_records.py', ROOT/'scripts/research_poc_followup_data_completion.py', ROOT/'skills/poc_followup_snapshots.py', ROOT/'skills/poc_followup_data_completion.py', ROOT/'docs/poc_followup_data_completion_20261014.json', ROOT/'scripts/research_poc_followup_merger.py', ROOT/'skills/poc_followup_merger_black.py', ROOT/'docs/poc_followup_merger_black_20261014.json', ROOT/'scripts/research_poc_followup_synchronized.py', ROOT/'skills/poc_followup_shared_acquisition.py', ROOT/'scripts/research_poc_followup_isolated.py', ROOT/'scripts/research_poc_followup_reuse.py', ROOT/'skills/poc_followup_reuse.py', ROOT/'scripts/research_poc_followup_profiles.py', ROOT/'skills/poc_followup_profile_schema.py', ROOT/'scripts/research_poc_followup_completion.py', ROOT/'skills/poc_followup.py', ROOT/'skills/poc_followup_data.py', ROOT/'scripts/research_poc_followup.py', ROOT/'skills/poc_followup_corporate.py',
            ROOT/'scripts/research_all_entry_accounts.py', ROOT/'scripts/research_legacy_return_comparison.py',
            ROOT/'scripts/research_legacy_return_comparison_v4.py', ROOT/'scripts/research_top3_validation_v3.py',
            ROOT/'skills/top3_validation.py', ROOT/'skills/top3_cost_audit.py', ROOT/'skills/top3_gap_audit.py',
            ROOT/'skills/legacy_merger_audit.py', ROOT/'skills/legacy_merger_settlement.py']
        refs.update({str(p.relative_to(ROOT)): digest(p) for p in dependencies})
        return runner, refs
    entry.configured_runner = configured
    return entry.run(output, [arm], candidate_bundle=bundle, fetch=fetch,
        **{k: ROOT/'docs'/(v+'.json') for k, v in TERMS.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['poc_up_red', 'poc_priority'], required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--profile-donors', type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        report = run(args.output, args.arm, args.mode, args.variant, donor_manifest=args.profile_donors, fetch=args.fetch)
    except Exception as exc:
        failure = dict(source='poc_risk_controls_20261014', arm=args.arm, mode=args.mode,
            variant=args.variant, completed=False, reason=str(exc), live_qualified=False)
        if not existed and not (args.output/'report.json').exists():
            write(args.output/'failure.json', failure)
            append_trial_registry(failure, registry_path=args.output/'trials.jsonl')
            append_trial_registry(failure)
        raise
    print(json.dumps(dict(all_completed=report['all_completed'], output=str(args.output))))
    raise SystemExit(0 if report['all_completed'] else 2)
