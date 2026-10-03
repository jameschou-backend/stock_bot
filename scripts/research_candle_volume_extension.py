#!/usr/bin/env python3
"""Replay registered candle/volume arms using the additive 8,000-attempt budget."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import research_candle_volume_account as sealed
from skills.volume_profile_budget_extension import (
    ExtendedAccountProfileData, EXTENSION_SOURCES, EVIDENCE, MAXIMUM, extension_sources,
)
from skills.volume_profile_selection import select_candidates

DEFAULT_ANCHORS = ROOT/'.cache/red-volume-exit-20261003/anchors-v2/report.json'
ANCHOR_ARMS = ('original', 'benchmark', 'poc_base')
INCOMPLETE_ARMS = ('poc_red_dry', 'poc_dry_weak', 'poc_red_dry_weak')


def validate_arms(arms):
    if not arms or len(arms) != len(set(arms)) or any(a not in sealed.ARMS for a in arms):
        raise ValueError('Use only the original eight registered candle/volume arms')
    # This supplement reproduces anchors or completes failures; it cannot replace
    # either of the two already complete variant results.
    if arms != ANCHOR_ARMS and arms != INCOMPLETE_ARMS:
        raise ValueError('Use all three anchors or exactly the three incomplete arms')


def require_extension_anchor_report(path, root=ROOT):
    refs = sealed.require_anchor_report(path, root)
    report = json.loads(Path(path).read_text())
    if report.get('profile_data', {}).get('maximum_adapter_requests') != MAXIMUM:
        raise ValueError('Anchor report was not produced by the fixed 8000 extension')
    ledger_name = str((EVIDENCE/'initial-attempt-ledger.json').relative_to(ROOT))
    ledger_sha = report.get('source_sha256', {}).get(ledger_name)
    if not ledger_sha or sealed.sha(root/ledger_name) != ledger_sha:
        raise ValueError('Extension initial ledger differs from the offline anchor')
    refs[ledger_name] = ledger_sha
    current = extension_sources(root)
    for name in EXTENSION_SOURCES:
        if report.get('source_sha256', {}).get(name) != current[name]:
            raise ValueError('Extension anchor source changed: '+name)
        refs[name] = current[name]
        suffix = '/source-snapshots/'+current[name]+'/'+name
        snapshots = {p: h for p, h in report['source_sha256'].items() if p.endswith(suffix)}
        if len(snapshots) != 1:
            raise ValueError('Extension anchor lacks immutable source snapshot: '+name)
        for snapshot, expected in snapshots.items():
            if expected != current[name] or sealed.sha(root/snapshot) != expected:
                raise ValueError('Extension anchor snapshot changed: '+snapshot)
        refs.update(snapshots)
    return refs


def run_extension(output, arms, *, anchor_report=DEFAULT_ANCHORS,
                  profile_fetch=False, execution_fetch=False, odd_fetch=False,
                  benchmark_limit_overlay=False):
    validate_arms(arms)
    if arms == ANCHOR_ARMS and (profile_fetch or execution_fetch or odd_fetch):
        raise ValueError('Extension anchors must run fully offline')
    # Fail before loading a provider or starting any acquisition.
    anchor_refs = require_extension_anchor_report(anchor_report) if arms == INCOMPLETE_ARMS else {}
    output = Path(output).resolve()
    output.relative_to(ROOT/'.cache/red-volume-exit-20261003')
    if output.exists():
        raise ValueError('Choose a new output directory')
    provider = ExtendedAccountProfileData(
        ROOT/'.cache/market-input-repair-20261002/inputs-v2', online=profile_fetch)
    for name, expected in anchor_refs.items():
        provider._mark(ROOT/name, expected)
    overlay = None
    if benchmark_limit_overlay:
        from skills.benchmark_limit_overlay import load_overlay
        overlay = load_overlay(ROOT)
    # The sealed runner owns the same persistent .run.lock and all financial
    # options. The wrapper has no dates, slots, capital, threshold or cap flags.
    return sealed.run(output, arms, provider, select_candidates,
                      execution_fetch=execution_fetch, limit_overlay=overlay,
                      odd_fetch=odd_fetch, anchor_report=anchor_report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', default=','.join(ANCHOR_ARMS))
    parser.add_argument('--anchor-report', type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument('--profile-fetch', action='store_true')
    parser.add_argument('--execution-fetch', action='store_true')
    parser.add_argument('--odd-fetch', action='store_true')
    parser.add_argument('--benchmark-limit-overlay', action='store_true')
    args = parser.parse_args(argv)
    output_existed = args.output.exists()
    try:
        report = run_extension(args.output, tuple(args.arms.split(',')),
            anchor_report=args.anchor_report, profile_fetch=args.profile_fetch,
            execution_fetch=args.execution_fetch, odd_fetch=args.odd_fetch,
            benchmark_limit_overlay=args.benchmark_limit_overlay)
    except Exception as exc:
        # Retain setup/publication failures as well as per-arm trials written by
        # the sealed runner. Never touch a prior output directory.
        target = args.output.resolve()
        failure = dict(completed=False, stage='data_extension_setup_or_publication',
                       reason=str(exc), source='candle_volume_data_extension_20261003',
                       arms=args.arms.split(','), live_qualified=False)
        try:
            target.relative_to(ROOT/'.cache/red-volume-exit-20261003')
            if not output_existed and not (target/'report.json').exists():
                sealed.write(target/'failure.json', failure)
                sealed.append_trial_registry(failure, registry_path=target/'trials.jsonl')
                sealed.append_trial_registry(failure)
        except ValueError:
            pass
        raise
    return 0 if report['all_completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
