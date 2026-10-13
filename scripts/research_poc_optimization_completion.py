#!/usr/bin/env python3
"""Complete the fixed trials with a fresh, bounded odd-data request scope.

The old receipts and authorization remain immutable. The shared origin hold,
normal-request checks and combined 100-attempt ceiling are preserved.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from scripts import research_poc_optimization as original
from skills.all_entry_account_data import AllEntryData, OddData, BASE as OLD_BASE
from skills.official_daily_acquisition import _write, digest
from skills.strategy_comparison_data import ComparisonDataRequired


class OptimizationOdd(OddData):
    def __init__(self, *, online=False):
        super().__init__(online=False)
        self.cache = original.BASE/'odd-completion-v1'
        self.auth = self.cache/'authorization.json'
        self.online = online
        self.expected_auth = dict(self.expected_auth,
            schema='poc_optimization_odd_authorization_v1',
            authorization_basis='current_user_go_for_strategy_optimization_and_necessary_data',
            request_description='Necessary preplanned odd orders for the fixed POC cooldown trials; combined old and new 100-attempt ceiling',
            prereg_path=str(original.SPEC.relative_to(ROOT)), prereg_sha256=digest(original.SPEC))
        if not self.auth.exists() and any((self.cache/'attempts').glob('*.json')):
            raise ValueError('Optimization odd attempts exist without authorization')
        if online and not self.auth.exists():
            with file_lock(self.cache/'.authorization.lock'):
                if not self.auth.exists():
                    _write(self.auth, dict(self.expected_auth, created_at=self._now()), exclusive=True)
        if self.auth.exists():
            self._authorization(fresh=False)

    def _dispatch_permission(self, request, state, hold, proof_path):
        previous = len(list((OLD_BASE/'odd-v1/attempts').glob('*.json')))
        current = len(list((self.cache/'attempts').glob('*.json')))
        if previous + current >= 100:
            raise ComparisonDataRequired('Combined original and optimization odd 100-attempt budget exhausted')
        return super()._dispatch_permission(request, state, hold, proof_path)


class OptimizationData(AllEntryData):
    def __init__(self, *, online=False):
        super().__init__(online=online)
        previous = self.intraday_odds.acquisition
        previous.online = False
        self.intraday_odds.legacy['all_entry_20261011'] = previous
        self.intraday_odds.acquisition = OptimizationOdd(online=online)
        self.intraday_odds.initial_attempts = len(list((self.intraday_odds.acquisition.cache/'attempts').glob('*.json')))
        self._bind(Path(__file__))
        self._bind(original.SPEC)
        self._merge_live_refs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['poc_priority'], required=True)
    parser.add_argument('--mode', choices=original.MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    existed = args.output.exists()
    original.AllEntryData = OptimizationData
    try:
        report = original.run(args.output, args.arm, args.mode, fetch=args.fetch)
    except Exception as exc:
        failure = dict(source='poc_optimization_completion_20261013', arm=args.arm, mode=args.mode,
            variant='cooldown5', completed=False, reason=str(exc), live_qualified=False)
        if not existed and not (args.output/'report.json').exists():
            original.write(args.output/'failure.json', failure)
            original.append_trial_registry(failure, registry_path=args.output/'trials.jsonl')
            original.append_trial_registry(failure)
        raise
    print(json.dumps(dict(all_completed=report['all_completed'], output=str(args.output))))
    raise SystemExit(0 if report['all_completed'] else 2)
