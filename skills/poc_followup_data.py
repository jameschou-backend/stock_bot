"""Demand-bound new data scope; every earlier receipt/failure keeps precedence."""
from pathlib import Path

from app.file_lock import file_lock
from skills.all_entry_account_data import AllEntryData, OddData, BASE as OLD_BASE
from skills.official_daily_acquisition import _write, digest
from skills.strategy_comparison_data import ComparisonDataRequired
from scripts.research_poc_optimization_completion import OptimizationOdd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'.cache/poc-followup-20261014'
SPEC = ROOT/'docs/prereg_poc_followup_20261014.json'


class FollowupOdd(OddData):
    def __init__(self, *, online=False):
        super().__init__(online=False)
        self.cache = BASE/'odd-v1'
        self.auth = self.cache/'authorization.json'
        self.online = online
        self.expected_auth = dict(self.expected_auth,
            schema='poc_followup_odd_authorization_v1',
            authorization_basis='user_requested_execution_exit_poc_correlation_and_missing_data',
            request_description='Only positive preplanned orders for 24 fixed follow-up cases; at most 100 additional market-day requests. Earlier receipts and failures retain precedence.',
            prereg_path=str(SPEC.relative_to(ROOT)), prereg_sha256=digest(SPEC))
        if not self.auth.exists() and any((self.cache/'attempts').glob('*.json')):
            raise ValueError('Follow-up attempts exist without authorization')
        if online and not self.auth.exists():
            with file_lock(self.cache/'.authorization.lock'):
                if not self.auth.exists():
                    _write(self.auth, dict(self.expected_auth, created_at=self._now()), exclusive=True)
        if self.auth.exists():
            self._authorization(fresh=False)

    def _dispatch_permission(self, request, state, hold, proof_path):
        old = [OLD_BASE/'odd-v1', ROOT/'.cache/poc-optimization-20261013/odd-completion-v1']
        if sum(len(list((p/'attempts').glob('*.json'))) for p in old) > 100:
            raise ComparisonDataRequired('Earlier combined scope exceeded its original ceiling')
        return super()._dispatch_permission(request, state, hold, proof_path)


class FollowupData(AllEntryData):
    def __init__(self, *, online=False):
        super().__init__(online=online)
        prior = self.intraday_odds.acquisition
        prior.online = False
        self.intraday_odds.legacy['all_entry_20261011'] = prior
        self.intraday_odds.legacy['optimization_20261013'] = OptimizationOdd(online=False)
        self.intraday_odds.acquisition = FollowupOdd(online=online)
        self.intraday_odds.initial_attempts = len(list((self.intraday_odds.acquisition.cache/'attempts').glob('*.json')))
        self._bind(Path(__file__))
        self._bind(ROOT/'scripts/research_poc_optimization_completion.py')
        self._bind(SPEC)
        self._merge_live_refs()
