"""Explicit additional profile allowance; old attempts and provider guards survive."""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from app.file_lock import file_lock
from skills.all_entry_account_data import LIMITS, TickData
from skills.all_entry_accounts import digest, write
from skills.poc_followup_shared_acquisition import SynchronizedFollowupData
from skills.strategy_comparison_data import ComparisonDataRequired

ROOT = Path(__file__).resolve().parents[1]
ADDENDUM = ROOT/'docs/poc_followup_data_completion_20261014.json'
ADDITIONAL = 6000


def validate_query(query):
    if (set(query) != {'dataset', 'data_id', 'start_date'}
            or query['dataset'] != 'TaiwanStockPriceTick'
            or not re.fullmatch(r'\d{4}', query['data_id'])
            or not '2023-11-01' <= query['start_date'] <= '2026-10-02'
            or date.fromisoformat(query['start_date']).isoformat() != query['start_date']):
        raise ValueError('Additional profile query exceeds registered scope')


class AdditionalProfileBudget:
    def __init__(self, previous, *, online=False, root=ROOT):
        self.previous, self.online, self.root = previous, online, Path(root)
        self.directory = self.root/'.cache/poc-followup-20261014/profile-allowance-v1'
        self.auth = self.directory/'authorization.json'
        self.refs = {}
        spec = self.root/'docs/poc_followup_data_completion_20261014.json'
        self.expected = dict(schema='poc_followup_additional_profiles_v1',
            preregistration=dict(path=str(spec.relative_to(self.root)), sha256=digest(spec)),
            additional_maximum=ADDITIONAL, old_maximum=LIMITS['profiles'],
            previous_authorization_sha256=digest(previous.auth), retries=0,
            previous_attempts_reset=False, hourly_limit_changed=False)
        if online and not self.auth.exists():
            with file_lock(self.directory/'.authorization.lock'):
                if not self.auth.exists():
                    write(self.auth, dict(self.expected, created_at=datetime.now(timezone.utc).isoformat()))
        if self.auth.exists():
            self.validate()

    def mark(self, path, expected=None):
        path = Path(path); actual = digest(path); name = str(path.relative_to(self.root))
        if (expected is not None and expected != actual) or (name in self.refs and self.refs[name] != actual):
            raise ValueError('Additional profile evidence changed: '+name)
        self.refs[name] = actual

    def validate(self):
        self.previous.validate()
        value = json.loads(self.auth.read_text())
        if any(value.get(k) != v for k, v in self.expected.items()):
            raise ValueError('Additional profile authorization changed')
        self.mark(self.auth)
        ref = self.expected['preregistration']; self.mark(self.root/ref['path'], ref['sha256'])

    def additional_count(self):
        return len(list((self.directory/'attempts').glob('*.json')))

    def can_reserve(self, purpose, query):
        if purpose != 'profiles':
            return
        validate_query(query)
        if not self.online or not self.auth.exists():
            raise ComparisonDataRequired('Additional profile scope is offline')
        self.validate()
        if self.previous.counts()['profiles'] >= LIMITS['profiles'] and self.additional_count() >= ADDITIONAL:
            raise ComparisonDataRequired('Additional 6000-profile allowance exhausted; request was not dispatched')

    def reserve(self, purpose, query):
        if purpose != 'profiles':
            return self.previous.reserve(purpose, query)
        with file_lock(self.directory/'.budget.lock', timeout=10):
            self.can_reserve(purpose, query)
            key = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()+'.json'
            path = self.directory/'attempts'/key
            old = self.previous.directory/'attempts'/'profiles'/key
            if path.exists() or old.exists():
                raise ComparisonDataRequired('Prior profile acquisition exists; no automatic retry')
            if self.previous.counts()['profiles'] < LIMITS['profiles']:
                return self.previous.reserve(purpose, query)
            write(path, dict(query=query, purpose=purpose, authorization_sha256=digest(self.auth),
                started_at=datetime.now(timezone.utc).isoformat(), retries=0))
            self.mark(path)

    def snapshot(self):
        self.validate()
        previous = self.previous.snapshot()
        for name, sha in self.previous.refs.items():
            self.mark(self.root/name, sha)
        for path in (self.directory/'attempts').glob('*.json'):
            row = json.loads(path.read_text()); validate_query(row['query'])
            if row['purpose'] != 'profiles' or row['authorization_sha256'] != digest(self.auth):
                raise ValueError('Additional profile attempt scope changed')
            self.mark(path)
        count = self.additional_count()
        if count > ADDITIONAL:
            raise ValueError('Additional profile allowance exceeded')
        return dict(previous=previous, additional_profile_maximum=ADDITIONAL,
            additional_profile_attempts=count,
            combined_profile_attempts=previous['purpose_attempts']['profiles']+count,
            previous_attempts_reset=False, hourly_limit_changed=False, source_sha256=dict(self.refs))


class CompletionTicks(TickData):
    def _fetch(self, query, dest, attempt):
        # Capacity is checked before a provider-attempt receipt is created.
        self.budget.can_reserve(self.purpose, query)
        from app.config import load_config
        from app.rate_limiter import get_rate_limiter
        if self.config is None:
            self.config = load_config()
        limiter = (self.limiter or get_rate_limiter)(min(6000, self.config.finmind_requests_per_hour))
        wait_for_shared_capacity(limiter)
        return super()._fetch(query, dest, attempt)


def wait_for_shared_capacity(limiter, *, sleep=time.sleep, clock=time.time):
    """Wait before dispatch; never retry a response or change the shared ledger."""
    started = clock()
    while True:
        stats = limiter.get_stats()
        if stats.remaining_requests >= 4 and stats.retry_after_seconds <= 0:
            return
        if clock()-started >= 3660:
            raise ComparisonDataRequired('Shared FinMind capacity still paused; no request dispatched')
        delay = min(60, max(1, stats.retry_after_seconds,
                           stats.window_start_time+3600-clock() if stats.remaining_requests < 4 else 0))
        print('waiting for shared FinMind capacity', round(delay), 'seconds; no request dispatched', flush=True)
        sleep(delay)


class CompletedProfileData(SynchronizedFollowupData):
    def __init__(self, *, online=False, donor_manifest):
        super().__init__(online=online, donor_manifest=donor_manifest)
        with self.shared_evidence.hold():
            self.budget = AdditionalProfileBudget(self.budget, online=online)
            self.profiles.raw_store = CompletionTicks('profiles', profiles=self.profiles,
                                                     budget=self.budget, online=online)
            self.profiles.maximum = self.profiles.raw_store.maximum = LIMITS['profiles']+ADDITIONAL
            self.board_ticks.store.budget = self.budget
            self.execution.budget = self.budget
            self._bind(Path(__file__)); self._bind(ADDENDUM)

    def profile_snapshot(self, output):
        result = super().profile_snapshot(output)
        result['maximum_profile_requests'] = LIMITS['profiles']+ADDITIONAL
        result['additional_profile_scope'] = dict(path=str(ADDENDUM.relative_to(ROOT)), sha256=digest(ADDENDUM))
        return result
