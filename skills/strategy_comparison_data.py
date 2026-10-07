"""Necessary evidence for the frozen eight-arm account comparison.

The financial/matching rules stay in RangeAccountData. New acquisitions have
their own durable scope; an absent source stops an arm instead of making its
orders appear unsuccessful. Old receipts, failed attempts and origin stops are
never reset. This module itself performs no acquisition at import time.
"""
from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import time

import pandas as pd
import requests

from app.file_lock import file_lock
from scripts.prepare_volume_profile import write
from skills.poc_executable_data import (
    ARM, RECEIPT_DIRS, TickReceipts, fetch_execution,
)
from skills.poc_latest_data import ROOT, merge_refs
from skills.poc_latest_execution import (
    DATASETS, END, START, LatestExecutionData,
)
from skills.poc_range_data import (
    RangeAccountData, RangeOddAcquisition, request_item,
)
from skills.poc_latest_odd import _epoch
from skills.official_daily_acquisition import _write, digest, read
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.volume_profile_data import QUALITY_REASONS, validate_event
from skills import strategy_comparison_corporate

BASE = '.cache/strategy-account-comparison-20261007'
PREREG = 'docs/prereg_strategy_account_comparison_20261007.md'
ORIGINAL_BUDGETS = {'board': 1800, 'profiles': 1800, 'financial': 400}
BUDGETS = {'board': 1200, 'profiles': 1800, 'financial': 1000}
BUDGET_ADDENDUM = 'docs/prereg_strategy_comparison_budget_reallocation_20261007.md'
MAXIMUM_FINMIND = 4000
MAXIMUM_ODD = 100
DAILY_REPORT = '.cache/poc-daily-opportunities-20261004/snapshots/20261004T063227825688Z/0063/report.json'
DAILY_REPORT_SHA = 'd5dcdaf1c2f42bda3ac0eadd8392e0fa86ca8c3703ee61f4b7b1f6da3421976d'


class ComparisonDataRequired(RuntimeError):
    """Necessary evidence is unresolved: the arm is incomplete, not a loser."""


class ComparisonBudget:
    """An immutable authorization plus one exclusive reservation per gateway call."""
    def __init__(self, root, *, online=False):
        self.root = Path(root).resolve()
        self.directory = self.root / BASE / 'finmind-budget-v1'
        self.online, self.refs = online, {}
        self.auth = self.directory / 'authorization.json'
        self.expected = dict(schema='strategy_comparison_finmind_authorization_v1',
            authorization_basis='current_user_requested_fair_poc_red_rsi_account_comparison',
            request_description_is_paraphrase=True, prereg_path=PREREG,
            prereg_sha256=digest(self.root / PREREG), start='2024-01-02', end=END,
            tick_warmup_start='2023-11-01', financial_start=START,
            purpose_maximums=ORIGINAL_BUDGETS, maximum_attempts=MAXIMUM_FINMIND,
            effective_shared_hourly_limit=5400, retries=0, previous_budgets_reset=False)
        self.mark(self.root / PREREG, self.expected['prereg_sha256'])
        if not self.auth.exists() and any((self.directory / 'attempts').glob('*/*.json')):
            raise ValueError('Comparison attempts exist without original authorization')
        if online and not self.auth.exists():
            with file_lock(self.directory / '.authorization.lock'):
                if not self.auth.exists():
                    _write(self.auth, dict(self.expected,
                        created_at=datetime.now(timezone.utc).isoformat()), exclusive=True)
        self.reallocation = self.directory / 'reallocation-v2-authorization.json'
        self.expected_reallocation = None
        if self.auth.exists():
            self.validate_authorization()
        self._initialize_reallocation()

    def _initialize_reallocation(self):
        if (sum(BUDGETS.values()) != MAXIMUM_FINMIND
                or sum(ORIGINAL_BUDGETS.values()) != MAXIMUM_FINMIND
                or BUDGETS['profiles'] != ORIGINAL_BUDGETS['profiles']
                or ORIGINAL_BUDGETS['board'] - BUDGETS['board'] != BUDGETS['financial'] - ORIGINAL_BUDGETS['financial']
                or BUDGETS['board'] > ORIGINAL_BUDGETS['board']):
            raise ValueError('Comparison reallocation changes total scope or the registered transfer')
        if not self.auth.exists():
            if self.reallocation.exists():
                raise ValueError('Comparison reallocation lacks its original authorization')
            return
        self.expected_reallocation = dict(schema='strategy_comparison_finmind_reallocation_v2',
            authorization_basis='continuation_of_user_requested_fair_comparison_without_strategy_changes',
            original_authorization=dict(path=str(self.auth.relative_to(self.root)), sha256=digest(self.auth)),
            addendum_path=BUDGET_ADDENDUM, addendum_sha256=digest(self.root / BUDGET_ADDENDUM),
            prior_purpose_maximums=ORIGINAL_BUDGETS, purpose_maximums=BUDGETS,
            maximum_attempts=MAXIMUM_FINMIND, effective_shared_hourly_limit=5400,
            retries=0, attempts_reset=False, original_sources_mutated=False,
            strategy_parameters_changed=False, genuine_missing_data_policy_changed=False)
        self.mark(self.root / BUDGET_ADDENDUM, self.expected_reallocation['addendum_sha256'])
        if self.online and not self.reallocation.exists():
            with file_lock(self.directory / '.budget.lock', timeout=10):
                if not self.reallocation.exists():
                    self.validate_authorization()
                    counts = self.counts()
                    if any(counts[k] > BUDGETS[k] for k in BUDGETS) or sum(counts.values()) > MAXIMUM_FINMIND:
                        raise ValueError('Cannot transfer already spent comparison allowance')
                    # Every existing reservation must belong to the original
                    # scope. Never mint a new scope to hide missing metadata.
                    original_sha = digest(self.auth)
                    for attempt in (self.directory / 'attempts').glob('*/*.json'):
                        row = read(attempt)
                        if row.get('authorization_sha256') != original_sha:
                            raise ValueError('Cannot reallocate an unrecognized prior reservation')
                        self.mark(attempt)
                    _write(self.reallocation, dict(self.expected_reallocation,
                        prior_attempts_at_reallocation=counts,
                        created_at=datetime.now(timezone.utc).isoformat()), exclusive=True)
        if self.reallocation.exists():
            self._validate_reallocation()

    def _validate_reallocation(self):
        self.validate_authorization()
        value = read(self.reallocation)
        if self.expected_reallocation is None or any(value.get(k) != v for k, v in self.expected_reallocation.items()):
            raise ValueError('Comparison reallocation authorization scope changed')
        prior = value.get('prior_attempts_at_reallocation')
        if (not isinstance(prior, dict) or set(prior) != set(BUDGETS)
                or any(type(prior[k]) is not int or not 0 <= prior[k] <= BUDGETS[k] for k in BUDGETS)):
            raise ValueError('Comparison reallocation baseline is malformed')
        self.mark(self.reallocation)
        self.mark(self.root / BUDGET_ADDENDUM, value['addendum_sha256'])

    def mark(self, path, expected=None):
        path = Path(path).resolve(); name = str(path.relative_to(self.root))
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Comparison budget source changed: ' + name)
        merge_refs(self.refs, {name: actual})
        return actual

    def validate_authorization(self):
        value = read(self.auth)
        if any(value.get(k) != v for k, v in self.expected.items()):
            raise ValueError('Comparison acquisition authorization scope changed')
        self.mark(self.auth)
        self.mark(self.root / PREREG, value['prereg_sha256'])

    def reserve(self, purpose, query):
        if purpose not in BUDGETS:
            raise ValueError('Unregistered comparison acquisition purpose')
        sid = query.get('data_id')
        if not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid):
            raise ValueError('Comparison acquisition needs exact stock identity')
        day = query.get('start_date')
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError('Comparison acquisition date is malformed')
        if purpose == 'financial':
            if query.get('dataset') not in DATASETS or day not in (START, '2026-09-10') or query.get('end_date') != END:
                raise ValueError('Comparison financial acquisition scope differs')
        elif query.get('dataset') != 'TaiwanStockPriceTick' or not '2023-11-01' <= day <= END or 'end_date' in query:
            raise ValueError('Comparison tick acquisition scope differs')
        if not self.online or not self.auth.exists():
            raise ComparisonDataRequired('Comparison acquisition is offline: ' + purpose)
        if not self.reallocation.exists():
            raise ComparisonDataRequired('Comparison reallocation authorization is missing')
        key = sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
        path = self.directory / 'attempts' / purpose / (key + '.json')
        with file_lock(self.directory / '.budget.lock', timeout=10):
            self._validate_reallocation()
            if path.exists():
                self.mark(path)
                raise ComparisonDataRequired('Comparison gateway attempt already reserved; no automatic retry')
            counts = self.counts()
            if counts[purpose] >= BUDGETS[purpose] or sum(counts.values()) >= MAXIMUM_FINMIND:
                raise ComparisonDataRequired('Comparison persistent ' + purpose + ' request budget exhausted')
            _write(path, dict(schema='strategy_comparison_finmind_attempt_v1', purpose=purpose,
                query=query, authorization_sha256=digest(self.reallocation), retries=0,
                started_at=datetime.now(timezone.utc).isoformat()), exclusive=True)
            self.mark(path)

    def counts(self):
        return {name: len(list((self.directory / 'attempts' / name).glob('*.json'))) for name in BUDGETS}

    def snapshot(self):
        accepted_auths = set()
        if self.auth.exists():
            self.validate_authorization()
            accepted_auths.add(digest(self.auth))
        if self.reallocation.exists():
            self._validate_reallocation()
            accepted_auths.add(digest(self.reallocation))
        original_counts = {name: 0 for name in BUDGETS}
        for path in (self.directory / 'attempts').glob('*/*.json'):
            record = read(path)
            purpose, authorization_sha = record.get('purpose'), record.get('authorization_sha256')
            if purpose not in BUDGETS or path.parent.name != purpose or authorization_sha not in accepted_auths:
                raise ValueError('Comparison reservation is outside its authorization')
            if authorization_sha == digest(self.auth):
                original_counts[purpose] += 1
            self.mark(path)
        counts = self.counts()
        if any(original_counts[k] > ORIGINAL_BUDGETS[k] for k in BUDGETS):
            raise ValueError('Original comparison acquisition budget was exceeded')
        if self.reallocation.exists():
            baseline = read(self.reallocation)['prior_attempts_at_reallocation']
            if original_counts != baseline:
                raise ValueError('Original reservations changed after comparison reallocation')
        elif original_counts != counts:
            raise ValueError('Comparison reservations require their reallocation authorization')
        active_limits = BUDGETS if self.reallocation.exists() else ORIGINAL_BUDGETS
        if any(counts[k] > active_limits[k] for k in BUDGETS) or sum(counts.values()) > MAXIMUM_FINMIND:
            raise ValueError('Comparison durable acquisition budget exceeded')
        return dict(purpose_maximums=active_limits, purpose_attempts=counts,
            original_purpose_maximums=ORIGINAL_BUDGETS, original_authorization_attempts=original_counts,
            maximum_attempts=MAXIMUM_FINMIND, attempted_calls=sum(counts.values()),
            authorization=str(self.auth.relative_to(self.root)) if self.auth.exists() else None,
            reallocation_authorization=str(self.reallocation.relative_to(self.root)) if self.reallocation.exists() else None,
            source_sha256=dict(self.refs))


class ComparisonTickReceipts(TickReceipts):
    """Old validated sources first; new board/profile receipts share the budget."""
    def __init__(self, root, directory, *, purpose, budget, online=False, **kwargs):
        super().__init__(root, directory, maximum=BUDGETS[purpose], online=False, **kwargs)
        self.purpose, self.budget, self.acquire_online = purpose, budget, online
        self.gateway = self.fetcher
        self.fetcher = self._gateway

    def _gateway(self, dataset, start, **kwargs):
        query = dict(dataset=dataset, data_id=kwargs['data_id'], start_date=start.isoformat())
        self.budget.reserve(self.purpose, query)
        if kwargs.get('max_retries') != 0 or kwargs.get('requests_per_hour', 6001) > 6000:
            raise ValueError('Comparison gateway retry/hourly settings changed')
        if self.gateway is None:
            from app.finmind import fetch_dataset
            return fetch_dataset(dataset, start, **kwargs)
        return self.gateway(dataset, start, **kwargs)

    def get(self, sid, day):
        # An older interrupted source must not be hidden by a new scope.
        key = sid + '-' + day
        # New sibling purposes are part of the same no-retry source boundary.
        # A profiles orphan cannot be hidden by requesting board, or vice versa.
        folders = (*RECEIPT_DIRS, *(BASE + '/' + name + '-v1/receipts' for name in ('board', 'profiles')))
        for folder in folders:
            parent = self.root / folder
            attempt = parent.parent / 'attempts' / (key + '.json')
            if attempt.exists() and not (parent / (key + '.json')).exists():
                self._mark(attempt)
                raise ComparisonDataRequired('Prior tick attempt is incomplete; no automatic retry: ' + key)
        with self._lock:
            result = super().get(sid, day)
            if result['status'] == 'not_requested':
                query = result['query']; dest = self.directory / 'receipts' / (key + '.json')
                # Share newly acquired sources across purposes without changing
                # the older source precedence or resurrecting failed queries.
                for purpose in ('board', 'profiles'):
                    source = self.root / BASE / (purpose + '-v1') / 'receipts' / (key + '.json')
                    if source == dest or not source.exists():
                        continue
                    item = self._validate(source, query)
                    if item.get('status') not in ('received', 'cached'):
                        raise ComparisonDataRequired('Prior comparison tick source is unavailable: ' + key)
                    write(dest, dict(item, reused_receipt=str(source.relative_to(self.root)),
                        reused_receipt_sha256=digest(source),
                        source_sha256={str(source.relative_to(self.root)): digest(source)}))
                    result = self._validate(dest, query)
                    break
                else:
                    if self.acquire_online:
                        result = self._fetch(query, dest, self.directory / 'attempts' / (key + '.json'))
            # A known disagreement remains quality evidence. Every other failure
            # needs evidence acquisition/resolution, not a losing account trade.
            if result['status'] not in ('received', 'cached', 'cached_versions_conflict'):
                raise ComparisonDataRequired('Necessary ' + self.purpose + ' tick unavailable: ' + key + ' ' + result['status'])
            return result


class ComparisonExecutionData(LatestExecutionData):
    """Reconstruct old financial receipts before using the new financial scope."""
    def __init__(self, root, *, budget, online=False, source_refs=None, config=None, fetcher=None):
        super().__init__(root, online=online, source_refs=source_refs,
                         maximum_requests=ORIGINAL_BUDGETS['financial'], config=config, fetcher=self._gateway)
        # The frozen parent constructor knows only its original 400 ceiling.
        # This explicit new-scope override retains the same on-disk attempts
        # and is additionally guarded by the reallocation's combined ledger.
        self.maximum = BUDGETS['financial']
        self.directory = self.root / BASE / 'execution-v1'
        self.dividend_directory = self.directory / 'dividends'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.budget, self.gateway, self.calls = budget, fetcher, 0

    def _gateway(self, dataset, start, end, **kwargs):
        self.budget.reserve('financial', dict(dataset=dataset, data_id=kwargs['data_id'],
            start_date=start.isoformat(), end_date=end.isoformat()))
        self.calls += 1
        return (self.gateway or fetch_execution)(dataset, start, end, **kwargs)

    def _request(self, sid, dataset, start):
        key = sid + '-' + dataset
        if any((self.directory / kind / (key + '.json')).exists() for kind in ('receipts', 'attempts')):
            try:
                return super()._request(sid, dataset, start)
            except ReplayDataUnavailable as exc:
                raise ComparisonDataRequired(str(exc)) from exc
        # The old executable store may contain additional symbols that are not
        # referenced by the older latest/broker report. Validate its raw chain.
        folder = self.root / '.cache/poc-executable-20261004/execution-v1'
        receipt, attempt = folder / 'receipts' / (key + '.json'), folder / 'attempts' / (key + '.json')
        if receipt.exists() or attempt.exists():
            donor = LatestExecutionData(self.root, online=False, source_refs=self.source_refs)
            donor.directory = folder
            try:
                frame = donor._request(sid, dataset, start)
            except ReplayDataUnavailable as exc:
                raise ComparisonDataRequired(str(exc)) from exc
            finally:
                merge_refs(self.refs, donor.refs)
            return frame
        try:
            return super()._request(sid, dataset, start)
        except ReplayDataUnavailable as exc:
            raise ComparisonDataRequired(str(exc)) from exc


class ComparisonOddAcquisition(RangeOddAcquisition):
    """Same guarded normal requests, separate explicit 100-attempt scope."""
    def __init__(self, root, *, online=False, session=None, clock=time.time, sleep=time.sleep):
        # Do not call the old online constructor or borrow its authorization.
        self.root = Path(root).resolve()
        self.cache, self.online = self.root / BASE / 'odd-v1', online
        self.clock, self.sleep, self.refs = clock, sleep, {}
        self.session = session if session is not None else requests.Session()
        if session is None:
            self.session.trust_env = False
        self.auth = self.cache / 'authorization.json'
        from skills.replay_market_feeds import URLS
        self.expected_auth = dict(schema='strategy_comparison_odd_authorization_v1',
            authorization_basis='current_user_requested_fair_poc_red_rsi_account_comparison',
            request_description='Necessary intraday odd market days for the registered eight-arm comparison',
            request_description_is_paraphrase=True, start='2024-01-02', end=END, urls=URLS,
            maximum_attempts=MAXIMUM_ODD, minimum_interval_seconds=3.1, retries=0,
            global_hold_remains=True, security_bypass_authorized=False,
            prereg_path=PREREG, prereg_sha256=digest(self.root / PREREG))
        self.mark(self.root / PREREG)
        self.authorization = None
        if not self.auth.exists() and any((self.cache / 'attempts').glob('*.json')):
            raise ValueError('Comparison odd attempts exist without original authorization')
        if online and not self.auth.exists():
            with file_lock(self.cache / '.authorization.lock'):
                if not self.auth.exists():
                    _write(self.auth, dict(self.expected_auth, created_at=self._now()), exclusive=True)
        if self.auth.exists():
            self._authorization(fresh=False)

    def _authorization(self, *, fresh):
        self.mark(self.auth)
        value = read(self.auth)
        if any(value.get(k) != v for k, v in self.expected_auth.items()):
            raise ValueError('Comparison odd authorization scope differs')
        if fresh and not 0 <= self.clock() - _epoch(value['created_at']) <= 86400:
            raise ComparisonDataRequired('Comparison odd authorization expired')
        self.mark(self.root / PREREG, value['prereg_sha256'])
        self.authorization = dict(path=str(self.auth.relative_to(self.root)), sha256=digest(self.auth),
                                  created_at=value['created_at'])
        return self.authorization

    def _dispatch_permission(self, request, state, hold, proof_path):
        # Parent calls this under both the experiment and shared origin locks.
        if len(list((self.cache / 'attempts').glob('*.json'))) >= MAXIMUM_ODD:
            raise ComparisonDataRequired('Comparison odd persistent 100-attempt budget exhausted')
        return super()._dispatch_permission(request, state, hold, proof_path)

    def snapshot(self):
        value = super().snapshot()
        value.update(schema='strategy_comparison_odd_acquisition_v1', maximum_attempts=MAXIMUM_ODD)
        return value


def require_profile_evidence(result, event):
    """Only observed, permanent quality unknowns may exclude a candidate."""
    sid, signal = validate_event(event)
    if result.get('stock_id') != sid or result.get('signal_date') != signal:
        raise ValueError('Comparison POC stock/date differs from candidate')
    if result.get('available') is True:
        if type(result.get('poc_up')) is not bool:
            raise ValueError('Comparison known POC direction is not boolean')
    elif result.get('available') is not False or result.get('reason') not in QUALITY_REASONS:
        raise ComparisonDataRequired('Necessary POC source unresolved: ' + sid + ' ' + signal + ' ' + str(result.get('reason')))
    return result


class StrategyComparisonData(RangeAccountData):
    """Drop-in RangeAccountData with strict completeness and new request scope."""
    def __init__(self, root=ROOT, *, online=False):
        root = Path(root).resolve()
        prereg_sha = digest(root / PREREG)
        # Always load the old provider offline. Its budgets/auth remain intact.
        super().__init__(root, online=False)
        self.root, self.online = root, online
        self.prereg_path, self.prereg_sha256 = root / PREREG, prereg_sha
        self._bind(self.prereg_path, prereg_sha); self._bind(Path(__file__))
        self._load_corporate_terms()
        self.budget = ComparisonBudget(root, online=online)
        index = root / '.cache/volume-profile-20261003/inventory/reuse-index.json'
        common = dict(reuse_index=self.profiles.reuse_index,
                      reuse_index_ref=(index, self.profiles.refs[str(index.relative_to(root))]),
                      budget=self.budget, online=online)
        self.profiles.directory = root / BASE / 'profiles-v1'
        self.profiles.maximum = BUDGETS['profiles']
        self.profiles.raw_store = ComparisonTickReceipts(root, self.profiles.directory,
                                                       purpose='profiles', **common)
        self.board_ticks.store = ComparisonTickReceipts(root, root / BASE / 'board-v1',
                                                      purpose='board', **common)
        self.execution = ComparisonExecutionData(root, budget=self.budget, online=online,
                                                source_refs=self.latest_refs)
        self.dividend_directory = self.execution.dividend_directory
        old_odd = self.intraday_odds.acquisition
        self.intraday_odds.legacy['range_original'] = old_odd
        self.intraday_odds.acquisition = ComparisonOddAcquisition(root, online=online)
        self.intraday_odds.initial_attempts = len(list((self.intraday_odds.acquisition.cache / 'attempts').glob('*.json')))
        self.profile_memo, self.profile_origins = {}, {}
        self.daily_profiles = None
        self._merge_live_refs()

    def _load_corporate_terms(self):
        rows, refs, qualifications = strategy_comparison_corporate.load_comparison_corporate_terms(self.root)
        for key, value in rows.items():
            if key in self.corporate_overrides and self.corporate_overrides[key] != value:
                raise ValueError('Comparison corporate supplement conflicts with existing terms: ' + key)
        self.corporate_overrides = dict(self.corporate_overrides, **deepcopy(rows))
        self.comparison_corporate_qualifications = qualifications
        self._bind(Path(strategy_comparison_corporate.__file__))
        for name, expected in refs.items():
            self._bind(self.root / name, expected)

    def _daily_profile(self, event):
        if self.daily_profiles is None:
            self.daily_profiles = {}
            path = self.root / DAILY_REPORT
            # A saved report is optional; the ordinary bounded builder remains
            # explicit when absent. A present damaged report is never ignored.
            if not path.exists():
                return None
            self._bind(path, DAILY_REPORT_SHA)
            report = read(path)
            self._bind(self.root / report['profiles']['path'], report['profiles']['sha256'])
            if report.get('end') != END or report.get('year') != 2026 or report.get('live_qualified') is not False:
                raise ValueError('Daily profile donor scope changed')
            # Bind the exact source closure, including every raw tape; source
            # code was snapshotted by the report and must not masquerade as data.
            for name, expected in report['source_sha256'].items():
                if name.startswith('.cache/'):
                    self._bind(self.root / name, expected)
            for name in ('skills/volume_profile.py', 'skills/poc_daily_opportunities.py'):
                self._bind(self.root / name, report['source_sha256'][name])
            for row in read(self.root / report['profiles']['path']):
                key = (row['stock_id'], row['signal_date'])
                if key in self.daily_profiles:
                    raise ValueError('Duplicate daily POC stock/date')
                self.daily_profiles[key] = row
        row = self.daily_profiles.get((event['members'][0], event['signal_date']))
        if row is None or row.get('status') == 'pending_data':
            return None
        row = deepcopy(row)
        i = self.calendar.index(event['signal_date'])
        if row.get('prior_dates') != self.calendar[i-20:i] or row.get('account_independent') is not True:
            raise ValueError('Cached POC window differs from frozen account calendar')
        if row.get('available') is True:
            if row.get('status') not in ('up', 'down') or (row['poc_after'] > row['poc_before']) != (row['status'] == 'up'):
                raise ValueError('Cached daily POC direction disagrees with profile')
            row['poc_up'] = row['status'] == 'up'
        else:
            row['poc_up'] = None
        row.update(event_id=event['event_id'], recoverable=False,
                   reused_daily_report=DAILY_REPORT, known_at_signal_realtime_receipts_verified=False)
        return row

    def profile(self, arm, event):
        if arm != ARM:
            raise ValueError('Unregistered comparison profile arm')
        sid, signal = validate_event(event)
        key = event['event_id']
        if key in self.profile_memo:
            return deepcopy(require_profile_evidence(self.profile_memo[key], event))
        if key in self.original_profiles:
            result = super().profile(arm, event)
            origin = 'frozen_original_account'
        elif (result := self._daily_profile(event)) is not None:
            origin = 'frozen_daily_profile'
        else:
            result = super().profile(arm, event)
            origin = 'necessary_lazy_profile'
        if (result.get('reason') == 'raw_tape_unavailable' and result.get('missing')
                and all(x.get('status') == 'cached_versions_conflict' for x in result['missing'])):
            result = dict(result, reason='ordinary_tape_conflict', recoverable=False,
                          issues=deepcopy(result['missing']))
        require_profile_evidence(result, event)
        self.profile_memo[key] = deepcopy(result)
        self.profile_origins[key] = origin
        self.used[arm][key] = deepcopy(result)
        return deepcopy(result)

    def finmind(self, sid, dataset):
        try:
            return super().finmind(sid, dataset)
        except ReplayDataUnavailable as exc:
            raise ComparisonDataRequired('Necessary financial evidence unresolved: ' + str(exc)) from exc

    def get_odd(self, day, sid, market, engine=None):
        # Old unfinished range requests are not absent market-day tables.
        key = request_item(day, market)['market'] + '-' + day
        for old in self.intraday_odds.legacy.values():
            orphan = old.cache / 'attempts' / (key + '.json')
            if orphan.exists() and not (old.cache / 'receipts' / (key + '.json')).exists():
                self._bind(orphan)
                raise ComparisonDataRequired('Prior odd attempt unfinished; no automatic retry: ' + key)
        try:
            return super().get_odd(day, sid, market, engine)
        except ReplayDataUnavailable as exc:
            # Fetched table quality can reject an order. Missing tables, expired
            # authorization, failed requests and holds stop this comparison arm.
            if any(reason in str(exc) for reason in (
                'table lacks stock', 'Conflicting official intraday rows',
                'Supplementary odd stock absent', 'Conflicting supplementary odd rows',
            )):
                raise
            raise ComparisonDataRequired('Necessary odd evidence unresolved: ' + str(exc)) from exc

    @property
    def finmind_calls(self):
        return self.board_ticks.store.calls + self.profiles.raw_store.calls + self.execution.calls

    def _merge_live_refs(self):
        super()._merge_live_refs()
        if hasattr(self, 'budget'):
            merge_refs(self.refs, self.budget.refs)

    def profile_snapshot(self, output):
        value = super().profile_snapshot(output)
        value.update(schema='strategy_comparison_account_data_v1',
            maximum_board_requests=BUDGETS['board'], maximum_profile_requests=BUDGETS['profiles'],
            maximum_financial_requests=BUDGETS['financial'], maximum_official_odd_requests=MAXIMUM_ODD,
            acquisition_budget=self.budget.snapshot(), profile_origins=deepcopy(self.profile_origins),
            corporate_settlement_qualifications=deepcopy(self.comparison_corporate_qualifications),
            unresolved_required_sources_policy='arm_incomplete', live_qualified=False)
        self._merge_live_refs()
        return value
