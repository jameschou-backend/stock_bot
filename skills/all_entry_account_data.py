"""Necessary evidence acquisition for the explicitly requested 31-rule study.

Old immutable receipts and failures retain precedence. New requests have their
own durable budget and use the existing shared FinMind gateway and TLS guards.
"""
from datetime import datetime, timezone, date
import hashlib
import json
from pathlib import Path
import re

from app.file_lock import file_lock
from skills.all_entry_accounts import ROOT, BASE, PREREG, END, digest, write
from skills.strategy_comparison_data import (
    StrategyComparisonData, ComparisonDataRequired, ComparisonTickReceipts,
    ComparisonExecutionData, ComparisonOddAcquisition,
)
from skills.poc_latest_execution import LatestExecutionData, DATASETS
from skills.poc_latest_data import merge_refs

LIMITS = dict(financial=6000, board=6000, profiles=8000)


class AccountBudget:
    def __init__(self, *, online=False):
        self.online, self.refs = online, {}
        self.directory = BASE / 'finmind-budget-v1'
        self.auth = self.directory / 'authorization.json'
        self.expected = dict(schema='all_entry_finmind_authorization_v1',
            authorization_basis='user_requested_29_entry_rules_plus_2_context_accounts_and_missing_data',
            prereg_path=str(PREREG.relative_to(ROOT)), prereg_sha256=digest(PREREG),
            start='2024-01-02', end=END, purpose_maximums=LIMITS,
            maximum_attempts=sum(LIMITS.values()), retries=0, previous_budgets_reset=False)
        if online and not self.auth.exists():
            with file_lock(self.directory / '.authorization.lock'):
                if not self.auth.exists():
                    write(self.auth, dict(self.expected, created_at=datetime.now(timezone.utc).isoformat()))
        if self.auth.exists():
            self.validate()

    def mark(self, path):
        name, actual = str(Path(path).relative_to(ROOT)), digest(path)
        if name in self.refs and self.refs[name] != actual:
            raise ValueError('Acquisition evidence changed: ' + name)
        self.refs[name] = actual

    def validate(self):
        value = json.loads(self.auth.read_text())
        if any(value.get(k) != v for k,v in self.expected.items()):
            raise ValueError('All-entry acquisition scope changed')
        self.mark(self.auth); self.mark(PREREG)

    def counts(self):
        return {k: len(list((self.directory/'attempts'/k).glob('*.json'))) for k in LIMITS}

    def reserve(self, purpose, query):
        sid, start = query.get('data_id'), query.get('start_date')
        if purpose not in LIMITS or not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid):
            raise ValueError('Unregistered all-entry acquisition')
        if date.fromisoformat(start).isoformat() != start:
            raise ValueError('Invalid acquisition date')
        if purpose == 'financial':
            if query.get('dataset') not in DATASETS or start not in ('2018-01-01', '2026-09-10') or query.get('end_date') != END:
                raise ValueError('Financial query exceeds fixed account period')
        elif query.get('dataset') != 'TaiwanStockPriceTick' or not '2023-11-01' <= start <= END or 'end_date' in query:
            raise ValueError('Tick query exceeds fixed account period')
        if not self.online or not self.auth.exists():
            raise ComparisonDataRequired('All-entry acquisition is offline')
        key = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
        path = self.directory/'attempts'/purpose/(key+'.json')
        with file_lock(self.directory/'.budget.lock', timeout=10):
            self.validate()
            if path.exists():
                raise ComparisonDataRequired('Prior acquisition exists; no automatic retry')
            if self.counts()[purpose] >= LIMITS[purpose]:
                raise ComparisonDataRequired('All-entry ' + purpose + ' persistent budget exhausted')
            write(path, dict(query=query, purpose=purpose, authorization_sha256=digest(self.auth),
                             started_at=datetime.now(timezone.utc).isoformat(), retries=0))
            self.mark(path)

    def snapshot(self):
        if self.auth.exists():
            self.validate()
        for path in (self.directory/'attempts').glob('*/*.json'):
            value = json.loads(path.read_text())
            if value['purpose'] != path.parent.name or value['authorization_sha256'] != digest(self.auth):
                raise ValueError('Acquisition receipt belongs to another scope')
            self.mark(path)
        counts = self.counts()
        if any(counts[k] > LIMITS[k] for k in LIMITS):
            raise ValueError('All-entry budget exceeded')
        return dict(purpose_maximums=LIMITS, purpose_attempts=counts,
                    maximum_attempts=sum(LIMITS.values()), attempted_calls=sum(counts.values()),
                    source_sha256=dict(self.refs))


class FinancialData(LatestExecutionData):
    def __init__(self, source_refs, budget, *, online=False):
        super().__init__(ROOT, online=online, source_refs=source_refs, fetcher=self._gateway)
        self.directory = BASE/'execution-v1'
        self.dividend_directory = self.directory/'dividends'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.maximum, self.budget, self.calls = LIMITS['financial'], budget, 0
        self.donor = ComparisonExecutionData(ROOT, budget=budget, online=False, source_refs=source_refs)

    def _gateway(self, dataset, start, end, **kwargs):
        from app.finmind import fetch_dataset
        self.budget.reserve('financial', dict(dataset=dataset, data_id=kwargs['data_id'],
                            start_date=start.isoformat(), end_date=end.isoformat()))
        self.calls += 1
        return fetch_dataset(dataset, start, end, **kwargs)

    def _request(self, sid, dataset, start):
        key = sid+'-'+dataset+'.json'
        directories = [self.donor.directory, ROOT/'.cache/poc-executable-20261004/execution-v1']
        if any((folder/kind/key).exists() for folder in directories for kind in ('attempts','receipts')):
            try:
                return self.donor._request(sid, dataset, start)
            finally:
                merge_refs(self.refs, self.donor.refs)
        return super()._request(sid, dataset, start)


class TickData(ComparisonTickReceipts):
    def __init__(self, purpose, *, profiles, budget, online):
        index = ROOT/'.cache/volume-profile-20261003/inventory/reuse-index.json'
        super().__init__(ROOT, BASE/(purpose+'-v1'), purpose=purpose, budget=budget,
            online=online, reuse_index=profiles.reuse_index,
            reuse_index_ref=(index, profiles.refs[str(index.relative_to(ROOT))]))
        self.maximum = LIMITS[purpose]

    def get(self, sid, day):
        # Reuse a sibling purpose only with its complete chain. An orphan or
        # empty receipt must never cause another download under a new purpose.
        key = sid+'-'+day+'.json'
        for purpose in ('board', 'profiles'):
            folder = BASE/(purpose+'-v1')
            receipt, attempt = folder/'receipts'/key, folder/'attempts'/key
            if folder == self.directory:
                continue
            if attempt.exists() and not receipt.exists():
                self._mark(attempt)
                raise ComparisonDataRequired('Sibling tick attempt is unfinished: '+key)
            if receipt.exists() and not (self.directory/'receipts'/key).exists():
                query = dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)
                row = self._validate(receipt, query)
                if row.get('status') not in ('received','cached'):
                    raise ComparisonDataRequired('Sibling tick source unavailable: '+key)
                from scripts.prepare_volume_profile import write as receipt_write
                receipt_write(self.directory/'receipts'/key, dict(row,
                    reused_receipt=str(receipt.relative_to(ROOT)), reused_receipt_sha256=digest(receipt),
                    source_sha256={str(receipt.relative_to(ROOT)):digest(receipt)}))
        return super().get(sid, day)


class OddData(ComparisonOddAcquisition):
    def _authorization(self, *, fresh):
        self.mark(self.auth)
        value = json.loads(self.auth.read_text())
        if any(value.get(k) != v for k, v in self.expected_auth.items()):
            raise ValueError('All-entry odd authorization scope differs')
        created = datetime.fromisoformat(value['created_at']).timestamp()
        if fresh and not 0 <= self.clock() - created <= 86400:
            raise ComparisonDataRequired('All-entry odd authorization expired')
        self.mark(self.root / value['prereg_path'], value['prereg_sha256'])
        self.authorization = dict(path=str(self.auth.relative_to(self.root)),
            sha256=digest(self.auth), created_at=value['created_at'])
        return self.authorization

    def __init__(self, *, online=False):
        # Initialize all guards offline; give this request its own auth/cache.
        super().__init__(ROOT, online=False)
        self.cache, self.online, self.refs = BASE/'odd-v1', online, {}
        self.auth = self.cache/'authorization.json'
        self.expected_auth = dict(self.expected_auth,
            schema='all_entry_odd_authorization_v1',
            authorization_basis='user_requested_31_account_backtests_and_necessary_evidence',
            request_description='Necessary intraday odd market days for the fixed 31-rule account comparison',
            prereg_path=str(PREREG.relative_to(ROOT)), prereg_sha256=digest(PREREG))
        # The inherited normal-request guard is stricter (100 attempts) than
        # the registered upper ceiling; no expired old authorization is used.
        if online and not self.auth.exists():
            write(self.auth, dict(self.expected_auth, created_at=self._now()))
        if self.auth.exists():
            self._authorization(fresh=False)


class AllEntryData(StrategyComparisonData):
    def __init__(self, *, online=False):
        super().__init__(ROOT, online=False)
        self.online = online
        self.prereg_path, self.prereg_sha256 = PREREG, digest(PREREG)
        self._bind(PREREG); self._bind(Path(__file__))
        self.budget = AccountBudget(online=online)
        self.profiles.directory, self.profiles.maximum = BASE/'profiles-v1', LIMITS['profiles']
        self.profiles.raw_store = TickData('profiles', profiles=self.profiles, budget=self.budget, online=online)
        self.board_ticks.store = TickData('board', profiles=self.profiles, budget=self.budget, online=online)
        self.execution = FinancialData(self.latest_refs, self.budget, online=online)
        self.dividend_directory = self.execution.dividend_directory
        self.intraday_odds.legacy['comparison_20261007'] = self.intraday_odds.acquisition
        self.intraday_odds.acquisition = OddData(online=online)
        self.intraday_odds.initial_attempts = len(list((self.intraday_odds.acquisition.cache/'attempts').glob('*.json')))
        self._merge_live_refs()

    def profile_snapshot(self, output):
        result = super().profile_snapshot(output)
        result.update(schema='all_entry_account_data_v1', maximum_board_requests=LIMITS['board'],
                      maximum_profile_requests=LIMITS['profiles'], maximum_financial_requests=LIMITS['financial'],
                      maximum_official_odd_requests=100)
        return result
