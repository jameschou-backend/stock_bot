"""Isolated, bounded evidence for the unchanged POC/red-candle execution study.

Normal-session prints are never substituted for odd-lot prints. A missing,
empty, conflicted or failed tape stops execution; it is not an unfilled order.
"""
from copy import deepcopy
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import threading

import pandas as pd

from app.file_lock import file_lock
from scripts.prepare_volume_profile import reuse, write
from scripts.research_exit_scenarios import read, sha
from scripts.research_volume_profile import audit_day
from skills.frozen_dividend_copy import ensure_dividend_copy
from skills.intraday_limit_replay import normalize_ticks
from skills.poc_latest_data import (
    ROOT, BUNDLE, OFFICIAL, LatestAccountData, LatestProfiles, merge_refs,
)
from skills.poc_latest_execution import LatestExecutionData, DATASETS, validate_frame, merge_dividends
from skills.replay_market_feeds import ReplayDataUnavailable

BASE = ROOT/'.cache/poc-executable-20261004'
PREREG = ROOT/'docs/prereg_poc_executable_20261004.md'
LATEST = ROOT/'.cache/poc-latest-20261003/run-v3/report.json'
END = '2026-10-02'
ARM = 'poc_red_executable'
RECEIPT_DIRS = (
    '.cache/poc-executable-20261004/board-v1/receipts',
    '.cache/poc-executable-20261004/profiles-v1/receipts',
    '.cache/poc-broker-account-20261004/profiles-v1/receipts',
    '.cache/poc-latest-20261003/profiles-v1/receipts',
    '.cache/volume-profile-account-20261003/profiles-v1/receipts',
    '.cache/volume-profile-20261003/tapes-v1/receipts',
)


def fetch_execution(*args, **kwargs):
    """The shared limiter applies the 10% reserve once, giving at most 5400/h."""
    from app.config import load_config
    from app.finmind import fetch_dataset
    kwargs['requests_per_hour'] = min(6000, load_config().finmind_requests_per_hour)
    return fetch_dataset(*args, **kwargs)


class TickReceipts:
    """Query-bound raw receipts with durable per-purpose request ceilings."""
    def __init__(self, root, directory, *, maximum, online=False, reuse_index=None,
                 reuse_index_ref=None, fetcher=None, config=None, limiter=None):
        self.root = Path(root).resolve()
        self.directory = Path(directory).resolve()
        self.directory.relative_to(self.root)
        if type(maximum) is not int or not 0 <= maximum <= 1800:
            raise ValueError('Invalid executable tick request ceiling')
        self.maximum, self.online = maximum, online
        self.reuse_index = reuse_index or {}
        self.reuse_index_ref = reuse_index_ref
        self.fetcher, self.config, self.limiter = fetcher, config, limiter
        self.refs, self.calls = {}, 0
        self._lock = threading.RLock()

    def _path(self, name):
        path = (self.root/name).resolve()
        path.relative_to(self.root)
        return path

    def _mark(self, name, expected=None):
        path = self._path(name)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Executable tick evidence hash changed: '+str(path))
        merge_refs(self.refs, {str(path.relative_to(self.root)):actual})
        return actual

    def _validate(self, path, query, seen=None):
        """Bind copied receipts, their original receipts and every raw ancestor."""
        path = self._path(path)
        seen = set() if seen is None else seen
        if path in seen:
            raise ValueError('Cyclic executable tick receipt chain')
        seen.add(path)
        self._mark(path)
        item = read(path)
        if item.get('query') != query:
            raise ValueError('Executable tick query identity changed')
        for name, expected in item.get('source_sha256', {}).items():
            self._mark(name, expected)
        if item.get('raw_path'):
            self._mark(item['raw_path'], item['raw_sha256'])
        if item.get('metadata_path'):
            self._mark(item['metadata_path'], item['metadata_sha256'])
            meta = read(self._path(item['metadata_path']))
            if meta.get('query') != query or meta.get('raw_sha256') != item.get('raw_sha256'):
                raise ValueError('Executable tick metadata identity differs')
        for source in item.get('sources', []):
            self._mark(source['path'], source['raw_sha256'])
            self._mark(source['metadata_path'], source['metadata_sha256'])
            meta = read(self._path(source['metadata_path']))
            if meta.get('query') != query or meta.get('raw_sha256') != source['raw_sha256']:
                raise ValueError('Reusable tick version identity differs')
        if item.get('attempt_path'):
            self._mark(item['attempt_path'], item['attempt_sha256'])
            if read(self._path(item['attempt_path'])).get('query') != query:
                raise ValueError('Executable tick reservation identity differs')
        if item.get('reused_receipt'):
            original = self._path(item['reused_receipt'])
            if item.get('reused_receipt_sha256'):
                self._mark(original, item['reused_receipt_sha256'])
            source = self._validate(original, query, seen)
            if any(source.get(k) != item.get(k) for k in ('status', 'raw_path', 'raw_sha256')):
                raise ValueError('Copied executable tick receipt differs from source')
        seen.remove(path)
        return item

    def get(self, sid, day):
        if not re.fullmatch(r'\d{4}', sid) or date.fromisoformat(day).isoformat() != day:
            raise ValueError('Invalid board tick identity')
        if not '2023-11-01' <= day <= END:
            raise ValueError('Executable tick date outside preregistered scope')
        query = dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)
        key = sid+'-'+day
        dest = self.directory/'receipts'/(key+'.json')
        with self._lock:
            if dest.exists():
                return self._validate(dest, query)
            # An interrupted local acquisition may not be hidden by later caches.
            attempt = self.directory/'attempts'/(key+'.json')
            if attempt.exists():
                self._mark(attempt)
                if read(attempt).get('query') != query:
                    raise ValueError('Executable orphan reservation identity differs')
                return dict(query=query, status='orphaned_started_attempt')
            if self.reuse_index.get(key, {}).get('content_conflict'):
                if self.reuse_index_ref:
                    self._mark(*self.reuse_index_ref)
                return dict(query=query, status='cached_versions_conflict')
            candidates = [self.root/folder/(key+'.json') for folder in RECEIPT_DIRS]
            candidates += sorted((self.root/'.cache/poc-daily-opportunities-20261004/data-v1/attempts').glob(key+'-*.json'), reverse=True)
            failed = []
            for source in candidates:
                if source == dest or not source.exists():
                    continue
                item = self._validate(source, query)
                if item.get('status') not in ('received', 'cached'):
                    failed.append(item)
                    continue
                saved = dict(item, reused_receipt=str(source.relative_to(self.root)),
                             reused_receipt_sha256=sha(source),
                             source_sha256={str(source.relative_to(self.root)):sha(source)})
                write(dest, saved)
                return self._validate(dest, query)
            if key in self.reuse_index:
                if self.reuse_index_ref:
                    self._mark(*self.reuse_index_ref)
                # The frozen helper verifies all versions and rejects known conflicts.
                item = reuse(self.reuse_index[key], query)
                item['sources'] = deepcopy(self.reuse_index[key]['sources'])
                if self.reuse_index_ref:
                    source_path, expected = self.reuse_index_ref
                    item['source_sha256'] = {str(self._path(source_path).relative_to(self.root)):expected}
                write(dest, item)
                return self._validate(dest, query)
            if failed:
                # Preserve a previous provider failure; do not silently retry it.
                return dict(query=query, status='prior_cached_attempt_unavailable',
                            prior_statuses=[p['status'] for p in failed])
            if not self.online:
                return dict(query=query, status='not_requested')
            return self._fetch(query, dest, attempt)

    def _fetch(self, query, dest, attempt):
        from app.config import load_config
        from app.finmind import fetch_dataset, FinMindError, FinMindQuotaError
        from app.rate_limiter import get_rate_limiter
        if self.config is None:
            self.config = load_config()
        limit = min(6000, self.config.finmind_requests_per_hour)
        with file_lock(self.directory/'.request.lock', timeout=5):
            if dest.exists():
                return self._validate(dest, query)
            if attempt.exists():
                self._mark(attempt)
                return dict(query=query, status='orphaned_started_attempt')
            stats = (self.limiter or get_rate_limiter)(limit).get_stats()
            if (stats.remaining_requests < 4 or stats.retry_after_seconds > 0 or
                    len(list(attempt.parent.glob('*.json'))) >= self.maximum):
                return dict(query=query, status='request_budget_or_quota_paused')
            reserved = dict(query=query, status='started', maximum_requests=self.maximum,
                            started_at=datetime.now(timezone.utc).isoformat(), retries=0)
            attempt.parent.mkdir(parents=True, exist_ok=True)
            with attempt.open('x') as handle:
                json.dump(reserved, handle)
        item = dict(query=query, attempt_path=str(attempt.relative_to(self.root)),
                    attempt_sha256=sha(attempt), started_at=reserved['started_at'])
        self.calls += 1
        try:
            raw = (self.fetcher or fetch_dataset)('TaiwanStockPriceTick', date.fromisoformat(query['start_date']),
                data_id=query['data_id'], token=self.config.finmind_token,
                requests_per_hour=limit, timeout=40, max_retries=0)
            path = self.directory/'raw'/(query['data_id']+'-'+query['start_date']+'.parquet')
            if path.exists():
                raise ValueError('Executable raw tape already exists without receipt')
            path.parent.mkdir(parents=True, exist_ok=True)
            raw.to_parquet(path, index=False)
            item.update(status='received' if len(raw) else 'empty', raw_path=str(path.relative_to(self.root)),
                        raw_sha256=sha(path), rows=len(raw), retrieved_at=raw.attrs.get('retrieved_at'),
                        cache_hit=bool(raw.attrs.get('cache_hit', False)))
        except FinMindQuotaError as exc:
            item.update(status='quota_paused', retry_after_seconds=exc.retry_after_seconds)
        except FinMindError:
            item.update(status='provider_error', error_type='FinMindError')
        except Exception as exc:
            item.update(status='failed', error_type=type(exc).__name__)
            write(dest, item)
            self._validate(dest, query)
            raise
        write(dest, item)
        return self._validate(dest, query)


class ExecutableProfiles(LatestProfiles):
    def __init__(self, *, online=False):
        super().__init__(BUNDLE, OFFICIAL, online=False)
        self.directory = BASE/'profiles-v1'
        self.maximum, self.online = 1800, online
        index = ROOT/'.cache/volume-profile-20261003/inventory/reuse-index.json'
        self.raw_store = TickReceipts(ROOT, self.directory, maximum=1800, online=online,
            reuse_index=self.reuse_index, reuse_index_ref=(index, self.refs[str(index.relative_to(ROOT))]))

    def _raw(self, coordinate):
        try:
            return self.raw_store.get(*coordinate)
        finally:
            merge_refs(self.refs, self.raw_store.refs)


class BoardTicks:
    def __init__(self, profiles, *, online=False):
        self.profiles, self.refs, self.queries, self.audits, self.loaded = profiles, {}, [], {}, {}
        index = ROOT/'.cache/volume-profile-20261003/inventory/reuse-index.json'
        self.store = TickReceipts(ROOT, BASE/'board-v1', maximum=600, online=online,
            reuse_index=profiles.reuse_index, reuse_index_ref=(index, profiles.refs[str(index.relative_to(ROOT))]))

    def get(self, sid, day, market):
        market = market.upper()
        key = (sid, day, market)
        if key in self.loaded:
            frame, digest = self.loaded[key]
            return frame.copy(deep=True), digest
        self.queries.append(dict(stock_id=sid, date=day, market=market))
        try:
            item = self.store.get(sid, day)
            if item['status'] not in ('received', 'cached'):
                raise ReplayDataUnavailable('Executable board tick unavailable: '+sid+' '+day+' '+item['status'])
            raw = pd.read_parquet(self.store._path(item['raw_path']))
            if raw.empty:
                raise ReplayDataUnavailable('Executable board tape is empty: '+sid+' '+day)
            official, exact = self.profiles._source(sid, day)
            if official is None or official['market'] != market:
                raise ReplayDataUnavailable('Executable board official identity unavailable: '+sid+' '+day)
            checked = audit_day(raw, sid, day, official, exact)
            self.audits[sid+'-'+day] = checked
            if not checked['status'].startswith('usable_'):
                raise ReplayDataUnavailable('Executable board tape quality conflict: '+sid+' '+day+' '+checked['status'])
            frame = normalize_ticks(raw, sid, day, market)
            # Keep real timestamps; order matching must select its committed window.
            frame.attrs['execution_audit'] = deepcopy(checked)
            self.loaded[key] = (frame, item['raw_sha256'])
            return frame.copy(deep=True), item['raw_sha256']
        finally:
            merge_refs(self.refs, self.store.refs)
            merge_refs(self.refs, self.profiles.refs)

    def audit_day(self, sid, day, market, tape, quote):
        key = (sid, day, market.upper())
        if key not in self.loaded:
            raise ValueError('Board audit requires the same loaded raw tape')
        original, digest = self.loaded[key]
        if not tape.equals(original):
            raise ValueError('Execution tape differs from bound source')
        check = self.audits[sid+'-'+day]
        regular = tape.loc[tape.time.ge(pd.Timedelta('09:00:00')) &
            tape.time.lt(pd.Timedelta('13:34:00')) & tape.shares.gt(0)]
        prices = dict(open=float(regular.price.iloc[0]), high=float(regular.price.max()),
                      low=float(regular.price.min()), close=float(regular.price.iloc[-1]))
        import math
        if any(not math.isfinite(float(quote[k])) or abs(prices[k]-float(quote[k]))>1e-6
               for k in prices) or not math.isfinite(float(quote['volume'])) or float(quote['volume'])<=0:
            raise ReplayDataUnavailable('Executable tape/raw account OHLC conflict: '+sid+' '+day)
        if int(tape.shares.sum()) > float(quote['volume']):
            raise ReplayDataUnavailable('Executable tape exceeds account all-session volume: '+sid+' '+day)
        return dict(raw_sha256=digest, official_status=check['status'],
            ordinary_volume_matched=check['ordinary_volume_matched'],
            ordinary_amount_matched=check['ordinary_amount_matched'],
            tick_sequence_complete=False, own_order_fill_proven=False,
            all_session_volume_bound_verified=True)


class ExecutableAccountData(LatestAccountData):
    def __init__(self, root=ROOT, *, online=False):
        super().__init__(root, online=False)
        self.online = online
        self.prereg_path, self.prereg_sha256 = PREREG, sha(PREREG)
        self._bind(PREREG, self.prereg_sha256)
        self._bind(Path(__file__))
        self._bind(LATEST, LATEST.with_suffix('.sha256').read_text().strip())
        report = read(LATEST)
        self.latest_refs = dict(report['source_sha256'])
        source = report['profile_data']['arms']['poc_red']
        self._bind(ROOT/source['path'], source['sha256'])
        self.original_profiles = {}
        for item in read(ROOT/source['path']):
            if item['event_id'] in self.original_profiles:
                raise ValueError('Duplicate frozen executable profile identity')
            self.original_profiles[item['event_id']] = item
        # The report's original data references remain bound to their exact bytes.
        # Current source files are already snapshotted by the parent runner.
        for name, expected in self.latest_refs.items():
            if name.startswith('.cache/') and '/source-snapshots/' not in name:
                self._bind(ROOT/name, expected)
        self.profiles = ExecutableProfiles(online=online)
        self.board_ticks = BoardTicks(self.profiles, online=online)
        self.ticks = self.board_ticks
        self.used = {ARM:{}}
        self.execution = LatestExecutionData(ROOT, online=online, source_refs=self.latest_refs,
                                            maximum_requests=300, fetcher=self._fetch_financial)
        self._financial_calls = 0
        self.execution.directory = BASE/'execution-v1'
        self.execution.dividend_directory = self.execution.directory/'dividends'
        self.execution.directory.mkdir(parents=True, exist_ok=True)
        self.dividend_directory = self.execution.dividend_directory
        self.execution_loaded = {}
        self.after_hours = None
        self._odd_initial_attempts = 0
        self.events, self.calendar = self.profiles.events, self.profiles.calendar
        self.latest_supported_end = self.latest_supported_account_end

    def _bind(self, path, expected=None):
        path = Path(path).resolve()
        path.relative_to(ROOT)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Executable source changed: '+str(path))
        merge_refs(self.refs, {str(path.relative_to(ROOT)):actual})

    def profile(self, arm, event):
        if arm != ARM or len(event.get('members', [])) != 1 or not event['signal_date'] < event['entry_date']:
            raise ValueError('Invalid executable POC candidate')
        eid, sid = event['event_id'], event['members'][0]
        if eid in self.original_profiles:
            result = deepcopy(self.original_profiles[eid])
            if result['stock_id'] != sid or result['signal_date'] != event['signal_date']:
                raise ValueError('Frozen executable POC stock/date mismatch')
        else:
            try:
                result = self.profiles(event)
            finally:
                merge_refs(self.refs, self.profiles.refs)
        self.used[arm][eid] = deepcopy(result)
        return result

    def finmind(self, sid, dataset):
        if not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid) or dataset not in DATASETS:
            raise ValueError('Unplanned executable financial query')
        key = (sid, dataset)
        if key not in self.execution_loaded:
            prior = ROOT/f'.cache/poc-latest-20261003/execution-v1/{sid}-{dataset}.parquet'
            name = str(prior.relative_to(ROOT))
            if name in self.latest_refs:
                self._bind(prior, self.latest_refs[name])
                meta = prior.with_suffix('.json')
                self._bind(meta, self.latest_refs[str(meta.relative_to(ROOT))])
                record = read(meta)
                if (record['stock_id'], record['dataset'], record['start'], record['end'], record['sha256']) != (
                        sid, dataset, '2018-01-01', END, sha(prior)):
                    raise ValueError('Frozen executable financial identity changed')
                frame = pd.read_parquet(prior)
                validate_frame(frame, sid, dataset, '2018-01-01', END)
            elif (reused := self._broker_financial(sid, dataset)) is not None:
                frame = reused
            else:
                receipt = self.execution.directory/'receipts'/(sid+'-'+dataset+'.json')
                if self.online and not receipt.exists():
                    from app.config import load_config
                    from app.rate_limiter import get_rate_limiter
                    stats = get_rate_limiter(min(6000, load_config().finmind_requests_per_hour)).get_stats()
                    if stats.remaining_requests < 4 or stats.retry_after_seconds > 0:
                        raise ReplayDataUnavailable('Executable financial shared quota paused')
                try:
                    frame = self.execution.finmind(sid, dataset)
                finally:
                    merge_refs(self.refs, self.execution.refs)
            if dataset == 'TaiwanStockDividend':
                target = self.dividend_directory/(sid+'.parquet')
                ensure_dividend_copy(frame, target, prepare=True)
                self._bind(target)
            self.execution_loaded[key] = frame.copy(deep=True)
        return self.execution_loaded[key].copy(deep=True)

    def _fetch_financial(self, *args, **kwargs):
        self._financial_calls += 1
        return fetch_execution(*args, **kwargs)

    def _broker_financial(self, sid, dataset):
        """Reconstruct a completed donor from raw and frozen rights, without writes."""
        folder = ROOT/'.cache/poc-broker-account-20261004/execution-v1'
        key = sid+'-'+dataset
        path = folder/(key+'.parquet')
        previous_receipt = folder/'receipts'/(key+'.json')
        if not path.exists():
            if previous_receipt.exists():
                self._bind(previous_receipt)
                raise ReplayDataUnavailable('Prior broker financial attempt is incomplete; no automatic retry: '+key)
            return None
        meta_path, receipt_path = path.with_suffix('.json'), folder/'receipts'/(key+'.json')
        attempt_path, raw_path = folder/'attempts'/(key+'.json'), folder/'raw'/(key+'.parquet')
        for source in (meta_path, receipt_path, attempt_path):
            self._bind(source)
        meta, receipt, attempt = map(read, (meta_path, receipt_path, attempt_path))
        old, old_refs = self.execution._old(sid, dataset)
        merge_refs(self.refs, self.execution.refs)
        start = '2026-09-10' if dataset == 'TaiwanStockPriceLimit' and old is not None else '2018-01-01'
        query = dict(stock_id=sid, dataset=dataset, start=start, end=END)
        if receipt.get('status') != 'received' or receipt.get('query') != query or attempt.get('query') != query:
            raise ValueError('Broker financial donor identity/status differs')
        self._bind(attempt_path, receipt['attempt_sha256'])
        self._bind(raw_path, receipt['raw_sha256'])
        raw = pd.read_parquet(raw_path)
        validate_frame(raw, sid, dataset, start, END)
        combined = merge_dividends(old, raw) if dataset == 'TaiwanStockDividend' else (
            raw.copy(deep=True) if old is None else pd.concat([old, raw], ignore_index=True))
        validate_frame(combined, sid, dataset, '2018-01-01', END)
        expected = dict(stock_id=sid, dataset=dataset, start='2018-01-01', end=END,
            historical_rights_frozen_through='2026-09-09' if old is not None else None,
            old_source_sha256=old_refs, new_query_start=start,
            preparation_rule='ex_date_components' if dataset == 'TaiwanStockDividend' else 'append_only',
            sha256=sha(path))
        if meta != expected or not combined.equals(pd.read_parquet(path)):
            raise ValueError('Broker financial donor reconstruction differs')
        self._bind(path, meta['sha256'])
        return combined

    def get_odd(self, day, sid, market, engine=None):
        if self.after_hours is None:
            from skills.poc_executable_odd import ExecutableOddData
            self.after_hours = ExecutableOddData(ROOT, online=self.online)
            self._odd_initial_attempts = self.after_hours.snapshot()['attempted_calls']
            self._bind(ROOT/'skills/poc_executable_odd.py')
        try:
            return self.after_hours.get(day, sid, market, engine)
        finally:
            merge_refs(self.refs, self.after_hours.refs)

    @property
    def network_calls(self):
        odd_calls = (self.after_hours.snapshot()['attempted_calls']-self._odd_initial_attempts
                     if self.after_hours else 0)
        return self.finmind_calls + odd_calls

    @property
    def finmind_calls(self):
        # Current-process gateway attempts; durable totals are separate below.
        return self.board_ticks.store.calls + self.profiles.raw_store.calls + getattr(self, '_financial_calls', 0)

    def profile_snapshot(self, output):
        output = Path(output)
        features = output/ARM/'profile-features.json'
        write(features, list(self.used[ARM].values()))
        self._bind(features)
        new = self.profiles.snapshot(output/'new-profiles')
        audit = output/'board-tape-audits.json'
        write(audit, dict(queries=self.board_ticks.queries, audits=self.board_ticks.audits))
        self._bind(audit)
        odd_snapshot = self.after_hours.snapshot() if self.after_hours else None
        self._merge_live_refs()
        return dict(schema='poc_executable_data_v1', profiles=new,
            board_attempts=len(list((self.board_ticks.store.directory/'attempts').glob('*.json'))),
            profile_attempts=len(list((self.profiles.directory/'attempts').glob('*.json'))),
            maximum_board_requests=600, maximum_profile_requests=1800, maximum_financial_requests=300,
            after_hours=odd_snapshot,
            tick_sequence_complete=False, own_order_fill_proven=False, live_qualified=False)

    def _merge_live_refs(self):
        for item in (self.profiles, self.board_ticks, self.execution, self.after_hours):
            if item is not None:
                merge_refs(self.refs, item.refs)

    def verify_sources(self):
        self._merge_live_refs()
        for name, expected in self.refs.items():
            if sha(ROOT/name) != expected:
                raise ValueError('Executable input changed: '+name)
