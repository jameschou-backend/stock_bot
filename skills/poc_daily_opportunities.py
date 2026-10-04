"""Account-independent, resumable daily POC with exact tick-price aggregation."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
import json
import math
import re
import time

import pandas as pd

from skills.poc_latest_data import LatestProfiles, BUNDLE, OFFICIAL, sealed_json, bind_closure
from skills.volume_profile_data import ROOT
from skills.volume_profile import build_volume_profile
from skills.independent_three_black import ThreeBlackPath, path_issue
from scripts.research_volume_profile import audit_day, constant_scale
from skills.intraday_limit_replay import normalize_ticks
from scripts.prepare_volume_profile import write, verify_saved, local
from skills.official_daily_acquisition import digest

BASE = ROOT/'.cache/poc-daily-opportunities-20261004'
INVENTORY = BASE/'inventory-v1'
PREREG = ROOT/'docs/prereg_poc_daily_opportunities_20261004.md'
PAYLOAD = ROOT/'.cache/poc-latest-20261003/explorer-v2/payload.json'
PAYLOAD_SHA = '0faa0d005c492d3a58293be0d10cee3d84bdd6cacba759953ba3bc5f653f87c8'
RECEIPT = PAYLOAD.with_name('receipt.json')
RECEIPT_SHA = '2014462759d7adddd3410bf2305b1518456cd9b9e01467b871ec7f1ac28c0a5e'
MAXIMUM_CALLS = 40000


def utc():
    return datetime.now(timezone.utc).isoformat()


def exact_price_sums(raw, sid, day, market):
    """Lossless price/volume sufficient statistics; timestamps remain observed."""
    ticks = normalize_ticks(raw, sid, day, market)
    ticks = ticks.loc[ticks.time.ge(pd.Timedelta('09:00:00')) &
                      ticks.time.lt(pd.Timedelta('13:34:00')) & ticks.shares.gt(0)].copy()
    if any(not math.isfinite(v) or v != int(v) for v in ticks.shares):
        raise ValueError('Board-tape aggregation requires exact integer shares')
    total = sum(int(v) for v in ticks.shares)
    if total >= 2**53:
        raise ValueError('Daily share sum exceeds exact float integer range')
    grouped = ticks.groupby('price', sort=True).agg(shares=('shares', 'sum'), time=('time', 'first'))
    grouped['timestamp'] = pd.Timestamp(day) + grouped.time
    result = grouped.reset_index()[['timestamp', 'price', 'shares']]
    return result, len(ticks), total


def base_row(event, calendar):
    sid, signal = event['stock_id'], event['signal_date']
    if not re.fullmatch(r'[1-9]\d{3}', sid) or date.fromisoformat(signal).year != 2026:
        raise ValueError('Daily POC requires a 2026 four-digit individual-stock signal')
    if signal not in calendar or calendar.index(signal) < 20:
        raise ValueError('Daily POC requires the canonical 20-session warmup')
    i = calendar.index(signal); prior = calendar[i-20:i]
    # Deliberately no entry_date, account positions, cash, or execution decision.
    return dict(signal_id=event['signal_id'], stock_id=sid, signal_date=signal,
                prior_dates=prior, window_start=prior[0], window_end=prior[-1],
                source_date_end=prior[-1], account_independent=True, reconstructed=True)


def assess(row, days):
    """Quality evidence wins over absence; no future/account inputs are accepted."""
    records = [days.get(row['stock_id']+'-'+d) for d in row['prior_dates']]
    bad = [r for r in records if r and r['status'] == 'unknown']
    if bad:
        return dict(row, status='unknown', available=False, reason='ordinary_tape_conflict',
                    issues=[r['audit'] for r in bad], computed_at=utc())
    missing = [d for d, r in zip(row['prior_dates'], records) if not r or r['status'] != 'usable']
    if missing:
        return dict(row, status='pending_data', available=False, reason='raw_tape_unavailable',
                    missing_dates=missing, computed_at=utc())
    totals = sum(r['shares'] for r in records)
    if totals >= 2**53:
        raise ValueError('Window exceeds exact integer aggregation range')
    ticks = pd.concat([r['prices'] for r in records], ignore_index=True)
    profile = build_volume_profile(ticks, signal_date=row['signal_date'],
        session_dates=row['prior_dates'], bins=40, value_fraction=.70,
        source_kind='authentic_regular_board_trade_ticks')
    if not profile['available']:
        raise ValueError('Audited positive daily tapes failed the fixed profile algorithm')
    return dict(row, status='up' if profile['poc_up'] else 'down', available=True, reason=None,
                poc_before=profile['first_half']['poc_price'],
                poc_after=profile['second_half']['poc_price'], computed_at=utc(),
                aggregation='exact_observed_daily_price_integer_share_sums',
                source_regular_tick_rows=sum(r['raw_rows'] for r in records),
                ordinary_daily_matched=all(r['audit']['ordinary_volume_matched'] and
                    r['audit']['ordinary_amount_matched'] for r in records))


class DailyProfiles(LatestProfiles):
    """Reuse immutable dated sources, never the account's lazy query decisions."""
    def __init__(self, directory=BASE/'data-v1'):
        super().__init__(BUNDLE, OFFICIAL, online=False)
        self.directory = local(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self._mark(PREREG); self._mark(Path(__file__))
        self._mark(PAYLOAD, PAYLOAD_SHA); self._mark(RECEIPT, RECEIPT_SHA)
        self.signals = json.loads(PAYLOAD.read_text())['signals']
        if len(self.signals) != 4612 or len({r['signal_id'] for r in self.signals}) != 4612:
            raise ValueError('Original candidate cohort changed')
        manifest = sealed_json(INVENTORY/'manifest.json', self.refs)
        for name, value in manifest['files_sha256'].items(): self._mark(INVENTORY/name, value)
        for name in ('report.json', 'cached-audit-report.json'):
            bind_closure(json.loads((INVENTORY/name).read_text()), self.refs)
        self.raw_index = json.loads((INVENTORY/'tick-reuse.json').read_text())
        self.days_cache = {}; self.unavailable = {}; self.rows = {}; self.structural = {}; self.allowed = set()
        self.attempt_count = len(list((self.directory/'attempts').glob('*.json')))
        self.attempt_paths = None
        self.calls_this_run = 0; self.quota_delay = 0.; self._config = None
        self._seed_rows()

    def _seed_rows(self):
        # One column/row scan per source instead of re-reading the same wide
        # historical parquet files for each of the 771 stocks.
        sids = sorted({e['stock_id'] for e in self.signals})
        positions = {d: i for i, d in enumerate(self.calendar)}
        first = self.calendar[min(positions[e['signal_date']] for e in self.signals)-20]
        frames = {name: pd.read_parquet(self.bundle/(name+'.parquet'), columns=['date']+sids).set_index('date')
                  for name in ('close-official', 'close-quality', 'eligibility')}
        for frame in frames.values(): frame.index = pd.to_datetime(frame.index)
        quotes = pd.read_parquet(self.bundle/'quotes-unmasked.parquet',
            filters=[('date', '>=', pd.Timestamp(first))], columns=['stock_id', 'date', 'close', 'high', 'low', 'volume', 'open'])
        quotes.date = pd.to_datetime(quotes.date)
        for sid, data in quotes[quotes.stock_id.isin(sids)].groupby('stock_id'):
            if data.date.duplicated().any(): raise ValueError('Ambiguous daily stock quote identity')
            data = data.set_index('date').reindex(self.days)
            self.paths[str(sid)] = ThreeBlackPath(self.days,
                *[frames[name][str(sid)].reindex(self.days).to_numpy(bool if name == 'eligibility' else float)
                  for name in ('close-official', 'close-quality', 'eligibility')],
                *[data[name].to_numpy(float) for name in ('close', 'high', 'low', 'volume', 'open')])
        identities = Counter(self.official.index)
        event_days = {str(sid): data.event_date.to_numpy() for sid, data in self.events.groupby('stock_id')}
        for event in self.signals:
            row = base_row(event, self.calendar); sid = row['stock_id']; signal = row['signal_date']
            if sid not in self.paths: raise ValueError('Candidate has no daily quote path')
            path = self.paths[sid]; i = positions[signal]; prior = row['prior_dates']
            actions = event_days.get(sid, [])
            action = bool(len(actions) and ((actions >= self.days[i-20].to_datetime64()) &
                                           (actions <= self.days[i].to_datetime64())).any())
            reason = None
            if action or not constant_scale(path.close[i-20:i+1], path.raw_close[i-20:i+1]):
                reason = 'corporate_action_or_nonconstant_price_scale'
            elif path_issue(path, i-20, i):
                reason = 'pre_signal_daily_path_invalid'
            elif any(identities[(sid, d)] != 1 for d in prior):
                reason = 'official_identity_missing_or_ambiguous'
            if reason:
                row.update(status='unknown', available=False, reason=reason, computed_at=utc())
                self.structural[row['signal_id']] = row
            else:
                self.allowed.update((sid, d) for d in prior)
            self.rows[row['signal_id']] = row
        self.paths.clear()
        identity = dict(schema='poc_daily_acquisition_plan_v1', payload_sha256=PAYLOAD_SHA,
                        inventory_sha256=digest(INVENTORY/'manifest.json'), prereg_sha256=digest(PREREG),
                        maximum_calls=MAXIMUM_CALLS, allowed_coordinates=sorted(self.allowed),
                        ordering='newest_signal_first_then_stock_id', max_retries=0)
        path = self.directory/'plan.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(identity)):
            raise ValueError('Existing daily acquisition plan changed')
        write(path, identity); self._mark(path)

    def _attempts(self, key):
        if self.attempt_paths is None:
            index = {}
            for p in sorted((self.directory/'attempts').glob('*.json')):
                index.setdefault(p.stem.rsplit('-', 1)[0], []).append(p)
            self.attempt_paths = index
        return self.attempt_paths.get(key, [])

    def _stored(self, sid, day):
        key = sid+'-'+day; query = dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)
        if key in self.raw_index:
            sources = self.raw_index[key]; chosen = None; canonical = None
            official, _ = self._source(sid, day)
            for source in sources:
                p = ROOT/source['path']; meta = ROOT/source['metadata_path']
                self._mark(p, source['raw_sha256']); self._mark(meta)
                receipt = json.loads(meta.read_text())
                if receipt.get('query') != query or receipt.get('raw_sha256') != source['raw_sha256']:
                    raise ValueError('Reuse receipt identity differs')
                # Different parquet encodings are acceptable only with identical normalized rows.
                if chosen is None or source['raw_sha256'] != chosen['raw_sha256']:
                    normalized = normalize_ticks(pd.read_parquet(p), sid, day, official['market'])
                    if canonical is not None and not canonical.equals(normalized):
                        raise ValueError('Conflicting normalized cached tape versions')
                    canonical = normalized
                if chosen is None: chosen = source
            return dict(query=query, status='cached', raw_path=chosen['path'], raw_sha256=chosen['raw_sha256'])
        receipt_path = self.directory/'receipts'/(key+'.json')
        attempts = self._attempts(key)
        if receipt_path.exists() or attempts:
            # A missing mutable pointer must not erase an already-started request.
            origin = receipt_path if receipt_path.exists() else attempts[-1]
            item = verify_saved(json.loads(origin.read_text()), query)
            # Mutable receipt pointer is not source provenance; immutable attempts below are.
            if not attempts or json.loads(attempts[-1].read_text()) != item:
                raise ValueError('Receipt differs from its last immutable acquisition attempt')
            for p in attempts: self._mark(p)
            if item.get('raw_path'): self._mark(ROOT/item['raw_path'], item['raw_sha256'])
            return item
        return None

    def _fetch(self, sid, day):
        if (sid, day) not in self.allowed: raise ValueError('Request outside frozen candidate windows')
        from app.config import load_config
        from app.finmind import fetch_dataset, FinMindError, FinMindQuotaError
        from app.rate_limiter import get_rate_limiter
        if self._config is None: self._config = load_config()
        limit = min(6000, self._config.finmind_requests_per_hour)
        stats = get_rate_limiter(limit).get_stats()
        if stats.remaining_requests <= 0 or stats.retry_after_seconds > 0:
            self.quota_delay = max(1., stats.retry_after_seconds)
            return None
        query = dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)
        key = sid+'-'+day
        prior = self._attempts(key)
        if prior:
            last = json.loads(prior[-1].read_text())
            if last.get('query') != query: raise ValueError('Orphan attempt query changed')
            for p in prior: self._mark(p)
            # Quota pauses alone may resume; errors/unknown crash outcomes stay pending.
            if last['status'] != 'quota_paused':
                self.unavailable[key] = last['status']
                return None
        number = len(prior)+1; attempt = self.directory/'attempts'/f'{key}-{number:04d}.json'
        pointer = self.directory/'receipts'/(key+'.json')
        item = dict(query=query, status='started', started_at=utc())
        with self._lock:
            if self.attempt_count >= MAXIMUM_CALLS: return None
            write(attempt, item); write(pointer, item)
            self.attempt_paths.setdefault(key, []).append(attempt)
            self.attempt_count += 1; self.calls_this_run += 1
        try:
            raw = fetch_dataset('TaiwanStockPriceTick', date.fromisoformat(day), data_id=sid,
                token=self._config.finmind_token, requests_per_hour=limit, timeout=40, max_retries=0)
        except FinMindQuotaError as exc:
            self.quota_delay = max(1., exc.retry_after_seconds)
            item.update(status='quota_paused', retry_after_seconds=self.quota_delay)
        except FinMindError:
            item.update(status='provider_error', error_type='FinMindError')
        else:
            path = self.directory/'raw'/(key+'.parquet'); path.parent.mkdir(exist_ok=True)
            temp = path.with_suffix('.tmp.parquet'); raw.to_parquet(temp, index=False); temp.replace(path)
            item.update(status='received' if len(raw) else 'empty', raw_path=str(path.relative_to(ROOT)),
                        raw_sha256=digest(path), rows=len(raw), retrieved_at=raw.attrs.get('retrieved_at'),
                        cache_hit=raw.attrs.get('cache_hit', False))
        item['completed_at'] = utc(); write(attempt, item); write(pointer, item)
        self._mark(attempt)
        if item.get('raw_path'): self._mark(ROOT/item['raw_path'], item['raw_sha256'])
        return item

    def day(self, sid, day, fetch=False):
        key = sid+'-'+day
        if key in self.days_cache: return self.days_cache[key]
        item = self._stored(sid, day)
        if (item is None or item['status'] == 'quota_paused') and fetch:
            item = self._fetch(sid, day)
        if not item or item['status'] not in ('received', 'cached'):
            if item and item['status'] != 'quota_paused':
                self.unavailable[key] = item['status']
            return None
        official, exact = self._source(sid, day)
        if official is None: raise ValueError('Dated market source disappeared after precheck')
        raw = pd.read_parquet(ROOT/item['raw_path']); checked = audit_day(raw, sid, day, official, exact)
        result = dict(status='unknown', audit=checked, raw_path=item['raw_path'], raw_sha256=item['raw_sha256'])
        if checked['status'].startswith('usable_'):
            prices, count, shares = exact_price_sums(raw, sid, day, official['market'])
            result.update(status='usable', prices=prices, raw_rows=count, shares=shares)
        self.days_cache[key] = result
        return result

    def offline(self):
        # Examine every already-present day before any network request, so known bad
        # days prune all overlapping windows even when their other days are missing.
        for sid, day in sorted(self.allowed):
            key = sid+'-'+day
            if (key in self.raw_index or (self.directory/'receipts'/(key+'.json')).exists()
                    or self._attempts(key)):
                self.day(sid, day)
        self.refresh()

    def refresh(self):
        for event in self.signals:
            key = event['signal_id']
            if key not in self.structural:
                self.rows[key] = assess(base_row(event, self.calendar), self.days_cache)

    def fill(self, maximum_new_calls, progress=None):
        """One bounded pass, latest first; return promptly at shared quota exhaustion."""
        if type(maximum_new_calls) is not int or maximum_new_calls < 0:
            raise ValueError('maximum_new_calls must be a nonnegative integer')
        self.quota_delay = 0.; initial = self.calls_this_run
        events = sorted(self.signals, key=lambda e: (-int(e['signal_date'].replace('-', '')), e['stock_id']))
        last_notice = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for event in events:
                row = self.rows[event['signal_id']]
                if row.get('status') != 'pending_data': continue
                pending = [d for d in row['prior_dates'] if row['stock_id']+'-'+d not in self.days_cache]
                while pending:
                    # A prior fetch can have invalidated this window. Only up to
                    # four in-flight days can be rendered unnecessary by a conflict.
                    if any(self.days_cache.get(row['stock_id']+'-'+d, {}).get('status') == 'unknown' for d in row['prior_dates']):
                        break
                    if any(row['stock_id']+'-'+d in self.unavailable for d in row['prior_dates']):
                        break
                    left = maximum_new_calls-(self.calls_this_run-initial)
                    if left <= 0 or self.quota_delay: break
                    batch, pending = pending[:min(4, left)], pending[min(4, left):]
                    list(pool.map(lambda d: self.day(row['stock_id'], d, fetch=True), batch))
                    if progress and time.monotonic()-last_notice >= 30:
                        progress(self); last_notice = time.monotonic()
                self.rows[event['signal_id']] = assess(base_row(event, self.calendar), self.days_cache)
                if self.calls_this_run-initial >= maximum_new_calls or self.quota_delay: break
        self.refresh()

    def snapshot(self, output):
        output = local(output)
        if (output/'report.json').exists(): raise ValueError('Snapshot must be immutable; choose a new output directory')
        for name in ('scripts/prepare_poc_daily_opportunities.py', 'tests/test_poc_daily_opportunities.py'):
            self._mark(ROOT/name)
        for path, expected in tuple(self.refs.items()): self._mark(ROOT/path, expected)
        profiles = output/'profiles.json'; write(profiles, list(self.rows.values()))
        audits = output/'day-audits.json'
        write(audits, {k: {n: v for n, v in r.items() if n != 'prices'} for k, r in self.days_cache.items()})
        counts = dict(Counter(r['status'] for r in self.rows.values()))
        report = dict(schema='poc_daily_profiles_v1', year=2026, end='2026-10-02', created_at=utc(),
            base_payload=dict(path=str(PAYLOAD.relative_to(ROOT)), sha256=PAYLOAD_SHA),
            base_receipt=dict(path=str(RECEIPT.relative_to(ROOT)), sha256=RECEIPT_SHA),
            profiles=dict(path=str(profiles.relative_to(ROOT)), sha256=digest(profiles)),
            source_sha256={**self.refs, str(audits.relative_to(ROOT)): digest(audits)},
            all_signals_materialized=True, all_profiles_available=counts.get('unknown', 0)+counts.get('pending_data', 0)==0,
            status_counts=counts, maximum_calls=MAXIMUM_CALLS, attempted_calls=self.attempt_count,
            unavailable_day_statuses=dict(Counter(self.unavailable.values())),
            quota_delay_seconds=self.quota_delay, account_results_changed=False, live_qualified=False)
        write(output/'report.json', report)
        (output/'report.sha256').write_text(digest(output/'report.json')+'\n')
        return report
