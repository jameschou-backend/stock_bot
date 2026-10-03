"""Causal, query-bound tick profiles for the preregistered cash-account study.

Only the shared FinMind client may acquire missing ticks. Provider errors and
request-budget exhaustion are unresolved data, never negative POC evidence.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import threading

import pandas as pd

from scripts.prepare_volume_profile import local, reuse, verify_saved, write
from scripts.research_volume_profile import audit_day, constant_scale
from scripts.research_early_signal_losses import digest
from skills.board_tape_reconciliation import parse_tpex
from skills.independent_three_black import ThreeBlackPath, path_issue
from skills.intraday_limit_replay import normalize_ticks
from skills.volume_profile import build_volume_profile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'.cache/volume-profile-account-20261003'
PILOT = ROOT/'.cache/volume-profile-20261003'
OFFICIAL = ROOT/'.cache/market-input-repair-20261002/quote-evidence'
QUALITY_REASONS = frozenset({
    'corporate_action_or_nonconstant_price_scale', 'pre_signal_daily_path_invalid',
    'official_identity_missing_or_ambiguous', 'ordinary_tape_conflict',
})


def validate_event(event):
    if len(event.get('members', [])) != 1 or not re.fullmatch(r'[1-9]\d{3}', event['members'][0]):
        raise ValueError('Profile requires one four-digit individual stock')
    sid, signal = event['members'][0], event['signal_date']
    date.fromisoformat(signal)
    if not signal < event['entry_date']:
        raise ValueError('Profile must precede the entry session')
    return sid, signal


def unknown(reason, **extra):
    return dict(available=False, poc_up=None, reason=reason,
                recoverable=reason not in QUALITY_REASONS, **extra)


class AccountProfileData:
    """Build profiles on demand; each exact source and failed request is retained."""
    def __init__(self, bundle, *, online=False, maximum_requests=4800, directory=BASE/'profiles-v1'):
        if type(maximum_requests) is not int or not 0 <= maximum_requests <= 4800:
            raise ValueError('Preregistered account request budget is at most 4800')
        self.bundle, self.directory = local(bundle), local(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.online, self.maximum = online, maximum_requests
        self.refs, self.profiles, self.audits, self.paths, self.parsed = {}, {}, {}, {}, {}
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._config = None
        self.manifest = self._json(self.bundle/'manifest.json')
        self._mark(self.bundle/'manifest.json', 'f62dbe32d263abb464c9f283879b7d6340b0667b6fa9e99404c6ece7e04663d6')
        for name, expected in self.manifest['files_sha256'].items():
            self._mark(self.bundle/name, expected)
        self.days = pd.DatetimeIndex(pd.read_parquet(self.bundle/'close-official.parquet', columns=['date']).date)
        self.calendar = self.days.strftime('%Y-%m-%d').tolist()
        self.events = pd.read_parquet(self.bundle/'events.parquet')
        self.events['event_date'] = pd.to_datetime(self.events.event_date)
        self.official_meta = self._json(OFFICIAL/'official-sources.json')
        self._mark(OFFICIAL/'official-sources.json', (OFFICIAL/'official-sources.sha256').read_text().strip())
        normalized = OFFICIAL/'official-normalized.parquet'
        self._mark(normalized, self.official_meta['output_sha256'][str(normalized.relative_to(ROOT))])
        frame = pd.read_parquet(normalized)
        frame = frame.loc[frame.date.ge('2023-11-01')]
        self.official = frame.set_index(['stock_id', 'date']).sort_index()
        self.reuse_index = self._json(PILOT/'inventory/reuse-index.json')
        self._mark(PILOT/'inventory/reuse-index.json', 'e289bfc2314449b4aae1e3214a8825edc03f89565cb0851228ddd2bf97d80bfc')
        self.pilot_report = self._json(PILOT/'tapes-v1/report.json')
        self._mark(PILOT/'tapes-v1/report.json', '1a184fb1d29918e8b1c941de40c70b7496df3762ffe26fbd78ea376d7598d4b7')
        self.receipt_hashes = self.pilot_report['receipt_sha256']
        self.exact_pilot = {(r['stock_id'], r['date'], r['market']):r for r in
                            self._json(PILOT/'peer-audit/ordinary-exact-reference.json')}
        self._mark(PILOT/'peer-audit/ordinary-exact-reference.json', '7660401aea0b46f615c48bd3c83fc787407d5c340a36525e9a4c187504b2ab9f')
        peer_manifest = self._json(PILOT/'peer-audit/manifest.json')
        self._mark(PILOT/'peer-audit/ordinary-exact-reference.json', peer_manifest['files_sha256']['ordinary-exact-reference.json'])
        self.peer_sources = self._json(PILOT/'peer-audit/source-hashes.json')
        sealed = self._json(PILOT/'research-v2/report.json')
        self._mark(PILOT/'research-v2/report.json', '4455d1261b4b923199aff63e0effc2bfecfcf5aef97ed10412f565a9666f1684')
        for name in ('manifest.json','source-hashes.json'):
            p = PILOT/'peer-audit'/name
            self._mark(p, sealed['source_sha256'][str(p.relative_to(ROOT))])
        for name in ('skills/volume_profile_data.py','skills/volume_profile.py',
                     'scripts/research_volume_profile.py','skills/board_tape_reconciliation.py',
                     'skills/independent_three_black.py','skills/intraday_limit_replay.py',
                     'scripts/prepare_volume_profile.py','tests/test_volume_profile_data.py'):
            self._mark(ROOT/name)
        self._mark(ROOT/'docs/prereg_volume_profile_account_poc_20261003.md',
            'e31cbc7d4c00d69c3a803b0f54c22394e1b6f2f3cc1912180fa0cce269a98832')
        if self.online:
            from app.config import load_config
            self._config = load_config()

    def _mark(self, path, expected=None):
        p = local(path); value = digest(p)
        if expected is not None and value != expected:
            raise ValueError('Profile input hash changed: '+str(p.relative_to(ROOT)))
        key = str(p.relative_to(ROOT))
        if key in self.refs and self.refs[key] != value:
            raise ValueError('Profile input changed during execution: '+key)
        self.refs[key] = value
        return value

    def _json(self, path):
        self._mark(path)
        return json.loads(Path(path).read_text())

    def _raw(self, coordinate):
        sid, day = coordinate; key = sid+'-'+day
        query = dict(dataset='TaiwanStockPriceTick', data_id=sid, start_date=day)
        dest = self.directory/'receipts'/(key+'.json')
        if dest.exists():
            item = verify_saved(json.loads(dest.read_text()), query)
        else:
            prior = PILOT/'tapes-v1/receipts'/(key+'.json')
            if prior.exists():
                self._mark(prior, self.receipt_hashes[str(prior.relative_to(ROOT))])
                item = verify_saved(json.loads(prior.read_text()), query)
                item = dict(item, reused_receipt=str(prior.relative_to(ROOT)))
            elif key in self.reuse_index:
                item = reuse(self.reuse_index[key], query)
            elif not self.online:
                return dict(query=query, status='not_requested')
            else:
                with self._lock:
                    # A lone reservation proves an earlier attempt even when a
                    # crash prevented a receipt. Never overwrite or retry it.
                    if dest.exists():
                        return verify_saved(json.loads(dest.read_text()), query)
                    reservation = self.directory/'attempts'/(key+'.json')
                    if reservation.exists():
                        prior_attempt = json.loads(reservation.read_text())
                        if prior_attempt['query'] != query:
                            raise ValueError('Reserved request identity changed')
                        self._mark(reservation)
                        return dict(query=query,status='orphaned_started_attempt')
                    attempts = len(list((self.directory/'attempts').glob('*.json')))
                    if self._stop.is_set() or attempts >= self.maximum:
                        return dict(query=query, status='request_budget_or_quota_paused')
                    item = dict(query=query, status='started', started_at=datetime.now(timezone.utc).isoformat())
                    # Write an immutable reservation before network I/O. No retries.
                    write(reservation, item)
                    write(dest, item)
                from app.config import load_config
                from app.finmind import fetch_dataset, FinMindQuotaError, FinMindError
                if self._config is None:
                    self._config = load_config()
                try:
                    raw = fetch_dataset('TaiwanStockPriceTick', date.fromisoformat(day), data_id=sid,
                        token=self._config.finmind_token, requests_per_hour=min(5400, self._config.finmind_requests_per_hour),
                        timeout=40, max_retries=0)
                except FinMindQuotaError as exc:
                    self._stop.set()
                    item.update(status='quota_paused', retry_after_seconds=exc.retry_after_seconds)
                except FinMindError:
                    item.update(status='provider_error', error_type='FinMindError')
                else:
                    path = self.directory/'raw'/(key+'.parquet'); path.parent.mkdir(exist_ok=True)
                    temporary = path.with_suffix('.tmp.parquet'); raw.to_parquet(temporary, index=False); temporary.replace(path)
                    item.update(status='received' if len(raw) else 'empty', raw_path=str(path.relative_to(ROOT)),
                        raw_sha256=digest(path), rows=len(raw), retrieved_at=raw.attrs.get('retrieved_at'),
                        cache_hit=raw.attrs.get('cache_hit', False))
            write(dest, item)
        self._mark(dest)
        if item.get('raw_path'):
            self._mark(ROOT/item['raw_path'], item['raw_sha256'])
        if item.get('metadata_path'):
            self._mark(ROOT/item['metadata_path'], item['metadata_sha256'])
        reservation = self.directory/'attempts'/(key+'.json')
        if reservation.exists(): self._mark(reservation)
        return item

    def _source(self, sid, day):
        if (sid, day) not in self.official.index:
            return None, None
        rows = self.official.loc[[(sid, day)]]
        if len(rows) != 1:
            return None, None
        source = rows.iloc[0].to_dict(); ident = source['source_id']
        meta = self.official_meta['sources'][ident]
        for pk, hk in (('path','sha256'), ('receipt','receipt_sha256')):
            self._mark(ROOT/meta[pk], meta[hk])
        exact = self.exact_pilot.get((sid, day, source['market']))
        if exact is not None:
            for path, expected in self.peer_sources.items():
                if path not in self.refs: self._mark(ROOT/path, expected)
        elif source['market'] == 'TPEX' and source['volume_scope'] == 'ordinary_session':
            if ident not in self.parsed:
                self.parsed[ident] = parse_tpex(json.loads((ROOT/meta['path']).read_text()), day)
            if sid in self.parsed[ident]:
                exact = dict(self.parsed[ident][sid], stock_id=sid, date=day, market='TPEX', source_ids=[ident])
        return source, exact

    def _path(self, sid):
        frames = {name:pd.read_parquet(self.bundle/(name+'.parquet'), columns=['date',sid]).set_index('date')
                  for name in ('close-official','close-quality','eligibility')}
        for frame in frames.values(): frame.index = pd.to_datetime(frame.index)
        # Column/row predicate pushdown avoids decoding every stock on each lookup.
        quotes = pd.read_parquet(self.bundle/'quotes-unmasked.parquet', filters=[('stock_id','=',sid)])
        quotes['date'] = pd.to_datetime(quotes.date)
        if quotes.date.duplicated().any() or not quotes.stock_id.eq(sid).all():
            raise ValueError('Ambiguous profile daily stock identity')
        quotes = quotes.set_index('date').reindex(self.days)
        return ThreeBlackPath(self.days,
            *[frames[name][sid].reindex(self.days).to_numpy(bool if name=='eligibility' else float)
              for name in ('close-official','close-quality','eligibility')],
            *[quotes[name].to_numpy(float) for name in ('close','high','low','volume','open')])

    def __call__(self, event):
        sid, signal = validate_event(event); key = sid+'-'+signal
        if key in self.profiles:
            return self.profiles[key]
        if signal not in self.calendar or self.calendar.index(signal) < 20:
            raise ValueError('Profile lacks the canonical 20-session calendar')
        i = self.calendar.index(signal); prior = self.calendar[i-20:i]
        if sid not in self.paths:
            self.paths[sid] = self._path(sid)
        path = self.paths[sid]
        base = dict(stock_id=sid, signal_date=signal, event_id=event['event_id'], prior_dates=prior)
        actions = self.events.loc[self.events.stock_id.eq(sid)&self.events.event_date.between(prior[0],signal)]
        if len(actions) or not constant_scale(path.close[i-20:i+1], path.raw_close[i-20:i+1]):
            result = unknown('corporate_action_or_nonconstant_price_scale', **base)
        elif path_issue(path, i-20, i):
            result = unknown('pre_signal_daily_path_invalid', **base)
        else:
            sources = [self._source(sid, day) for day in prior]
            if any(o is None for o, _ in sources):
                result = unknown('official_identity_missing_or_ambiguous', **base)
            else:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    receipts = list(pool.map(self._raw, [(sid, day) for day in prior]))
                missing = [dict(date=day, status=r['status']) for day,r in zip(prior,receipts)
                           if r['status'] not in ('received','cached')]
                if missing:
                    result = unknown('raw_tape_unavailable', missing=missing, **base)
                else:
                    selected, checks = [], []
                    for day, receipt, (official, exact) in zip(prior,receipts,sources):
                        raw = pd.read_parquet(ROOT/receipt['raw_path'])
                        # Explicit quality statuses are data evidence. Schema,
                        # identity and programming exceptions must halt the run.
                        checked = audit_day(raw, sid, day, official, exact)
                        self.audits[sid+'-'+day] = checked; checks.append(checked)
                        if checked['status'].startswith('usable_'):
                            ticks = normalize_ticks(raw,sid,day,official['market'])
                            ticks = ticks.loc[ticks.time.ge(pd.Timedelta('09:00:00'))&ticks.time.lt(pd.Timedelta('13:34:00'))&ticks.shares.gt(0)].copy()
                            ticks['timestamp'] = pd.Timestamp(day)+ticks.time
                            selected.append(ticks[['timestamp','price','shares']])
                    if len(selected) != 20:
                        result = unknown('ordinary_tape_conflict', issues=[c for c in checks if not c['status'].startswith('usable_')], **base)
                    else:
                        profile = build_volume_profile(pd.concat(selected,ignore_index=True),signal_date=signal,
                            session_dates=prior,bins=40,value_fraction=.70,source_kind='authentic_regular_board_trade_ticks')
                        if not profile['available']:
                            raise ValueError('Audited positive tapes failed the fixed profile algorithm')
                        result = dict(base,available=True,poc_up=profile['poc_up'],reason=None,recoverable=False,
                            profile=profile,ordinary_daily_matched=all(c['ordinary_volume_matched'] and c['ordinary_amount_matched'] for c in checks))
        self.profiles[key] = result
        write(self.directory/'features'/(key+'.json'), result)
        print('profile',sid,signal,'up='+str(result['poc_up']),result['reason'] or '',flush=True)
        return result

    def snapshot(self, output):
        output = local(output)
        for path, expected in tuple(self.refs.items()):
            self._mark(ROOT/path, expected)
        write(output/'profile-features.json', list(self.profiles.values()))
        write(output/'profile-daily-audit.json', list(self.audits.values()))
        report = dict(schema='causal_account_profiles_v1',profiles=len(self.profiles),
            known=sum(p['available'] for p in self.profiles.values()),
            unknown_reasons=dict(Counter(p['reason'] for p in self.profiles.values() if not p['available'])),
            ordinary_daily_matched=sum(p.get('ordinary_daily_matched',False) for p in self.profiles.values()),
            dataset='TaiwanStockPriceTick', maximum_adapter_requests=self.maximum,
            adapter_requests=len(list((self.directory/'attempts').glob('*.json'))),
            source_sha256=dict(self.refs),live_qualified=False)
        write(output/'profile-data-report.json',report)
        return report
