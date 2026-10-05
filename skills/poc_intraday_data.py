"""Official intraday odd daily data for a separate HL2 execution estimate.

Each origin's first necessary day is its single normal probe. Existing origin
holds are retained, and failures are immutable rather than retried. This module
does not alter the sealed latest-account adapter or borrow its authorization.
"""
from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import time

import requests

from app.file_lock import file_lock
from skills.official_daily_acquisition import (
    OfficialDailyAcquisition, _security_response, _write, digest, read,
)
from skills.poc_latest_odd import _epoch, wrapper
from skills.replay_market_feeds import URLS, ReplayDataUnavailable, parse_odd

START, END = '2024-01-02', '2026-10-02'
BASE = '.cache/poc-intraday-20261005/odd-v1'
PREREG = 'docs/prereg_poc_intraday_20261005.md'
MAXIMUM = 400


def request_item(day, market):
    stamp = date.fromisoformat(day)
    market = str(market).upper()
    if stamp.isoformat() != day or not START <= day <= END or market.lower() not in URLS:
        raise ValueError('Intraday odd request outside registered market/date scope')
    return dict(market=market, date=day, url=URLS[market.lower()],
                params=dict(date=stamp.strftime('%Y%m%d' if market == 'TWSE' else '%Y/%m/%d'), response='json'))


def validate_plans(day, sid, plans):
    if not isinstance(sid, str) or len(sid) != 4 or not sid.isdigit():
        raise ValueError('Intraday odd demand needs a four-digit stock')
    if not isinstance(plans, list) or not plans:
        raise ValueError('Intraday odd demand needs precommitted positive odd-share orders')
    for plan in plans:
        qty, odd = plan.get('planned_qty'), plan.get('odd_qty')
        if (plan.get('date') != day or plan.get('stock_id') != sid
                or type(qty) is not int or type(odd) is not int or not 0 < odd <= qty
                or odd >= 1000):
            raise ValueError('Intraday odd precommitted quantity or identity differs')
        board = plan.get('board_qty')
        if board is not None and (type(board) is not int or board < 0 or board % 1000 or board + odd != qty):
            raise ValueError('Intraday odd board and odd quantities disagree')
    return deepcopy(plans)


def requirement(day, sid, engine):
    values = getattr(engine, 'day_plans', {})
    if not isinstance(values, dict):
        raise ValueError('Intraday odd requires the frozen day-plan mapping')
    plans = [p for p in values.values() if p.get('date') == day and p.get('stock_id') == sid
             and p.get('planned_qty', 0) > 0 and p.get('odd_qty', 0) > 0]
    return validate_plans(day, sid, plans)


class IntradayOddAcquisition:
    def __init__(self, root, *, online=False, session=None, clock=time.time, sleep=time.sleep):
        self.root = Path(root).resolve()
        self.cache, self.online = self.root / BASE, online
        self.clock, self.sleep, self.refs = clock, sleep, {}
        self.session = session if session is not None else requests.Session()
        if session is None:
            self.session.trust_env = False
        self.auth = self.cache / 'authorization.json'
        self.expected_auth = dict(schema='poc_intraday_odd_authorization_v1',
            authorization_basis='current_user_requested_intraday_odd_cash_account_backtest',
            request_description='Necessary official intraday odd-lot daily evidence for this original POC plus red-candle intraday-odd account backtest',
            request_description_is_paraphrase=True, start=START, end=END, urls=URLS,
            maximum_attempts=MAXIMUM, minimum_interval_seconds=3.1, retries=0,
            global_hold_remains=True, security_bypass_authorized=False,
            prereg_path=PREREG, prereg_sha256=digest(self.root / PREREG))
        self.mark(self.root / PREREG)
        self.authorization = None
        if online and not self.auth.exists():
            with file_lock(self.cache / '.authorization.lock'):
                if not self.auth.exists():
                    _write(self.auth, dict(self.expected_auth, created_at=self._now()), exclusive=True)
        if self.auth.exists():
            self._authorization(fresh=False)

    def _now(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).isoformat()

    def path(self, value):
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Intraday odd evidence escapes repository')
        return path

    def mark(self, path, expected=None):
        path = self.path(path)
        actual, name = digest(path), str(path.relative_to(self.root))
        if expected is not None and actual != expected:
            raise ValueError('Intraday odd source hash differs: ' + name)
        if name in self.refs and self.refs[name] != actual:
            raise ValueError('Intraday odd source mutated: ' + name)
        self.refs[name] = actual
        return actual

    def _authorization(self, *, fresh):
        self.mark(self.auth)
        value = read(self.auth)
        if any(value.get(k) != v for k, v in self.expected_auth.items()):
            raise ValueError('Intraday odd authorization scope differs')
        if fresh and not 0 <= self.clock() - _epoch(value['created_at']) <= 86400:
            raise ReplayDataUnavailable('Intraday odd authorization expired')
        self.mark(self.root / PREREG, value['prereg_sha256'])
        self.authorization = dict(path=str(self.auth.relative_to(self.root)), sha256=digest(self.auth),
                                  created_at=value['created_at'])
        return self.authorization

    def _hold(self, request):
        _, _, hold = OfficialDailyAcquisition._origin_paths(self, request)
        if not hold.exists():
            return None
        value = read(hold)
        if value.get('status') != 'blocked':
            raise ValueError('Unrecognized intraday odd origin hold')
        self.mark(hold)
        for mapping in ('evidence_sha256', 'evidence'):
            for name, expected in value.get(mapping, {}).items():
                self.mark(self.path(name), expected)
        return digest(hold)

    def cached(self, path, day, market, seen=None):
        request = request_item(day, market)
        path = self.path(path)
        if path != self.cache / 'receipts' / (request['market'] + '-' + day + '.json'):
            raise ValueError('Intraday odd receipt path identity differs')
        seen = set() if seen is None else set(seen)
        if path in seen:
            raise ValueError('Cyclic intraday odd evidence')
        seen.add(path)
        self.mark(path); self.mark(path.with_suffix('.sha256'))
        if path.with_suffix('.sha256').read_text().strip() != digest(path):
            raise ValueError('Intraday odd receipt hash differs')
        r = read(path)
        if r.get('accepted') is not True:
            raise ReplayDataUnavailable('Prior intraday odd request failed; no automatic retry: ' + market + ' ' + day)
        if (r.get('schema') != 'poc_intraday_odd_receipt_v1' or any(r.get(k) != v for k, v in request.items())
                or r.get('status') != 'verified_odd_market_day' or r.get('http_status') != 200
                or r.get('security_denied') is not False or r.get('response_present') is not True
                or r.get('redirect_statuses') != [] or r.get('automatic_redirects_disabled') is not True
                or r.get('global_hold_unchanged') is not True):
            raise ValueError('Intraday odd accepted receipt identity/transport differs')
        auth = self._authorization(fresh=False)
        if r['authorization'] != auth:
            raise ValueError('Intraday odd receipt belongs to another authorization')
        start, end = _epoch(r['started_at']), _epoch(r['retrieved_at'])
        if not 0 <= start - _epoch(auth['created_at']) <= 86400 or end < start:
            raise ValueError('Intraday odd receipt authorization time differs')
        raw = self.path(r['raw_path'])
        if raw != self.cache / 'raw' / (market + '-' + day + '.bin'):
            raise ValueError('Intraday odd raw path identity differs')
        self.mark(raw, r['raw_sha256'])
        attempt_path = self.cache / 'attempts' / (market + '-' + day + '.json')
        self.mark(attempt_path, r['attempt_sha256']); attempt = read(attempt_path)
        if attempt.get('schema') != 'poc_intraday_odd_attempt_v1' or any(
                r.get(k) != v for k, v in attempt.items() if k != 'schema'):
            raise ValueError('Intraday odd receipt differs from pre-dispatch attempt')
        helper = self.cache / 'source-snapshots' / (r['helper_sha256'] + '.py')
        self.mark(helper, r['helper_sha256'])
        demand_path = self.path(r['demand_path']); self.mark(demand_path, r['demand_sha256'])
        demand = read(demand_path)
        if (demand.get('schema') != 'poc_intraday_odd_demand_v1' or demand.get('date') != day
                or demand.get('market') != market or demand_path != self.cache / 'demands' /
                (market + '-' + day + '-' + demand['stock_id'] + '.json')):
            raise ValueError('Intraday odd demand identity differs')
        validate_plans(day, demand['stock_id'], demand['precommitted_plans'])
        if self._hold(request) != r['global_hold_sha256']:
            raise ValueError('Intraday odd global hold changed')
        proof = r.get('recovery_proof')
        if r['request_kind'] == 'single_normal_probe':
            if proof is not None:
                raise ValueError('Intraday odd first probe must not borrow another proof')
        elif r['request_kind'] == 'necessary_preplanned_day':
            if not isinstance(proof, dict):
                raise ValueError('Intraday odd endpoint proof is missing')
            parent = self.path(proof['path']); self.mark(parent, proof['sha256']); prior = read(parent)
            if (prior.get('request_kind') != 'single_normal_probe' or prior.get('url') != r['url']
                    or prior.get('market') != market or prior.get('authorization') != auth
                    or prior.get('retrieved_at') != proof.get('retrieved_at')
                    or not 0 <= start - _epoch(prior['retrieved_at']) <= 86400):
                raise ValueError('Intraday odd endpoint proof identity/time differs')
            self.cached(parent, prior['date'], market, seen)
        else:
            raise ValueError('Unknown intraday odd request kind')
        rows = parse_odd(wrapper(r, json.loads(raw.read_bytes())), market.lower(), day)
        if len(rows) != r['rows'] or demand['stock_id'] not in rows:
            raise ValueError('Intraday odd demand stock or row count differs')
        return rows

    def _dispatch_permission(self, request, state, hold, proof_path):
        auth = self._authorization(fresh=True)
        if state.get('in_flight'):
            raise ReplayDataUnavailable('Unfinished shared origin request; no automatic recovery')
        self._hold(request)
        blocks = OfficialDailyAcquisition._blocks(self, state, hold)
        probe_key = sha256((auth['sha256'] + '\n' + request['url']).encode()).hexdigest()
        if not proof_path.exists():
            if probe_key in state.get('probes', {}):
                raise ReplayDataUnavailable('Intraday odd single normal probe already consumed')
            if any(_epoch(b['observed_at']) >= _epoch(auth['created_at']) for b in blocks):
                raise ReplayDataUnavailable('Newer origin stop prohibits intraday odd probe')
            return True, None, probe_key
        self.mark(proof_path); proof = read(proof_path)
        parent = self.path(proof['path']); self.mark(parent, proof['sha256']); previous = read(parent)
        self.cached(parent, previous['date'], request['market'])
        if (previous['request_kind'] != 'single_normal_probe' or previous['url'] != request['url']
                or previous['retrieved_at'] != proof['retrieved_at']
                or not 0 <= self.clock() - _epoch(previous['retrieved_at']) <= 86400):
            raise ReplayDataUnavailable('Intraday odd exact endpoint proof is invalid or stale')
        if any(_epoch(b['observed_at']) >= _epoch(previous['retrieved_at']) for b in blocks):
            raise ReplayDataUnavailable('Newer origin stop invalidates intraday odd proof')
        return False, proof, probe_key

    def get(self, day, sid, market, engine=None):
        request = request_item(day, market); market = request['market']
        if not isinstance(sid, str) or len(sid) != 4 or not sid.isdigit():
            raise ValueError('Intraday odd requires a four-digit stock')
        key = market + '-' + day
        receipt = self.cache / 'receipts' / (key + '.json')
        if receipt.exists():
            rows = self.cached(receipt, day, market)
            if sid not in rows:
                raise ReplayDataUnavailable('Intraday odd table lacks stock: ' + key + ' ' + sid)
            return deepcopy(rows[sid])
        if not self.online:
            raise ReplayDataUnavailable('Intraday official odd day missing: ' + key)
        plans = requirement(day, sid, engine)
        state_path, origin_lock, hold = OfficialDailyAcquisition._origin_paths(self, request)
        attempt = self.cache / 'attempts' / (key + '.json')
        proof_path = self.cache / ('proof-' + market + '.json')
        with file_lock(self.cache / '.dispatch.lock'), file_lock(origin_lock):
            if receipt.exists():
                rows = self.cached(receipt, day, market)
                if sid not in rows:
                    raise ReplayDataUnavailable('Intraday odd table lacks stock: ' + key + ' ' + sid)
                return deepcopy(rows[sid])
            if attempt.exists():
                raise ReplayDataUnavailable('Previous intraday odd attempt unfinished; no retry')
            if len(list((self.cache / 'attempts').glob('*.json'))) >= MAXIMUM:
                raise ReplayDataUnavailable('Intraday odd persistent 400-attempt budget exhausted')
            state = read(state_path) if state_path.exists() else dict(probes={})
            is_probe, proof, probe_key = self._dispatch_permission(request, state, hold, proof_path)
            before_hold = digest(hold) if hold.exists() else None
            wait = max(0, 3.1 - (self.clock() - state.get('last_start_epoch', 0)))
            if wait:
                self.sleep(wait + .001)
            # Expiry or an independent new global hold during sleep forbids GET.
            if (digest(hold) if hold.exists() else None) != before_hold:
                raise ReplayDataUnavailable('Intraday odd global hold changed during wait')
            is_probe, proof, probe_key = self._dispatch_permission(request, state, hold, proof_path)
            demand = self.cache / 'demands' / (key + '-' + sid + '.json')
            demand_record = dict(schema='poc_intraday_odd_demand_v1', date=day, market=market,
                                 stock_id=sid, precommitted_plans=plans)
            if demand.exists() and read(demand) != demand_record:
                raise ValueError('Intraday odd previously frozen demand differs')
            if not demand.exists():
                _write(demand, demand_record, exclusive=True)
            self.mark(demand)
            helper_hash = digest(Path(__file__))
            helper = self.cache / 'source-snapshots' / (helper_hash + '.py')
            if not helper.exists():
                helper.parent.mkdir(parents=True, exist_ok=True)
                with helper.open('xb') as stream:
                    stream.write(Path(__file__).read_bytes())
            self.mark(helper, helper_hash)
            record = dict(schema='poc_intraday_odd_attempt_v1', **request,
                authorization=self.authorization, request_kind='single_normal_probe' if is_probe else 'necessary_preplanned_day',
                started_at=self._now(), demand_path=str(demand.relative_to(self.root)), demand_sha256=digest(demand),
                global_hold_sha256=before_hold, helper_sha256=helper_hash, recovery_proof=proof)
            _write(attempt, record, exclusive=True); self.mark(attempt)
            record.update(schema='poc_intraday_odd_receipt_v1', attempt_sha256=digest(attempt), accepted=False,
                          http_status=None, response_present=False, security_denied=False,
                          automatic_redirects_disabled=True, redirect_statuses=[])
            state['last_start_epoch'] = self.clock()
            state['in_flight'] = dict(started_at=record['started_at'], identity=key)
            if is_probe:
                state.setdefault('probes', {})[probe_key] = dict(started_at=record['started_at'], identity=key)
            _write(state_path, state)
            def retain_response(response):
                raw = self.cache / 'raw' / (key + '.bin'); raw.parent.mkdir(parents=True, exist_ok=True)
                with raw.open('xb') as stream:
                    stream.write(response.content)
                self.mark(raw)
                record.update(response_present=True, http_status=response.status_code,
                    raw_path=str(raw.relative_to(self.root)), raw_sha256=digest(raw),
                    security_denied=_security_response(response),
                    redirect_statuses=[r.status_code for r in getattr(response, 'history', [])])
            try:
                response = self.session.get(request['url'], params=request['params'], timeout=(10, 30), allow_redirects=False)
                retain_response(response)
                if record['security_denied']:
                    record['status'] = 'origin_stopped'
                elif response.status_code != 200:
                    record['status'] = 'http_error_no_retry'
                else:
                    rows = parse_odd(wrapper(record, json.loads(response.content)), market.lower(), day)
                    if not rows or sid not in rows:
                        raise ValueError('Official odd table lacks demanded stock')
                    record.update(accepted=True, status='verified_odd_market_day', rows=len(rows))
            except requests.RequestException as exc:
                attached = getattr(exc, 'response', None)
                if attached is not None and not record['response_present']:
                    retain_response(attached)
                record.update(status='request_failed_no_retry', error_type=type(exc).__name__)
            except (ReplayDataUnavailable, ValueError, KeyError, TypeError) as exc:
                record.update(status='schema_error_no_retry', error_type=type(exc).__name__)
            record['retrieved_at'] = self._now()
            record['global_hold_unchanged'] = (digest(hold) if hold.exists() else None) == before_hold
            if not record['global_hold_unchanged']:
                record.update(accepted=False, status='global_hold_changed')
            _write(receipt, record, exclusive=True)
            receipt.with_suffix('.sha256').write_text(digest(receipt) + '\n')
            self.mark(receipt); self.mark(receipt.with_suffix('.sha256'))
            if record['security_denied']:
                state['stopped'] = dict(observed_at=record['retrieved_at'], kind='security_or_rate_limit',
                    receipt_path=str(receipt.relative_to(self.root)), receipt_sha256=digest(receipt))
            state.pop('in_flight', None); _write(state_path, state)
            if is_probe and record['accepted']:
                _write(proof_path, dict(path=str(receipt.relative_to(self.root)), sha256=digest(receipt),
                                      retrieved_at=record['retrieved_at']), exclusive=True)
        rows = self.cached(receipt, day, market)
        return deepcopy(rows[sid])

    def snapshot(self):
        receipts = []
        for path in sorted((self.cache / 'receipts').glob('*.json')):
            self.mark(path); self.mark(path.with_suffix('.sha256'))
            if path.with_suffix('.sha256').read_text().strip() != digest(path):
                raise ValueError('Intraday odd snapshot receipt hash differs')
            value = read(path)
            if value.get('accepted') is True:
                self.cached(path, value['date'], value['market'])
            receipts.append(dict(path=str(path.relative_to(self.root)), sha256=digest(path),
                                 accepted=value.get('accepted') is True, status=value.get('status')))
        return dict(schema='poc_intraday_odd_snapshot_v1', maximum_attempts=MAXIMUM,
            attempted_calls=len(list((self.cache / 'attempts').glob('*.json'))),
            accepted_count=sum(r['accepted'] for r in receipts), failed_count=sum(not r['accepted'] for r in receipts),
            receipts=receipts, source_sha256=dict(self.refs), live_qualified=False)


# Compose existing validated intraday tables only. Ordinary prints and the new
# after-hours TWT53U/odd endpoint are deliberately absent from this inventory.
from skills.poc_broker_odd import SupplementaryOddSources, OddMarketDayMissing
from skills.poc_broker_odd_acquisition import BrokerOddData
from skills.poc_latest_odd import LatestOddData
from skills.poc_executable_data import ExecutableAccountData

INVENTORY_SOURCE = '.cache/poc-broker-account-20261004/odd-source-inventory.json'
INVENTORY = '.cache/poc-intraday-20261005/odd-source-inventory.json'
VOLUME_SCOPE = 'intraday_odd_session'


def intraday_row(value, day, sid, market):
    """Label an already parsed official daily row; never manufacture ticks."""
    market = market.lower()
    if value.get('source_date') != day or value.get('market') != market:
        raise ValueError('Intraday daily row market/date differs')
    if value.get('after_hours') or value.get('volume_scope') not in (None, VOLUME_SCOPE):
        raise ValueError('Non-intraday evidence cannot be relabelled as intraday')
    shares = value.get('odd_shares')
    if type(shares) is not int or shares < 0:
        raise ValueError('Intraday shares must be explicit nonnegative integers')
    prices = [value.get(k) for k in ('odd_low', 'odd_last', 'odd_high')]
    import math
    if shares == 0:
        if any(p is not None for p in prices):
            raise ValueError('Intraday official zero volume has trade prices')
    elif any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0 for p in prices) or prices != sorted(prices):
        raise ValueError('Intraday daily trade prices are invalid')
    result = deepcopy(value)
    # The old generic parser exposes .05 as an informational daily ceiling;
    # this experiment has its own registered .01 execution model.
    result.pop('daily_participation_ceiling', None)
    result.update(stock_id=sid, volume_scope=VOLUME_SCOPE, volume_unit='shares',
        price_unit='TWD_per_share', after_hours=False,
        evidence_status='official_intraday_daily_table',
        execution_evidence='intraday_odd_daily_hl2_proxy',
        intraday_tick_verified=False, actual_fill_verified=False, live_qualified=False)
    return result


class IntradayOddData:
    """Frozen inventory plus exact receipt reuse, then bounded necessary GETs.

    A missing whole market-day permits another source; a missing stock, bad
    provenance, failed receipt or conflicting row does not. Every available
    source for a queried market-day must agree. Reusing a table does not renew
    any old authorization/probe. New requests use only this experiment's guard.
    """
    def __init__(self, root, *, online=False, source_refs=None, acquisition=None):
        self.root = Path(root).resolve()
        self.refs, self.queries, self.selected_sources = {}, [], {}
        if source_refs is None:
            target, original = self.root/INVENTORY, self.root/INVENTORY_SOURCE
            with file_lock(target.with_suffix('.lock')):
                if not target.exists():
                    values = read(original)
                    _write(target, values, exclusive=True)
                    _write(target.with_suffix('.origin.json'), dict(path=INVENTORY_SOURCE,
                        sha256=digest(original), inventory_sha256=digest(target)), exclusive=True)
                origin = read(target.with_suffix('.origin.json'))
                self._mark(original, origin['sha256'])
                self._mark(target, origin['inventory_sha256'])
                self._mark(target.with_suffix('.origin.json'))
                source_refs = read(target)
        self.supplement = SupplementaryOddSources(self.root, source_refs=source_refs)
        self.legacy = {}
        for name, folder, cls in (
            ('latest', '.cache/poc-latest-20261003/odd-v1', LatestOddData),
            ('broker', '.cache/poc-broker-account-20261004/odd-acquisition-v1', BrokerOddData)):
            receipt_dir = self.root/folder/'receipts'
            if receipt_dir.is_dir() and any(receipt_dir.glob('*.json')):
                # Old clients are read-only. Never spend their expired auth or
                # make missing dates retry through an older experiment.
                self.legacy[name] = cls(self.root, online=False)
        self.acquisition = acquisition or IntradayOddAcquisition(self.root, online=online)
        self.initial_attempts = len(list((self.acquisition.cache/'attempts').glob('*.json')))
        self._merge()

    def _mark(self, path, expected=None):
        path = Path(path).resolve(); name = str(path.relative_to(self.root)); actual = digest(path)
        if expected is not None and actual != expected or name in self.refs and self.refs[name] != actual:
            raise ValueError('Intraday source hash changed: '+name)
        self.refs[name] = actual

    def _merge(self):
        for source in (self.supplement, *self.legacy.values(), self.acquisition):
            for name, expected in source.refs.items():
                if name in self.refs and self.refs[name] != expected:
                    raise ValueError('Intraday source identity conflict: '+name)
                self.refs[name] = expected

    @property
    def calls(self):
        return len(list((self.acquisition.cache/'attempts').glob('*.json'))) - self.initial_attempts

    def get(self, day, sid, market, engine=None):
        request = request_item(day, market); market = request['market']
        if not isinstance(sid, str) or len(sid) != 4 or not sid.isdigit():
            raise ValueError('Intraday request requires a four-digit stock')
        values, selected = [], []
        key = market+'-'+day
        try:
            try:
                values.append(self.supplement.get(day, sid, market))
                selected.append('frozen_inventory')
            except OddMarketDayMissing:
                pass
            except ReplayDataUnavailable as exc:
                if any(t in str(exc) for t in ('changed', 'hash', 'outside sealed', 'escapes', 'unsafe', 'ancestor', 'receipt chain')):
                    raise ValueError('Intraday frozen source integrity failure: '+str(exc)) from exc
                raise
            for name, source in self.legacy.items():
                path = source.cache/'receipts'/(key+'.json')
                if path.exists():
                    rows = source.cached(path, day, market)
                    if sid not in rows:
                        raise ReplayDataUnavailable('Existing intraday table lacks stock: '+key+' '+sid)
                    values.append(rows[sid]); selected.append(name)
            path = self.acquisition.cache/'receipts'/(key+'.json')
            if path.exists() or not values:
                values.append(self.acquisition.get(day, sid, market, engine))
                selected.append('intraday_experiment')
            if any(value != values[0] for value in values[1:]):
                raise ReplayDataUnavailable('Conflicting official intraday rows: '+key+' '+sid)
            row = intraday_row(values[0], day, sid, market)
            self.selected_sources[key+':'+sid] = selected
            self.queries.append(dict(date=day, stock_id=sid, market=market, sources=selected))
            return row
        finally:
            self._merge()

    def snapshot(self):
        acquired = self.acquisition.snapshot(); self._merge()
        return dict(schema='poc_intraday_daily_sources_v1', queries=deepcopy(self.queries),
            selected_sources=deepcopy(self.selected_sources),
            inventory_market_days=len(self.supplement.available_market_days),
            selected_inventory_sources=deepcopy(self.supplement.selected_sources),
            new_acquisition=acquired, source_sha256=dict(self.refs),
            volume_scope=VOLUME_SCOPE, participation=.01,
            price_assumption='intraday_daily_high_low_midpoint_proxy',
            intraday_tick_verified=False, actual_fill_verified=False, live_qualified=False)


class IntradayAccountData(ExecutableAccountData):
    """Keep the existing ordinary/POC/financial data; substitute only odd scope."""
    def __init__(self, root=None, *, online=False):
        if root is None:
            root = Path(__file__).resolve().parents[1]
        super().__init__(root, online=online)
        self.intraday_odds = IntradayOddData(root, online=online)
        self.prereg_path = Path(root)/PREREG
        self.prereg_sha256 = digest(self.prereg_path)
        self._bind(self.prereg_path, self.prereg_sha256)
        self._bind(Path(__file__))
        for name in ('poc_broker_odd.py', 'poc_broker_odd_acquisition.py',
                     'poc_latest_odd.py', 'replay_market_feeds.py'):
            self._bind(Path(root)/'skills'/name)
        self._merge_live_refs()

    def get_odd(self, day, sid, market, engine=None):
        try:
            return self.intraday_odds.get(day, sid, market, engine)
        finally:
            self._merge_live_refs()

    def _merge_live_refs(self):
        super()._merge_live_refs()
        if hasattr(self, 'intraday_odds'):
            self.intraday_odds._merge()
            for name, expected in self.intraday_odds.refs.items():
                if name in self.refs and self.refs[name] != expected:
                    raise ValueError('Intraday account source conflict: '+name)
                self.refs[name] = expected

    @property
    def network_calls(self):
        return self.finmind_calls + self.intraday_odds.calls

    def profile_snapshot(self, output):
        result = super().profile_snapshot(output)
        if result.pop('after_hours') is not None:
            raise ValueError('Intraday account unexpectedly used after-hours data')
        result['schema'] = 'poc_intraday_account_data_v1'
        result['intraday_odd'] = self.intraday_odds.snapshot()
        self._merge_live_refs()
        return result
