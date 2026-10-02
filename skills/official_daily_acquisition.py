"""Resumable primary daily tables with explicit, endpoint-scoped recovery.

This module never removes an origin hold. A caller must explicitly authorize a
single ordinary probe; even its successful result only covers that endpoint.
"""
from datetime import date, datetime, timezone
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse
import json
import math
import os
import tempfile
import time

import requests

from app.file_lock import file_lock
from skills.market_input_validation import parse_market_day


URLS = {
    'TWSE': 'https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX',
    'TPEX': 'https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php',
}
USER_REQUEST = '但仍缺 3,745張官方市場日表 你可以補？'


class AcquisitionBlocked(ValueError):
    pass


def require(value, message):
    if not value:
        raise AcquisitionBlocked(message)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)+'\n'


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as stream:
            stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
        return
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
        name = stream.name
    os.replace(name, path)


def _bound(root, path):
    path = Path(path)
    path = (path if path.is_absolute() else root/path).resolve()
    require(path.is_relative_to(root), 'Source escapes repository')
    return path


def _epoch(stamp):
    result = datetime.fromisoformat(stamp)
    require(result.tzinfo is not None, 'Timestamp must contain a timezone')
    return result.timestamp()


def request_item(market, day):
    market = market.upper()
    require(market in URLS, 'Unsupported market')
    d = date.fromisoformat(day)
    require(day == d.isoformat(), 'Date must be canonical YYYY-MM-DD')
    params = (dict(date=day.replace('-', ''), type='ALLBUT0999', response='json')
              if market == 'TWSE' else
              dict(l='zh-tw', d=f'{d.year-1911}/{d.month:02d}/{d.day:02d}', o='json', se='EW'))
    item = dict(market=market, date=day, url=URLS[market], params=params)
    item['identity'] = sha256(encoded(item).encode()).hexdigest()
    return item


def create_plan(root, cache, items, source_sha256):
    root = Path(root).resolve()
    cache = _bound(root, cache)
    entries = [request_item(i['market'], i['date']) for i in items]
    require(entries and len({i['identity'] for i in entries}) == len(entries), 'Empty/duplicate plan')
    for path, expected in source_sha256.items():
        require(digest(_bound(root, path)) == expected, 'Plan source changed')
    value = dict(schema='official_daily_plan_v1', entries=entries,
                 source_sha256=source_sha256, max_requests=len(entries),
                 minimum_start_interval_seconds=3.1, automatic_retries=0)
    path = cache/'plan.json'
    with file_lock(cache/'acquisition.lock'):
        if path.exists():
            require(path.with_suffix('.sha256').read_text().strip() == digest(path), 'Plan hash changed')
            require(read(path) == value, 'Immutable acquisition plan differs')
        else:
            _write(path, value, exclusive=True)
            path.with_suffix('.sha256').write_text(digest(path)+'\n')
    return path


class OfficialDailyAcquisition:
    def __init__(self, root, cache, *, authorization_path=None, recovery_proofs=None,
                 session=None, min_interval=3.1, clock=time.time, sleep=time.sleep):
        self.root = Path(root).resolve()
        self.cache = _bound(self.root, cache)
        self.plan_path = self.cache/'plan.json'
        self.plan_hash = digest(self.plan_path)
        require(self.plan_hash == self.plan_path.with_suffix('.sha256').read_text().strip(), 'Plan hash changed')
        self.plan = read(self.plan_path)
        require(self.plan['schema'] == 'official_daily_plan_v1', 'Unsupported acquisition plan')
        self.items = {r['identity']: r for r in self.plan['entries']}
        require(len(self.items) == len(self.plan['entries']) == self.plan['max_requests'], 'Plan request budget differs')
        for item in self.items.values():
            require(item == request_item(item['market'], item['date']), 'Plan endpoint/query differs')
        for path, expected in self.plan['source_sha256'].items():
            require(digest(_bound(self.root, path)) == expected, 'Plan source changed')
        require(math.isfinite(min_interval) and min_interval >= 3.1, 'Origin interval must be at least 3.1 seconds')
        self.interval, self.clock, self.sleep = min_interval, clock, sleep
        self.authorization = _bound(self.root, authorization_path) if authorization_path else None
        self.proofs = {m.upper(): _bound(self.root, p) for m,p in (recovery_proofs or {}).items()}
        self.session = session or requests.Session()

    def _auth(self):
        require(self.authorization is not None, 'Current user-request authorization record required')
        record = read(self.authorization)
        require(record.get('schema') == 'official_daily_authorization_v1'
                and record.get('scope') == 'missing_official_daily_tables'
                and record.get('security_bypass_authorized') is False
                and record.get('user_request') == USER_REQUEST
                and record.get('plan_sha256') == self.plan_hash,
                'Authorization must describe only this user request and immutable plan')
        require(0 <= self.clock()-_epoch(record['created_at']) <= 86400, 'Current authorization is stale')
        return dict(path=str(self.authorization.relative_to(self.root)), sha256=digest(self.authorization),
                    created_at=record['created_at'])

    def _validate_item(self, item):
        require(digest(self.plan_path) == self.plan_hash, 'Immutable plan changed during acquisition')
        require(item == request_item(item['market'], item['date'])
                and self.items.get(item['identity']) == item, 'Request is outside immutable plan')

    def _origin_paths(self, item):
        host = urlparse(item['url']).hostname
        base = self.root/'.cache/official-daily-origin-dispatch'
        return (base/(host+'.json'), base/(host+'.lock'),
                self.root/'.cache/official-origin-holds'/(host+'.json'))

    def _blocks(self, state, hold):
        blocks = []
        if hold.exists():
            data = read(hold)
            require(data.get('status') == 'blocked', 'Unrecognized existing origin hold')
            for path, expected in data.get('evidence_sha256', {}).items():
                require(digest(_bound(self.root, path)) == expected, 'Origin hold evidence changed')
            blocks.append(dict(observed_at=data['observed_at'], kind='global_hold'))
        if state.get('stopped'):
            stopped = state['stopped']
            require(digest(_bound(self.root, stopped['receipt_path'])) == stopped['receipt_sha256'],
                    'Shared origin stop evidence changed')
            blocks.append(state['stopped'])
        if state.get('in_flight'):
            blocks.append(dict(observed_at=state['in_flight']['started_at'], kind='interrupted_origin_dispatch'))
        return blocks

    def _receipt(self, path):
        require(path.with_suffix('.sha256').read_text().strip() == digest(path), 'Receipt hash changed')
        r = read(path)
        require(r.get('schema') == 'official_daily_receipt_v1' and r.get('accepted') is True,
                'A successful complete official receipt is required')
        require(r.get('plan_sha256') == self.plan_hash, 'Receipt covers another acquisition plan')
        item = request_item(r['market'], r['date'])
        require(all(r.get(k) == v for k,v in item.items()) and r['http_status'] == 200
                and r['automatic_redirects_disabled'] is True and r.get('redirect_statuses') == []
                and r.get('security_denied') is False and r.get('status') == 'verified_market_day',
                'Receipt endpoint, date or transport differs')
        raw = _bound(self.root, r['raw_path'])
        require(digest(raw) == r['raw_sha256'], 'Official payload changed')
        rows = parse_market_day(read(raw), r['market'], r['date'])
        require(rows and r['rows'] == len(rows)
                and r['volume_scope'] == next(iter(rows.values()))['volume_scope'], 'Receipt table scope differs')
        return dict(r, receipt_path=str(path.relative_to(self.root)), receipt_sha256=digest(path))

    def _recover(self, item, blocks):
        proof_path = self.proofs.get(item['market'])
        require(proof_path is not None, 'Origin hold active; no endpoint recovery proof')
        proof = self._receipt(proof_path)
        auth = self._auth()
        require(proof.get('request_kind') == 'single_normal_probe'
                and proof.get('authorization') == auth and proof['url'] == item['url']
                and proof.get('plan_sha256') == self.plan_hash, 'Recovery proof covers another endpoint/request')
        observed = _epoch(proof['retrieved_at'])
        require(0 <= self.clock()-observed <= 86400, 'Endpoint recovery proof is stale')
        require(all(observed > _epoch(b['observed_at']) for b in blocks), 'Newer origin stop invalidates recovery proof')

    def fetch(self, item):
        return self._dispatch(item, probe=False)

    def probe(self, item, *, allow_probe=False):
        require(allow_probe is True, 'Single normal probe is disabled by default')
        return self._dispatch(item, probe=True)

    def _dispatch(self, item, *, probe):
        self._validate_item(item)
        key = item['identity']
        receipt_path = self.cache/'receipts'/(key+'.json')
        attempt_path = self.cache/'attempts'/(key+'.json')
        state_path, origin_lock, hold_path = self._origin_paths(item)
        with file_lock(self.cache/('acquisition-'+item['market']+'.lock')):
            if receipt_path.exists():
                value = read(receipt_path)
                if value.get('accepted'):
                    return self._receipt(receipt_path)
                require(receipt_path.with_suffix('.sha256').read_text().strip() == digest(receipt_path), 'Receipt hash changed')
                return dict(value, resumed_without_retry=True)
            if attempt_path.exists():
                return dict(accepted=False, identity=key, status='interrupted_no_retry')
            with file_lock(origin_lock):
                state = read(state_path) if state_path.exists() else dict(probes={})
                blocks = self._blocks(state, hold_path)
                auth = self._auth() if probe else None
                if probe:
                    # One ordinary GET per current request/endpoint, shared across caches.
                    probe_key = sha256((auth['sha256']+'\n'+item['url']).encode()).hexdigest()
                    require(probe_key not in state.get('probes', {}), 'Single normal probe already consumed')
                    require(all(_epoch(auth['created_at']) > _epoch(b['observed_at']) for b in blocks),
                            'Origin stopped after this authorization; no new probe')
                elif blocks:
                    self._recover(item, blocks)
                elapsed = self.clock()-state.get('last_start_epoch', float('-inf'))
                wait = max(0, self.interval-elapsed)
                if wait:
                    self.sleep(wait+0.001)  # Avoid floating-point rounding below the minimum.
                started = self.clock()
                stamp = datetime.fromtimestamp(started, timezone.utc).isoformat()
                attempt = dict(schema='official_daily_attempt_v1', **item, started_at=stamp,
                    request_kind='single_normal_probe' if probe else 'planned_missing_day',
                    plan_sha256=self.plan_hash, authorization=auth)
                _write(attempt_path, attempt, exclusive=True)
                state['last_start_epoch'] = started
                state['in_flight'] = dict(started_at=stamp, identity=key)
                if probe:
                    state.setdefault('probes', {})[probe_key] = dict(started_at=stamp, identity=key)
                _write(state_path, state)
                result = dict(attempt,
                    accepted=False, http_status=None, automatic_redirects_disabled=True, redirect_statuses=[])
                result['schema'] = 'official_daily_receipt_v1'
                try:
                    response = self.session.get(item['url'], params=item['params'],
                        timeout=(10,30), allow_redirects=False)
                    raw_path = self.cache/'raw'/(key+'.bin')
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with raw_path.open('xb') as stream:
                        stream.write(response.content); stream.flush(); os.fsync(stream.fileno())
                    body = response.content.lower()
                    security = (response.status_code in (401,403,428,429)
                        or 300 <= response.status_code < 400 or bool(getattr(response, 'history', []))
                        or any(marker in body for marker in (b'for security reasons', b'captcha',
                            b'challenge-platform', b'cf-chl-', b'access denied', '因為安全性考量'.encode())))
                    result.update(http_status=response.status_code,
                        raw_path=str(raw_path.relative_to(self.root)), raw_sha256=digest(raw_path),
                        bytes=len(response.content), security_denied=security,
                        redirect_statuses=[r.status_code for r in getattr(response,'history',[])])
                    if security:
                        result['status'] = 'origin_stopped'
                    elif response.status_code != 200:
                        result['status'] = 'http_error_no_retry'
                    else:
                        try:
                            rows = parse_market_day(json.loads(response.content), item['market'], item['date'])
                            require(rows, 'Empty primary market table')
                            result.update(accepted=True, status='verified_market_day', rows=len(rows),
                                volume_scope=next(iter(rows.values()))['volume_scope'])
                        except (ValueError,KeyError,TypeError,UnicodeDecodeError) as exc:
                            result.update(status='schema_error_no_retry', error_type=type(exc).__name__, error=str(exc))
                except requests.RequestException as exc:
                    result.update(status='transport_error_no_retry', error_type=type(exc).__name__)
                result['retrieved_at'] = datetime.fromtimestamp(self.clock(),timezone.utc).isoformat()
                _write(receipt_path, result, exclusive=True)
                receipt_path.with_suffix('.sha256').write_text(digest(receipt_path)+'\n')
                if result.get('security_denied'):
                    state['stopped'] = dict(observed_at=result['retrieved_at'], kind='security_or_rate_limit',
                        receipt_path=str(receipt_path.relative_to(self.root)), receipt_sha256=digest(receipt_path))
                state.pop('in_flight', None)
                _write(state_path, state)
                if result['accepted']:
                    if probe:
                        self.proofs[item['market']] = receipt_path
                    return self._receipt(receipt_path)
                return dict(result, receipt_path=str(receipt_path.relative_to(self.root)), receipt_sha256=digest(receipt_path))

    def export_manifest(self, path):
        path = _bound(self.root, path)
        with ExitStack() as locks:
            # Stable order prevents deadlocks; readers cannot observe half a receipt.
            for market in sorted(URLS):
                locks.enter_context(file_lock(self.cache/('acquisition-'+market+'.lock')))
            accepted = []
            refs = {str(self.plan_path.relative_to(self.root)): self.plan_hash}
            for receipt_path in sorted((self.cache/'receipts').glob('*.json')):
                if read(receipt_path).get('accepted'):
                    row = self._receipt(receipt_path)
                    accepted.append(row)
                    refs[row['raw_path']] = row['raw_sha256']
                    refs[row['receipt_path']] = row['receipt_sha256']
            value = dict(schema='official_market_supplement_v1', entries=accepted,
                source_sha256=refs, accepted_market_days=len(accepted),
                live_qualified=False, automatic_retries=0)
            _write(path, value, exclusive=True)
            path.with_suffix('.sha256').write_text(digest(path)+'\n')
            return value
