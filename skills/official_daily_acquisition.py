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
TRANSPORT_RECOVERY_KIND = 'reviewed_transport_recovery'


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


def validate_transport_failure(receipt, item, plan_hash, *, receipt_sha256=None, legacy_failure_sha256=None):
    require(receipt.get('schema') == 'official_daily_receipt_v1'
            and all(receipt.get(k) == v for k,v in item.items())
            and receipt.get('plan_sha256') == plan_hash
            and receipt.get('accepted') is False
            and receipt.get('request_kind') == 'planned_missing_day'
            and receipt.get('status') == 'transport_error_no_retry'
            and receipt.get('error_type') in ('ConnectionError','Timeout','ConnectTimeout','ReadTimeout')
            and receipt.get('http_status') is None and not receipt.get('security_denied')
            and ('exception_response_present' not in receipt or receipt['exception_response_present'] is False)
            and receipt.get('automatic_redirects_disabled') is True and receipt.get('redirect_statuses') == []
            and not any(k in receipt for k in ('raw_path','raw_sha256','bytes')),
            'Only a completed response-free or explicitly acknowledged legacy connection failure can receive recovery')
    if 'exception_response_present' not in receipt:
        require(isinstance(legacy_failure_sha256,str) and bool(receipt_sha256)
                and legacy_failure_sha256 == receipt_sha256,
                'Legacy exception response presence is unknown; explicit reviewed failure SHA is required')
    else:
        require(legacy_failure_sha256 is None, 'Legacy acknowledgment does not apply to a recorded exception response')


def _security_response(response):
    body = response.content.lower()
    return (response.status_code in (401,403,428,429) or 300 <= response.status_code < 400
            or bool(getattr(response,'history',[]))
            or any(marker in body for marker in (b'for security reasons',b'captcha',
                b'challenge-platform',b'cf-chl-',b'access denied','因為安全性考量'.encode())))


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

    def _transport_base(self, item, legacy_failure_sha256=None):
        receipt = self.cache/'receipts'/(item['identity']+'.json')
        attempt = self.cache/'attempts'/(item['identity']+'.json')
        require(receipt.is_file() and attempt.is_file(), 'Interrupted/missing attempt cannot be recovered')
        require(receipt.with_suffix('.sha256').read_text().strip() == digest(receipt), 'Base failure receipt changed')
        value, started = read(receipt), read(attempt)
        validate_transport_failure(value,item,self.plan_hash,receipt_sha256=digest(receipt),
                                   legacy_failure_sha256=legacy_failure_sha256)
        require(all(started.get(k) == value.get(k) for k in
                    (*item,'plan_sha256','request_kind','authorization','started_at'))
                and started.get('schema') == 'official_daily_attempt_v1', 'Base failure attempt differs')
        return dict(base_receipt_path=str(receipt.relative_to(self.root)),base_receipt_sha256=digest(receipt),
                    base_attempt_path=str(attempt.relative_to(self.root)),base_attempt_sha256=digest(attempt))

    def _linked_recovery(self, item):
        folder = self.cache/'recoveries'/item['identity']
        value = self._receipt(folder/'receipt.json')
        links = self._transport_base(item,value.get('legacy_failure_sha256'))
        attempt = read(folder/'attempt.json')
        require(all(value.get(k) == attempt.get(k) == v for k,v in links.items())
                and attempt.get('schema') == 'official_daily_attempt_v1'
                and all(value.get(k) == attempt.get(k) == v for k,v in item.items())
                and value.get('request_kind') == attempt.get('request_kind') == TRANSPORT_RECOVERY_KIND
                and all(attempt.get(k) == value.get(k) for k in
                    (*item,'plan_sha256','started_at','authorization','recovery_reason',
                     'recovery_proof_path','recovery_proof_sha256','legacy_failure_sha256'))
                and isinstance(value.get('recovery_reason'),str) and value['recovery_reason'].strip(),
                'Recovery is not linked to the preserved original failure')
        proof_path = _bound(self.root,value['recovery_proof_path'])
        require(digest(proof_path) == value['recovery_proof_sha256'], 'Recovery endpoint proof changed')
        proof = self._receipt(proof_path)
        require(proof['request_kind'] == 'single_normal_probe' and proof['url'] == item['url']
                and proof['authorization'] == value['authorization']
                and 0 <= _epoch(value['started_at'])-_epoch(proof['retrieved_at']) <= 86400
                and _epoch(value['started_at']) >= _epoch(read(self.root/links['base_receipt_path'])['retrieved_at']),
                'Recovery endpoint proof/timing differs')
        auth = value['authorization']
        require(digest(_bound(self.root,auth['path'])) == auth['sha256'], 'Recovery authorization changed')
        require(value['raw_path'] == str((folder/'raw.bin').relative_to(self.root)), 'Recovery raw path differs')
        return value

    def recover_transport(self, item, reason, *, legacy_failure_sha256=None):
        require(isinstance(reason,str) and bool(reason.strip()) and len(reason) <= 2000,
                'Explicit agent review reason required for the one transport recovery')
        return self._dispatch(item,probe=False,recovery_reason=reason,legacy_failure_sha256=legacy_failure_sha256)

    def inspect_transport_recovery(self, item):
        """Read and verify linked recovery evidence without dispatching a request."""
        self._validate_item(item)
        return self._linked_recovery(item)

    def _dispatch(self, item, *, probe, recovery_reason=None, legacy_failure_sha256=None):
        self._validate_item(item)
        key = item['identity']
        receipt_path = self.cache/'receipts'/(key+'.json')
        attempt_path = self.cache/'attempts'/(key+'.json')
        recovery = recovery_reason is not None
        if recovery:
            folder = self.cache/'recoveries'/key
            receipt_path, attempt_path = folder/'receipt.json',folder/'attempt.json'
        state_path, origin_lock, hold_path = self._origin_paths(item)
        with file_lock(self.cache/('acquisition-'+item['market']+'.lock')):
            links = self._transport_base(item,legacy_failure_sha256) if recovery else {}
            if receipt_path.exists():
                value = read(receipt_path)
                if value.get('accepted'):
                    return self._linked_recovery(item) if recovery else self._receipt(receipt_path)
                require(receipt_path.with_suffix('.sha256').read_text().strip() == digest(receipt_path), 'Receipt hash changed')
                return dict(value, resumed_without_retry=True)
            if attempt_path.exists():
                return dict(accepted=False, identity=key, status='interrupted_no_retry')
            with file_lock(origin_lock):
                state = read(state_path) if state_path.exists() else dict(probes={})
                blocks = self._blocks(state, hold_path)
                auth = self._auth() if probe or recovery else None
                if probe:
                    # One ordinary GET per current request/endpoint, shared across caches.
                    probe_key = sha256((auth['sha256']+'\n'+item['url']).encode()).hexdigest()
                    require(probe_key not in state.get('probes', {}), 'Single normal probe already consumed')
                    require(all(_epoch(auth['created_at']) > _epoch(b['observed_at']) for b in blocks),
                            'Origin stopped after this authorization; no new probe')
                elif blocks or recovery:
                    self._recover(item, blocks)
                if recovery:
                    recovery_key = links['base_receipt_sha256']
                    require(recovery_key not in state.get('transport_recoveries',{}), 'One transport recovery already consumed')
                elapsed = self.clock()-state.get('last_start_epoch', float('-inf'))
                wait = max(0, self.interval-elapsed)
                if wait:
                    self.sleep(wait+0.001)  # Avoid floating-point rounding below the minimum.
                # Waiting must not extend authorization/proof expiry. A global
                # hold may also have appeared while this origin's lock was held.
                current_blocks = self._blocks(state,hold_path)
                if probe or recovery:
                    require(self._auth() == auth, 'Authorization changed during origin wait')
                if probe:
                    require(all(_epoch(auth['created_at']) > _epoch(b['observed_at']) for b in current_blocks),
                            'Origin stopped during probe wait')
                elif current_blocks or recovery:
                    self._recover(item,current_blocks)
                started = self.clock()
                stamp = datetime.fromtimestamp(started, timezone.utc).isoformat()
                attempt = dict(schema='official_daily_attempt_v1', **item, started_at=stamp,
                    request_kind=('single_normal_probe' if probe else
                                  TRANSPORT_RECOVERY_KIND if recovery else 'planned_missing_day'),
                    plan_sha256=self.plan_hash, authorization=auth)
                if recovery:
                    proof = self.proofs[item['market']]
                    attempt.update(links,recovery_reason=recovery_reason,
                        recovery_proof_path=str(proof.relative_to(self.root)),recovery_proof_sha256=digest(proof),
                        legacy_failure_sha256=legacy_failure_sha256)
                _write(attempt_path, attempt, exclusive=True)
                state['last_start_epoch'] = started
                state['in_flight'] = dict(started_at=stamp, identity=key)
                if probe:
                    state.setdefault('probes', {})[probe_key] = dict(started_at=stamp, identity=key)
                if recovery:
                    state.setdefault('transport_recoveries',{})[recovery_key] = dict(started_at=stamp,identity=key)
                _write(state_path, state)
                result = dict(attempt,
                    accepted=False, http_status=None, automatic_redirects_disabled=True, redirect_statuses=[])
                result['schema'] = 'official_daily_receipt_v1'
                try:
                    response = self.session.get(item['url'], params=item['params'],
                        timeout=(10,30), allow_redirects=False)
                    raw_path = folder/'raw.bin' if recovery else self.cache/'raw'/(key+'.bin')
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with raw_path.open('xb') as stream:
                        stream.write(response.content); stream.flush(); os.fsync(stream.fileno())
                    security = _security_response(response)
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
                    attached = getattr(exc,'response',None)
                    result.update(error_type=type(exc).__name__,exception_response_present=attached is not None)
                    if attached is None:
                        result['status'] = 'transport_error_no_retry'
                    else:
                        # requests.Response(4xx) is falsey; never discard its denial evidence.
                        raw_path = folder/'raw.bin' if recovery else self.cache/'raw'/(key+'.bin')
                        raw_path.parent.mkdir(parents=True,exist_ok=True)
                        with raw_path.open('xb') as stream:
                            stream.write(attached.content); stream.flush(); os.fsync(stream.fileno())
                        security = _security_response(attached)
                        result.update(status='origin_stopped' if security else 'exception_response_no_retry',
                            http_status=attached.status_code,security_denied=security,
                            raw_path=str(raw_path.relative_to(self.root)),raw_sha256=digest(raw_path),
                            bytes=len(attached.content),
                            redirect_statuses=[r.status_code for r in getattr(attached,'history',[])])
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
                    return self._linked_recovery(item) if recovery else self._receipt(receipt_path)
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
                receipt = read(receipt_path)
                if receipt.get('accepted'):
                    row = self._receipt(receipt_path)
                else:
                    recovery_path = self.cache/'recoveries'/receipt_path.stem/'receipt.json'
                    if not recovery_path.exists() or not read(recovery_path).get('accepted'):
                        continue
                    item = request_item(receipt['market'],receipt['date'])
                    row = self._linked_recovery(item)
                    for prefix in ('base_receipt','base_attempt','recovery_proof'):
                        refs[row[prefix+'_path']] = row[prefix+'_sha256']
                    attempt = recovery_path.with_name('attempt.json')
                    refs[str(attempt.relative_to(self.root))] = digest(attempt)
                    refs[row['authorization']['path']] = row['authorization']['sha256']
                accepted.append(row)
                refs[row['raw_path']] = row['raw_sha256']
                refs[row['receipt_path']] = row['receipt_sha256']
            value = dict(schema='official_market_supplement_v1', entries=accepted,
                source_sha256=refs, accepted_market_days=len(accepted),
                live_qualified=False, automatic_retries=0)
            _write(path, value, exclusive=True)
            path.with_suffix('.sha256').write_text(digest(path)+'\n')
            return value
