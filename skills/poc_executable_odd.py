"""Bounded, demand-bound after-hours odd-lot single-auction research evidence.

The 14:30 table supports an explicit conservative fill simulation, never proof
that an actual order won auction allocation. Intraday odd tables are rejected.
Original research adapters and global source holds are not modified.
"""
from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import time
import math
import re
from decimal import Decimal, ROUND_FLOOR

import requests

from app.file_lock import file_lock
from skills.official_daily_acquisition import (
    OfficialDailyAcquisition, _security_response, _write, digest, read,
)
from skills.poc_latest_odd import _epoch, wrapper
from skills.replay_market_feeds import ReplayDataUnavailable, _number

START, END = '2024-01-02', '2026-10-02'
BASE = '.cache/poc-executable-20261004/odd-v1'
PREREG = 'docs/prereg_poc_executable_20261004.md'
MAXIMUM = 400
URLS = {
    'twse': 'https://www.twse.com.tw/rwd/zh/afterTrading/TWT53U',
    'tpex': 'https://www.tpex.org.tw/www/zh-tw/afterTrading/odd',
}
SCOPE = 'after_hours_odd_single_auction'
FIELDS = {
    'twse': ['證券代號', '證券名稱', '成交股數', '成交筆數', '成交金額', '成交價',
             '最後揭示買價', '最後揭示買量', '最後揭示賣價', '最後揭示賣量'],
    'tpex': ['代號', '名稱', '成交股數', '成交筆數', '成交金額', '成交價格(元)',
             '未成交買價', '未成交買量', '未成交賣價', '未成交賣量'],
}


def _twse_summary_rows(table):
    """Only the unique final all-securities total may have an empty code.

    Named total columns include all securities, not just our four-digit stock
    subset. Foreign-currency ETF amounts are excluded only where the official
    table explicitly declares that convention.
    """
    data = table['data']
    blank = [i for i, row in enumerate(data)
             if isinstance(row, list) and row and not str(row[0]).strip()]
    if not blank:
        return data
    if blank != [len(data)-1]:
        raise ReplayDataUnavailable('After-hours total must be the unique last blank-code row')
    total = data[-1]
    if len(total) != len(FIELDS['twse']) or str(total[1]).strip() != '合計' or len(data) < 2:
        raise ReplayDataUnavailable('Unrecognized after-hours blank-code row')
    if any(_number(total[i], missing=True) is not None for i in (5,6,8)):
        raise ReplayDataUnavailable('After-hours total cannot have an auction or quote price')
    notes = table.get('notes', [])
    foreign_amounts_excluded = ('不加計外幣交易證券交易金額。' in notes
        and any('ETF證券代號第六碼為K、M、S、C者' in str(n) for n in notes))
    for index in (2,3,4,7,9):
        observed = Decimal(0)
        for row in data[:-1]:
            if not isinstance(row, list) or len(row) != len(FIELDS['twse']):
                raise ReplayDataUnavailable('After-hours row width differs')
            if index == 4 and foreign_amounts_excluded and re.fullmatch(r'\d{5}[KMSC]', str(row[0]).strip()):
                continue
            observed += Decimal(str(_number(row[index], quantity=index != 4)))
        expected = Decimal(str(_number(total[index], quantity=index != 4)))
        if observed != expected:
            raise ReplayDataUnavailable('After-hours total does not equal all source rows')
    return data[:-1]


def parse_after_hours(record, market, day):
    """Validate a dated full-market auction; an absent security is not zero.

    Units come from the exact official column names (shares, NTD per share).
    The TWSE unit hint, when supplied, must also agree. All rows are structurally
    checked; only four-digit stock/benchmark rows are retained.
    """
    try:
        item = request_item(day, market)
        market = item['market'].lower()
        if not isinstance(record, dict):
            raise ReplayDataUnavailable('After-hours record is not an object')
        expected = dict(schema=1, provider=market, day=day, url=item['url'],
                        params=item['params'], http_status=200)
        if any(record.get(k) != v for k, v in expected.items()) or not record.get('retrieved_at'):
            raise ReplayDataUnavailable('After-hours source provenance differs')
        _epoch(record['retrieved_at'])
        data = record['payload']
        if (not isinstance(data, dict) or data.get('date') != day.replace('-', '')
                or str(data.get('stat')).lower() != 'ok'):
            raise ReplayDataUnavailable('After-hours date or status differs')
        if market == 'twse':
            table = data
            title = f'{int(day[:4])-1911}年{day[5:7]}月{day[8:]}日 盤後零股交易行情單'
            if data.get('type') != 'ALL' or ('hints' in data and data['hints'] != '單位：元、股'):
                raise ReplayDataUnavailable('After-hours TWSE scope or units differ')
            count = data.get('total')
        else:
            if (not isinstance(data.get('tables'), list) or len(data['tables']) != 1
                    or not isinstance(data['tables'][0], dict)
                    or data.get('template') != '/template/afterTrading/odd'):
                raise ReplayDataUnavailable('Wrong TPEx after-hours table')
            table = data['tables'][0]
            if table.get('date') != f'{int(day[:4])-1911}/{day[5:7]}/{day[8:]}':
                raise ReplayDataUnavailable('TPEx after-hours inner date differs')
            title, count = '盤後零股每日收盤行情', table.get('totalCount')
        fields = FIELDS[market]
        if (table.get('title') != title or table.get('fields') != fields or type(count) is not int
                or count <= 0 or not isinstance(table.get('data'), list) or count != len(table['data'])):
            raise ReplayDataUnavailable('After-hours schema or row count differs')
        detail = _twse_summary_rows(table) if market == 'twse' else table['data']
        rows, seen = {}, set()
        for raw in detail:
            if not isinstance(raw, list) or len(raw) != len(fields) or not isinstance(raw[0], str):
                raise ReplayDataUnavailable('After-hours row width differs')
            sid = str(raw[0]).strip()
            if not sid or sid in seen:
                raise ReplayDataUnavailable('Duplicate or empty after-hours security')
            seen.add(sid)
            if not re.fullmatch(r'[0-9]{4}', sid):
                continue  # Four-digit common-stock universe plus the 0050 benchmark.
            qty, trades = _number(raw[2], quantity=True), _number(raw[3], quantity=True)
            amount, price = _number(raw[4]), _number(raw[5], missing=True)
            if (qty > 0 and (price is None or price <= 0 or trades <= 0 or trades > qty)
                    or qty == 0 and (price is not None or trades != 0 or amount != 0)):
                raise ReplayDataUnavailable('After-hours price, amount and volume conflict')
            rows[sid] = dict(stock_id=sid, odd_high=price, odd_low=price, auction_price=price,
                odd_shares=qty, auction_trades=trades, auction_amount=amount,
                source_date=day, market=market, after_hours=True, volume_scope=SCOPE,
                price_unit='TWD_per_share', volume_unit='shares', auction_time='14:30:00')
        if not rows:
            raise ReplayDataUnavailable('After-hours table has no supported securities')
        return rows
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ReplayDataUnavailable):
            raise
        raise ReplayDataUnavailable('Malformed after-hours auction evidence') from exc


def match_after_hours(row, side, limit_price, quantity, *, used_shares=0, participation=.01):
    """One 14:30 auction, strict price-through and at most 1% observed shares.

    `used_shares` is previous simulated fills for this security/day, across all
    orders and both sides. Same-price queue allocation is deliberately uncredited.
    Auction price is the only price simulated; no intraday HL2 is constructed.
    """
    if (side not in ('buy', 'sell') or type(quantity) is not int or not 0 < quantity < 1000
            or type(used_shares) is not int or used_shares < 0
            or isinstance(limit_price, bool) or not isinstance(limit_price, (int, float))
            or not math.isfinite(limit_price) or limit_price <= 0
            or isinstance(participation, bool) or not isinstance(participation, (int, float))
            or not math.isfinite(participation) or not 0 < participation <= .01):
        raise ValueError('Invalid preplanned after-hours odd order or participation')
    if not isinstance(row, dict):
        raise ReplayDataUnavailable('Missing after-hours auction evidence')
    qty, price = row.get('odd_shares'), row.get('auction_price')
    if (row.get('volume_scope') != SCOPE or row.get('after_hours') is not True
            or row.get('volume_unit') != 'shares' or row.get('price_unit') != 'TWD_per_share'
            or row.get('auction_time') != '14:30:00' or type(qty) is not int or qty < 0
            or row.get('odd_high') != price or row.get('odd_low') != price
            or (qty == 0 and price is not None)
            or (qty > 0 and (isinstance(price, bool) or not isinstance(price, (int, float))
                              or not math.isfinite(price) or price <= 0))):
        raise ReplayDataUnavailable('Missing or inconsistent after-hours auction evidence')
    total_capacity = int((Decimal(qty)*Decimal(str(participation))).to_integral_value(rounding=ROUND_FLOOR))
    if used_shares > total_capacity:
        raise ValueError('Previously used after-hours shares exceed auction capacity')
    through = price is not None and (Decimal(str(price)) < Decimal(str(limit_price)) if side == 'buy'
                                    else Decimal(str(price)) > Decimal(str(limit_price)))
    available = max(0, total_capacity-used_shares) if through else 0
    filled = min(quantity, available)
    reason = ('no_auction_trades' if qty == 0 else 'same_price_queue_uncredited' if price == limit_price
              else 'limit_not_through_auction' if not through else 'participation_capacity_exhausted'
              if not filled else 'partial_auction_capacity' if filled < quantity else None)
    return dict(filled_qty=filled, capacity_qty=available, auction_capacity_qty=total_capacity,
        eligible_shares=qty if through else 0, source_volume=qty, auction_price=price,
        reference_price=price if filled else None, last_fill_time='14:30:00' if filled else None,
        participation_limit=participation, used_shares=used_shares, failure=reason,
        evidence_status='complete_auction_table', execution_evidence='after_hours_strict_price_through_simulation',
        actual_fill_verified=False, live_qualified=False)


def request_item(day, market):
    stamp = date.fromisoformat(day)
    market = str(market).upper()
    if stamp.isoformat() != day or not START <= day <= END or market.lower() not in URLS:
        raise ValueError('Executable after-hours odd request outside registered market/date scope')
    return dict(market=market, date=day, url=URLS[market.lower()],
                params=dict(date=stamp.strftime('%Y%m%d' if market == 'TWSE' else '%Y/%m/%d'),
                            response='json', type='ALL' if market == 'TWSE' else 'Daily'))


def validate_plans(day, sid, plans):
    if not isinstance(sid, str) or len(sid) != 4 or not sid.isdigit():
        raise ValueError('Executable after-hours odd demand needs a four-digit stock')
    if not isinstance(plans, list) or not plans:
        raise ValueError('Executable after-hours odd demand needs precommitted positive odd-share orders')
    for plan in plans:
        qty, odd = plan.get('planned_qty'), plan.get('odd_qty')
        if (plan.get('date') != day or plan.get('stock_id') != sid
                or type(qty) is not int or type(odd) is not int or not 0 < odd <= qty
                or odd >= 1000):
            raise ValueError('Executable after-hours odd precommitted quantity or identity differs')
        board = plan.get('board_qty')
        if board is not None and (type(board) is not int or board < 0 or board % 1000 or board + odd != qty):
            raise ValueError('Executable after-hours odd board and odd quantities disagree')
        signal, price = plan.get('signal_date'), plan.get('odd_limit')
        if (plan.get('side') not in ('buy', 'sell') or not isinstance(signal, str)
                or date.fromisoformat(signal).isoformat() != signal or signal >= day
                or plan.get('odd_order_time') != '13:40:00'
                or plan.get('odd_expires_at') != '14:30:00'
                or isinstance(price, bool) or not isinstance(price, (int, float))
                or not math.isfinite(price) or price <= 0):
            raise ValueError('Executable after-hours odd requires prior signal, limit and auction window')
        tick = ('.01' if price < 50 else '.05') if sid == '0050' else next(
            unit for ceiling, unit in [(10,'.01'),(50,'.05'),(100,'.1'),(500,'.5'),
                                       (1000,'1'),(math.inf,'5')] if price < ceiling)
        if Decimal(str(price)) % Decimal(tick):
            raise ValueError('Executable after-hours odd limit violates stock tick size')
    return deepcopy(plans)


def demand_outcomes(rows, sid, plans):
    """Bind the dispatch demand to source-supported simulated auction outcomes."""
    used, outcomes = 0, []
    for plan in plans:
        outcome = match_after_hours(rows[sid], plan['side'], plan['odd_limit'],
                                    plan['odd_qty'], used_shares=used)
        outcomes.append(dict(side=plan['side'], requested_qty=plan['odd_qty'],
                             limit_price=plan['odd_limit'], **outcome))
        used += outcome['filled_qty']
    return outcomes


def requirement(day, sid, engine):
    values = getattr(engine, 'day_plans', {})
    if not isinstance(values, dict):
        raise ValueError('Executable after-hours odd requires the frozen day-plan mapping')
    plans = [p for p in values.values() if p.get('date') == day and p.get('stock_id') == sid
             and p.get('planned_qty', 0) > 0 and p.get('odd_qty', 0) > 0]
    return validate_plans(day, sid, plans)


class ExecutableOddData:
    def __init__(self, root, *, online=False, session=None, clock=time.time, sleep=time.sleep):
        self.root = Path(root).resolve()
        self.cache, self.online = self.root / BASE, online
        self.clock, self.sleep, self.refs = clock, sleep, {}
        self.session = session if session is not None else requests.Session()
        if session is None:
            self.session.trust_env = False
        self.auth = self.cache / 'authorization.json'
        self.expected_auth = dict(schema='poc_executable_odd_authorization_v1',
            authorization_basis='current_user_requested_poc_red_candle_executable_limit_account_backtest',
            request_description='Necessary official after-hours odd-lot evidence for this POC plus red-candle preplanned limit account backtest',
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
            raise ValueError('Executable after-hours odd evidence escapes repository')
        return path

    def mark(self, path, expected=None):
        path = self.path(path)
        actual, name = digest(path), str(path.relative_to(self.root))
        if expected is not None and actual != expected:
            raise ValueError('Executable after-hours odd source hash differs: ' + name)
        if name in self.refs and self.refs[name] != actual:
            raise ValueError('Executable after-hours odd source mutated: ' + name)
        self.refs[name] = actual
        return actual

    def _authorization(self, *, fresh):
        self.mark(self.auth)
        value = read(self.auth)
        if any(value.get(k) != v for k, v in self.expected_auth.items()):
            raise ValueError('Executable after-hours odd authorization scope differs')
        if fresh and not 0 <= self.clock() - _epoch(value['created_at']) <= 86400:
            raise ReplayDataUnavailable('Executable after-hours odd authorization expired')
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
            raise ValueError('Unrecognized executable after-hours odd origin hold')
        self.mark(hold)
        for mapping in ('evidence_sha256', 'evidence'):
            for name, expected in value.get(mapping, {}).items():
                self.mark(self.path(name), expected)
        return digest(hold)

    def _supplement_path(self, day, market):
        item = request_item(day, market)
        return self.cache / 'schema-supplements' / (item['market']+'-'+day+'.json')

    def cached(self, path, day, market, seen=None):
        """Use an explicitly issued offline supplement without erasing failure."""
        path = self.path(path)
        supplement = self._supplement_path(day, market)
        if path == supplement:
            return self._verified_supplement(path, day, market, seen)
        expected = self.cache/'receipts'/(request_item(day,market)['market']+'-'+day+'.json')
        if path != expected:
            raise ValueError('Executable after-hours odd receipt path identity differs')
        if read(path).get('accepted') is not True and supplement.exists():
            return self._verified_supplement(supplement, day, market, seen)
        return self._cached_source(path, day, market, seen)

    def _proof_source(self, path, day, market, seen=None):
        """A supplement proves original bytes; its timestamp is not a new probe."""
        self.cached(path, day, market, seen)
        value = read(path)
        if value.get('schema') == 'poc_executable_odd_schema_supplement_v1':
            return read(self.path(value['base_receipt_path']))
        return value

    def _cached_source(self, path, day, market, seen=None, *, permit_schema_failure=False):
        request = request_item(day, market)
        market = request['market']
        path = self.path(path)
        if path != self.cache / 'receipts' / (request['market'] + '-' + day + '.json'):
            raise ValueError('Executable after-hours odd receipt path identity differs')
        seen = set() if seen is None else set(seen)
        if path in seen:
            raise ValueError('Cyclic executable after-hours odd evidence')
        seen.add(path)
        self.mark(path); self.mark(path.with_suffix('.sha256'))
        if path.with_suffix('.sha256').read_text().strip() != digest(path):
            raise ValueError('Executable after-hours odd receipt hash differs')
        r = read(path)
        schema_failure = (permit_schema_failure and r.get('accepted') is False
                          and r.get('status') == 'schema_error_no_retry')
        if r.get('accepted') is not True and not schema_failure:
            raise ReplayDataUnavailable('Prior executable after-hours odd request failed; no automatic retry: ' + market + ' ' + day)
        if (r.get('schema') != 'poc_executable_odd_receipt_v1' or any(r.get(k) != v for k, v in request.items())
                or r.get('status') != ('schema_error_no_retry' if schema_failure else 'verified_after_hours_odd_market_day')
                or r.get('http_status') != 200
                or r.get('security_denied') is not False or r.get('response_present') is not True
                or r.get('redirect_statuses') != [] or r.get('automatic_redirects_disabled') is not True
                or r.get('global_hold_unchanged') is not True):
            raise ValueError('Executable after-hours odd accepted receipt identity/transport differs')
        auth = self._authorization(fresh=False)
        if r['authorization'] != auth:
            raise ValueError('Executable after-hours odd receipt belongs to another authorization')
        start, end = _epoch(r['started_at']), _epoch(r['retrieved_at'])
        if not 0 <= start - _epoch(auth['created_at']) <= 86400 or end < start:
            raise ValueError('Executable after-hours odd receipt authorization time differs')
        raw = self.path(r['raw_path'])
        if raw != self.cache / 'raw' / (market + '-' + day + '.bin'):
            raise ValueError('Executable after-hours odd raw path identity differs')
        self.mark(raw, r['raw_sha256'])
        if _security_response(SimpleNamespace(content=raw.read_bytes(),status_code=200,history=[])):
            raise ValueError('Security response cannot be reclassified as auction evidence')
        attempt_path = self.cache / 'attempts' / (market + '-' + day + '.json')
        self.mark(attempt_path, r['attempt_sha256']); attempt = read(attempt_path)
        if attempt.get('schema') != 'poc_executable_odd_attempt_v1' or any(
                r.get(k) != v for k, v in attempt.items() if k != 'schema'):
            raise ValueError('Executable after-hours odd receipt differs from pre-dispatch attempt')
        helper = self.cache / 'source-snapshots' / (r['helper_sha256'] + '.py')
        self.mark(helper, r['helper_sha256'])
        demand_path = self.path(r['demand_path']); self.mark(demand_path, r['demand_sha256'])
        demand = read(demand_path)
        if (demand.get('schema') != 'poc_executable_odd_demand_v1' or demand.get('date') != day
                or demand.get('market') != market or demand_path != self.cache / 'demands' /
                (market + '-' + day + '-' + demand['stock_id'] + '.json')):
            raise ValueError('Executable after-hours odd demand identity differs')
        validate_plans(day, demand['stock_id'], demand['precommitted_plans'])
        if self._hold(request) != r['global_hold_sha256']:
            raise ValueError('Executable after-hours odd global hold changed')
        proof = r.get('recovery_proof')
        if r['request_kind'] == 'single_normal_probe':
            if proof is not None:
                raise ValueError('Executable after-hours odd first probe must not borrow another proof')
        elif r['request_kind'] == 'necessary_preplanned_day':
            if not isinstance(proof, dict):
                raise ValueError('Executable after-hours odd endpoint proof is missing')
            parent = self.path(proof['path']); self.mark(parent, proof['sha256'])
            prior = self._proof_source(parent, read(parent)['date'], market, seen)
            if (prior.get('request_kind') != 'single_normal_probe' or prior.get('url') != r['url']
                    or prior.get('market') != market or prior.get('authorization') != auth
                    or prior.get('retrieved_at') != proof.get('retrieved_at')
                    or not 0 <= start - _epoch(prior['retrieved_at']) <= 86400):
                raise ValueError('Executable after-hours odd endpoint proof identity/time differs')
        else:
            raise ValueError('Unknown executable after-hours odd request kind')
        rows = parse_after_hours(wrapper(r, json.loads(raw.read_bytes())), market.lower(), day)
        if demand['stock_id'] not in rows or (not schema_failure and len(rows) != r['rows']):
            raise ValueError('Executable after-hours odd demand stock or row count differs')
        if schema_failure:
            return rows
        if (r.get('volume_scope') != SCOPE or r.get('auction_time') != '14:30:00'
                or r.get('demand_outcomes') != demand_outcomes(rows, demand['stock_id'], demand['precommitted_plans'])
                or r.get('actual_fill_verified') is not False):
            raise ValueError('Executable after-hours odd demand outcome differs')
        return rows

    def _verified_supplement(self, path, day, market, seen=None):
        path = self.path(path); item = request_item(day,market); market = item['market']
        if path != self._supplement_path(day,market):
            raise ValueError('Offline schema supplement path differs')
        seen = set() if seen is None else set(seen)
        if path in seen:
            raise ValueError('Cyclic executable after-hours schema supplement')
        seen.add(path)
        self.mark(path); self.mark(path.with_suffix('.sha256'))
        if path.with_suffix('.sha256').read_text().strip() != digest(path):
            raise ValueError('Offline schema supplement hash differs')
        value = read(path)
        if (value.get('schema') != 'poc_executable_odd_schema_supplement_v1'
                or any(value.get(k) != v for k,v in item.items())
                or value.get('request_kind') != 'offline_schema_reparse'
                or value.get('accepted') is not True or value.get('status') != 'verified_existing_http200_bytes'
                or value.get('network_requests') != 0 or value.get('original_failure_preserved') is not True
                or value.get('volume_scope') != SCOPE or value.get('auction_time') != '14:30:00'
                or value.get('actual_fill_verified') is not False or value.get('live_qualified') is not False
                or not isinstance(value.get('reason'),str) or not value['reason'].strip()):
            raise ValueError('Offline schema supplement scope/claim differs')
        base = self.path(value['base_receipt_path'])
        if base != self.cache/'receipts'/(market+'-'+day+'.json'):
            raise ValueError('Offline schema supplement original identity differs')
        verifier = type(self)(self.root, online=False, clock=self.clock, sleep=self.sleep)
        verifier.mark(base,value['base_receipt_sha256'])
        rows = verifier._cached_source(base, day, market, seen, permit_schema_failure=True)
        original = read(base)
        if original.get('accepted') is not False or original.get('status') != 'schema_error_no_retry':
            raise ValueError('Offline supplement requires an original schema failure')
        required = ('authorization','raw_path','raw_sha256','attempt_sha256','demand_path','demand_sha256',
                    'started_at','retrieved_at','global_hold_sha256')
        if any(value.get(k) != original[k] for k in required):
            raise ValueError('Offline schema supplement changed original evidence')
        if (value.get('original_request_kind') != original['request_kind']
                or _epoch(value['created_at']) < _epoch(original['retrieved_at'])):
            raise ValueError('Offline schema supplement time or original request kind differs')
        helper = self.cache/'source-snapshots'/(value['helper_sha256']+'.py')
        verifier.mark(helper,value['helper_sha256'])
        demand = read(self.path(original['demand_path']))
        if (value.get('rows') != len(rows) or value.get('demand_outcomes') !=
                demand_outcomes(rows,demand['stock_id'],demand['precommitted_plans'])
                or value.get('source_sha256') != verifier.refs):
            raise ValueError('Offline schema supplement outcome or source closure differs')
        for name,expected in verifier.refs.items():
            self.mark(self.path(name),expected)
        return rows

    def reparse_schema(self, day, market, *, reason):
        """Explicit, append-only local repair; never dispatch or refresh age.

        Only a response-present HTTP200 schema failure is eligible. Original
        receipt, attempt, demand, bytes, authorization and source hold remain
        byte-for-byte unchanged. A repaired first probe retains its ORIGINAL
        response timestamp for the separate 24-hour dispatch freshness check.
        """
        item = request_item(day,market); market = item['market']
        if not isinstance(reason,str) or not reason.strip():
            raise ValueError('Offline schema repair requires a concrete reason')
        base = self.cache/'receipts'/(market+'-'+day+'.json')
        target = self._supplement_path(day,market)
        _,origin_lock,_ = OfficialDailyAcquisition._origin_paths(self,item)
        with file_lock(self.cache/'.dispatch.lock'), file_lock(origin_lock):
            if target.exists():
                self._verified_supplement(target,day,market)
                value = read(target)
                self._publish_reparse_proof(target,value)
                return value
            verifier = type(self)(self.root, online=False, clock=self.clock, sleep=self.sleep)
            rows = verifier._cached_source(base,day,market,permit_schema_failure=True)
            original = read(base)
            if original.get('accepted') is not False or original.get('status') != 'schema_error_no_retry':
                raise ValueError('Offline reparse only repairs an original schema failure')
            helper_hash = digest(Path(__file__))
            helper = self.cache/'source-snapshots'/(helper_hash+'.py')
            if not helper.exists():
                helper.parent.mkdir(parents=True,exist_ok=True)
                with helper.open('xb') as stream:stream.write(Path(__file__).read_bytes())
            verifier.mark(helper,helper_hash)
            demand = read(self.path(original['demand_path']))
            value = dict(schema='poc_executable_odd_schema_supplement_v1',**item,
                request_kind='offline_schema_reparse', original_request_kind=original['request_kind'],
                accepted=True,status='verified_existing_http200_bytes',network_requests=0,
                original_failure_preserved=True,reason=reason.strip(),created_at=self._now(),
                base_receipt_path=str(base.relative_to(self.root)),base_receipt_sha256=digest(base),
                helper_sha256=helper_hash,rows=len(rows),volume_scope=SCOPE,auction_time='14:30:00',
                demand_outcomes=demand_outcomes(rows,demand['stock_id'],demand['precommitted_plans']),
                source_sha256=dict(verifier.refs),actual_fill_verified=False,live_qualified=False)
            for field in ('authorization','raw_path','raw_sha256','attempt_sha256','demand_path','demand_sha256',
                          'started_at','retrieved_at','global_hold_sha256'):
                value[field] = original[field]
            _write(target,value,exclusive=True)
            target.with_suffix('.sha256').write_text(digest(target)+'\n')
            self._verified_supplement(target,day,market)
            self._publish_reparse_proof(target,value)
            return value

    def _publish_reparse_proof(self, target, value):
        if value['original_request_kind'] != 'single_normal_probe':
            return
        proof_path = self.cache/('proof-'+value['market']+'.json')
        proof = dict(path=str(target.relative_to(self.root)),sha256=digest(target),
                     retrieved_at=value['retrieved_at'])
        if proof_path.exists() and read(proof_path) != proof:
            raise ValueError('Offline schema repair cannot replace an endpoint proof')
        if not proof_path.exists():_write(proof_path,proof,exclusive=True)
        self.mark(proof_path)

    def _dispatch_permission(self, request, state, hold, proof_path):
        auth = self._authorization(fresh=True)
        if state.get('in_flight'):
            raise ReplayDataUnavailable('Unfinished shared origin request; no automatic recovery')
        self._hold(request)
        blocks = OfficialDailyAcquisition._blocks(self, state, hold)
        probe_key = sha256((auth['sha256'] + '\n' + request['url']).encode()).hexdigest()
        if not proof_path.exists():
            if probe_key in state.get('probes', {}):
                raise ReplayDataUnavailable('Executable after-hours odd single normal probe already consumed')
            if any(_epoch(b['observed_at']) >= _epoch(auth['created_at']) for b in blocks):
                raise ReplayDataUnavailable('Newer origin stop prohibits executable after-hours odd probe')
            return True, None, probe_key
        self.mark(proof_path); proof = read(proof_path)
        parent = self.path(proof['path']); self.mark(parent, proof['sha256'])
        previous = self._proof_source(parent, read(parent)['date'], request['market'])
        if (previous['request_kind'] != 'single_normal_probe' or previous['url'] != request['url']
                or previous['retrieved_at'] != proof['retrieved_at']
                or not 0 <= self.clock() - _epoch(previous['retrieved_at']) <= 86400):
            raise ReplayDataUnavailable('Executable after-hours odd exact endpoint proof is invalid or stale')
        if any(_epoch(b['observed_at']) >= _epoch(previous['retrieved_at']) for b in blocks):
            raise ReplayDataUnavailable('Newer origin stop invalidates executable after-hours odd proof')
        return False, proof, probe_key

    def get(self, day, sid, market, engine=None):
        request = request_item(day, market); market = request['market']
        if not isinstance(sid, str) or len(sid) != 4 or not sid.isdigit():
            raise ValueError('Executable after-hours odd requires a four-digit stock')
        key = market + '-' + day
        receipt = self.cache / 'receipts' / (key + '.json')
        plans = requirement(day, sid, engine) if engine is not None else None
        if receipt.exists():
            rows = self.cached(receipt, day, market)
            if sid not in rows:
                raise ReplayDataUnavailable('Executable after-hours odd table lacks stock: ' + key + ' ' + sid)
            if plans is None:
                # Pure offline replay may use the original bound demand. A
                # different stock requires its own frozen execution plan.
                demand = read(self.path(read(receipt)['demand_path']))
                if demand['stock_id'] != sid:
                    raise ValueError('Executable after-hours odd new stock requires a precommitted plan')
            return deepcopy(rows[sid])
        if not self.online:
            raise ReplayDataUnavailable('Executable official after-hours odd day missing: ' + key)
        if plans is None:
            plans = requirement(day, sid, engine)
        state_path, origin_lock, hold = OfficialDailyAcquisition._origin_paths(self, request)
        attempt = self.cache / 'attempts' / (key + '.json')
        proof_path = self.cache / ('proof-' + market + '.json')
        with file_lock(self.cache / '.dispatch.lock'), file_lock(origin_lock):
            if receipt.exists():
                rows = self.cached(receipt, day, market)
                if sid not in rows:
                    raise ReplayDataUnavailable('Executable after-hours odd table lacks stock: ' + key + ' ' + sid)
                return deepcopy(rows[sid])
            if attempt.exists():
                raise ReplayDataUnavailable('Previous executable after-hours odd attempt unfinished; no retry')
            if len(list((self.cache / 'attempts').glob('*.json'))) >= MAXIMUM:
                raise ReplayDataUnavailable('Executable after-hours odd persistent 400-attempt budget exhausted')
            state = read(state_path) if state_path.exists() else dict(probes={})
            is_probe, proof, probe_key = self._dispatch_permission(request, state, hold, proof_path)
            before_hold = digest(hold) if hold.exists() else None
            wait = max(0, 3.1 - (self.clock() - state.get('last_start_epoch', 0)))
            if wait:
                self.sleep(wait + .001)
            # Expiry or an independent new global hold during sleep forbids GET.
            if (digest(hold) if hold.exists() else None) != before_hold:
                raise ReplayDataUnavailable('Executable after-hours odd global hold changed during wait')
            is_probe, proof, probe_key = self._dispatch_permission(request, state, hold, proof_path)
            demand = self.cache / 'demands' / (key + '-' + sid + '.json')
            demand_record = dict(schema='poc_executable_odd_demand_v1', date=day, market=market,
                                 stock_id=sid, precommitted_plans=plans)
            if demand.exists() and read(demand) != demand_record:
                raise ValueError('Executable after-hours odd previously frozen demand differs')
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
            record = dict(schema='poc_executable_odd_attempt_v1', **request,
                authorization=self.authorization, request_kind='single_normal_probe' if is_probe else 'necessary_preplanned_day',
                started_at=self._now(), demand_path=str(demand.relative_to(self.root)), demand_sha256=digest(demand),
                global_hold_sha256=before_hold, helper_sha256=helper_hash, recovery_proof=proof)
            _write(attempt, record, exclusive=True); self.mark(attempt)
            record.update(schema='poc_executable_odd_receipt_v1', attempt_sha256=digest(attempt), accepted=False,
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
                    rows = parse_after_hours(wrapper(record, json.loads(response.content)), market.lower(), day)
                    if not rows or sid not in rows:
                        raise ValueError('Official odd table lacks demanded stock')
                    record.update(accepted=True, status='verified_after_hours_odd_market_day', rows=len(rows),
                        volume_scope=SCOPE, auction_time='14:30:00', actual_fill_verified=False,
                        demand_outcomes=demand_outcomes(rows, sid, plans))
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
                raise ValueError('Executable after-hours odd snapshot receipt hash differs')
            value = read(path)
            supplement = self._supplement_path(value['date'],value['market'])
            reparsed = supplement.exists()
            if value.get('accepted') is True or reparsed:
                self.cached(path, value['date'], value['market'])
            receipts.append(dict(path=str(path.relative_to(self.root)), sha256=digest(path),
                accepted=value.get('accepted') is True, status=value.get('status'),
                offline_schema_reparsed=reparsed,
                effective_accepted=value.get('accepted') is True or reparsed,
                supplement_path=str(supplement.relative_to(self.root)) if reparsed else None,
                supplement_sha256=digest(supplement) if reparsed else None))
        return dict(schema='poc_executable_odd_snapshot_v1', maximum_attempts=MAXIMUM,
            attempted_calls=len(list((self.cache / 'attempts').glob('*.json'))),
            accepted_count=sum(r['accepted'] for r in receipts), failed_count=sum(not r['accepted'] for r in receipts),
            offline_schema_reparsed_count=sum(r['offline_schema_reparsed'] for r in receipts),
            effective_accepted_count=sum(r['effective_accepted'] for r in receipts),
            unresolved_failure_count=sum(not r['effective_accepted'] for r in receipts),
            receipts=receipts, source_sha256=dict(self.refs), volume_scope=SCOPE,
            actual_fill_verified=False, live_qualified=False)
