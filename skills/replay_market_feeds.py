"""Date-verified execution evidence for preparation and strictly offline replay.

Odd-lot daily volume is a participation ceiling, not executable book depth.
Neither a last trade nor a last quote guarantees a fill at the replay time.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid

import requests

from app.file_lock import file_lock
from app.finmind import FinMindError, FinMindQuotaError, fetch_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / '.cache/million-replay-inputs/execution-feeds'
START, END = date(2022, 1, 1), date(2026, 9, 9)
URLS = {'twse': 'https://www.twse.com.tw/rwd/zh/afterTrading/TWTC7U',
        'tpex': 'https://www.tpex.org.tw/www/zh-tw/afterTrading/oddQuote'}
FIELDS = {
    'twse': {'stock_id': '證券代號', 'odd_shares': '成交股數', 'odd_last': '當日最後一次成交價',
             'odd_high': '當日最高價', 'odd_low': '當日最低價', 'odd_bid': '最後揭示買價',
             'odd_ask': '最後揭示賣價', 'bid_qty': '最後揭示買量', 'ask_qty': '最後揭示賣量'},
    'tpex': {'stock_id': '代號', 'odd_shares': '成交股數', 'odd_last': '最後成交價',
             'odd_high': '最高', 'odd_low': '最低', 'odd_bid': '最後買價',
             'odd_ask': '最後賣價', 'bid_qty': '最後買量(股)', 'ask_qty': '最後賣量(股)'}}
LIMITATION = ('Odd-lot daily volume is not auction volume or executable depth; '
              '5% daily participation is a research ceiling, not a fill guarantee.')


class ReplayDataUnavailable(RuntimeError):
    """Missing, changed, or invalid source evidence; never a no-trade signal."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    def invalid(value):
        raise ReplayDataUnavailable('Non-finite JSON value in execution evidence')
    try:
        return json.loads(Path(path).read_text(), parse_constant=invalid)
    except (OSError, ValueError, TypeError) as exc:
        raise ReplayDataUnavailable('Execution evidence cannot be read') from exc


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _day(value):
    if isinstance(value, datetime):
        if value.tzinfo is not None or value.time() != datetime.min.time():
            raise ValueError('A timezone-naive date-only value is required')
        value = value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Date must be YYYY-MM-DD')
    return date.fromisoformat(value)


def _sid(sid):
    if not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid):
        raise ValueError('Only four-digit Taiwan stock IDs, including 0050, are supported')
    return sid


def _number(value, *, quantity=False, missing=False):
    if isinstance(value, bool):
        raise ReplayDataUnavailable('Boolean market number is invalid')
    text = '' if value is None else str(value).strip().replace(',', '')
    if text in {'', '-', '--', '---', '----'}:
        if missing:
            return None
        raise ReplayDataUnavailable('Required market number is missing')
    try:
        number = float(text)
    except (ValueError, TypeError) as exc:
        raise ReplayDataUnavailable('Invalid numeric market field') from exc
    if (not math.isfinite(number) or number < 0
            or (quantity and (not number.is_integer() or number > 2**53 - 1))):
        raise ReplayDataUnavailable('Market number must be finite and nonnegative')
    if quantity:
        return int(number)
    return None if number == 0 and missing else number


def _params(market, day):
    return {'date': day.strftime('%Y%m%d' if market == 'twse' else '%Y/%m/%d'), 'response': 'json'}


def parse_odd(record, market, day):
    """Validate source date and named columns before retaining four-digit rows."""
    day = _day(day)
    if (record.get('schema') != 1 or record.get('provider') != market or record.get('day') != day.isoformat()
            or record.get('url') != URLS[market] or record.get('params') != _params(market, day)
            or record.get('http_status') != 200 or not record.get('retrieved_at')):
        raise ReplayDataUnavailable('Odd-lot request provenance or HTTP status is invalid')
    payload = record.get('payload')
    if (not isinstance(payload, dict) or payload.get('date') != day.strftime('%Y%m%d')
            or str(payload.get('stat', '')).lower() != 'ok'):
        raise ReplayDataUnavailable('Odd-lot response date/status does not match requested date')
    if market == 'twse':
        table = payload
        expected_title = f'{day.year - 1911}年{day.month:02d}月{day.day:02d}日 盤中零股交易行情單'
        if table.get('title') != expected_title:
            raise ReplayDataUnavailable('TWSE odd-lot title/date is inconsistent')
    else:
        tables = payload.get('tables')
        if not isinstance(tables, list) or len(tables) != 1 or not isinstance(tables[0], dict):
            raise ReplayDataUnavailable('TPEx odd-lot table is missing or ambiguous')
        table = tables[0]
        if table.get('date') != day.strftime('%Y%m%d') or table.get('title') != '盤中零股每日收盤行情':
            raise ReplayDataUnavailable('TPEx odd-lot table date/title does not match requested date')
    fields, source_rows = table.get('fields'), table.get('data')
    if (not isinstance(fields, list) or any(not isinstance(field, str) for field in fields)
            or not isinstance(source_rows, list)):
        raise ReplayDataUnavailable('Odd-lot fields/data schema is missing')
    fields = [field.strip() for field in fields]
    required = {value for key, value in FIELDS[market].items() if key not in {'bid_qty', 'ask_qty'}}
    if len(set(fields)) != len(fields) or not required.issubset(fields):
        raise ReplayDataUnavailable('Odd-lot required named fields changed')
    count_field = 'total' if market == 'twse' else 'totalCount'
    if count_field in table and _number(table[count_field], quantity=True) != len(source_rows):
        raise ReplayDataUnavailable('Odd-lot response row count is incomplete')
    rows = {}
    for values in source_rows:
        if not isinstance(values, list) or len(values) != len(fields):
            raise ReplayDataUnavailable('Odd-lot row does not match field names')
        source = dict(zip(fields, values))
        sid = str(source[FIELDS[market]['stock_id']]).strip()
        # The official response includes non-four-digit ETFs and warrants. They
        # remain in raw evidence, but are outside this replay's declared scope.
        if not re.fullmatch(r'\d{4}', sid):
            continue
        if sid in rows:
            raise ReplayDataUnavailable('Duplicate odd-lot stock/date')
        row = {}
        for key, name in FIELDS[market].items():
            if key == 'stock_id':
                continue
            if key in {'bid_qty', 'ask_qty'} and name not in source:
                row[key] = None
            else:
                row[key] = _number(source[name], quantity=key in {'odd_shares', 'bid_qty', 'ask_qty'},
                                   missing=key != 'odd_shares')
        traded = row['odd_shares'] > 0
        prices = [row[key] for key in ('odd_low', 'odd_last', 'odd_high')]
        if traded and (any(value is None or value <= 0 for value in prices) or prices != sorted(prices)):
            raise ReplayDataUnavailable('Positive odd-lot volume lacks coherent trade prices')
        if not traded and any(value is not None for value in prices):
            raise ReplayDataUnavailable('Zero odd-lot volume conflicts with recorded trade prices')
        for side in ('bid', 'ask'):
            if row[side + '_qty'] is not None and row[side + '_qty'] > 0 and row['odd_' + side] is None:
                raise ReplayDataUnavailable('Positive last quote quantity has no corresponding price')
        rows[sid] = {**row, 'source_date': day.isoformat(), 'market': market,
                     'daily_participation_ceiling': .05, 'execution_limitation': LIMITATION}
    return rows


def parse_limits(record, sid):
    if (record.get('schema') != 1 or record.get('provider') != 'finmind'
            or record.get('dataset') != 'TaiwanStockPriceLimit' or record.get('stock_id') != sid
            or record.get('start_date') != START.isoformat() or record.get('end_date') != END.isoformat()
            or record.get('status') != 'success' or not isinstance(record.get('data'), list)):
        raise ReplayDataUnavailable('FinMind limit source/query is invalid')
    result = {}
    for row in record['data']:
        if not isinstance(row, dict) or not {'stock_id', 'date', 'limit_up', 'limit_down'}.issubset(row):
            raise ReplayDataUnavailable('FinMind limit schema is incomplete')
        try:
            day = _day(row['date'])
        except ValueError as exc:
            raise ReplayDataUnavailable('FinMind limit date is invalid') from exc
        if row['stock_id'] != sid or not START <= day <= END or day.isoformat() in result:
            raise ReplayDataUnavailable('FinMind limit date/stock coverage is inconsistent')
        upper, lower = _number(row['limit_up']), _number(row['limit_down'])
        if (upper == 0) != (lower == 0) or upper < lower:
            raise ReplayDataUnavailable('Invalid limit range; both zero means no price limit')
        result[day.isoformat()] = {'upper': upper, 'lower': lower}
    return dict(sorted(result.items()))


class ReplayMarketFeeds:
    def __init__(self, cache_dir=DEFAULT_CACHE, *, offline=False, token=None,
                 http_get=None, finmind_fetch=None, official_min_interval=1.5):
        if not isinstance(offline, bool):
            raise ValueError('offline must be a boolean')
        if not math.isfinite(official_min_interval) or official_min_interval < 0:
            raise ValueError('official_min_interval must be finite and nonnegative')
        self.cache_dir, self.offline, self.token = Path(cache_dir), offline, token
        self.http_get = http_get or requests.Session().get
        self.finmind_fetch = finmind_fetch or fetch_dataset
        self.interval, self.last_http = official_min_interval, 0.
        if not offline:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with file_lock(self.cache_dir / 'feeds.lock', timeout=30):
                if not (self.cache_dir / 'index.json').exists():
                    self._initialize()

    def _initialize(self):
        state = {'schema': 1, 'parser_sha256': _sha(__file__), 'entries': {}, 'files_sha256': {},
                 'request_counters': {'official_http_requests': 0, 'finmind_fetch_calls': 0,
                                      'finmind_requests': 0, 'finmind_cache_hits': 0,
                                      'finmind_quota_failures_with_unknown_charge': 0},
                 'limitations': [LIMITATION, 'Missing dates are unavailable, never inferred from yesterday close.']}
        for path in sorted((self.cache_dir / 'probes').glob('*.json')):
            probe = _read(path)
            state['files_sha256'][str(path.relative_to(self.cache_dir))] = _sha(path)
            if probe.get('provider') in URLS:
                state['request_counters']['official_http_requests'] += 1
            elif probe.get('provider') == 'finmind':
                state['request_counters']['finmind_fetch_calls'] += 1
                state['request_counters']['finmind_requests'] += int(probe.get('finmind_requests', 0))
                state['request_counters']['finmind_cache_hits'] += int(bool(probe.get('attrs', {}).get('cache_hit')))
        _write(self.cache_dir / 'index.json', state)

    def _state(self):
        state = _read(self.cache_dir / 'index.json')
        if state.get('schema') != 1 or state.get('parser_sha256') != _sha(__file__):
            raise ReplayDataUnavailable('Execution-feed parser changed; prepare evidence in a new cache directory')
        return state

    def _verify_file(self, state, name):
        path = self.cache_dir / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or name not in state['files_sha256']
                or not path.is_file() or _sha(path) != state['files_sha256'][name]):
            raise ReplayDataUnavailable('Execution-feed file changed or is missing: ' + name)
        return path

    def _save(self, state, name, value):
        _write(self.cache_dir / name, value)
        state['files_sha256'][name] = _sha(self.cache_dir / name)
        _write(self.cache_dir / 'index.json', state)

    def _cached(self, state, key):
        entry = state['entries'].get(key)
        if entry is None:
            return None
        raw_path = self._verify_file(state, entry['raw_file'])
        normalized = _read(self._verify_file(state, entry['rows_file']))
        if normalized.get('raw_sha256') != _sha(raw_path):
            raise ReplayDataUnavailable('Execution-feed raw/normalized provenance disagrees')
        return normalized['rows']

    def _store(self, state, key, stem, record, rows):
        raw_name, rows_name = stem + '.raw.json', stem + '.rows.json'
        self._save(state, raw_name, record)
        self._save(state, rows_name, {'schema': 1, 'raw_sha256': state['files_sha256'][raw_name], 'rows': rows})
        dates = sorted(rows) if key.startswith('limits:') else [record['day']]
        state['entries'][key] = {'raw_file': raw_name, 'rows_file': rows_name,
            'row_count': len(rows), 'first_date': dates[0] if dates else None,
            'last_date': dates[-1] if dates else None, 'empty_verified_response': not rows}
        _write(self.cache_dir / 'index.json', state)

    def _probe(self, state, name):
        path = 'probes/' + name + '.json'
        if path in state['files_sha256']:
            return _read(self._verify_file(state, path))
        return None

    def get_odd(self, day, sid, market):
        day, sid = _day(day), _sid(sid)
        market = str(market).lower()
        if market not in URLS:
            raise ValueError('market must be twse or tpex')
        key, stem = f'odd:{market}:{day.isoformat()}', f'odd-{market}-{day.isoformat()}'
        if self.offline:
            rows = self._cached(self._state(), key)
            if rows is None:
                raise ReplayDataUnavailable('Offline replay is missing odd-lot evidence: ' + key)
            return deepcopy(rows.get(sid))
        with file_lock(self.cache_dir / 'feeds.lock', timeout=30):
            state = self._state()
            rows = self._cached(state, key)
            if rows is not None:
                return deepcopy(rows.get(sid))
            record = self._probe(state, stem)
            if record is None:
                record = {'schema': 1, 'provider': market, 'day': day.isoformat(), 'url': URLS[market],
                          'params': _params(market, day), 'retrieved_at': datetime.now(timezone.utc).isoformat()}
                time.sleep(max(0, self.interval - (time.monotonic() - self.last_http)))
                state['request_counters']['official_http_requests'] += 1
                _write(self.cache_dir / 'index.json', state)
                try:
                    response = self.http_get(URLS[market], params=record['params'], timeout=30)
                    self.last_http = time.monotonic()
                    record['http_status'] = response.status_code
                    if response.status_code != 200:
                        raise ReplayDataUnavailable(f'Official odd-lot HTTP {response.status_code}')
                    record['payload'] = response.json()
                except (requests.RequestException, ValueError, ReplayDataUnavailable) as exc:
                    record['error_type'] = type(exc).__name__
                    self._save(state, f'failures/{stem}-{uuid.uuid4().hex}.json', record)
                    raise ReplayDataUnavailable(f'Official odd-lot data unavailable ({type(exc).__name__})') from None
            try:
                rows = parse_odd(record, market, day)
            except ReplayDataUnavailable:
                self._save(state, f'failures/{stem}-{uuid.uuid4().hex}.json', record)
                raise
            self._store(state, key, stem, record, rows)
            return deepcopy(rows.get(sid))

    def get_limits(self, sid):
        sid = _sid(sid)
        key, stem = 'limits:' + sid, 'limits-' + sid
        if self.offline:
            rows = self._cached(self._state(), key)
            if rows is None:
                raise ReplayDataUnavailable('Offline replay is missing price-limit evidence: ' + sid)
            return deepcopy(rows)
        with file_lock(self.cache_dir / 'feeds.lock', timeout=30):
            state = self._state()
            rows = self._cached(state, key)
            if rows is not None:
                return deepcopy(rows)
            record = self._probe(state, stem)
            if record is None:
                token = self.token
                if token is None:
                    from app.config import load_config
                    token = load_config().finmind_token
                record = {'schema': 1, 'provider': 'finmind', 'dataset': 'TaiwanStockPriceLimit',
                    'stock_id': sid, 'start_date': START.isoformat(), 'end_date': END.isoformat(),
                    'retrieved_at': datetime.now(timezone.utc).isoformat()}
                state['request_counters']['finmind_fetch_calls'] += 1
                _write(self.cache_dir / 'index.json', state)
                try:
                    # The shared limiter reserves 10% of 6000, hence 5400/h.
                    frame = self.finmind_fetch('TaiwanStockPriceLimit', START, END, token=token,
                        data_id=sid, requests_per_hour=6000, max_retries=0, timeout=30)
                    record.update(status='success', data=json.loads(frame.to_json(orient='records')),
                                  attrs={key: frame.attrs.get(key) for key in ('retrieved_at', 'cache_hit', 'source')})
                    hit = bool(frame.attrs.get('cache_hit', False))
                    state['request_counters']['finmind_cache_hits'] += int(hit)
                    state['request_counters']['finmind_requests'] += int(not hit)
                except FinMindQuotaError:
                    # The shared client can raise before HTTP or after HTTP
                    # 402/429. Its exception does not identify which happened.
                    state['request_counters']['finmind_quota_failures_with_unknown_charge'] += 1
                    record.update(status='error', error_type='FinMindQuotaError')
                    self._save(state, f'failures/{stem}-{uuid.uuid4().hex}.json', record)
                    raise  # Preserve pause/retry-after semantics; never retry here.
                except FinMindError as exc:
                    state['request_counters']['finmind_requests'] += 1
                    record.update(status='error', error_type=type(exc).__name__)
                    self._save(state, f'failures/{stem}-{uuid.uuid4().hex}.json', record)
                    raise ReplayDataUnavailable('FinMind price-limit source failed: ' + type(exc).__name__) from None
            try:
                rows = parse_limits(record, sid)
            except ReplayDataUnavailable:
                self._save(state, f'failures/{stem}-{uuid.uuid4().hex}.json', record)
                raise
            self._store(state, key, stem, record, rows)
            return deepcopy(rows)

    def manifest(self):
        """Return verified hashes/counters; offline calls neither fetch nor write."""
        state = self._state()
        for name in state['files_sha256']:
            self._verify_file(state, name)
        return {**deepcopy(state), 'manifest_sha256': _sha(self.cache_dir / 'index.json'),
                'finmind_requests_upper_bound': (state['request_counters']['finmind_requests']
                    + state['request_counters']['finmind_quota_failures_with_unknown_charge']),
                'cache_directory': str(self.cache_dir.resolve())}
