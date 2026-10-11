"""Explicit official daily aggregates for explicitly sealed missing execution coordinates.

No ticks, timestamps, price-level volumes or POC profiles are synthesized.
The daily-range price and ordinary-volume capacity equations remain unchanged.
"""
from dataclasses import dataclass
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR
import inspect
import json
from pathlib import Path
import textwrap

import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from scripts.prepare_twse_board_reconciliation_v2 import query
from skills.all_entry_accounts import digest
from skills.board_tape_reconciliation_v2 import parse_twse_complete, parse_basket_detail, basket_requests, basket_key
from scripts.audit_twse_board_reconciliation_v2 import detail_item
from types import FunctionType
from skills.poc_executable_replay import validate_tape
from skills import poc_range_execution as execution
from skills import poc_range_audit as frozen_audit
from skills.all_entry_resource_rejections import resource_auditor
from skills.poc_range_data import intraday_row, requirement
from skills.replay_market_feeds import parse_odd

MANIFEST_SHA = '7dbf8530d37df87b1a7bf3f9de793772a9b4efa648cb36e9675ea63714f64274'
EXECUTION_SHA = 'dd241d0fa88ae5e82460b86a182821ce1cc32f4ae664f180d3d939f66f4e0cd1'
BASE_MANIFEST = '.cache/all-entry-accounts-20261011/ordinary-components-v1/manifest.json'
STATUS = 'official_ordinary_session_decomposition'
WINDOW = 'official_regular_session_including_delayed_close'


@dataclass(frozen=True)
class OfficialRegularDay:
    stock_id: str
    date: str
    market: str
    shares: int
    open_cents: int
    high_cents: int
    low_cents: int
    close_cents: int
    amount_cents: int
    source_sha256: str


def validate_board(value):
    if not isinstance(value, OfficialRegularDay):
        return validate_tape(value)
    if (type(value.shares) is not int or value.shares <= 0 or value.shares % 1000
            or any(type(getattr(value, f)) is not int or getattr(value, f) <= 0
                   for f in ('open_cents','high_cents','low_cents','close_cents','amount_cents'))
            or not value.low_cents <= min(value.open_cents, value.close_cents)
            <= max(value.open_cents, value.close_cents) <= value.high_cents
            or not value.low_cents * value.shares <= value.amount_cents <= value.high_cents * value.shares
            or not value.source_sha256):
        raise ValueError('Invalid official ordinary-session aggregate')


def board_prices(value):
    if isinstance(value, OfficialRegularDay):
        validate_board(value)
        return pd.Series([value.low_cents / 100, value.high_cents / 100])
    return value.price


def digest_fields(value, sha):
    if isinstance(value, OfficialRegularDay):
        return dict(ticks_sha256=None, official_daily_sha256=sha)
    return dict(ticks_sha256=sha)


def parse_verified_basket(payload, request, day):
    """Map the observed constituent-table schema without treating its M selector as a parent."""
    if payload.get('fields') == ['證券代號','證券名稱','成交價','成交量','成交值']:
        if (payload.get('selectType') != 'M'
                or payload.get('title') != f"序號：{request['identity']['sub']} {int(day[:4])-1911}年{day[5:7]}月{day[8:]}日 鉅額交易日成交資訊-股票組合"):
            raise ValueError('Unexpected constituent table identity or title')
        payload = deepcopy(payload)
        payload['fields'] = ['證券代號','證券名稱','成交價','成交股數','成交金額']
        payload.pop('selectType')
    # The original validator independently checks all four source identities,
    # date, exact security count, amounts, quantities and parent totals.
    return parse_basket_detail(payload, request, day)


def parse_complete(parts, details, day):
    private = FunctionType(parse_twse_complete.__code__,
        dict(parse_twse_complete.__globals__, parse_basket_detail=parse_verified_basket))
    return private(parts, details, day)


def load_official_completion(root, document):
    root, document = Path(root).resolve(), Path(document).resolve()
    sources = Sources(root)
    doc = json.loads(sources.bind(str(document.relative_to(root)), digest(document)).read_text())
    if (doc.get('schema') != 'all_entry_official_execution_completion_v1'
            or doc.get('revision') != 'v2_complete_basket_constituents'
            or any(doc.get(k) is not False for k in ('account_model_changed','actual_fill_verified','live_qualified'))):
        raise ValueError('Official completion scope or reviewed source revision changed')
    manifest_sha = doc['manifest_sha256']
    manifest = json.loads(sources.bind(doc['manifest_path'], manifest_sha).read_text())
    baseline = json.loads(sources.bind(BASE_MANIFEST, MANIFEST_SHA).read_text())
    if (manifest.get('schema') != 'all_entry_verified_ordinary_components_v1'
            or manifest.get('synthesized_tick_rows') is not False
            or doc.get('coordinates') != sorted(manifest['days'])
            or any(manifest['days'].get(k) != row for k,row in baseline['days'].items())
            or doc.get('added_coordinates') != sorted(set(manifest['days'])-set(baseline['days']))):
        raise ValueError('Official component manifest differs or rewrites sealed previous evidence')
    for name, sha in manifest['source_sha256'].items(): sources.bind(name, sha)
    result = {}
    for identity,row in manifest['days'].items():
        key = row['stock_id'], row['date'], row['market']
        if (identity != row['stock_id']+'-'+row['date'] or key[2] != 'TWSE'
                or len(key[0]) != 4 or not key[0].isdigit() or key in result):
            raise ValueError('Official coordinate is invalid or duplicated')
        parts = {}
        for kind, record in row['components'].items():
            if record['request'] != query(kind, row['date']):
                raise ValueError('Official component endpoint or requested date differs')
            for name, sha in record['source_sha256'].items(): sources.bind(name, sha)
            raw = json.loads(sources.bind(record['raw_path'], record['raw_sha256']).read_text())
            parts[kind] = raw['payload'] if record['record_wrapper'] else raw
        needed = {basket_key(r):r for r in basket_requests(parts['block_basket'],row['date'])}
        if set(row.get('baskets',{})) != set(needed):
            raise ValueError('Every basket requires its exact constituent report')
        details = {}
        for key_id,record in row.get('baskets',{}).items():
            requested = detail_item(row['date'],needed[key_id])
            if record['request'] != requested:
                raise ValueError('Basket constituent request differs from official parent')
            for name,expected in ((record['raw_path'],record['raw_sha256']),
                                  (record['receipt_path'],record['receipt_sha256'])):
                if manifest['source_sha256'].get(name) != expected:
                    raise ValueError('Basket source escapes the sealed evidence inventory')
                sources.bind(name,expected)
            receipt = json.loads((root/record['receipt_path']).read_text())
            if (receipt.get('http_status') != 200 or receipt.get('security_denied') is not False
                    or receipt.get('automatic_redirects_disabled') is not True
                    or receipt.get('redirect_statuses') != []
                    or receipt.get('raw_sha256') != record['raw_sha256']
                    or any(receipt.get(k) != requested[k] for k in ('url','params','date'))):
                raise ValueError('Basket HTTP provenance does not match the exact request')
            # Earlier schema rejection is retained in the receipt. Acceptance is
            # decided offline here by explicit field mapping and parent totals.
            details[key_id] = json.loads((root/record['raw_path']).read_text())
        actual = parse_complete(parts, details, row['date'])[row['stock_id']]
        if actual != row['ordinary']:
            raise ValueError('Official ordinary-session decomposition changed')
        value = OfficialRegularDay(*key, **{f:actual[f] for f in
            ('shares','open_cents','high_cents','low_cents','close_cents','amount_cents')},
            source_sha256=manifest_sha)
        validate_board(value)
        result[key] = value
    return result, dict(sources.used)


def load_completion_odd(root, document):
    """Reuse the same already bound intraday component, with native odd parsing."""
    root = Path(root).resolve()
    _, refs = load_official_completion(root, document)
    doc = json.loads(Path(document).read_text())
    manifest = json.loads((root/doc['manifest_path']).read_text())
    result = {}
    for row in manifest['days'].values():
        record = row['components']['intraday_odd']
        raw = json.loads((root/record['raw_path']).read_text())
        if record['record_wrapper']:
            envelope = raw
        else:
            receipts = [name for name in record['source_sha256'] if '/receipts/' in name]
            if len(receipts) != 1:
                raise ValueError('Official intraday component lacks a unique HTTP receipt')
            receipt = json.loads((root/receipts[0]).read_text())
            if (receipt.get('accepted') is not True or receipt.get('security_denied') is not False
                    or receipt.get('raw_sha256') != record['raw_sha256']):
                raise ValueError('Official intraday receipt conflicts with its payload')
            envelope = dict(schema=1, provider='twse', day=row['date'], payload=raw,
                **{k:receipt[k] for k in ('url','params','retrieved_at','http_status')})
        parsed = parse_odd(envelope, 'twse', row['date'])
        result[row['stock_id'], row['date'], row['market']] = intraday_row(
            parsed[row['stock_id']], row['date'], row['stock_id'], row['market'])
    return result, refs


class CompletedBoardEvidence:
    def __init__(self, original, root, document):
        self.original, self.root, self.document = original, root, document
        self.days, self.refs = load_official_completion(root, document)
        self.queries = original.queries
        self._seen = set()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def get(self, sid, day, market):
        key = sid, day, market.upper()
        if key in self.days:
            if key not in self._seen:
                self.queries.append(dict(stock_id=sid, date=day, market=key[2],
                    source_kind='official_ordinary_daily_aggregate'))
                self._seen.add(key)
            return self.days[key], self.days[key].source_sha256
        try:
            return self.original.get(sid, day, market)
        finally:
            self.refs.update(self.original.refs)

    def audit_day(self, sid, day, market, value, quote):
        key = sid, day, market.upper()
        if key not in self.days:
            return self.original.audit_day(sid, day, market, value, quote)
        rebuilt, refs = load_official_completion(self.root, self.document)
        self.refs.update(refs)
        if value != rebuilt[key] or not isinstance(value, OfficialRegularDay):
            raise ValueError('Official execution aggregate differs from bound primary sources')
        validate_board(value)
        for name in ('open','high','low','close'):
            if not execution.finite(float(quote[name])) or abs(float(quote[name])-getattr(value, name+'_cents')/100)>1e-8:
                raise ValueError('Official ordinary OHLC conflicts with the sealed account quote')
        if not execution.finite(quote['volume']) or quote['volume'] < value.shares:
            raise ValueError('Official ordinary quantity exceeds sealed all-session total')
        return dict(official_daily_sha256=value.source_sha256, official_status=STATUS,
            ordinary_volume_matched=True, ordinary_amount_matched=True,
            tick_sequence_complete=False, own_order_fill_proven=False,
            all_session_volume_bound_verified=True, synthesized_tick_rows=False)


def match_board(value, side, limit_price, quantity, prior_adv, *, fraction=.5, participation=.01):
    if not isinstance(value, OfficialRegularDay):
        return execution.match_range_board(value, side, limit_price, quantity, prior_adv,
            fraction=fraction, participation=participation)
    validate_board(value)
    execution._validate_order(side, limit_price, quantity, fraction, participation, board=True)
    if not execution.finite(prior_adv) or prior_adv <= 0:
        raise ValueError('Invalid prior ADV for range capacity')
    day_cap = value.shares // 100000 * 1000
    adv_cap = int((Decimal(str(prior_adv))*Decimal('.01')/1000).to_integral_value(rounding=ROUND_FLOOR))*1000
    result = execution._range_result(value.shares, value.high_cents/100, value.low_cents/100,
        side, limit_price, quantity, fraction, min(day_cap, adv_cap), board=True)
    result.update(regular_daily_capacity_qty=day_cap, prior_adv_capacity_qty=adv_cap,
        allocations=[], source_window=WINDOW, evidence_status=STATUS)
    return result


def replace_once(code, before, after):
    if code.count(before) != 1:
        raise ValueError('Reviewed official aggregate adaptation site changed')
    return code.replace(before, after)


def execution_mixin():
    if digest(Path(execution.__file__)) != EXECUTION_SHA:
        raise ValueError('Frozen daily-range execution changed')
    code = textwrap.dedent(inspect.getsource(execution.RangeOrders._execute_order))
    code = replace_once(code, "tape.price.lt(bounds['lower']-1e-8).any() or tape.price.gt(bounds['upper']+1e-8).any()",
        "board_prices(tape).lt(bounds['lower']-1e-8).any() or board_prices(tape).gt(bounds['upper']+1e-8).any()")
    code = replace_once(code, 'row.update(ticks_sha256=digest, tape_evidence=evidence,',
        'row.update(**digest_fields(tape,digest), tape_evidence=evidence,')
    namespace = dict(execution.__dict__, match_range_board=match_board,
                     board_prices=board_prices, digest_fields=digest_fields)
    exec(compile(code, execution.__file__, 'exec'), namespace)
    return type('OfficialAggregateExecution', (execution.RangeOrders,),
                {'_execute_order':namespace['_execute_order']})


def official_auditor():
    # Resource audit supplies a private, hash-checked copy; never patch a shared
    # imported auditor or use the execution matcher to certify itself.
    wrapper = resource_auditor()
    original = inspect.getclosurevars(wrapper).nonlocals['original']
    namespace = original.__globals__
    code = inspect.getsource(frozen_audit.audit_range_children)
    code = code.replace("source.loc[source.date.eq(pd.Timestamp(day)) & source.stock_id.eq(sid)]",
                        '_indexed_quote_rows(source, day, sid)')
    code = replace_once(code, "    required = {(key, channel) for key, p in plans.items()\n",
        "    rejected = _verified_resource_rejections(account)\n"
        "    required = {(key, channel) for key, p in plans.items() if key not in rejected\n")
    code = replace_once(code, "            if not p['planned_qty']:\n",
        "            if key in rejected:\n                continue\n            if not p['planned_qty']:\n")
    code = replace_once(code, "if digest != row['ticks_sha256']:",
        "if (digest != row.get('official_daily_sha256' if isinstance(tape, OfficialRegularDay) else 'ticks_sha256')"
        " or isinstance(tape, OfficialRegularDay) and row.get('ticks_sha256') is not None"
        " or not isinstance(tape, OfficialRegularDay) and row.get('official_daily_sha256') is not None):")
    code = code.replace('tape.price.lt(', 'board_prices(tape).lt(').replace('tape.price.gt(', 'board_prices(tape).gt(')
    normal = """            regular = [t for t in tape.itertuples(index=False)
                       if pd.Timedelta('09:00:00') <= t.time < pd.Timedelta('13:34:00') and t.shares > 0]
            volume = sum(int(t.shares) for t in regular)
            high = max(float(t.price) for t in regular) if volume else None
            low = min(float(t.price) for t in regular) if volume else None
"""
    alternative = """            if isinstance(tape, OfficialRegularDay):
                # audit_day just re-read and subtracted all six primary tables.
                # There is no invented print sequence behind these values.
                volume = tape.shares
                high, low = tape.high_cents / 100, tape.low_cents / 100
            else:
""" + textwrap.indent(normal, '    ')
    code = replace_once(code, normal, alternative)
    code = replace_once(code, "row.get('source_window') != '09:00:00<=time<13:34:00_including_delayed_close'",
        "row.get('source_window') != (OFFICIAL_WINDOW if isinstance(tape, OfficialRegularDay) else '09:00:00<=time<13:34:00_including_delayed_close')")
    code = replace_once(code, 'fraction=fraction, capacity=cap, board=True)',
        "fraction=fraction, capacity=cap, board=True, evidence_status=OFFICIAL_STATUS if isinstance(tape, OfficialRegularDay) else 'parent_quality_checked_regular_tape')")
    code = replace_once(code, "'source_hash_verification_required_by_caller', 'live_qualified'):",
        "'source_hash_verification_required_by_caller', 'live_qualified', 'official_daily_sha256',\n"
        "                          'ticks_sha256', 'tape_evidence', 'evidence_status', 'source_window'):")
    namespace.update(OfficialRegularDay=OfficialRegularDay, board_prices=board_prices,
        validate_tape=validate_board, OFFICIAL_STATUS=STATUS, OFFICIAL_WINDOW=WINDOW)
    exec(compile(code, namespace['__file__'], 'exec'), namespace)
    metadata = inspect.getsource(frozen_audit._proxy_metadata)
    metadata = replace_once(metadata, 'capacity, board):',
        "capacity, board, evidence_status='parent_quality_checked_regular_tape'):")
    metadata = replace_once(metadata,
        "('parent_quality_checked_regular_tape' if board else 'official_intraday_daily_table')",
        "(evidence_status if board else 'official_intraday_daily_table')")
    exec(compile(metadata, namespace['__file__'], 'exec'), namespace)
    failure = inspect.getsource(frozen_audit._rebuild_failure)
    failure = failure.replace('tape.price.lt(', 'board_prices(tape).lt(').replace('tape.price.gt(', 'board_prices(tape).gt(')
    failure = failure.replace("source.loc[source.date.eq(pd.Timestamp(day)) & source.stock_id.eq(sid)]",
                              '_indexed_quote_rows(source, day, sid)')
    exec(compile(failure, namespace['__file__'], 'exec'), namespace)
    return wrapper


def install_official_execution(runner, provider, document, root):
    if not hasattr(runner, '_all_entry_indexed_audit'):
        raise ValueError('Install the frozen private audit binding before official evidence')
    provider.board_ticks = CompletedBoardEvidence(provider.board_ticks, root, document)
    provider.ticks = provider.board_ticks
    old_odd = provider.get_odd
    def get_odd(day, sid, market, engine=None):
        key = sid, day, market.upper()
        if key not in provider.board_ticks.days:
            return old_odd(day, sid, market, engine=engine)
        requirement(day, sid, engine)
        values, refs = load_completion_odd(root, document)
        for name, sha in refs.items(): provider._bind(Path(root)/name, sha)
        return deepcopy(values[key])
    provider.get_odd = get_odd
    runner._all_entry_indexed_audit = official_auditor()
    original = runner.engine_types
    # Preserve the rollback wrapper ahead of the replaced executor. Placing a
    # new executor above the finished engine would bypass whole-order rollback.
    orders = type('OfficialRangeGapOrders', (execution.RangeGapOrders, execution_mixin()), {})
    def engine_types(execution_orders, *args, **kwargs):
        if execution_orders is not execution.RangeGapOrders:
            raise ValueError('Unexpected frozen execution layer')
        return original(orders, *args, **kwargs)
    runner.engine_types = engine_types
    return dict(provider.board_ticks.refs)
