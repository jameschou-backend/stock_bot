"""Append-only prospective evidence, separate from real fills and old paper NAV.

SQLite transactions serialize writers. Hashes detect edits, not malicious DB
administrators. No market snapshot is ever promoted to an actual fill.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT/'.cache/forward-validation/journal.sqlite3'
RULES = {
    'version': 'leader-forward-v1', 'initial_cash': 1000000, 'slots': 3,
    'selection': 'fixed leader_now, ON on original signal date, entry next market session',
    'signal_lag_sessions': 1, 'holding_sessions': 63, 'stop_loss': .12,
    'allocation': 'prior_close_nav_divided_by_3', 'idle_asset': '0050',
    'order_type': 'limit_day', 'entry_limit': 'signal_day_raw_close_no_intraday_repricing', 'buy_retries_next_day': False,
    'exit_retry': 'remaining_quantity_next_open_session',
    'commission': .001425, 'minimum_commission': 20, 'stock_sell_tax': .003,
    'etf_sell_tax': .001, 'snapshot_max_age_seconds': 30,
    'fill_policy': 'record_evidence_only_no_automatic_fills',
    'quote_volume_unit': 'provider_unspecified_do_not_assume_shares',
    'odd_lot_quote_source': 'not_connected', 'live_qualified': False,
}
CODE = ('app/forward_journal.py', 'app/forward_service.py', 'scripts/prepare_forward_signals.py',
        'scripts/forward_validation.py', 'skills/diffusion_signals.py',
        'skills/regime_state.py', 'scripts/prepare_million_signals.py')


def now():
    return datetime.now(timezone.utc)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None: raise ValueError('Timestamp must include timezone')
    return result.astimezone(timezone.utc)


@contextmanager
def connection(path=DEFAULT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=5)
    try:
        con.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, event_key TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, recorded_at TEXT NOT NULL, body TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL)')
        con.execute('BEGIN IMMEDIATE')
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def read_events(con):
    out, previous = [], ''
    for seq, key, kind, at, body, prev, sha in con.execute('SELECT * FROM events ORDER BY seq'):
        row = dict(seq=seq, event_key=key, kind=kind, recorded_at=at,
                   body=json.loads(body), previous_hash=prev)
        if prev != previous or digest(row) != sha or seq != len(out)+1:
            raise ValueError('Forward journal integrity check failed')
        out.append(row | {'hash':sha})
        previous = sha
    return out


def append(con, key, kind, body, clock=now):
    rows = read_events(con)
    existing = next((r for r in rows if r['event_key'] == key), None)
    if existing:
        if existing['kind'] != kind or existing['body'] != body:
            raise ValueError('Frozen event cannot be overwritten; append a new observation')
        return existing
    row = dict(seq=len(rows)+1, event_key=key, kind=kind, recorded_at=clock().isoformat(),
               body=body, previous_hash=rows[-1]['hash'] if rows else '')
    sha = digest(row)
    con.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',
                (row['seq'], key, kind, row['recorded_at'], encode(body), row['previous_hash'], sha))
    return row | {'hash':sha}


def initialize(con, clock=now):
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in CODE}
    payload = dict(rules=RULES, code_sha256=hashes)
    rows = read_events(con)
    if rows:
        first = rows[0]
        if first['kind'] != 'rules' or first['body'] != payload:
            raise ValueError('Frozen rules/code changed. Start a separate journal; keep this evidence.')
        return first
    return append(con, 'rules', 'rules', payload, clock)


def freeze(con, status, signals, source_hash, clock=now):
    """Only same-day signals written before their next entry can be prospective."""
    initialize(con, clock)
    local = clock().astimezone(ZoneInfo('Asia/Taipei'))
    today = str(local.date())
    reasons = list(status.get('problems', []))
    if not status.get('data_ready'):
        reasons.append('market_data_not_ready')
    if signals.get('strategy') != RULES['selection'] or signals.get('source_kind') != 'original_rule_forward_extension':
        reasons.append('original_strategy_source_missing')
    if signals.get('signal_end') != today or status.get('price_date') != today:
        reasons.append('signal_or_price_not_today')
    if local.hour < 18:
        reasons.append('daily_source_publication_window_not_finished')
    entries = [e for e in signals.get('entries', []) if e.get('signal_date') == today]
    if any(e.get('entry_date', '') <= today or e.get('entry_date') != signals.get('next_session') for e in entries):
        reasons.append('entry_must_be_a_future_session')
    # Save failed attempts too. They are gaps, not zero-return/no-signal days.
    body = dict(signal_date=today, source_sha256=source_hash, data_status=status,
                candidates=entries, eligible=not reasons, reasons=sorted(set(reasons)),
                qualification='prospective_signal' if not reasons else 'blocked_observation')
    key = ('signal:'+today) if not reasons else ('blocked:'+today+':'+digest(body))
    return append(con, key, 'signal' if not reasons else 'blocked', body, clock)


def number(value):
    try: v = Decimal(str(value))
    except Exception: raise ValueError('Invalid numeric value') from None
    if not v.is_finite() or v <= 0: raise ValueError('Value must be positive and finite')
    return v


def order(con, signal_hash, stock_id, side, channel, qty, limit_price, entry_date, clock=now):
    initialize(con, clock)
    rows = read_events(con)
    signal = next((r for r in rows if r['hash'] == signal_hash and r['kind'] == 'signal'), None)
    if not signal or not signal['body']['eligible']:
        raise ValueError('A prospective frozen original-strategy signal is required')
    if side != 'buy':
        raise ValueError('Exit orders require a reconciled position ledger; currently blocked')
    entries = [e for e in signal['body']['candidates'] if e.get('members') == [stock_id] and e.get('entry_date') == entry_date]
    if not entries or not re.fullmatch(r'\d{4}', stock_id):
        raise ValueError('Stock is not an eligible frozen candidate')
    local = clock().astimezone(ZoneInfo('Asia/Taipei'))
    if str(local.date()) > entry_date or (str(local.date()) == entry_date and local.hour >= 9):
        raise ValueError('Plan must be recorded before entry session opens; no backdated orders')
    if type(qty) is not int or qty <= 0 or channel not in ('board', 'odd'):
        raise ValueError('Invalid order quantity/channel')
    if channel == 'board' and qty % 1000 or channel == 'odd' and qty >= 1000:
        raise ValueError('Board lot must be 1000 multiples; odd lot must be 1..999 shares')
    price = number(limit_price)
    value = price*qty
    fee = max(Decimal(20), (value*Decimal('.001425')).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    if value+fee > Decimal(RULES['initial_cash'])/3:
        raise ValueError('Order exceeds one-third starting capital; later NAV allocation is not yet enabled')
    key = f'order:{signal_hash}:{stock_id}:{channel}'
    body = dict(signal_hash=signal_hash, stock_id=stock_id, side=side, channel=channel,
                qty=qty, limit_price=str(price), entry_date=entry_date,
                reserved_cash=str(value+fee), status='planned', filled_qty=0)
    existing = next((r for r in rows if r['event_key'] == key), None)
    if existing: return append(con, key, 'order', body, clock)
    cancelled = {r['body']['order_hash'] for r in rows if r['kind'] == 'cancel'}
    active = [r['body'] for r in rows if r['kind'] == 'order' and r['hash'] not in cancelled]
    if len({o['stock_id'] for o in active} | {stock_id}) > RULES['slots']:
        raise ValueError('Three stock slots are already reserved')
    if sum(Decimal(o['reserved_cash']) for o in active if o['stock_id']==stock_id)+value+fee > Decimal(RULES['initial_cash'])/3:
        raise ValueError('Combined board/odd orders exceed the stock budget')
    if sum(Decimal(o['reserved_cash']) for o in active)+value+fee > RULES['initial_cash']:
        raise ValueError('Insufficient unreserved paper cash')
    return append(con, key, 'order', body, clock)


def cancel(con, order_hash, reason, clock=now):
    initialize(con, clock)
    if not reason.strip(): raise ValueError('Cancellation reason required')
    if not any(r['hash'] == order_hash and r['kind'] == 'order' for r in read_events(con)):
        raise ValueError('Unknown order')
    return append(con, 'cancel:'+order_hash, 'cancel', dict(order_hash=order_hash, reason=reason), clock)


def snapshot(con, frame, clock=now):
    initialize(con, clock)
    required = {'stock_id', 'date', 'buy_price', 'buy_volume', 'sell_price', 'sell_volume'}
    if not required.issubset(frame.columns): raise ValueError('Snapshot missing required quote fields')
    if frame.stock_id.duplicated().any(): raise ValueError('Duplicate snapshot stock')
    received = clock()
    rows = []
    for r in frame.to_dict('records'):
        if not re.fullmatch(r'\d{4}', str(r['stock_id'])): continue
        # Provider labels date as trade time, not quote time. Never certify depth freshness.
        trade_time = datetime.fromisoformat(str(r['date']))
        if trade_time.tzinfo is None: trade_time = trade_time.replace(tzinfo=ZoneInfo('Asia/Taipei'))
        age = (received-trade_time).total_seconds()
        rows.append(dict(stock_id=str(r['stock_id']), source_trade_at=trade_time.isoformat(),
            bid=r['buy_price'], bid_volume_provider_units=r['buy_volume'],
            ask=r['sell_price'], ask_volume_provider_units=r['sell_volume'],
            trade_age_seconds=age, stale_trade=not 0 <= age <= RULES['snapshot_max_age_seconds'],
            depth_time_verified=False, odd_lot_compatible=False))
    body = dict(source='FinMind/taiwan_stock_tick_snapshot', retrieved_at=frame.attrs.get('retrieved_at'),
                cache_hit=frame.attrs.get('cache_hit'), quotes=rows, fill_evidence=False)
    return append(con, 'quote:'+digest(body), 'quote', body, clock)


def summary(path=DEFAULT_PATH):
    with connection(path) as con:
        rows = read_events(con)
    return dict(events=len(rows), rules_frozen=bool(rows), live_qualified=False,
        prospective_days=sum(r['kind']=='signal' for r in rows),
        blocked_attempts=sum(r['kind']=='blocked' for r in rows),
        orders=sum(r['kind']=='order' for r in rows), confirmed_fills=sum(r['kind']=='fill' for r in rows),
        latest=rows[-1] if rows else None, rows=rows,
        limitations=['尚未接入零股即時報價及成交回報', '沒有成交證據，不計算策略報酬',
                     '0050買入、持倉出場、公司行動與複利帳本尚未接入此驗證流程'])


def record_fill(con, order_hash, qty, price, fee, tax, executed_at, evidence, clock=now):
    """Record supplied paper-execution evidence; never infer fills from quotes."""
    initialize(con, clock)
    rows = read_events(con)
    source = next((r for r in rows if r['kind']=='order' and r['hash']==order_hash), None)
    if source is None: raise ValueError('Unknown paper order')
    body = source['body']
    if type(qty) is not int or qty<=0: raise ValueError('Positive integer fill quantity required')
    if body['channel']=='board' and qty%1000: raise ValueError('Board fill requires whole lots')
    at = timestamp(executed_at)
    if at < timestamp(source['recorded_at']) or at > clock(): raise ValueError('Fill time must follow order and cannot be in the future')
    if str(at.astimezone(ZoneInfo('Asia/Taipei')).date()) != body['entry_date']:
        raise ValueError('Day order cannot fill on another date')
    if not isinstance(evidence,dict) or evidence.get('source')!='paper_execution_report' or not evidence.get('report_id'):
        raise ValueError('Paper execution report with unique report_id is required; quotes are not fills')
    px=number(price)
    if body['side']=='buy' and px>Decimal(body['limit_price']): raise ValueError('Fill exceeds limit')
    costs=[Decimal(str(v)) for v in (fee,tax)]
    if any(not v.is_finite() or v<0 for v in costs): raise ValueError('Invalid fees/tax')
    if body['side']=='buy' and costs[1]!=0: raise ValueError('Buy transaction must not include sell tax')
    key='fill:'+str(evidence['report_id'])
    payload=dict(order_hash=order_hash, stock_id=body['stock_id'], side=body['side'], channel=body['channel'],
        qty=qty, price=str(px), fee=str(costs[0]), tax=str(costs[1]), executed_at=at.isoformat(),
        evidence=evidence, evidence_authenticity='user_supplied_not_broker_authenticated')
    if any(r['event_key']==key for r in rows): return append(con,key,'fill',payload,clock)
    cancelled=[r for r in rows if r['kind']=='cancel' and r['body']['order_hash']==order_hash]
    if cancelled: raise ValueError('Cancelled order requires reconciliation before accepting a late report')
    filled=sum(r['body']['qty'] for r in rows if r['kind']=='fill' and r['body']['order_hash']==order_hash)
    if filled+qty>body['qty']: raise ValueError('Fill exceeds remaining quantity')
    cash=Decimal(RULES['initial_cash'])
    for r in rows:
        if r['kind']=='fill':
            f=r['body'];cash-=Decimal(f['price'])*f['qty']+Decimal(f['fee'])+Decimal(f['tax'])
    if cash<px*qty+sum(costs): raise ValueError('Paper account cannot overdraw')
    return append(con,key,'fill',payload,clock)
