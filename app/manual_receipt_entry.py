"""Human units -> existing read-only receipt contract; never broker I/O."""
from copy import deepcopy
from datetime import date, time
from decimal import Decimal, InvalidOperation
import hashlib
import re

from app.manual_execution_audit import audit
from skills.contingent_execution import Plan, integer


def cents(value):
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('金額請填數字，單位為元') from exc
    if not amount.is_finite() or amount < 0 or amount*100 != (amount*100).to_integral_value():
        raise ValueError('金額須為非負數，最多小數兩位')
    return int(amount*100)


def shares(value, minimum=0):
    if not re.fullmatch(r'[0-9]+', str(value).strip()):
        raise ValueError('股數須為整數，單位是股')
    return integer(int(str(value).strip()), minimum)


def holdings(rows):
    result = {}
    for row in rows:
        sid = str(row.get('股票代號', '')).strip()
        if not sid and str(row.get('股數', '')).strip() in ('', '0', 'None'):
            continue
        if not re.fullmatch(r'[0-9]{4}', sid) or sid in result:
            raise ValueError('庫存代號需四碼且不可重複')
        result[sid] = shares(row['股數'], 1)
    return result


def new_document(session, opening_cash, slots, holding_rows, rights_rows, plan_rows):
    date.fromisoformat(session)
    plans = []
    for row in plan_rows:
        if not str(row.get('委託代號', '')).strip():
            raise ValueError('每筆計畫需填委託代號；請刪除未使用的空白列')
        side = {'買進': 'buy', '賣出': 'sell'}.get(row['買賣'])
        channel = {'整股': 'board', '零股': 'odd'}.get(row['盤別'])
        plan = Plan(order_id=str(row['委託代號']).strip(), stock_id=str(row['股票代號']).strip(),
            side=side, channel=channel, qty=shares(row['股數'], 1),
            limit_cents=cents(row['限價（元）']), budget_cents=cents(row['預算（元）']),
            signal_date=str(row['訊號日']))
        plan.validate(session)
        plans.append(vars(plan))
    if not plans or len({p['order_id'] for p in plans}) != len(plans):
        raise ValueError('至少需一筆計畫，委託代號不可重複')
    held, rights = holdings(holding_rows), holdings(rights_rows)
    integer(slots, 1)
    if len(set(held) | set(rights)) > slots:
        raise ValueError('期初股票與待交付股票已超過持股名額')
    return dict(schema='dahu_manual_receipts_v1', session=session, slots=slots,
        opening_cash_cents=cents(opening_cash), opening_holdings=held,
        undelivered_stock_rights=rights, plans=plans, events=[])


def append_event(document, kind, event_id, occurred, received, evidence, fields):
    if not isinstance(evidence, bytes) or not evidence or len(evidence) > 10_000_000:
        raise ValueError('每筆回報需附原始證據檔，大小須為1 byte至10MB')
    def stamp(clock):
        parsed = time.fromisoformat(clock)
        if parsed.tzinfo is not None:
            raise ValueError('請輸入台北時間，不需填時區')
        return document['session']+'T'+parsed.isoformat()+'+08:00'
    common = dict(id=event_id.strip(), kind=kind, occurred_at=stamp(occurred),
        received_at=stamp(received), source_sha256=hashlib.sha256(evidence).hexdigest())
    if set(fields) & set(common):
        raise ValueError('不可覆蓋回報基本欄位')
    candidate = deepcopy(document)
    candidate['events'].append(dict(**common, **fields))
    # Every addition must reconcile the entire prefix. Failure leaves draft intact.
    audit(candidate)
    return candidate
