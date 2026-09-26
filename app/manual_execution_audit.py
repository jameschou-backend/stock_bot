"""Read-only sequence reconciliation for manually transcribed Dahu Tou receipts.

No brokerage connection or account mutation. Source digests identify supplied
evidence, not its authenticity. Integer TWD cents and shares only.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re

from skills.contingent_execution import Plan, integer
from skills.manual_security_ids import valid_manual_security_id

TZ = ZoneInfo('Asia/Taipei')


def timestamp(value, session):
    at = datetime.fromisoformat(value)
    if at.tzinfo is None or str(at.astimezone(TZ).date()) != session:
        raise ValueError('回報需含時區，且屬於核對交易日')
    return at


def audit(document):
    if document.get('schema') != 'dahu_manual_receipts_v1':
        raise ValueError('需使用大戶投手動回報核對範本 v1')
    if document.get('example_only') or not document.get('events'):
        raise ValueError('範本或空白紀錄不是成交證據，請整理實際回報後再核對')
    session = document['session']; date.fromisoformat(session)
    plans = {}
    for raw in document['plans']:
        plan = Plan(**raw); plan.validate(session)
        if plan.order_id in plans:
            raise ValueError('事前委託編號重複')
        plans[plan.order_id] = plan
    holdings = dict(document['opening_holdings'])
    rights = dict(document['undelivered_stock_rights'])
    for sid,qty in (list(holdings.items())+list(rights.items())):
        if not valid_manual_security_id(sid):
            raise ValueError('庫存股票代號需四碼或00631L')
        integer(qty,1)
    cash = integer(document['opening_cash_cents'])
    slots = integer(document['slots'],1)
    if len(set(holdings)|set(rights)) > slots:
        raise ValueError('期初持股超過名額限制')
    orders, seen, decisions = {}, set(), []
    available, last_received, last_resource, last_snapshot = 0, None, None, None
    source_hashes = set()
    for event in document['events']:
        identity = event['id']; kind = event['kind']
        fields = {
            'funds_snapshot': {'available_cents','covers_through'},
            'submit': {'order_id'}, 'fill': {'order_id','qty','price_cents','fee_cents','tax_cents'},
            'cancel_request': {'order_id'},
            'cancel_ack': {'order_id','request_id','cumulative_filled_qty','cancelled_qty'},
            'cancel_rejected': {'order_id','request_id'},
        }
        if kind not in fields or set(event) != fields[kind] | {'id','kind','occurred_at','received_at','source_sha256'}:
            raise ValueError('回報欄位不符：'+str(kind))
        if not isinstance(identity,str) or not identity.strip() or identity in seen:
            raise ValueError('回報編號缺漏或重複')
        received = timestamp(event['received_at'],session)
        occurred = timestamp(event['occurred_at'],session)
        if occurred > received or received > datetime.now(TZ) or (last_received and received <= last_received):
            raise ValueError('回報收到時間須嚴格遞增，且不得早於事件時間')
        if not re.fullmatch(r'[0-9a-f]{64}',event.get('source_sha256','')):
            raise ValueError('每筆回報需原始證據 SHA256；不可用報價代替成交')
        source_hashes.add(event['source_sha256'])
        seen.add(identity); last_received = received
        order = orders.get(event.get('order_id'))
        if kind == 'funds_snapshot':
            if event.get('covers_through') != (last_resource[0] if last_resource else None):
                raise ValueError('可用額度快照未涵蓋最新委託／成交／撤單回報')
            if last_resource and occurred <= last_resource[1]:
                raise ValueError('可用額度快照早於最新回報，不能釋放預算')
            reserved = sum(o['reserved'] for o in orders.values())
            supplied = integer(event['available_cents'])
            if supplied > cash-reserved:
                raise ValueError('券商可用額度超過本策略帳本餘額；需核對其他資金或委託')
            available = supplied; last_snapshot = (identity,received)
        elif kind == 'submit':
            oid = event['order_id']; plan = plans.get(oid)
            if plan is None or oid in orders:
                raise ValueError('未對應事前委託，或重複送單')
            if last_resource and occurred <= last_resource[1]:
                raise ValueError('新委託早於依賴回報的收到時間')
            if plan.side == 'buy':
                if last_snapshot and occurred <= last_snapshot[1]:
                    raise ValueError('新買單早於可用額度快照的收到時間')
                occupied = set(holdings) | set(rights) | {o['plan'].stock_id for o in orders.values()
                    if o['plan'].side == 'buy' and o['remaining'] > 0 and not o['terminal']}
                if plan.stock_id not in occupied and len(occupied) >= slots:
                    raise ValueError('舊部位或未終結買单仍占名額，不能補位')
                if not last_snapshot or available < plan.budget_cents:
                    raise ValueError('尚無足夠且已核對的可用買進額度')
                available -= plan.budget_cents
            else:
                pending = sum(o['remaining'] for o in orders.values()
                    if o['plan'].side == 'sell' and o['plan'].stock_id == plan.stock_id and not o['terminal'])
                if pending+plan.qty > holdings.get(plan.stock_id,0):
                    raise ValueError('賣單超過庫存或與未終結賣單重複')
            orders[oid] = dict(plan=plan,remaining=plan.qty,filled=0,
                reserved=plan.budget_cents,sent_at=occurred,pending_cancel=None,terminal=None)
        elif kind in ('fill','cancel_request','cancel_ack','cancel_rejected'):
            if order is None or order['terminal'] or occurred <= order['sent_at']:
                raise ValueError('未知／已終結委託或事件早於送單；需先更正回報')
            plan = order['plan']
            if kind == 'fill':
                qty = integer(event['qty'],1); price = integer(event['price_cents'],1)
                fee = integer(event['fee_cents']); tax = integer(event['tax_cents'])
                if qty > order['remaining'] or (plan.channel=='board' and qty%1000):
                    raise ValueError('成交超過剩餘股數或盤別不符')
                if plan.side == 'buy':
                    paid = qty*price+fee+tax
                    if tax or price > plan.limit_cents or paid > order['reserved']:
                        raise ValueError('買進限價、稅費或預留預算不符')
                    holdings[plan.stock_id] = holdings.get(plan.stock_id,0)+qty
                    cash -= paid; order['reserved'] -= paid
                else:
                    if price < plan.limit_cents or qty > holdings.get(plan.stock_id,0):
                        raise ValueError('賣出限價或庫存不符')
                    cash += qty*price-fee-tax
                    if cash < 0:
                        raise ValueError('賣出費稅造成現金不足，需另行對帳')
                    holdings[plan.stock_id] -= qty
                    if not holdings[plan.stock_id]:
                        del holdings[plan.stock_id]
                order['filled'] += qty; order['remaining'] -= qty
                if not order['remaining']:
                    order['terminal'] = 'filled'; order['reserved'] = 0
                # Sell receipts and released buy budgets do not credit availability.
                available = min(available,max(0,cash-sum(o['reserved'] for o in orders.values())))
            elif kind == 'cancel_request':
                if order['pending_cancel'] is not None:
                    raise ValueError('重複撤單請求')
                order['pending_cancel'] = (identity,received)
            else:
                pending = order['pending_cancel']
                if not pending or event['request_id'] != pending[0] or occurred <= pending[1]:
                    raise ValueError('撤單回報沒有對應請求或時序不符')
                if kind == 'cancel_ack':
                    if (integer(event['cumulative_filled_qty']) != order['filled'] or
                            integer(event['cancelled_qty'],1) != order['remaining']):
                        raise ValueError('撤單累計成交／取消量不符，可能有尚未收到的成交')
                    order['terminal'] = 'cancelled'; order['reserved'] = 0
                order['pending_cancel'] = None
        else:
            raise ValueError('不支援的回報類型：'+str(kind))
        if kind != 'funds_snapshot':
            last_resource = (identity,received)
        decisions.append(dict(id=identity,kind=kind,cash_cents=cash,available_cents=available,
            reserved_cents=sum(o['reserved'] for o in orders.values()),holdings=dict(holdings)))
    return dict(schema='dahu_manual_audit_v1',sequence_consistent=True,
        classification='user_transcribed_not_broker_authenticated',broker_execution_verified=False,
        live_qualified=False,events=len(decisions),source_sha256=sorted(source_hashes),
        document_sha256=hashlib.sha256(json.dumps(document,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        decisions=decisions,open_orders=[oid for oid,o in orders.items() if not o['terminal']],
        note='僅核對提供的回報時序與帳本；未證明回報真實或完整，也不代表券商接受新單。')
