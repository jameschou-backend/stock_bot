"""Transactional cash ledger and plans; this module never submits broker orders."""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal, InvalidOperation
import re
from uuid import uuid4
from sqlalchemy import select
from app.workbench_models import WorkbenchAccount, WorkbenchFill, WorkbenchPlan

D = Decimal


def amount(value, places='0.01'):
    try:
        out = D(str(value))
        if not out.is_finite() or not 0 <= out < D('1e14'):
            raise ValueError('金額必須是有限的非負數，且小於 100 兆')
        return out.quantize(D(places))
    except InvalidOperation:
        raise ValueError('金額格式或範圍錯誤') from None


def validate_stock(stock_id):
    if not re.fullmatch(r'\d{4}', stock_id):
        raise ValueError('請輸入四碼股票代號')


def account_row(session, account_id, lock=False):
    if account_id not in ('paper', 'real'):
        raise ValueError('帳本必須為 paper 或 real')
    stmt = select(WorkbenchAccount).where(WorkbenchAccount.account_id == account_id)
    if lock:
        stmt = stmt.with_for_update()
    row = session.execute(stmt).scalar_one_or_none()
    if row is None:
        raise ValueError('請先設定這份帳本的期初現金')
    return row


def initialize_account(session, account_id, initial_cash):
    if account_id not in ('paper', 'real'):
        raise ValueError('帳本必須為 paper 或 real')
    if session.get(WorkbenchAccount, account_id):
        raise ValueError('帳本已建立，期初現金不可覆寫')
    cash = amount(initial_cash)
    if not 0 < cash <= D('1e10'):
        raise ValueError('期初現金必須大於零，且不超過 100 億')
    session.add(WorkbenchAccount(account_id=account_id, initial_cash=cash))
    session.flush()


def ledger_state(initial_cash, fills, quotes=None):
    cash, realized, total_fees = D(str(initial_cash)), D(0), D(0)
    holdings = {}
    for fill in fills:
        sid = fill.stock_id
        position = holdings.setdefault(sid, {'qty': 0, 'basis': D(0)})
        value = D(str(fill.price)) * fill.qty
        costs = D(str(fill.fee)) + D(str(fill.tax))
        total_fees += costs
        if fill.side == 'buy':
            cash -= value + costs
            position['qty'] += fill.qty
            position['basis'] += value + costs
        else:
            if fill.qty > position['qty']:
                raise ValueError('成交紀錄包含超賣，請核對帳本')
            basis = position['basis'] * fill.qty / position['qty']
            realized += value - costs - basis
            position['qty'] -= fill.qty
            position['basis'] -= basis
            cash += value - costs
        if cash < 0:
            raise ValueError('可用現金不足，現金帳本不支援融資')
    positions, missing = [], []
    market_value, unrealized = D(0), D(0)
    for sid, position in holdings.items():
        if not position['qty']:
            continue
        quote = (quotes or {}).get(sid)
        mv = None if quote is None else D(str(quote['close'])) * position['qty']
        pnl = None if mv is None else mv - position['basis']
        if mv is None:
            missing.append(sid)
        else:
            market_value += mv
            unrealized += pnl
        positions.append({'stock_id': sid, 'qty': position['qty'],
                          'cost_basis': float(position['basis']),
                          'average_cost': float(position['basis'] / position['qty']),
                          'price': None if quote is None else quote['close'],
                          'price_date': None if quote is None else str(quote['date']),
                          'unrealized_pnl': None if pnl is None else float(pnl)})
    return {'cash': float(cash), 'realized_pnl': float(realized), 'fees_and_tax': float(total_fees),
            'market_value': None if missing else float(market_value),
            'unrealized_pnl': None if missing else float(unrealized),
            'net_pnl': None if missing else float(realized + unrealized),
            'equity': None if missing else float(cash + market_value),
            'positions': positions, 'missing_quotes': missing}


def fills_for(session, account_id):
    return list(session.execute(select(WorkbenchFill).where(WorkbenchFill.account_id == account_id)
        .order_by(WorkbenchFill.sequence_no)).scalars())


def plans_for(session, account_id):
    return list(session.execute(select(WorkbenchPlan).where(WorkbenchPlan.account_id == account_id)
        .order_by(WorkbenchPlan.created_at.desc())).scalars())


def reservation(plan):
    qty = plan.qty - plan.filled_qty
    if plan.status != 'open' or qty <= 0:
        return D(0)
    value = D(str(plan.entry_price)) * qty
    return value + max(D(20), value * D('.001425'))


def preview_plan(cash, equity, entry, stop, risk_pct, position_pct, lot_size=1, current_exposure=0):
    entry, stop = amount(entry, '.0001'), amount(stop, '.0001')
    if not 0 < stop < entry:
        raise ValueError('停損價必須大於零且低於預計買價')
    if not 0 < risk_pct <= 5 or not 0 < position_pct <= 100 or lot_size not in (1, 1000):
        raise ValueError('風險比例需為 0–5%，部位比例為 0–100%，單位為 1 或 1000 股')
    equity, cash = amount(equity), amount(cash)
    # Reserve both minimum commissions and a 0.6% round-trip slippage stress.
    # Gaps can still exceed this estimate; this is a sizing budget, not an exit guarantee.
    risk_per_share = entry - stop + entry * D('.01185')
    risk_budget = equity * D(str(risk_pct)) / 100
    position_budget=max(D(0),equity * D(str(position_pct)) / 100-amount(current_exposure))
    qty = int(min(D(10000000),equity * D(str(risk_pct)) / 100 / risk_per_share,
                  position_budget / entry,
                  max(D(0), cash - 20) / (entry * D('1.001425'))))
    qty = qty // lot_size * lot_size
    def loss(n):
        return ((entry-stop) * n + max(D(20),entry*n*D('.001425'))
                + max(D(20),stop*n*D('.001425')) + stop*n*D('.003') + entry*n*D('.006'))
    while qty > 0 and loss(qty) > risk_budget:
        qty -= lot_size
    return {'qty': qty, 'estimated_cash': float(entry * qty + max(D(20), entry * qty * D('.001425'))) if qty else 0,
            'estimated_loss': float(loss(qty)) if qty else 0, 'entry_price': float(entry), 'stop_price': float(stop)}


def create_plan(session, account_id, stock_id, entry_price, stop_price, qty, reason=''):
    validate_stock(stock_id)
    account = account_row(session, account_id, lock=True)
    entry, stop = amount(entry_price, '.0001'), amount(stop_price, '.0001')
    if not 0 < qty <= 10000000 or int(qty) != qty or not 0 < stop < entry:
        raise ValueError('數量需為正整數，停損價需低於買價')
    if len(reason) > 500:
        raise ValueError('計畫說明最多 500 字')
    state = ledger_state(account.initial_cash, fills_for(session, account_id))
    reserved = sum((reservation(p) for p in plans_for(session, account_id)), D(0))
    plan = WorkbenchPlan(plan_id=uuid4().hex, account_id=account_id, stock_id=stock_id,
                         entry_price=entry, stop_price=stop, qty=qty, filled_qty=0,
                         status='open', reason=reason)
    if reservation(plan) + reserved > D(str(state['cash'])):
        raise ValueError('可用現金不足，其他未完成計畫已預留資金')
    session.add(plan)
    session.flush()
    return plan.plan_id


def cancel_plan(session, account_id, plan_id):
    account_row(session, account_id, lock=True)
    plan = session.get(WorkbenchPlan, plan_id)
    if not plan or plan.account_id != account_id:
        raise ValueError('找不到這份計畫')
    if plan.status == 'open':
        plan.status = 'cancelled'
    session.flush()


def record_fill(session, account_id, fill_id, stock_id, side, qty, price, fee, tax,
                executed_at, plan_id=None):
    validate_stock(stock_id)
    if not re.fullmatch(r'[0-9a-f]{32}', fill_id):
        raise ValueError('成交識別碼格式錯誤')
    if side not in ('buy', 'sell') or not 0 < qty <= 10000000 or int(qty) != qty:
        raise ValueError('成交方向或數量錯誤')
    price, fee, tax = amount(price, '.0001'), amount(fee), amount(tax)
    if price <= 0:
        raise ValueError('成交價必須大於零')
    if executed_at.tzinfo is not None:
        from datetime import timezone
        executed_at = executed_at.astimezone(timezone.utc).replace(tzinfo=None)
    if executed_at > datetime.utcnow():
        raise ValueError('成交時間不可在未來')
    account = account_row(session, account_id, lock=True)
    values = dict(account_id=account_id, stock_id=stock_id, side=side, qty=qty,
                  price=price, fee=fee, tax=tax, executed_at=executed_at, plan_id=plan_id)
    existing = session.get(WorkbenchFill, fill_id)
    if existing:
        if any(getattr(existing, k) != v for k, v in values.items()):
            raise ValueError('相同成交識別碼對應不同內容')
        return fill_id
    fills = fills_for(session, account_id)
    if fills and executed_at < fills[-1].executed_at:
        raise ValueError('請依成交時間順序記帳；此筆早於現有最後一筆')
    plan = None
    if plan_id:
        plan = session.get(WorkbenchPlan, plan_id)
        if not plan or plan.account_id != account_id or plan.stock_id != stock_id or side != 'buy':
            raise ValueError('成交與計畫不符')
        if plan.status != 'open' or qty > plan.qty - plan.filled_qty:
            raise ValueError('成交數量超過計畫剩餘數量，或計畫已結束')
    fill = WorkbenchFill(fill_id=fill_id, sequence_no=len(fills)+1, **values)
    # Validate the entire cash/position transition before mutating anything.
    ledger_state(account.initial_cash, [*fills, fill])
    if plan:
        plan.filled_qty += qty
        if plan.filled_qty == plan.qty:
            plan.status = 'filled'
    session.add(fill)
    session.flush()
    return fill_id
