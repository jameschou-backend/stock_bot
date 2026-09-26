"""Research-only, single-day confirmation gate; no broker or market data I/O.

Inputs are trusted, normalized confirmations from a separately validated adapter.
Passing these invariants alone does not authenticate a fill or qualify a backtest.
Money and prices are integer cents, quantity is shares, clocks are microseconds
since midnight. Equal-time prices cannot fill a newly submitted order.
"""
from dataclasses import dataclass
from datetime import date
from skills.manual_security_ids import valid_manual_security_id

OPEN = (9*3600+60)*1_000_000
CUTOFF = (13*3600+25*60)*1_000_000


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Expected integer at or above minimum')
    return value


@dataclass(frozen=True)
class Plan:
    order_id: str
    stock_id: str
    side: str
    channel: str
    qty: int
    limit_cents: int
    budget_cents: int
    signal_date: str

    def validate(self, execution_date):
        if (not self.order_id or not valid_manual_security_id(self.stock_id)
                or self.side not in ('buy', 'sell') or self.channel not in ('board', 'odd')
                or date.fromisoformat(self.signal_date) >= date.fromisoformat(execution_date)):
            raise ValueError('Invalid identity, channel, side or signal date')
        integer(self.qty, 1); integer(self.limit_cents, 1); integer(self.budget_cents)
        if self.channel == 'board' and self.qty % 1000:
            raise ValueError('Board quantity must be a multiple of 1000')
        if self.channel == 'odd' and self.qty >= 1000:
            raise ValueError('One odd order must be under 1000 shares')
        if self.side == 'buy' and self.budget_cents < self.qty*self.limit_cents:
            raise ValueError('Budget must include limit notional and costs')
        if self.side == 'sell' and self.budget_cents:
            raise ValueError('Sell order does not reserve cash')


class ConfirmationGate:
    def __init__(self, execution_date, plans, holdings, available_cents, slots=3):
        date.fromisoformat(execution_date)
        self.plans = tuple(plans)
        for plan in self.plans:
            plan.validate(execution_date)
        if len({p.order_id for p in self.plans}) != len(self.plans):
            raise ValueError('Duplicate order ID')
        self.holdings = dict(holdings)
        for sid, qty in self.holdings.items():
            if not valid_manual_security_id(sid):
                raise ValueError('Invalid holding stock ID')
            integer(qty, 1)
        self.available = integer(available_cents)
        self.slots = integer(slots, 1)
        if len(self.holdings) > slots:
            raise ValueError('Opening holdings exceed slots')
        planned_sells = {}
        for plan in self.plans:
            if plan.side == 'sell':
                planned_sells[plan.stock_id] = planned_sells.get(plan.stock_id, 0)+plan.qty
        if any(qty > self.holdings.get(sid, 0) for sid, qty in planned_sells.items()):
            raise ValueError('Sell plans exceed opening holdings')
        self.orders, self.receivables, self.seen = {}, {}, set()
        self.clock, self.closed = 0, False
        self.events = []

    def _time(self, at):
        integer(at)
        if self.closed or at < self.clock or at >= CUTOFF:
            raise ValueError('Out-of-order or out-of-session event')

    def submit_ready(self, at):
        self._time(at)
        if at < OPEN:
            raise ValueError('Order before start time')
        self.clock = at
        sent = []
        # Submit known exits first. This does not confirm them or credit cash.
        ordered = [p for p in self.plans if p.side == 'sell'] + [p for p in self.plans if p.side == 'buy']
        for plan in ordered:
            if plan.order_id in self.orders:
                continue
            if plan.side == 'buy':
                members = set(self.holdings) | {o['plan'].stock_id for o in self.orders.values()
                    if o['plan'].side == 'buy' and o['remaining']}
                if (plan.stock_id not in members and len(members) >= self.slots
                        or self.available < plan.budget_cents):
                    break
                self.available -= plan.budget_cents
            self.orders[plan.order_id] = dict(plan=plan, sent_at=at, remaining=plan.qty,
                                              reserved=plan.budget_cents)
            sent.append(plan.order_id)
        self.events.append(dict(kind='submit', at=at, order_ids=sent, available=self.available))
        return sent

    def confirm_fill(self, fill_id, order_id, at, qty, price_cents, cash_cents):
        self._time(at)
        if not fill_id or fill_id in self.seen or order_id not in self.orders:
            raise ValueError('Missing/duplicate fill ID or unknown order')
        order = self.orders[order_id]; plan = order['plan']
        integer(qty, 1); integer(price_cents, 1); integer(cash_cents, 1)
        if (at <= order['sent_at'] or qty > order['remaining']
                or plan.channel == 'board' and qty % 1000):
            raise ValueError('Fill precedes order or exceeds remaining/channel quantity')
        gross = qty*price_cents
        if plan.side == 'buy':
            if price_cents > plan.limit_cents or cash_cents < gross or cash_cents > order['reserved']:
                raise ValueError('Buy violates limit, costs or reservation')
        elif (price_cents < plan.limit_cents or cash_cents > gross
                or qty > self.holdings.get(plan.stock_id, 0)):
            raise ValueError('Sell violates limit, costs or holdings')
        # Validate the entire event before mutating balances or deduplication IDs.
        self.clock = at; self.seen.add(fill_id); order['remaining'] -= qty
        if plan.side == 'sell':
            self.holdings[plan.stock_id] -= qty
            if not self.holdings[plan.stock_id]:
                del self.holdings[plan.stock_id]
            self.receivables[fill_id] = cash_cents
        else:
            self.holdings[plan.stock_id] = self.holdings.get(plan.stock_id, 0)+qty
            order['reserved'] -= cash_cents
            if not order['remaining']:
                self.available += order['reserved']; order['reserved'] = 0
        self.events.append(dict(kind='fill', fill_id=fill_id, order_id=order_id, at=at,
                                qty=qty, price_cents=price_cents, cash_cents=cash_cents))

    def confirm_available(self, confirmation_id, fill_id, at, cents):
        self._time(at); integer(cents, 1)
        if (not confirmation_id or confirmation_id in self.seen
                or cents > self.receivables.get(fill_id, 0)):
            raise ValueError('Unbacked or duplicate available-funds confirmation')
        self.clock = at; self.seen.add(confirmation_id)
        self.receivables[fill_id] -= cents; self.available += cents
        self.events.append(dict(kind='available', confirmation_id=confirmation_id,
                                fill_id=fill_id, at=at, cents=cents))

    def close(self):
        if self.closed:
            raise ValueError('Session already closed')
        self.closed = True; self.clock = CUTOFF
        # Day orders expire; no post-cutoff trading is permitted.
        for order in self.orders.values():
            self.available += order['reserved']; order['reserved'] = 0
        self.events.append(dict(kind='close', at=CUTOFF, available=self.available))
