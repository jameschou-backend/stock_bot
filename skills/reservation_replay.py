"""One decision-protocol contrast; retain the sealed daily/odd execution model.

This does not qualify historical intraday execution. No missing odd evidence
is replaced with board trades, and no old research module is changed.
"""
import math
from collections import defaultdict

from skills.five_axis_replay import FiveAxisReplay
from skills.execution_stress import StressBenchmark, audit_stress
from skills.million_replay import COMMISSION, SLIPPAGE, MIN_FEE, MIN_IDLE_BUY, money


class ReserveDecisions:
    def __init__(self, *args, reserve_before_open=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.reserve_before_open = reserve_before_open
        self.plans, self.reservations = [], {}
        self.active_reservation = None
        self.benchmark_attempted = False

    def corporate_day(self, day):
        opening_cash = self.cash
        opening_members = {s for s in self.holdings if s != '0050'}
        income = super().corporate_day(day)
        if not self.reserve_before_open:
            return income
        self.reservations = {}
        self.benchmark_attempted = False
        # No same-day sale proceeds, dividend payments or rejected-order funds.
        available = min(opening_cash, self.cash)
        previous = self.days[self.positions[day]-1]
        candidates = self.events.get(day, [])
        if self.benchmark:
            candidates = [dict(event_id='benchmark', members=['0050'],
                               signal_date=str(previous.date()))]
        chosen = []
        for event in candidates:
            sid, identity = event['members'][0], event['event_id']
            price = self.prior(day, sid)
            eligible = self.benchmark or (sid not in opening_members
                and len(opening_members) < self.slots
                and math.isfinite(float(self.amount20.at[day, sid]))
                and self.amount20.at[day, sid] >= 50e6)
            budget = min(available, self.previous_nav / (1 if self.benchmark else self.slots))
            if self.benchmark and budget < MIN_IDLE_BUY:
                eligible = False
            # Same legacy sizing formula and single-share precision, even under
            # combined stress. Only when cash/slots are decided changes.
            qty = max(0, math.floor((budget-2*MIN_FEE)/(price*(1+COMMISSION+SLIPPAGE)))) if eligible and price else 0
            plan = dict(date=str(day.date()), signal_date=event['signal_date'],
                        stock_id=sid, event_id=identity, planned_qty=qty,
                        sizing_price=price, reserved_cash=budget if qty else 0.)
            if qty:
                available = money(available-budget)
                opening_members.add(sid)
                self.reservations[identity] = dict(plan, remaining=budget)
                chosen.append(event)
            else:
                plan['rejection'] = 'opening_cash_slots_or_legacy_eligibility'
            self.plans.append(plan)
        if not self.benchmark:
            self.events[day] = chosen
        return income

    def _affordable(self, qty, step, price, cash, sid):
        if self.active_reservation is not None:
            cash = min(cash, self.reservations[self.active_reservation]['remaining'])
        return super()._affordable(qty, step, price, cash, sid)

    def cash_move(self, day, kind, change, **extra):
        if kind == 'buy' and self.active_reservation is not None:
            reserve = self.reservations[self.active_reservation]
            if change > 0 or -change > reserve['remaining']+.005:
                raise ValueError('Buy exceeds prior reserved cash')
            reserve['remaining'] = money(reserve['remaining']+change)
        return super().cash_move(day, kind, change, **extra)

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if not self.reserve_before_open or side != 'buy':
            return super().order(day, sid, side, qty, reason, event_id, signal_date)
        plan = self.reservations[event_id]
        if plan['stock_id'] != sid or self.active_reservation is not None:
            raise ValueError('Reservation/order identity mismatch')
        self.active_reservation = event_id
        try:
            return super().order(day, sid, side, plan['planned_qty'], reason,
                                 event_id, plan['signal_date'])
        finally:
            self.active_reservation = None

    def buy_etf(self, day, reason):
        if not self.reserve_before_open or not self.benchmark:
            return super().buy_etf(day, reason)
        if self.benchmark_attempted or 'benchmark' not in self.reservations:
            return
        self.benchmark_attempted = True
        self.holdings.setdefault('0050', dict(qty=0, event_id='benchmark', due_index=None))
        self.corporate.prepare('0050')
        plan = self.reservations['benchmark']
        self.order(day, '0050', 'buy', plan['planned_qty'], reason, 'benchmark', plan['signal_date'])


class ReservedCapacityReplay(ReserveDecisions, FiveAxisReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, arm='capacity', **kwargs)


class ReservedBenchmark(ReserveDecisions, StressBenchmark):
    pass


def audit_reservations(account, plans):
    result = audit_stress(account)
    byday, spent = defaultdict(list), defaultdict(float)
    for plan in plans:
        if plan['signal_date'] >= plan['date']:
            raise ValueError('Noncausal reservation')
        byday[plan['date']].append(plan)
    keys = {(p['date'], p['event_id']): p for p in plans if p['planned_qty']}
    for trade in account['trades']:
        if trade['side'] == 'buy':
            key = (trade['date'], trade['event_id'])
            if key not in keys:
                raise ValueError('Buy has no prior reservation')
            spent[key] += -trade['cash_change']
    cash = account['settings']['initial_cash']
    for day in account['daily']:
        if sum(p['reserved_cash'] for p in byday[day['date']]) > cash+.01:
            raise ValueError('Reservations use later cash')
        cash = day['cash']
    for key, amount in spent.items():
        if amount > keys[key]['reserved_cash']+.01:
            raise ValueError('Buy exceeds frozen budget')
    for key, plan in keys.items():
        orders = [o for o in account['orders'] if (o['date'], o['event_id']) == key and o['side']=='buy']
        if orders and sum(o['requested_qty'] for o in orders) != plan['planned_qty']:
            raise ValueError('Order quantity differs from prior plan')
    result['precommitted_budget_and_quantity'] = True
    return result
