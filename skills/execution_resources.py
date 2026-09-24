"""Separate daily cash inflows, slot reuse, and unused order budget reuse."""
import math
from collections import defaultdict

from skills.five_axis_replay import FiveAxisReplay
from skills.execution_stress import StressBenchmark, audit_stress
from skills.million_replay import COMMISSION, SLIPPAGE, MIN_FEE, money


class ResourceDecisions:
    def __init__(self, *args, opening_cash_only=False, lock_slots=False, lock_unused=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.opening_cash_only, self.lock_slots, self.lock_unused = opening_cash_only, lock_slots, lock_unused
        self.resource_plans = []
        self.active_budget = None

    def corporate_day(self, day):
        self.opening_limit = self.cash
        self.opening_remaining = self.cash
        self.locked_unused = 0.
        self.occupied = {sid for sid in self.holdings if sid != '0050'}
        return super().corporate_day(day)

    def _affordable(self, qty, step, price, cash, sid):
        if self.active_budget is not None:
            cash = min(cash, self.active_budget)
        return super()._affordable(qty, step, price, cash, sid)

    def cash_move(self, day, kind, change, **extra):
        if kind == 'buy' and self.active_budget is not None:
            if change > 0 or -change > self.active_budget+.005:
                raise ValueError('Buy exceeds active resource budget')
            self.active_budget = money(self.active_budget+change)
            self.opening_remaining = money(self.opening_remaining+change)
        return super().cash_move(day,kind,change,**extra)

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if side != 'buy' or not (self.opening_cash_only or self.lock_slots or self.lock_unused):
            return super().order(day,sid,side,qty,reason,event_id,signal_date)
        if signal_date is None and self.benchmark and sid == '0050':
            index = self.positions[day]
            if index == 0:
                raise ValueError('Benchmark resource sizing needs a prior market day')
            signal_date = str(self.days[index-1].date())
        if self.active_budget is not None or not signal_date or signal_date >= str(day.date()):
            raise ValueError('Resource order is nested or lacks a prior signal')
        before = sorted(self.occupied)
        price = self.prior(day,sid)
        usable = min(self.cash,self.opening_remaining) if self.opening_cash_only else self.cash
        usable = max(0.,money(usable-self.locked_unused))
        sizing_budget = min(usable,self.previous_nav/(1 if self.benchmark else self.slots))
        # Without U, preserve legacy gap-price affordability; only C limits the
        # account's cash source. Do not inadvertently add a position-spend cap.
        budget = sizing_budget if self.lock_unused else usable
        failure = None
        # Preserve legacy sizing exactly when only slots change.
        if self.opening_cash_only or self.lock_unused:
            allowed = max(0,math.floor((sizing_budget-2*MIN_FEE)/(price*(1+COMMISSION+SLIPPAGE)))) if price else 0
            qty = min(qty,allowed)
        if not qty:
            failure = failure or 'resource_cash_locked'
        elif self.lock_slots and sid != '0050':
            if sid in self.occupied or len(self.occupied) >= self.slots:
                failure = 'resource_slots_locked'
            else:
                self.occupied.add(sid)
        plan = dict(date=str(day.date()),signal_date=signal_date,stock_id=sid,event_id=event_id,
            opening_cash=self.opening_limit,available_before=usable,budget=budget,planned_qty=qty,
            occupied_before=before,locked_unused_before=self.locked_unused)
        if failure:
            self.orders.append(dict(date=str(day.date()),stock_id=sid,name=self.names.get(sid,sid),
                event_id=event_id,signal_date=signal_date,side='buy',channel='event',
                requested_qty=qty,filled_qty=0,reason=reason,failure=failure))
            plan.update(spent=0.,locked_after=self.locked_unused,failure=failure)
            self.resource_plans.append(plan)
            return 0
        self.active_budget = budget
        try:
            filled = super().order(day,sid,side,qty,reason,event_id,signal_date)
            spent = money(budget-self.active_budget)
            if self.lock_unused:
                self.locked_unused = money(self.locked_unused+self.active_budget)
            plan.update(spent=spent,locked_after=self.locked_unused,filled_qty=filled)
            self.resource_plans.append(plan)
            return filled
        finally:
            self.active_budget = None


class ResourceCapacityReplay(ResourceDecisions, FiveAxisReplay):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,arm='capacity',**kwargs)


class ResourceBenchmark(ResourceDecisions, StressBenchmark):
    pass


def audit_resources(account, plans, *, opening_cash_only, lock_slots, lock_unused):
    result = audit_stress(account)
    byday = defaultdict(list)
    for plan in plans:
        if plan['signal_date'] >= plan['date'] or plan['spent'] > plan['budget']+.01:
            raise ValueError('Resource timing/budget audit failed')
        if lock_slots and plan.get('filled_qty',0) and plan['stock_id']!='0050':
            if plan['stock_id'] in plan['occupied_before'] or len(plan['occupied_before']) >= account['settings']['slots']:
                raise ValueError('Filled buy reused a locked slot')
        byday[plan['date']].append(plan)
    for day,items in byday.items():
        buys = [r for r in account['trades'] if r['date']==day and r['side']=='buy']
        if abs(sum(-r['cash_change'] for r in buys)-sum(p['spent'] for p in items))>.02:
            raise ValueError('Resource plans do not reconcile to buy ledger')
        if opening_cash_only and sum(-r['cash_change'] for r in buys)>items[0]['opening_cash']+.01:
            raise ValueError('Daily buys used same-day inflows')
        if lock_unused:
            for before,after in zip(items,items[1:]):
                if after['locked_unused_before'] != before['locked_after']:
                    raise ValueError('Unused budget released before day end')
    result['resource_policy_reconciled'] = True
    return result
