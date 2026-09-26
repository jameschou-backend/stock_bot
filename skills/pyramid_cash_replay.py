"""One predeclared addition per existing cohort, using only prior-day information."""
from copy import deepcopy
import math

from skills.million_replay import money
from skills.support_risk_replay import SupportRiskReplay


def addition_capacity(cost, price, ratio, held, nav, available, sid, maximum=None):
    budget = max(0., min(available, nav*.1, nav/5-held*price)) if price else 0.
    if not price or not ratio or not 0 < ratio < 1:
        return dict(budget=budget, qty=0, planned_loss=None)
    ceiling = int(budget/price)//1000*1000
    if maximum is not None:
        ceiling = min(ceiling, maximum)
    for qty in range(ceiling, 0, -1000):
        paid = -cost(price, qty, 'buy', sid)['cash_change']
        loss = money(held*price+paid-cost(price*ratio, held+qty, 'sell', sid)['cash_change'])
        if paid <= budget+1e-8 and loss <= nav*.02+1e-8:
            return dict(budget=budget, qty=qty, planned_loss=loss)
    return dict(budget=budget, qty=0, planned_loss=None)


class PyramidCashReplay(SupportRiskReplay):
    def __init__(self, *args, pyramid_enabled, **kwargs):
        if type(pyramid_enabled) is not bool or kwargs.get('technical_mode') != 'support_risk2':
            raise ValueError('Addition experiment requires explicit switch and support_risk2 base')
        super().__init__(*args, **kwargs)
        self.pyramid_enabled = pyramid_enabled
        self.pyramid_pending, self.pyramid_done = {}, set()
        self.pyramid_decisions = []

    def execute_addition(self, day, sid, identity, signal, capacity):
        if self.active_budget is not None or self.residual_spend_left is not None:
            raise ValueError('Addition cannot nest in another cash reservation')
        budget, qty = capacity['budget'], capacity['qty']
        available = max(0., money(min(self.cash, self.opening_remaining)-self.locked_unused))
        plan = dict(date=str(day.date()), signal_date=signal['signal_date'], stock_id=sid,
            event_id=identity, opening_cash=self.opening_limit, available_before=available,
            budget=budget, planned_qty=qty, occupied_before=sorted(self.entry_slot_members()),
            locked_unused_before=self.locked_unused, kind='pyramid_add')
        first_decision, first_trade = len(self.board_decisions), len(self.trades)
        self.active_budget = budget
        try:
            # This is an existing position, not a new slot. Keep the original
            # identity, price-limit, participation, rounding and cash execution layers.
            filled = self._execute_order(day, sid, 'buy', qty, 'pyramid_add', identity, signal['signal_date'])
            decisions, trades = self.board_decisions[first_decision:], self.trades[first_trade:]
            if len(decisions) != 1 or sum(t['qty'] for t in trades) != filled:
                raise ValueError('Addition lacks one exact final board execution decision')
            decisions[0].update(filled_qty=filled, trade_sequences=[t['sequence'] for t in trades])
            self.locked_unused = money(self.locked_unused+self.active_budget)
            plan.update(spent=money(budget-self.active_budget), locked_after=self.locked_unused, filled_qty=filled)
            self.resource_plans.append(plan)
            return filled
        finally:
            self.active_budget = None

    def buy_etf(self, day, reason):
        if self.pyramid_enabled and reason == 'idle_cash':
            self.additions(day)
        return super().buy_etf(day, reason)

    def additions(self, day):
        cohorts = {c['event_id']: c for c in self.cohorts}
        present = {h['event_id']: sid for sid, h in self.holdings.items()
                   if sid != '0050' and cohorts[h['event_id']]['entry_date'] < str(day.date())}
        for identity in sorted((set(present)|set(self.pyramid_pending))-self.pyramid_done):
            sid = cohorts[identity]['stock_id']
            h = self.holdings.get(sid)
            if h and h['event_id'] != identity:
                h = None
            state = self.exit_states[identity]
            pending = self.pyramid_pending.get(identity)
            row = dict(date=str(day.date()), stock_id=sid, event_id=identity,
                       pending_before=deepcopy(pending), filled_qty=0)
            if h is None or state['trigger_reason'] or not h['qty']:
                row['status'] = 'cancelled_exit_or_no_physical_shares'
                self.pyramid_pending.pop(identity, None)
                self.pyramid_decisions.append(row)
                continue
            rights = sum(r.get('qty', 0) for r in self.receivables if r['event_id']==identity and r['kind']=='shares')
            if rights or h['qty'] % 1000:
                row['status'] = 'cancelled_unsettled_or_nonboard_shares'
                self.pyramid_pending.pop(identity, None)
                self.pyramid_decisions.append(row)
                continue
            context = self.technical_signals.technical_context(self.positions[day], sid)
            if pending is None and (context['adjusted_close'] is None or state['entry_price'] is None
                    or context['adjusted_close'] < state['entry_price']*1.1-1e-12 or context['breakout20'] is not True):
                row['status'] = 'no_strong_signal'
                self.pyramid_decisions.append(row)
                continue
            amount = float(self.amount20.at[day, sid])
            if not math.isfinite(amount) or amount < 50_000_000:
                row['status'] = 'cancelled_prior_liquidity'
                self.pyramid_pending.pop(identity, None)
                self.pyramid_decisions.append(row)
                continue
            price = self.prior(day, sid)
            available = max(0., money(min(self.cash, self.opening_remaining)-self.locked_unused))
            if pending is None:
                ratio = max(state['entry_price']*.88, self.support_floors[identity])/context['adjusted_close']
                capacity = addition_capacity(self._costs, price, ratio, h['qty'], self.previous_nav, available, sid)
                row.update(context=context, capacity=capacity)
                if not capacity['qty']:
                    row['status'] = 'no_capacity'
                    self.pyramid_decisions.append(row)
                    continue
                pending = dict(signal_date=context['signal_date'], created_date=str(day.date()),
                    target_index=self.positions[day]+int(self.execution_factors['entry_delay']),
                    stop_ratio=ratio, original_qty=capacity['qty'])
                self.pyramid_pending[identity] = pending
                row['created_instruction'] = deepcopy(pending)
            if self.positions[day] < pending['target_index']:
                row['status'] = 'waiting_extra_entry_delay'
            else:
                capacity = addition_capacity(self._costs, price, pending['stop_ratio'], h['qty'],
                    self.previous_nav, available, sid, pending['original_qty'])
                row['execution_capacity'] = capacity
                row['filled_qty'] = self.execute_addition(day, sid, identity, pending, capacity) if capacity['qty'] else 0
                row['status'] = 'filled' if row['filled_qty'] else 'unfilled'
                if row['filled_qty']:
                    self.pyramid_done.add(identity)
                self.pyramid_pending.pop(identity)
            self.pyramid_decisions.append(row)

    def run(self):
        result = super().run()
        if self.pyramid_enabled:
            result['settings']['pyramid_policy'] = 'one_strong_addition_cash_v1'
        return result
