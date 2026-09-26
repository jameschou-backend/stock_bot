"""Preregistered support exits and cost-inclusive planned-risk quantity caps.

The sealed account loop, corporate settlement and resource layers remain intact.
The control arm must reproduce that entire account, not merely its return.
"""
import math

import pandas as pd

from skills.million_replay import money
from skills.residual_slot_replay import ResidualSlotReplay
from skills.technical_signals import TechnicalSignals

ARMS = ('control', 'support20', 'risk2', 'support_risk2')


def planned_loss(cost, price, stop, qty, sid):
    if qty == 0:
        return 0.
    return money(-cost(price, qty, 'buy', sid)['cash_change']
                 - cost(stop, qty, 'sell', sid)['cash_change'])


def risk_quantity(cost, price, stop, requested, budget, sid):
    if (type(requested) is not int or requested < 0 or
            any(not math.isfinite(x) for x in (price, stop, budget)) or
            not 0 < stop < price or budget < 0):
        raise ValueError('Invalid planned-risk inputs')
    for qty in range(requested // 1000 * 1000, 0, -1000):
        if planned_loss(cost, price, stop, qty, sid) <= budget + 1e-8:
            return qty
    return 0


class SupportRiskReplay(ResidualSlotReplay):
    def __init__(self, *args, technical_mode, technical_signals, **kwargs):
        if technical_mode not in ARMS or not isinstance(technical_signals, TechnicalSignals):
            raise ValueError('A registered mode and causal technical signals are required')
        super().__init__(*args, residual_policy='release', **kwargs)
        if (not technical_signals.days.equals(self.days) or
                not technical_signals.adjusted_close.equals(self.exit_signals.adjusted_close)):
            raise ValueError('Technical and exit signals must share the same frozen prices')
        self.technical_mode, self.technical_signals = technical_mode, technical_signals
        self.use_support = technical_mode in ('support20', 'support_risk2')
        self.use_risk = technical_mode in ('risk2', 'support_risk2')
        self.entry_technical, self.support_floors = {}, {}
        self.technical_entries, self.support_decisions = [], []

    def original_context(self, signal_date, day, sid):
        signal = pd.Timestamp(signal_date)
        if signal not in self.positions or signal >= day:
            raise ValueError('Technical entry requires the original prior signal date')
        return self.technical_signals.technical_context(self.positions[signal] + 1, sid)

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if self.technical_mode == 'control' or side != 'buy' or sid == '0050':
            return super().order(day, sid, side, qty, reason, event_id, signal_date)
        context = self.original_context(signal_date, day, sid)
        price = self.prior(day, sid)
        ratio, stop, loss, why = None, None, None, 'original_quantity'
        allowed = qty
        if self.use_support and not context['risk_available']:
            allowed, why = 0, 'missing_or_nonpositive_support_distance'
        elif self.use_risk:
            close = context['adjusted_close']
            if close is None or not price:
                allowed, why = 0, 'missing_planned_risk_price'
            else:
                ratio = max(.88, context['support20']/close) if self.use_support else .88
                stop = price * ratio
                allowed = risk_quantity(self._costs, price, stop, qty, self.previous_nav*.02, sid)
                loss = planned_loss(self._costs, price, stop, allowed, sid)
                why = 'risk_quantity_cap' if allowed < qty//1000*1000 else 'original_board_quantity'
        row = dict(date=str(day.date()), stock_id=sid, event_id=event_id, signal_date=signal_date,
            context=context, reference_price=price, opening_nav=self.previous_nav,
            requested_qty=qty, allowed_qty=allowed, planned_stop_ratio=ratio,
            planned_stop_price=stop, planned_loss=loss, risk_budget=self.previous_nav*.02,
            slippage=self.stress_slippage, reason=why)
        filled = super().order(day, sid, side, allowed, reason, event_id, signal_date)
        row['filled_qty'] = filled
        self.technical_entries.append(row)
        if filled and self.use_support:
            self.entry_technical[event_id] = context
            self.support_floors[event_id] = context['support20']
        return filled

    def corporate_day(self, day):
        latched = {key for key, state in self.exit_states.items() if state['trigger_reason']}
        begin = len(self.exit_decisions)
        income = super().corporate_day(day)
        if not self.use_support:
            return income
        index = self.positions[day]
        for row in self.exit_decisions[begin:]:
            identity, sid = row['event_id'], row['stock_id']
            if identity not in self.support_floors:
                raise ValueError('Active support cohort lacks its original entry signal')
            context = self.technical_signals.technical_context(index, sid)
            before = self.support_floors[identity]
            after = max(before, context['support20']) if context['support_available'] else before
            self.support_floors[identity] = after
            price = context['adjusted_close']
            broken = price is not None and price < after
            state = self.exit_states[identity]
            if identity not in latched and state['trigger_reason'] != 'loss12' and broken:
                state.update(trigger_reason='support20', signal_date=str(self.days[index-1].date()),
                             target_date=str(day.date()), target_index=index)
                row.update(exit=True, reason='support20', phase='exiting', extend=False,
                           first_signal_date=state['signal_date'], target_date=state['target_date'])
                # The ancestor already applied the stress to its own triggers.
                # This new trigger enters that same one-day convention exactly once.
                if self.execution_factors['exit_delay']:
                    state['original_target_date'] = str(day.date())
                    state['target_index'] = index + 1
                    state['target_date'] = str(self.days[index+1].date()) if index+1 < len(self.days) else None
                    self.delayed_exits.add(identity)
                holding = self.holdings.get(sid)
                if holding and holding['event_id'] == identity:
                    holding['due_index'] = state['target_index']
            self.support_decisions.append(dict(date=str(day.date()), event_id=identity, stock_id=sid,
                signal_date=context['signal_date'], initial_signal_date=self.entry_technical[identity]['signal_date'],
                support_available=context['support_available'], observed_support=context['support20'],
                floor_before=before, floor_after=after, signal_close=price, broken=broken,
                previously_latched=identity in latched, trigger_reason=state['trigger_reason'],
                first_signal_date=state['signal_date'], target_date=state['target_date']))
        return income

    def run(self):
        result = super().run()
        if self.technical_mode != 'control':
            result['settings']['technical_mode'] = self.technical_mode
        return result
