"""Research-only cash execution stress and causal account exposure controls."""
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from skills.cash_allocation_replay import CashAllocationReplay
from skills.execution_stress import StressOrder
from skills.million_replay import COMMISSION, MIN_FEE, SLIPPAGE
from skills.regime_state import _validate_index


def risk_schedule(close, mode):
    """Close-of-day observations; callers must execute on a later session."""
    if mode not in ('none', 'trend60', 'shock'):
        raise ValueError('Unknown account risk mode')
    _validate_index(close.index)
    valid = close.where(np.isfinite(close) & close.gt(0))
    mean = valid.rolling(60, min_periods=60).mean()
    ret = valid / valid.shift(1) - 1
    vol = ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    ret5 = valid / valid.shift(5) - 1
    dd = valid / valid.rolling(63, min_periods=63).max() - 1
    rows, cap, recovery = [], 1., []
    for day in close.index:
        known = pd.notna(valid.at[day]) and pd.notna(mean.at[day])
        if mode == 'shock':
            known = known and all(pd.notna(x.at[day]) for x in (vol, ret5, dd))
        reason = 'normal'
        if mode == 'none':
            desired = 1.
        elif not known:
            desired, reason = None, 'missing_risk_data'
        elif mode == 'shock' and (ret5.at[day] <= -.05 or dd.at[day] <= -.10):
            desired, reason = .25, 'market_shock'
        elif valid.at[day] < mean.at[day] or (mode == 'shock' and vol.at[day] > .30):
            desired, reason = .5, 'market_weak_or_volatile'
        else:
            desired = 1.
        if desired is None:
            recovery = []
            effective = None
        else:
            if desired <= cap:
                cap, recovery = desired, []
            else:
                recovery.append(desired)
                if len(recovery) >= 5:
                    cap, recovery = min(recovery[-5:]), []
            effective = cap
        rows.append(dict(date=str(day.date()), cap=effective, desired_cap=desired,
                         reason=reason, recovery_sessions=len(recovery)))
    return pd.DataFrame(rows).set_index('date')


class CashRiskReplay(StressOrder, CashAllocationReplay):
    def __init__(self, *args, stress_mode='control', risk_mode='none', **kwargs):
        super().__init__(*args, allocation_mode='cash', **kwargs)
        self._setup_stress(stress_mode)
        self.risk_mode = risk_mode
        self.risk = risk_schedule(self.exit_signals.adjusted_close['0050'], risk_mode)
        self.risk_decisions, self.delayed_exits = [], set()
        if stress_mode in ('entry_delay', 'combined'):
            shifted = defaultdict(list)
            for day, entries in self.events.items():
                i = self.positions[day] + 1
                if i >= len(self.days):
                    raise ValueError('No calendar session for delayed entry')
                for original in entries:
                    event = deepcopy(original)
                    event['original_entry_date'] = event['entry_date']
                    event['entry_date'] = str(self.days[i].date())
                    shifted[self.days[i]].append(event)
            self.events = shifted

    def corporate_day(self, day):
        income = super().corporate_day(day)
        if self.stress_mode in ('exit_delay', 'combined'):
            for identity, state in self.exit_states.items():
                if state['target_index'] is not None and identity not in self.delayed_exits:
                    state['original_target_date'] = state['target_date']
                    state['target_index'] += 1
                    i = state['target_index']
                    state['target_date'] = str(self.days[i].date()) if i < len(self.days) else None
                    self.delayed_exits.add(identity)
                holding = self.holdings.get(state['stock_id'])
                if holding and holding['event_id'] == identity and state['target_index'] is not None:
                    holding['due_index'] = state['target_index']
        return income

    def control(self, day):
        i = self.positions[day]
        if not i:
            return dict(cap=None, date=None, reason='no_previous_session')
        date = str(self.days[i-1].date())
        row = self.risk.loc[date].to_dict()
        row['date'] = date
        row['cap'] = None if pd.isna(row['cap']) else float(row['cap'])
        return row

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if self.risk_mode != 'none' and side == 'buy' and sid != '0050':
            control = self.control(day)
            price = self.prior(day, sid)
            cap = control['cap']
            if cap is None or not price:
                allowed = 0
            else:
                # Only previous close and post-action shares; no current-day prices.
                values = [(h['qty'], self.prior(day, stock)) for stock, h in self.holdings.items()
                          if stock != '0050' and h['qty']]
                if any(mark is None for _, mark in values):
                    allowed = 0
                else:
                    exposure = sum(n * mark for n, mark in values)
                    budget = min(self.previous_nav * cap / self.slots,
                                 max(0., self.previous_nav * cap - exposure))
                    allowed = max(0, math.floor((budget-2*MIN_FEE)/(price*(1+COMMISSION+SLIPPAGE))))
            if qty > allowed:
                self.risk_decisions.append(dict(date=str(day.date()), signal_date=control['date'],
                    stock_id=sid, action='limit_new_buy', cap=cap, requested_qty=qty,
                    allowed_qty=allowed, reason=control['reason']))
            qty = min(qty, allowed)
        return super().order(day, sid, side, qty, reason, event_id, signal_date)

    def buy_etf(self, day, reason):
        super().buy_etf(day, reason)
        if self.risk_mode == 'none' or reason != 'idle_cash':
            return
        control = self.control(day)
        cap = control['cap']
        if cap is None or cap >= 1:
            return
        # This hook runs after market P&L recognition, so gap losses cannot vanish.
        budget = self.previous_nav * cap / self.slots
        for sid, holding in list(self.holdings.items()):
            if sid == '0050' or not holding['qty']:
                continue
            # An existing stop remains latched, including its execution delay.
            state = self.exit_states.get(holding['event_id'])
            if state and state['trigger_reason']:
                continue
            price = self.prior(day, sid)
            if not price:
                continue
            qty = max(0, holding['qty'] - math.floor(budget / price))
            if qty:
                filled = self.order(day, sid, 'sell', qty, 'account_risk_trim',
                                    holding['event_id'], control['date'])
                self.risk_decisions.append(dict(date=str(day.date()), signal_date=control['date'],
                    stock_id=sid, action='reduce_existing', cap=cap, requested_qty=qty,
                    filled_qty=filled, remaining_qty=qty-filled, reason=control['reason']))
