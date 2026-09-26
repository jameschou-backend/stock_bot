"""Prior-only volatility caps over the sealed five-slot cash account."""
import math

import numpy as np
import pandas as pd

from skills.million_replay import money
from skills.residual_slot_replay import ResidualSlotReplay

TARGETS = (None, .30, .40, .50)


def prior_volatility(close, calendar):
    prices = close.reindex(calendar)
    valid = prices.where(np.isfinite(prices) & prices.gt(0))
    returns = valid/valid.shift(1)-1
    return returns.rolling(20, min_periods=20).std(ddof=1).shift(1)*math.sqrt(252)


def budget_cap(equal_budget, volatility, target):
    if not math.isfinite(volatility) or volatility < 0:
        return 0., 'missing_prior_volatility'
    ratio = min(1., target/volatility) if volatility else 1.
    return max(0., equal_budget*ratio), 'scaled' if ratio < 1 else 'equal_cap'


class VolatilityBudgetReplay(ResidualSlotReplay):
    def __init__(self, *args, volatility_target, **kwargs):
        if isinstance(volatility_target, bool) or volatility_target not in TARGETS:
            raise ValueError('Unsupported preregistered volatility target')
        super().__init__(*args, residual_policy='release', **kwargs)
        self.volatility_target = volatility_target
        self.entry_volatility = prior_volatility(self.exit_signals.adjusted_close, self.days)
        self.volatility_decisions = []
        self.volatility_spend_left = None

    def _affordable(self, qty, step, price, cash, sid):
        if self.volatility_spend_left is not None:
            cash = min(cash, self.volatility_spend_left)
        return super()._affordable(qty, step, price, cash, sid)

    def cash_move(self, day, kind, change, **extra):
        if kind == 'buy' and self.volatility_spend_left is not None:
            if change > 0 or -change > self.volatility_spend_left+.005:
                raise ValueError('Buy exceeded volatility risk budget')
            self.volatility_spend_left = money(self.volatility_spend_left+change)
        return super().cash_move(day, kind, change, **extra)

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if self.volatility_target is None or side != 'buy' or sid == '0050':
            return super().order(day, sid, side, qty, reason, event_id, signal_date)
        vol = float(self.entry_volatility.at[day, sid])
        budget, why = budget_cap(self.residual_budget, vol, self.volatility_target)
        row = dict(date=str(day.date()), event_id=event_id, stock_id=sid, signal_date=signal_date,
            reference_date=str(self.days[self.positions[day]-1].date()),
            annual_volatility=vol if math.isfinite(vol) else None, target=self.volatility_target,
            equal_budget=self.residual_budget, budget=budget, requested_qty=qty,
            reason=why)
        before = len(self.trades)
        self.volatility_spend_left = budget
        try:
            filled = super().order(day,sid,side,qty,reason,event_id,signal_date)
        finally:
            self.volatility_spend_left = None
        row.update(filled_qty=filled, spent=sum(-t['cash_change'] for t in self.trades[before:]))
        self.volatility_decisions.append(row)
        return filled

    def run(self):
        account = super().run()
        if self.volatility_target is not None:
            account['settings']['entry_volatility_target'] = self.volatility_target
        return account


def audit_volatility_budget(account, decisions, snapshots, close, calendar, target):
    vol = prior_volatility(close, calendar)
    slots = {r['date']:r for r in snapshots}
    fills = {(r['date'],r['event_id']) for r in account['trades'] if r['side']=='buy'}
    checked = set()
    days = pd.DatetimeIndex(calendar)
    for row in decisions:
        key=(row['date'],row['event_id']);day=pd.Timestamp(row['date']);sid=row['stock_id']
        if key in checked or row['reference_date'] != str(days[days.get_loc(day)-1].date()):
            raise ValueError('Duplicate or future volatility decision')
        observed=float(vol.at[day,sid])
        wanted=observed if math.isfinite(observed) else None
        budget,why=budget_cap(slots[row['date']]['new_position_budget'],observed,target)
        trades=[t for t in account['trades'] if (t['date'],t['event_id'])==key and t['side']=='buy']
        spent=sum(-t['cash_change'] for t in trades)
        if (row['annual_volatility']!=wanted or row['target']!=target or row['reason']!=why
                or abs(row['equal_budget']-slots[row['date']]['new_position_budget'])>.005
                or abs(row['budget']-budget)>.005 or abs(row['spent']-spent)>.005
                or spent>budget+.005 or row['filled_qty']!=sum(t['qty'] for t in trades)):
            raise ValueError('Volatility cap or final cash spend did not reconstruct')
        checked.add(key)
    if not fills.issubset(checked):
        raise ValueError('Buy lacks a volatility decision')
    return dict(prior_volatility_rebuilt=True, capped_spend_rebuilt=True, gap_cannot_enlarge_budget=True)
