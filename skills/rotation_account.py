"""Cash-conserving fractional-notional account for daily rotation research.

This is an accounting model, not exchange fill simulation: lot sizes, suspensions,
price limits and capacity still need separate execution validation.
"""
import math


class RotationAccount:
    def __init__(self, initial_cash):
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError('initial_cash must be finite and positive')
        self.cash = float(initial_cash)
        self.holdings = {}
        self.total_cost = 0.0
        self.stale_mark_count = 0

    @staticmethod
    def _validate(price, rate):
        if not math.isfinite(price) or price <= 0 or not math.isfinite(rate) or not 0 <= rate < 1:
            raise ValueError('Price must be positive and cost rate must be in [0, 1)')

    def buy(self, sid, price, budget, rate):
        self._validate(price, rate)
        if sid in self.holdings or not math.isfinite(budget) or budget <= 0 or self.cash <= 1e-8:
            raise ValueError('Invalid/duplicate purchase or insufficient cash')
        spent = min(float(budget), self.cash)
        notional = spent / (1 + rate)
        fee = spent - notional
        self.holdings[sid] = dict(units=notional / price, spent=spent, last=price, entry_cost=fee)
        self.cash -= spent
        self.total_cost += fee
        return spent

    def nav(self, prices):
        value = self.cash
        for sid, holding in self.holdings.items():
            if sid in prices:
                self._validate(prices[sid], 0)
                holding['last'] = prices[sid]
            else:
                self.stale_mark_count += 1
            value += holding['units'] * holding['last']
        return value

    def sell(self, sid, price, rate):
        self._validate(price, rate)
        holding = self.holdings.pop(sid)
        gross = holding['units'] * price
        fee = gross * rate
        proceeds = gross - fee
        self.cash += proceeds
        self.total_cost += fee
        return dict(realized_pnl_pct=proceeds / holding['spent'] - 1,
                    entry_cash=holding['spent'], exit_cash=proceeds,
                    total_cost=holding['entry_cost'] + fee,
                    entry_cost=holding['entry_cost'], exit_cost=fee, adjusted_units=holding['units'])
