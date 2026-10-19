"""One fixed, causal MA120 control on new purchase budgets only."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
import math

import numpy as np
import pandas as pd

from skills.mixed_odd_replay import sized_quantity
from skills.top3_validation import MODES

VARIANT = 'market120_half'
WINDOW = 120


def market_state(adjusted, signal):
    """Require a current quote; suspensions are not forward-filled observations."""
    stamp = pd.Timestamp(signal)
    series = adjusted['0050'].loc[:stamp]
    current = series.get(stamp)
    values = series.loc[np.isfinite(series) & series.gt(0)].tail(WINDOW)
    known = current is not None and math.isfinite(current) and current > 0 and len(values) == WINDOW
    average = float(values.mean()) if known else None
    state = 'unknown' if not known else 'weak' if current < average else 'normal'
    return dict(state=state, factor={'unknown': 0., 'weak': .5, 'normal': 1.}[state],
        close=float(current) if current is not None and math.isfinite(current) else None,
        average=average, observations=len(values),
        window_start=str(values.index[0].date()) if len(values) else None,
        window_end=str(values.index[-1].date()) if len(values) else None)


def audit_market_sizing(account, adjusted):
    """Independently reconstruct market windows, integer sizing and every plan."""
    if account['settings']['risk_control_variant'] != VARIANT:
        raise ValueError('Unregistered market sizing variant')
    plans = {(p['date'], p['event_id'], p['side']): p for p in account['tick_plans']}
    if len(plans) != len(account['tick_plans']):
        raise ValueError('Duplicate market sizing plan')
    seen, counts, first = set(), dict(normal=0, weak=0, unknown=0), None
    fee, slip = (Decimal(str(x)) for x in MODES[account['settings']['top3_cost_mode']])
    for r in account['market_sizing_records']:
        before, after = r['before'], r['after']
        key = after['date'], after['event_id'], after['side']
        if key in seen or after != plans.get(key) or not before['signal_date'] < before['date']:
            raise ValueError('Market sizing identity or cutoff differs')
        seen.add(key)
        expected = deepcopy(before)
        eligible = before['side'] == 'buy' and before['planned_qty'] > 0
        expected['market_sizing_applied'] = eligible
        if not eligible:
            if r['market'] is not None or r['bounds'] is not None:
                raise ValueError('Existing holding or blocked order received market sizing')
        else:
            if before['signal_date'] != before['reference_date']:
                raise ValueError('Market sizing requires the previous completed session')
            # Scalar selection and summation independently check the vector implementation.
            stamp = pd.Timestamp(before['signal_date'])
            raw = [(d, float(p)) for d, p in adjusted['0050'].items() if d <= stamp]
            values = [(d, p) for d, p in raw if math.isfinite(p) and p > 0][-WINDOW:]
            current = next((p for d, p in raw if d == stamp), None)
            known = current is not None and math.isfinite(current) and current > 0 and len(values) == WINDOW
            average = sum(p for d, p in values)/WINDOW if known else None
            state = 'unknown' if not known else 'weak' if current < average else 'normal'
            factor = {'unknown': 0., 'weak': .5, 'normal': 1.}[state]
            market = r['market']
            source = dict(state=state, factor=factor,
                close=current if current is not None and math.isfinite(current) else None,
                observations=len(values), window_start=str(values[0][0].date()) if values else None,
                window_end=str(values[-1][0].date()) if values else None)
            if any(market.get(k) != v for k, v in source.items()) or (
                    (average is None) != (market.get('average') is None)) or (
                    average is not None and not math.isclose(average, market['average'], rel_tol=1e-12)):
                raise ValueError('Market source window differs')
            counts[state] += 1
            if factor != 1.:
                first = min(first or before['date'], before['date'])
                budget = before['sizing_budget']*factor
                maximum = max(0, math.floor((budget-40)/(before['prior_reference']*1.005925)))
                price = Decimal(str(r['bounds']['upper']))
                n = min(maximum, max(0, int(budget/float(price))))
                while n:
                    needed = Decimal(0)
                    for q in (n//1000*1000, n%1000):
                        if q:
                            gross = price*q
                            needed += gross + max(Decimal(20), (gross*fee).quantize(Decimal(1), rounding=ROUND_HALF_UP)) + (gross*slip).quantize(Decimal(1), rounding=ROUND_CEILING)
                    if needed <= Decimal(str(budget))+Decimal('.0000001'):
                        break
                    n -= 1
                expected.update(sizing_budget=budget, planned_qty=n, board_qty=n//1000*1000, odd_qty=n%1000,
                    rejection=None if n else 'market_history_unknown' if state == 'unknown' else 'market_budget_below_one_share')
        if expected != after:
            raise ValueError('Market sizing changed an unrelated plan field or quantity')
    if seen != set(plans):
        raise ValueError('Missing market sizing plan evidence')
    return dict(passed=True, plans_checked=len(seen), states=counts,
        reduced_buy_plans=counts['weak'], unknown_buy_plans=counts['unknown'],
        first_trigger=first, causal_inputs=True, independent_sizing=True,
        original_reservations_preserved=True, sell_plans_unchanged=True)


def install(runner, variant, metadata):
    if variant != VARIANT:
        raise ValueError('Unregistered market sizing variant')
    original = runner.engine_types

    class MarketSizing:
        def __init__(self, *args, **kwargs):
            self.market_sizing_records = []
            super().__init__(*args, **kwargs)

        def _plan(self, day, sid, side, eid, signal, qty, budget, opening_cash, failure):
            super()._plan(day, sid, side, eid, signal, qty, budget, opening_cash, failure)
            p = self.day_plans[(eid, side)]
            before = deepcopy(p)
            p['market_sizing_applied'] = side == 'buy' and p['planned_qty'] > 0
            state, bounds = None, None
            if p['market_sizing_applied']:
                state = market_state(self.exit_signals.adjusted_close, signal)
                bounds = self.feeds.get_limits(sid)[str(day.date())]
                if state['factor'] != 1.:
                    p['sizing_budget'] *= state['factor']
                    maximum = max(0, math.floor((p['sizing_budget']-40)/(p['prior_reference']*1.005925)))
                    n = sized_quantity(maximum, bounds['upper'], p['sizing_budget'], self, sid)
                    p.update(planned_qty=n, board_qty=n//1000*1000, odd_qty=n%1000,
                        rejection=None if n else 'market_history_unknown' if state['state'] == 'unknown' else 'market_budget_below_one_share')
            self.market_sizing_records.append(dict(before=before, after=deepcopy(p),
                market=state, bounds=deepcopy(bounds)))
            self.tick_plans[-1] = deepcopy(p)

        def run(self):
            result = super().run()
            result['settings']['risk_control_variant'] = variant
            result['market_sizing_records'] = self.market_sizing_records
            result['market_sizing_audit'] = audit_market_sizing(result, self.exit_signals.adjusted_close)
            return result

    def engines(*args, **kwargs):
        stock, benchmark = original(*args, **kwargs)
        return type('MarketSizing'+stock.__name__, (MarketSizing, stock), {}), benchmark
    runner.engine_types = engines
    writer = runner.write

    def write(path, value):
        if 'account' in value or str(path).endswith('/report.json'):
            value['risk_controls'] = metadata
        return writer(path, value)
    runner.write = write
