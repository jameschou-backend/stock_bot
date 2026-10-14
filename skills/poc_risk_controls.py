"""Two isolated pre-order controls, preserving frozen accounts and reservations."""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR
import inspect
import math
from types import FunctionType

import numpy as np
import pandas as pd

from skills.intraday_limit_replay import limit_price
from skills.mixed_odd_replay import sized_quantity

VARIANTS = ('chase3', 'vol20')


def volatility(close, sid, signal):
    values = close.loc[:pd.Timestamp(signal), sid].tail(21)
    if len(values) != 21 or not np.isfinite(values).all() or not values.gt(0).all():
        return None
    return float(values.pct_change(fill_method=None).iloc[1:].std(ddof=1)*math.sqrt(252))


def buy_cap(reference, sid, bounds):
    cap = limit_price(float(Decimal(str(reference))*Decimal('1.03')), sid, 'buy')
    return min(bounds['upper'], max(bounds['lower'], cap))


def audit_controls(account, adjusted):
    """Recompute causal inputs with scalar arithmetic, then compare entire plans."""
    variant = account['settings']['risk_control_variant']
    records = account['risk_control_records']
    plans = {(p['date'], p['event_id'], p['side']): p for p in account['tick_plans']}
    wanted = {k for k, p in plans.items() if p['side'] == 'buy' and p['risk_control_applied']}
    seen = set()
    for r in records:
        before, after = r['before'], r['after']
        key = after['date'], after['event_id'], after['side']
        if key in seen or after != plans[key] or not after['signal_date'] < after['date']:
            raise ValueError('Risk control plan identity or cutoff changed')
        seen.add(key)
        expected = deepcopy(before)
        expected['risk_control_applied'] = True
        bounds = r['bounds']
        if variant == 'chase3':
            price = Decimal(str(before['prior_reference']))*Decimal('1.03')
            tick = next(Decimal(t) for ceiling, t in [(10, '.01'), (50, '.05'),
                (100, '.1'), (500, '.5'), (1000, '1'), (math.inf, '5')] if price < ceiling)
            cap = float((price/tick).to_integral_value(rounding=ROUND_FLOOR)*tick)
            cap = min(bounds['upper'], max(bounds['lower'], cap))
            expected.update(limit_price=cap, odd_limit=cap)
        else:
            prices = adjusted.loc[:pd.Timestamp(before['signal_date']), before['stock_id']].tail(21).tolist()
            known = len(prices) == 21 and all(math.isfinite(p) and p > 0 for p in prices)
            sigma = None
            if known:
                returns = [prices[i]/prices[i-1]-1 for i in range(1, 21)]
                mean = sum(returns)/20
                sigma = math.sqrt(sum((v-mean)**2 for v in returns)/19*252)
            if (sigma is None) != (r['annual_volatility'] is None) or (
                    sigma is not None and not math.isclose(sigma, r['annual_volatility'], abs_tol=1e-12)):
                raise ValueError('Volatility source window differs')
            factor = 0 if sigma is None else min(1., .30/sigma) if sigma > 0 else 1.
            budget = before['sizing_budget']*factor
            if not math.isclose(budget, after['sizing_budget'], abs_tol=1e-7):
                raise ValueError('Volatility budget differs')
            # Independently find the largest affordable integer, including two minima.
            from skills.top3_validation import MODES
            from decimal import ROUND_HALF_UP, ROUND_CEILING
            fee, slip = map(lambda x: Decimal(str(x)), MODES[account['settings']['top3_cost_mode']])
            maximum = max(0, math.floor((budget-40)/(before['prior_reference']*1.005925)))
            price = Decimal(str(bounds['upper']))
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
            expected.update(sizing_budget=after['sizing_budget'], planned_qty=n,
                board_qty=n//1000*1000, odd_qty=n%1000,
                rejection=None if n else 'volatility_history_unknown' if sigma is None else 'volatility_budget_below_one_share')
        if expected != after:
            raise ValueError('Risk control changed an unrelated plan field')
    if seen != wanted:
        raise ValueError('Missing risk-control plan evidence')
    return dict(plans_checked=len(seen), causal_inputs=True, independent_sizing=True,
                original_reservations_preserved=True, isolated_variant=variant)


def _namespace(runner):
    found, visited = {}, set()
    def visit(fn):
        if not isinstance(fn, FunctionType) or id(fn) in visited:
            return
        visited.add(id(fn))
        ns = fn.__globals__
        if ns.get('__name__') == '_all_entry_independent_audit':
            found[id(ns)] = ns
        for cell in fn.__closure__ or ():
            visit(cell.cell_contents)
    visit(runner._all_entry_indexed_audit)
    if len(found) != 1:
        raise ValueError('Expected one private execution auditor')
    return next(iter(found.values()))


def install(runner, variant, metadata):
    if variant not in VARIANTS:
        raise ValueError('Unregistered risk control')
    original = runner.engine_types
    class Control:
        def __init__(self, *args, **kwargs):
            self.risk_control_records = []
            super().__init__(*args, **kwargs)

        def _plan(self, day, sid, side, eid, signal, qty, budget, opening_cash, failure):
            super()._plan(day, sid, side, eid, signal, qty, budget, opening_cash, failure)
            p = self.day_plans[(eid, side)]
            p['risk_control_applied'] = False
            if side == 'buy' and p['planned_qty']:
                before = deepcopy(p)
                bounds = self.feeds.get_limits(sid)[str(day.date())]
                sigma = None
                if variant == 'chase3':
                    cap = buy_cap(p['prior_reference'], sid, bounds)
                    p.update(limit_price=cap, odd_limit=cap)
                else:
                    sigma = volatility(self.exit_signals.adjusted_close, sid, signal)
                    factor = 0 if sigma is None else min(1., .30/sigma) if sigma > 0 else 1.
                    p['sizing_budget'] *= factor
                    maximum = max(0, math.floor((p['sizing_budget']-40)/(p['prior_reference']*1.005925)))
                    n = sized_quantity(maximum, bounds['upper'], p['sizing_budget'], self, sid)
                    p.update(planned_qty=n, board_qty=n//1000*1000, odd_qty=n%1000,
                        rejection=None if n else 'volatility_history_unknown' if sigma is None else 'volatility_budget_below_one_share')
                p['risk_control_applied'] = True
                self.risk_control_records.append(dict(before=before, after=deepcopy(p),
                    bounds=deepcopy(bounds), annual_volatility=sigma))
            self.tick_plans[-1] = deepcopy(p)

        def run(self):
            result = super().run()
            result['settings']['risk_control_variant'] = variant
            result['risk_control_records'] = self.risk_control_records
            result['risk_control_audit'] = audit_controls(result, self.exit_signals.adjusted_close)
            return result

    def engines(*args, **kwargs):
        stock, benchmark = original(*args, **kwargs)
        return type('RiskControl'+stock.__name__, (Control, stock), {}), benchmark
    runner.engine_types = engines
    if variant == 'chase3':
        ns = _namespace(runner)
        # Preserve all source, cash, fill, participation and exclusion assertions.
        # Only the independently checked preknown limit policy differs. Quantity
        # is deliberately still sized at the original legal upper bound.
        from skills import poc_range_audit
        for name in ('audit_range_children', 'audit_range_execution'):
            original_fn = getattr(poc_range_audit, name)
            code = inspect.getsource(original_fn)
            field = 'limits' if name == 'audit_range_children' else 'bounds'
            old = f"bound = {field}['upper' if p['side'] == 'buy' else 'lower']"
            if code.count(old) != 1:
                raise ValueError('Frozen limit audit seam changed')
            code = code.replace(old, old+f"\n            sizing_bound = bound\n            if p['side'] == 'buy':\n                bound = _risk_buy_cap(p['prior_reference'], sid, {field})")
            code = code.replace('sized_quantity(maximum, bound,', 'sized_quantity(maximum, sizing_bound,')
            code = code.replace("source.loc[source.date.eq(pd.Timestamp(day)) & source.stock_id.eq(sid)]", "_indexed_quote_rows(source, day, sid)")
            code = code.replace("source.loc[source.date.eq(pd.Timestamp(p['reference_date'])) & source.stock_id.eq(sid)]", "_indexed_quote_rows(source, p['reference_date'], sid)")
            ns['_risk_buy_cap'] = buy_cap
            exec(compile(code, __file__, 'exec'), ns)
        # Rebind the frozen audit function captured by corporate wrappers too.
        def rebound(fn):
            if not isinstance(fn, FunctionType):
                return fn
            if fn.__globals__ is ns and fn.__name__ == 'audit_range_execution':
                return ns['audit_range_execution']
            for cell in fn.__closure__ or ():
                if isinstance(cell.cell_contents, FunctionType):
                    cell.cell_contents = rebound(cell.cell_contents)
            return fn
        runner._all_entry_indexed_audit = rebound(runner._all_entry_indexed_audit)
    writer = runner.write
    def write(path, value):
        if 'account' in value or str(path).endswith('/report.json'):
            value['risk_controls'] = metadata
        return writer(path, value)
    runner.write = write
