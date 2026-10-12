"""Fixed HL2 study and fail-closed prospective qualification, separate from live orders."""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from types import FunctionType

ARMS = ('obv20_breakout', 'poc_up_red', 'entry_peer_narrow', 'benchmark')
MODES = {'baseline': (.001425, .0045), 'fee2': (.00285, .0045),
         'slip2': (.001425, .009), 'combined': (.00285, .009)}


def cost_function(mode):
    """Clone the sealed cost primitive with private constants; no shared mutation."""
    from skills.million_replay import costs
    fee, slip = MODES[mode]
    namespace = dict(costs.__globals__, COMMISSION=fee, SLIPPAGE=slip)
    return FunctionType(costs.__code__, namespace, costs.__name__, costs.__defaults__, costs.__closure__)


def verify_costs(account, mode):
    """Scalar Decimal oracle independent of the engine's cost callable."""
    fee, slip = map(lambda x: Decimal(str(x)), MODES[mode])
    for trade in account['trades']:
        gross = Decimal(str(trade['reference_price'])) * trade['qty']
        commission = max(Decimal(20), (gross*fee).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        tax = ((gross*Decimal('.001' if trade['stock_id']=='0050' else '.003')).quantize(
            Decimal(1), rounding=ROUND_FLOOR) if trade['side']=='sell' else Decimal(0))
        slippage = (gross*slip).quantize(Decimal(1), rounding=ROUND_CEILING)
        for name, expected in (('commission', commission), ('tax', tax), ('slippage', slippage)):
            if Decimal(str(trade[name])) != expected:
                raise ValueError('Independent stress cost mismatch: '+name)
    return {'trades_checked': len(account['trades']), 'mode': mode}


def _fractions(buy, sell):
    if (buy, sell) != (.5, .3):
        raise ValueError('Top-three study requires buy HL2 and unchanged sell30')


def install(runner, mode):
    """Install only in the private runner after corporate/execution adapters."""
    if mode not in MODES:
        raise ValueError('Unknown fixed cost stress')
    cost = cost_function(mode)
    original = runner.engine_types

    class CostsAndHL2:
        def __init__(self, *args, buy_fraction=.5, sell_fraction=.3, **kwargs):
            _fractions(buy_fraction, sell_fraction)
            # The frozen constructor admits its original registered pair only.
            # It merely stores fractions. Set the new registered buy value before run.
            super().__init__(*args, buy_fraction=.7, sell_fraction=.3, **kwargs)
            self.buy_fraction = .5

        def _costs(self, price, qty, side, sid):
            return cost(price, qty, side, sid)

        def run(self):
            account = super().run()
            account['settings'].update(commission=MODES[mode][0], slippage=MODES[mode][1],
                buy_fraction=.5, sell_fraction=.3, top3_cost_mode=mode)
            verify_costs(account, mode)
            return account

    def engines(*args, **kwargs):
        return tuple(type('Top3'+cls.__name__, (CostsAndHL2, cls), {}) for cls in original(*args, **kwargs))
    runner.engine_types = engines
    # Corporate/resource wrappers close over the private independent auditor.
    # Reach only closures, never mutate imported module globals.
    visited, namespaces = set(), {}
    def visit(function):
        if not isinstance(function, FunctionType) or id(function) in visited:
            return
        visited.add(id(function))
        ns = function.__globals__
        if ns.get('__name__') == '_all_entry_independent_audit':
            namespaces[id(ns)] = ns
        for cell in function.__closure__ or ():
            visit(cell.cell_contents)
    visit(runner._all_entry_indexed_audit)
    if len(namespaces) != 1:
        raise ValueError('Expected one isolated independent audit namespace')
    namespace = next(iter(namespaces.values()))
    namespace.update(costs=cost_function(mode), _fractions=_fractions)


def qualification(rows, forward=None):
    """No missing field, historical split, or hindsight fill can grant paper access."""
    forward = forward or {}
    complete = set(rows) == set(MODES) and all(r.get('completed') is True for r in rows.values())
    historical = complete and all(r['total_return'] > 0 and r['excess_return'] > 0
        and r['max_drawdown'] >= r['benchmark_max_drawdown'] for r in rows.values())
    unseen = (forward.get('version_locked_before_signals') is True
        and forward.get('source_verified') is True and forward.get('sessions', 0) >= 126
        and forward.get('settled_cohorts', 0) >= 30
        and forward.get('net_return', 0) > 0 and forward.get('excess_return', 0) > 0
        and forward.get('drawdown_not_worse_than_benchmark') is True)
    executable = forward.get('causal_fill_verified') is True
    checks = dict(complete_stress_matrix=complete, historical_stress_pass=bool(historical),
        unseen_validation=bool(unseen), causal_fill_verified=executable)
    return dict(checks=checks, paper_enabled=all(checks.values()), live_qualified=False,
        missing=[key for key, value in checks.items() if not value])
