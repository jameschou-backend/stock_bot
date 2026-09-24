"""One execution-policy contrast; preserve the sealed sizing/accounting engine."""
from collections import Counter

from skills.conservative_diversification import ConservativeDiversification
from skills.execution_resources import ResourceBenchmark


class BoardOnlyOrders:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board_decisions = []

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        # This seam runs AFTER ResourceDecisions sizes and reserves the order.
        # Rounding before that layer would allow a later cash cap to create odd lots.
        if type(qty) is not int or qty < 0:
            raise ValueError('Order shares must be integer and nonnegative')
        eligible = min(qty, self.holdings.get(sid, {}).get('qty', 0)) if side == 'sell' else qty
        board, remainder = eligible // 1000 * 1000, eligible % 1000
        if sid != '0050' and side == 'sell' and reason == 'scheduled_exit' and hasattr(self, 'exit_states'):
            state = self.exit_states.get(event_id)
            if not state or not state['trigger_reason']:
                raise ValueError('Stock sale requires a latched exit')
            reason, signal_date = state['trigger_reason'], state['signal_date']
        filled = super()._execute_order(day, sid, side, board, reason, event_id, signal_date) if board else 0
        failure = None
        if remainder:
            failure = 'board_only_odd_remainder' if board else 'board_only_below_one_lot'
            self.orders.append(dict(date=str(day.date()), stock_id=sid, name=self.names.get(sid, sid),
                side=side, channel='odd', requested_qty=remainder, filled_qty=0, reason=reason,
                event_id=event_id, signal_date=signal_date, failure=failure))
        self.board_decisions.append(dict(date=str(day.date()), stock_id=sid, side=side,
            event_id=event_id, signal_date=signal_date, reason=reason, requested_qty=qty,
            eligible_qty=eligible, board_qty=board, rejected_odd_qty=remainder,
            filled_qty=filled, failure=failure))
        return filled

    def run(self):
        account = super().run()
        account['settings']['execution_policy'] = 'board_only'
        return account


class BoardOnlyReplay(BoardOnlyOrders, ConservativeDiversification):
    def __init__(self, *args, **kwargs):
        if 'position_count' in kwargs:
            raise ValueError('Board-only research fixes position_count=5')
        super().__init__(*args, position_count=5, **kwargs)


class BoardOnlyBenchmark(BoardOnlyOrders, ResourceBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, opening_cash_only=True, lock_unused=True, **kwargs)


def audit_board_only(account, decisions, plans):
    """Additional policy audit; the caller also reconstructs every share and cash flow."""
    if account['settings'].get('execution_policy') != 'board_only':
        raise ValueError('Missing board-only policy identity')
    if any(t['channel'] != 'board' or t['qty'] % 1000 for t in account['trades']):
        raise ValueError('Board-only account contains an odd-lot fill')
    keys = ('date', 'stock_id', 'side', 'event_id', 'signal_date', 'reason')
    def identity(row):
        return tuple(row[k] for k in keys)
    rejected = Counter()
    requested = Counter()
    for decision in decisions:
        qty = decision['eligible_qty']
        board, odd = qty // 1000 * 1000, qty % 1000
        if (decision['board_qty'] != board or decision['rejected_odd_qty'] != odd
                or decision['filled_qty'] > board or decision['filled_qty'] % 1000):
            raise ValueError('Board-only rounding decision differs')
        expected = ('board_only_odd_remainder' if board else 'board_only_below_one_lot') if odd else None
        if decision['failure'] != expected:
            raise ValueError('Board-only rejection reason differs')
        if odd:
            rejected[(*identity(decision), odd, expected)] += 1
        requested[identity(decision)] += board
    actual = Counter()
    for order in account['orders']:
        if (order.get('failure') or '').startswith('board_only_'):
            if order['channel'] != 'odd' or order['filled_qty'] != 0:
                raise ValueError('Board-only rejected remainder generated a fill')
            actual[(*identity(order), order['requested_qty'], order['failure'])] += 1
        elif order['channel'] == 'odd':
            raise ValueError('Unexpected odd-lot order in board-only account')
    if rejected != actual:
        raise ValueError('Board-only rejected quantities do not reconcile')
    fills = Counter()
    for trade in account['trades']:
        fills[identity(trade)] += trade['qty']
    if any(qty > requested[key] for key, qty in fills.items()):
        raise ValueError('Board-only fills exceed rounded decisions')
    decision_buys = {(r['date'], r['stock_id'], r['event_id']): r for r in decisions if r['side'] == 'buy'}
    for plan in plans:
        key = (plan['date'], plan['stock_id'], plan['event_id'])
        if not plan.get('failure') and (key not in decision_buys
                or decision_buys[key]['requested_qty'] != plan['planned_qty']):
            raise ValueError('Board-only rounding did not follow resource sizing')
    return dict(board_fills_only=True, rejected_remainders_reconciled=True,
        rounding_after_resource_sizing=True, odd_lot_fills=0)


def execution_summary(account, decisions=()):
    """Separate unsellable residual exposure from full-lot tradable holdings."""
    last_day = account['daily'][-1]['date']
    residual = [dict(date=r['date'], stock_id=r['stock_id'], event_id=r['event_id'],
        qty=r['qty'], residual_qty=r['qty'] % 1000, residual_market_value=r['qty'] % 1000*r['price'])
        for r in account['holdings'] if r['qty'] % 1000]
    final = [r for r in residual if r['date'] == last_day]
    last = account['daily'][-1]
    return dict(odd_lot_fills=sum(t['channel'] == 'odd' for t in account['trades']),
        rejected_remainder_orders=sum(bool(r['rejected_odd_qty']) for r in decisions),
        rejected_remainder_shares=sum(r['rejected_odd_qty'] for r in decisions),
        below_lot_zero_fill_attempts=sum(r['failure'] == 'board_only_below_one_lot' for r in decisions),
        below_lot_buy_attempts=sum(r['side'] == 'buy' and r['failure'] == 'board_only_below_one_lot' for r in decisions),
        residual_holding_days=len(residual),
        residual_only_slot_days=sum(0 < r['qty'] < 1000 and r['stock_id'] != '0050' for r in account['holdings']),
        final_residuals=final, final_residual_market_value=sum(r['residual_market_value'] for r in final),
        final_cash=last['cash'], final_market_value=last['market_value'], final_receivable=last['receivable'],
        traded_notional_initial_equity=sum(t['gross'] for t in account['trades'])/account['settings']['initial_cash'])
