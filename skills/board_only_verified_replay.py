"""Evidence-date and final-fill audits layered above the immutable board-only study."""
from skills.board_only_replay import BoardOnlyReplay, BoardOnlyBenchmark, audit_board_only
from skills.replay_market_feeds import ReplayDataUnavailable


class VerifiedBoardOnlyOrders:
    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if type(qty) is not int or qty < 0:
            raise ValueError('Order shares must be integer and nonnegative')
        eligible = min(qty, self.holdings.get(sid, {}).get('qty', 0)) if side == 'sell' else qty
        if eligible >= 1000 and all(self.raw(day, sid, key) for key in ('close', 'volume', 'high', 'low')):
            if not self.feeds.get_limits(sid).get(str(day.date())):
                raise ReplayDataUnavailable(f'Offline replay is missing price-limit date: {sid} {day.date()}')
        return super()._execute_order(day, sid, side, qty, reason, event_id, signal_date)

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        first_decision, first_trade = len(self.board_decisions), len(self.trades)
        filled = super().order(day, sid, side, qty, reason, event_id, signal_date)
        decisions, trades = self.board_decisions[first_decision:], self.trades[first_trade:]
        if len(decisions) > 1 or (trades and not decisions):
            raise ValueError('An order must have exactly one board decision for its fills')
        if decisions:
            # StressOrder.order may settle a negative-proceeds sale AFTER its
            # _execute_order returns. Only the outermost order sees final fills.
            decisions[0]['filled_qty'] = sum(trade['qty'] for trade in trades)
            decisions[0]['trade_sequences'] = [trade['sequence'] for trade in trades]
            if decisions[0]['filled_qty'] != filled:
                raise ValueError('Final board decision differs from returned order quantity')
        return filled


class BoardOnlyVerifiedReplay(VerifiedBoardOnlyOrders, BoardOnlyReplay):
    pass


class BoardOnlyVerifiedBenchmark(VerifiedBoardOnlyOrders, BoardOnlyBenchmark):
    pass


def audit_verified_board_only(account, decisions, plans):
    checked = audit_board_only(account, decisions, plans)
    trades = {trade['sequence']: trade for trade in account['trades']}
    matched = set()
    keys = ('date', 'stock_id', 'side', 'event_id', 'signal_date', 'reason')
    for decision in decisions:
        sequences = decision.get('trade_sequences')
        if not isinstance(sequences, list) or len(sequences) != len(set(sequences)):
            raise ValueError('Board decision lacks unique final trade sequences')
        if any(sequence not in trades or sequence in matched for sequence in sequences):
            raise ValueError('Board decision references an unknown or reused trade')
        fills = [trades[sequence] for sequence in sequences]
        if any(any(trade[key] != decision[key] for key in keys) for trade in fills):
            raise ValueError('Board decision and final trade identities differ')
        if decision['filled_qty'] != sum(trade['qty'] for trade in fills):
            raise ValueError('Board decision final filled quantity differs from trades')
        matched.update(sequences)
    if matched != set(trades):
        raise ValueError('Final board trade has no audited decision')
    checked['final_fill_decisions_exact'] = True
    return checked
