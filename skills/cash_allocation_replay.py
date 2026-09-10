"""Compare where idle capital waits while preserving the stock exit policy.

The inherited run loop invokes buy_etf only after corporate actions and market
P&L have been accounted for. Allocation trades use that hook, so a same-day ETF
sale cannot remove opening shares before their price move is recognized.
"""
from __future__ import annotations

from skills.scenario_exit_replay import ExitSignals, ScenarioExitReplay


ALLOCATION_MODES = ('always_0050', 'cash', 'trend_0050')


class CashAllocationReplay(ScenarioExitReplay):
    """Keep loss12/63-session stock exits and vary only idle-capital policy.

    ``allocation_decisions`` is a separate audit trail; the returned account
    retains its existing schema so always_0050 reproduces loss12 exactly.
    ETF targets are reconsidered each day. An unfilled OFF sale does not latch
    through a later ON or UNKNOWN observation; stock exit latches are unchanged.
    """

    def __init__(self, *args, exit_signals: ExitSignals,
                 allocation_mode='always_0050', **kwargs):
        if allocation_mode not in ALLOCATION_MODES:
            raise ValueError('Unknown idle-capital allocation policy: '+str(allocation_mode))
        if kwargs.pop('mode', 'loss12') != 'loss12':
            raise ValueError('Allocation research fixes stock exits to loss12')
        super().__init__(*args, exit_signals=exit_signals, mode='loss12', **kwargs)
        self.allocation_mode = allocation_mode
        self.allocation_decisions = []

    def buy_etf(self, day, reason):
        index = self.positions[day]
        signal_date = str(self.days[index-1].date()) if index else None
        state = self.exit_signals.trend.state.iloc[index-1] if index else 'UNKNOWN'
        if state not in ('ON', 'OFF', 'UNKNOWN'):
            raise ValueError('Allocation trend state must be ON, OFF or UNKNOWN')
        cash_before = self.cash
        quantity_before = self.holdings.get('0050', {}).get('qty', 0)
        trade_start, order_start = len(self.trades), len(self.orders)

        if self.allocation_mode == 'always_0050':
            action, side = 'buy_0050', 'buy'
            super().buy_etf(day, reason)
        elif self.allocation_mode == 'cash':
            action, side = 'hold_cash', None
        elif state == 'ON':
            action, side = 'buy_0050', 'buy'
            super().buy_etf(day, reason)
        elif state == 'OFF':
            action, side = 'sell_0050', 'sell'
            if quantity_before:
                self.order(day, '0050', 'sell', quantity_before,
                           'parking_trend_off', 'benchmark', signal_date)
        else:
            # Missing yesterday's close supplies no new allocation instruction.
            # The stock-entry loop can still fund an entry from any existing ETF.
            action, side = 'retain_unknown', None

        self.allocation_decisions.append(dict(
            date=str(day.date()), signal_date=signal_date, market_state=state,
            allocation_mode=self.allocation_mode, hook_reason=reason,
            action=action, side=side, cash_before=cash_before,
            etf_qty_before=quantity_before,
            requested_qty=sum(row['requested_qty'] for row in self.orders[order_start:]),
            filled_qty=sum(row['qty'] for row in self.trades[trade_start:]),
            cash_after=self.cash, etf_qty_after=self.holdings.get('0050', {}).get('qty', 0)))
