"""Lagged exit decisions on top of the unchanged integer-share account.

Only the date/reason of a stock exit changes. Price signals are adjusted close
ratios; execution, capacity, corporate rights and all cash accounting remain
with Replay. No execution-day close is read by the exit policy.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
import math

import numpy as np
import pandas as pd

from skills.exit_policy import MODES, decide_exit
from skills.million_replay import Replay, UnresolvedAction
from skills.regime_state import _validate_index, build_trend


def _finite(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


class ExitSignals:
    """Precompute causal windows once for all seven account comparisons."""

    def __init__(self, adjusted_close: pd.DataFrame, days: pd.DatetimeIndex):
        days = pd.DatetimeIndex(days)
        _validate_index(days)
        if not isinstance(adjusted_close, pd.DataFrame):
            raise ValueError('adjusted_close must be a DataFrame')
        _validate_index(adjusted_close.index)
        if not adjusted_close.columns.is_unique or '0050' not in adjusted_close:
            raise ValueError('Unique stock columns including 0050 are required')
        if not all(isinstance(sid, str) for sid in adjusted_close.columns):
            raise ValueError('Stock columns must be strings')
        if not pd.api.types.is_numeric_dtype(adjusted_close.dtypes.iloc[0]):
            raise ValueError('Adjusted prices must be numeric or explicitly missing')
        try:
            if any(pd.api.types.is_bool_dtype(t) or pd.api.types.is_complex_dtype(t)
                   or not pd.api.types.is_numeric_dtype(t) for t in adjusted_close.dtypes):
                raise ValueError('Adjusted prices must be real numbers or explicitly missing')
            close = adjusted_close.reindex(days).astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError('Adjusted prices must be real numbers or explicitly missing') from exc
        self.days = days
        self.adjusted_close = close.where(np.isfinite(close) & close.gt(0))
        self.ma20 = self.adjusted_close.rolling(20, min_periods=20).mean()
        # Exactly twenty audited market rows; pct_change's historical default
        # forward-fill must not fabricate returns around missing quotes.
        returns20 = self.adjusted_close / self.adjusted_close.shift(20) - 1
        relative20 = returns20.sub(returns20['0050'], axis=0)
        self.relative20 = relative20.where(np.isfinite(relative20))
        below = self.adjusted_close.lt(self.ma20)
        self.below_two = below & below.shift(1, fill_value=False)
        self.trend = build_trend(self.adjusted_close['0050'])
        off = self.trend.state.eq('OFF')
        self.market_off_two = off & off.shift(1, fill_value=False)
        self.strong = (self.adjusted_close.ge(self.ma20)
                       & self.ma20.gt(self.ma20.shift(5))
                       & self.relative20.gt(0)).mul(self.trend.state.eq('ON'), axis=0)

    def price(self, index, sid):
        if index < 0 or index >= len(self.days):
            return None
        if sid not in self.adjusted_close:
            raise ValueError('Exit signal stock missing from adjusted matrix: '+sid)
        return _finite(self.adjusted_close.iloc[index][sid])

    def context(self, index, sid, state):
        """index is today's execution row; all signal reads stop at index-1."""
        signal_index = index - 1
        age = index - state['entry_index']
        if age < 1:
            raise ValueError('Exit decisions require a session after actual entry')
        current = self.price(signal_index, sid)
        available = current is not None
        if available:
            state['peak_price'] = max(state['peak_price'] or current, current)
        entry, peak = state['entry_price'], state['peak_price']
        relative = _finite(self.relative20.iloc[signal_index][sid])
        return dict(held_sessions=int(age), has_signal=available,
                    entry_return=current/entry-1 if available and entry else None,
                    peak_return=peak/entry-1 if peak and entry else None,
                    peak_drawdown=current/peak-1 if available and peak else None,
                    relative20=relative,
                    below_ma20_two=bool(age >= 2 and self.below_two.iloc[signal_index][sid]),
                    market_off_two=bool(self.market_off_two.iloc[signal_index]),
                    strong_trend=bool(available and self.strong.iloc[signal_index][sid]))


class FractionalCashActions:
    """Translate an issuer's rounded fractional cash into a separate right.

    The old account can already keep cash with an unknown payment date as a
    receivable. Use that capability without tying fractional cash payment to
    delivery of whole new shares. No payment date or processing-fee waiver is
    inferred. Only the explicitly marked issuer policy uses this adapter.
    """

    def __init__(self, provider, account):
        self.provider, self.account = provider, account

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def on_date(self, sid, day):
        rows = self.provider.on_date(sid, day)
        terms = getattr(self.provider, 'overrides', {}).get(f'{sid}-{day}', {})
        if 'fractional_cash_rounding' not in terms:
            return rows
        if terms['fractional_cash_rounding'] != 'floor_ntd':
            raise UnresolvedAction(f'Unsupported fractional rounding: {sid} {day}')
        payment = terms.get('fractional_cash_pay_date')
        if payment is not None:
            try:
                stamp = pd.Timestamp(payment)
                valid = (isinstance(payment, str) and not pd.isna(stamp) and str(stamp.date()) == payment
                         and stamp.tzinfo is None and stamp == stamp.normalize() and payment >= day)
            except (ValueError, TypeError):
                valid = False
            if not valid:
                raise UnresolvedAction(f'Invalid fractional payment date: {sid} {day}')
        result = []
        for supplied in rows:
            row = dict(supplied)
            if row['kind'] == 'stock_dividend':
                if terms.get('certificate_trading_modeled') is False:
                    row['certificate_restriction'] = {
                        key: terms[key] for key in (
                            'certificate_delivery_date', 'ordinary_share_available_date',
                            'valuation_basis', 'certificate_trading_modeled')}
                quantity = self.account.holdings[sid]['qty']
                face = row.get('fractional_cash_per_share')
                if (isinstance(face, bool) or not isinstance(face, (int, float, Decimal))
                        or not math.isfinite(face) or face < 0):
                    raise UnresolvedAction(f'Fractional face value missing: {sid} {day}')
                new = Decimal(quantity) * Decimal(str(row['shares_per_share']))
                fraction = new - new.to_integral_value(rounding=ROUND_FLOOR)
                amount = (fraction * Decimal(str(face))).to_integral_value(rounding=ROUND_FLOOR)
                # Whole-share delivery no longer creates a second fraction
                # payment. The independent cash entitlement below owns it.
                row['fractional_cash_per_share'] = 0.
                row['fractional_settlement'] = dict(fraction=float(fraction), face_value=float(face),
                    gross_cash_amount=float(amount), rounding='floor_ntd', payment_date=payment,
                    payment_date_verified=payment is not None,
                    processing_or_remittance_fees_verified=False)
                if amount > 0:
                    result.append(dict(action_id=row['action_id']+'-fractional-cash', stock_id=sid,
                        date=day, kind='cash_dividend', distribution_type='fractional_share_cash',
                        # This effective rate only adapts the existing journal
                        # algebra. The issuer rate is the explicit face value.
                        cash_per_share=float(amount/Decimal(quantity)), pay_date=payment,
                        source=row.get('source'), gross_cash_amount=float(amount),
                        fractional_settlement=dict(row['fractional_settlement'])))
            result.append(row)
        return result


class ScenarioExitReplay(Replay):
    """Keep an exit instruction latched until executable shares have sold.

    Original cohorts retain their 63-session due fields for traceability.
    exit_states and exit_decisions are separate from the unchanged account
    schema, so fixed63 can reproduce the previous sealed account exactly.
    """

    def __init__(self, *args, exit_signals: ExitSignals, mode='fixed63', **kwargs):
        if mode not in MODES:
            raise ValueError('Unknown scenario exit policy: '+str(mode))
        super().__init__(*args, **kwargs)
        if self.horizon != 63 or self.benchmark:
            raise ValueError('Scenario research requires the stock account and original 63-day horizon')
        if not isinstance(exit_signals, ExitSignals) or not exit_signals.days.equals(self.days):
            raise ValueError('Exit signals and account must share the same audited calendar')
        required = {e['members'][0] for group in self.events.values() for e in group}
        if not required.issubset(exit_signals.adjusted_close.columns):
            raise ValueError('Candidate stock missing from adjusted exit signals')
        self.exit_signals, self.mode = exit_signals, mode
        self.exit_states, self.exit_decisions = {}, []
        self.corporate = FractionalCashActions(self.corporate, self)

    def corporate_day(self, day):
        index = self.positions[day]
        # Include pending shares: the original sale can precede delivery.
        active = {h['event_id'] for sid, h in self.holdings.items() if sid != '0050'}
        active |= {r['event_id'] for r in self.receivables if r.get('qty', 0) > 0}
        for cohort in self.cohorts:
            identity, sid = cohort['event_id'], cohort['stock_id']
            if identity not in active:
                continue
            if identity not in self.exit_states:
                entry_index = self.positions[pd.Timestamp(cohort['entry_date'])]
                entry_price = self.exit_signals.price(entry_index, sid)
                self.exit_states[identity] = dict(stock_id=sid, event_id=identity,
                    entry_index=entry_index, entry_price=entry_price, peak_price=entry_price,
                    trigger_reason=None, signal_date=None, target_date=None, target_index=None)
            state = self.exit_states[identity]
            context = self.exit_signals.context(index, sid, state)
            if state['trigger_reason']:
                decision = dict(exit=True, reason=state['trigger_reason'], phase='exiting', extend=False)
            else:
                decision = decide_exit(context, self.mode)
                if decision['exit']:
                    state.update(trigger_reason=decision['reason'],
                        signal_date=str(self.days[index-1].date()),
                        target_date=str(day.date()), target_index=index)
            self.exit_decisions.append(dict(date=str(day.date()), signal_date=str(self.days[index-1].date()),
                stock_id=sid, event_id=identity, mode=self.mode, **context, **decision,
                signal_close=self.exit_signals.price(index-1, sid),
                ma20=_finite(self.exit_signals.ma20.iloc[index-1][sid]),
                market_state=self.exit_signals.trend.state.iloc[index-1],
                entry_price=state['entry_price'], peak_price=state['peak_price'],
                first_signal_date=state['signal_date'], target_date=state['target_date']))
            # The ordinary run loop will sell when its due date arrives.
            if sid in self.holdings and self.holdings[sid]['event_id'] == identity:
                if state['target_index'] is not None:
                    self.holdings[sid]['due_index'] = state['target_index']
                elif decision['extend']:
                    self.holdings[sid]['due_index'] = index + 1
        income = super().corporate_day(day)
        # Late shares can recreate a holding with its original 63-day due
        # date. Restore the first latched instruction after that delivery.
        for sid, holding in self.holdings.items():
            state = self.exit_states.get(holding['event_id'])
            if state and state['target_index'] is not None:
                holding['due_index'] = state['target_index']
        return income

    def order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if self.mode != 'fixed63' and sid != '0050' and side == 'sell' and reason == 'scheduled_exit':
            state = self.exit_states.get(event_id)
            if not state or state['trigger_reason'] is None:
                raise ValueError('Stock sale requires a prior latched exit decision')
            reason, signal_date = state['trigger_reason'], state['signal_date']
        return super().order(day, sid, side, qty, reason, event_id, signal_date)
