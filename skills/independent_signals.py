"""Unit-return diagnostics, deliberately separate from executable cash accounts."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from skills.exit_policy import decide_exit
from skills.million_replay import COMMISSION, SLIPPAGE


@dataclass(frozen=True)
class SignalPath:
    days: pd.DatetimeIndex
    close: np.ndarray
    other: np.ndarray
    eligible: np.ndarray
    raw_close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray

    def validate(self):
        if self.days.hasnans or not self.days.is_unique or not self.days.is_monotonic_increasing:
            raise ValueError('Unique ordered calendar required')
        if any(len(v) != len(self.days) for v in (
            self.close, self.other, self.eligible, self.raw_close,
            self.high, self.low, self.volume,
        )):
            raise ValueError('Path columns must share the calendar')
        if pd.isna(self.eligible).any():
            raise ValueError('Eligibility cannot be unknown')
        if self.eligible.dtype != np.dtype(bool):
            raise ValueError('Eligibility must contain explicit booleans')


def net_unit_return(ratio):
    """Proportional fees only; never an integer-share cash-account result."""
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('Positive finite price ratio required')
    return ratio * (1 - COMMISSION - SLIPPAGE - .003) / (1 + COMMISSION + SLIPPAGE) - 1


def _decision(age, current, anchor):
    valid = bool(np.isfinite(current) and current > 0 and np.isfinite(anchor) and anchor > 0)
    return decide_exit(dict(
        held_sessions=age, has_signal=valid,
        entry_return=float(current / anchor - 1) if valid else None,
        peak_return=None, peak_drawdown=None, relative20=None,
        below_ma20_two=False, market_off_two=False, strong_trend=False,
    ), 'loss12')


def observe(path, entry_index):
    """T+1 HL2 buy; close-only trigger followed by the NEXT session HL2 sell."""
    path.validate()
    if type(entry_index) is not int or not 1 <= entry_index < len(path.days):
        raise ValueError('Entry needs a previous signal session')
    s, last = entry_index, len(path.days) - 1
    result = dict(status='unknown', reason=None, exit_signal_date=None,
                  exit_date=None, observed_end_date=None, issue=None)
    anchor = path.close[s]
    end, trigger = last, None
    # Decisions are evaluated at j's close, and request execution at j+1.
    for j in range(s, min(s + 62, last) + 1):
        d = _decision(j + 1 - s, path.close[j], anchor)
        if d['exit']:
            trigger = j
            result.update(reason=d['reason'], exit_signal_date=str(path.days[j].date()))
            end = j + 1 if j < last else last
            break
    closed = trigger is not None and trigger < last
    result.update(observed_end_date=str(path.days[end].date()),
                  exit_date=str(path.days[end].date()) if closed else None,
                  held_sessions=end - s + 1)
    sl = slice(s, end + 1)
    c, o = path.close[sl], path.other[sl]
    raw, high, low, vol = [v[sl] for v in (path.raw_close, path.high, path.low, path.volume)]
    if not all((np.isfinite(v) & (v > 0)).all() for v in (c, o, raw, high, low)):
        result['issue'] = 'missing_or_invalid_price'
    elif not path.eligible[sl].all():
        result['issue'] = 'historical_identity_or_eligibility'
    elif not ((low <= raw) & (raw <= high)).all():
        result['issue'] = 'raw_ohlc_conflict'
    elif not (np.isfinite(vol) & (vol >= 0)).all():
        result['issue'] = 'missing_or_invalid_volume'
    elif not vol[0] > 0 or (closed and not vol[-1] > 0):
        result['issue'] = 'no_volume_on_assumed_fill'
    else:
        a, b = c[1:] / c[:-1] - 1, o[1:] / o[:-1] - 1
        if ((abs(a) > .20) | (abs(b) > .20) | (abs(a - b) > .005)).any():
            result['issue'] = 'daily_adjustment_conflict'
        elif abs(c[-1] / c[0] - o[-1] / o[0]) > .02:
            result['issue'] = 'cumulative_adjustment_conflict'
    if result['issue']:
        # Missing earlier prices could hide the first trigger. Do not retain a
        # subsequently observed instruction as a verified original exit.
        result.update(reason=None, exit_date=None, exit_signal_date=None)
        return result
    raw_entry = float((path.high[s] + path.low[s]) / 2)
    adjusted_entry = raw_entry * float(path.close[s] / path.raw_close[s])
    raw_end = float((path.high[end] + path.low[end]) / 2) if closed else float(path.raw_close[end])
    adjusted_end = raw_end * float(path.close[end] / path.raw_close[end])
    gross = adjusted_end / adjusted_entry - 1
    pnl = net_unit_return(1 + gross)
    # Exit day close is not known before a hypothetical intraday sale.
    history = path.close[s:end] if closed else path.close[s:end + 1]
    gains = history / adjusted_entry - 1
    result.update(
        status='closed' if closed else 'pending_exit' if trigger is not None else 'open',
        raw_entry_price=raw_entry, raw_exit_price=raw_end if closed else None,
        raw_mark_price=None if closed else raw_end,
        entry_adjusted_price=float(adjusted_entry), end_adjusted_price=float(adjusted_end),
        gross_return=float(gross), net_return=float(pnl) if closed else None,
        unrealized_net_return=float(pnl) if not closed else None,
        outcome=('profit' if pnl > 1e-12 else 'loss' if pnl < -1e-12 else 'flat') if closed else 'unrealized',
        peak_close_return=float(max(gains)), trough_close_return=float(min(gains)),
        entry_single_price=bool(path.high[s] == path.low[s]),
        exit_single_price=bool(closed and path.high[end] == path.low[end]),
        entry_volume=float(path.volume[s]), exit_volume=float(path.volume[end]) if closed else None,
    )
    for pct in (5, 10, 20):
        hits = np.flatnonzero(gains >= pct / 100 - 1e-12)
        result[f'first_{pct}pct_day'] = int(hits[0] + 1) if len(hits) else None
    return result
