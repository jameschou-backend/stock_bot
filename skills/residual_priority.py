"""Rescore fixed leader events using a causal, unscaled residual-return sum.

This module never changes an event's stock, dates or members. It is a same-day
processing priority experiment, not a new stock universe or group-leader search.
See docs/prereg_capacity_20260910.md for the two non-overlapping windows.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import math
import re

import numpy as np
import pandas as pd


FIT_RETURNS = 126
SCORE_RETURNS = 20
MIN_MARKET_VARIANCE = 1e-12
MAX_ABS_RETURN = .20
MAX_BASIS_DIFFERENCE = .005


def _date(value, name):
    if not isinstance(value, (str, date, datetime, pd.Timestamp, np.datetime64)):
        raise ValueError(f'{name} must be a timezone-naive date')
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a timezone-naive date') from exc
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError(f'{name} must be a timezone-naive date')
    return result


def _validate_prices(close, other_close):
    if (not isinstance(close, pd.DataFrame) or not isinstance(other_close, pd.DataFrame)
            or not isinstance(close.index, pd.DatetimeIndex) or close.index.tz is not None
            or close.index.hasnans or not close.index.is_unique or not close.index.is_monotonic_increasing
            or not close.index.equals(close.index.normalize()) or not close.columns.is_unique
            or '0050' not in close.columns
            or any(not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid) for sid in close.columns)
            or not close.index.equals(other_close.index) or not close.columns.equals(other_close.columns)):
        raise ValueError('Price bases require identical ordered unique date-only indices and four-digit columns including 0050')


def rescore_events(close: pd.DataFrame, events: list[dict], other_close: pd.DataFrame) -> dict:
    """Return rescored copies, explicit rejections, and one diagnostic per event.

    For signal row T, fit OLS on r[T-145:T-19] (126 simple daily returns),
    then sum residuals on r[T-19:T+1] (20 returns, including T). This needs
    147 consecutive supplied market-row prices, T-146 through T. Both price
    bases must have finite positive quotes throughout; no forward filling.

    All 146 returns in both assets/bases also pass the fixed anomaly limits.
    The 0050 fit-return *population variance* must exceed 1e-12. Missing or
    degenerate data is rejected, never replaced with the original priority.
    Negative and zero scores remain eligible; no extra score filter is added.

    Entry dates must follow signals, but need not appear in the supplied price
    prefix: execution-calendar checks belong to the portfolio engine. Only
    an exact signal-date row is used; missing signal dates are never shifted.
    """
    _validate_prices(close, other_close)
    if not isinstance(events, list):
        raise ValueError('events must be a list of dictionaries')
    positions = {day: i for i, day in enumerate(close.index)}
    normalized, ids = [], set()
    for event in events:
        if not isinstance(event, dict) or not {'event_id', 'signal_date', 'entry_date', 'members', 'priority'}.issubset(event):
            raise ValueError('Each event requires event_id, signal_date, entry_date, members and priority')
        identity = event['event_id']
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError('Event IDs must be nonempty unique strings')
        ids.add(identity)
        members = event['members']
        if (not isinstance(members, (list, tuple)) or len(members) != 1
                or not isinstance(members[0], str) or not re.fullmatch(r'\d{4}', members[0])
                or members[0] == '0050'):
            raise ValueError('Each event must have one four-digit stock member excluding 0050')
        signal = _date(event['signal_date'], 'signal_date')
        entry = _date(event['entry_date'], 'entry_date')
        if entry <= signal:
            raise ValueError('entry_date must be strictly after signal_date')
        try:
            priority = float(event['priority'])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Original priority must be finite') from exc
        if isinstance(event['priority'], (bool, np.bool_)) or not math.isfinite(priority):
            raise ValueError('Original priority must be finite')
        normalized.append((event, signal, members[0], priority))

    rescored, rejected, diagnostics = [], [], []
    needed_returns = FIT_RETURNS + SCORE_RETURNS
    for original, signal, sid, original_priority in normalized:
        diagnostic = {'event_id': original['event_id'], 'stock_id': sid,
                      'signal_date': str(signal.date()), 'entry_date': str(_date(original['entry_date'], 'entry_date').date()),
                      'fit_start': None, 'fit_end': None, 'score_start': None, 'score_end': None,
                      'fit_observations': None, 'score_observations': None,
                      'required_fit_observations': FIT_RETURNS, 'required_score_observations': SCORE_RETURNS,
                      'alpha': None, 'beta': None, 'market_fit_variance': None,
                      'original_priority': original_priority, 'residual_score': None,
                      'missing_price_dates': [], 'anomaly_dates': [], 'reason': None}
        if signal not in positions:
            reason = 'signal_date_not_trading_session'
        elif sid not in close.columns:
            reason = 'stock_not_in_prices'
        elif positions[signal] < needed_returns:
            reason = 'insufficient_history'
            diagnostic['available_prices'] = positions[signal] + 1
        else:
            end = positions[signal]
            first = end - needed_returns
            window_days = close.index[first:end + 1]
            diagnostic.update(fit_start=str(window_days[1].date()),
                              fit_end=str(window_days[FIT_RETURNS].date()),
                              score_start=str(window_days[FIT_RETURNS + 1].date()),
                              score_end=str(window_days[-1].date()))
            # Convert only this event's past window. A bad future quote must
            # never invalidate an earlier score or make a prefix behave differently.
            frames = [frame.iloc[first:end + 1].loc[:, ['0050', sid]] for frame in (close, other_close)]
            values = [frame.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float) for frame in frames]
            invalid = [~np.isfinite(value) | (value <= 0) for value in values]
            if any(mask.any() for mask in invalid):
                bad_rows = np.any(invalid[0] | invalid[1], axis=1)
                diagnostic['missing_price_dates'] = [str(day.date()) for day in window_days[bad_rows]]
                reason = 'missing_prices'
            else:
                with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                    returns, other_returns = [value[1:] / value[:-1] - 1 for value in values]
                if not np.isfinite(returns).all() or not np.isfinite(other_returns).all():
                    reason = 'missing_returns'
                else:
                    diagnostic.update(fit_observations=FIT_RETURNS, score_observations=SCORE_RETURNS)
                    anomalous = ((np.abs(returns) > MAX_ABS_RETURN)
                                 | (np.abs(other_returns) > MAX_ABS_RETURN)
                                 | (np.abs(returns - other_returns) > MAX_BASIS_DIFFERENCE))
                    if anomalous.any():
                        diagnostic['anomaly_dates'] = [str(day.date()) for day in window_days[1:][anomalous.any(axis=1)]]
                        reason = 'price_anomaly'
                    else:
                        x, y = returns[:FIT_RETURNS, 0], returns[:FIT_RETURNS, 1]
                        x_centered, y_centered = x - x.mean(), y - y.mean()
                        variance = float(np.mean(x_centered * x_centered))
                        diagnostic['market_fit_variance'] = variance
                        if not math.isfinite(variance) or variance <= MIN_MARKET_VARIANCE:
                            reason = 'benchmark_variance_too_small'
                        else:
                            beta = float(np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered))
                            alpha = float(y.mean() - beta * x.mean())
                            if not math.isfinite(alpha) or not math.isfinite(beta):
                                reason = 'nonfinite_regression'
                            else:
                                diagnostic.update(alpha=alpha, beta=beta)
                                score = float(np.sum(returns[FIT_RETURNS:, 1] - alpha - beta * returns[FIT_RETURNS:, 0]))
                                if not math.isfinite(score):
                                    reason = 'nonfinite_score'
                                else:
                                    diagnostic['residual_score'] = score
                                    reason = 'scored'
        diagnostic['reason'] = reason
        diagnostics.append(diagnostic)
        if reason == 'scored':
            copy = deepcopy(original)
            copy['priority'] = diagnostic['residual_score']
            rescored.append(copy)
        else:
            rejected.append({**deepcopy(original), 'reason': reason})
    return {'events': rescored, 'rejections': rejected, 'diagnostics': diagnostics}
