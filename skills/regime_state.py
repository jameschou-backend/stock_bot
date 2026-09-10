"""Causal 0050 trend observations and explicitly delayed research decisions.

This module uses supplied daily prices only. A missing current quote produces
UNKNOWN even after 120 earlier observations, and no state is filled forward.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime

import numpy as np
import pandas as pd


STATES = frozenset({'ON', 'OFF', 'UNKNOWN'})
LOOKBACK = 120


def _validate_index(index):
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is not None
            or index.hasnans or not index.is_unique
            or not index.is_monotonic_increasing
            or not index.equals(index.normalize())):
        raise ValueError('An ordered, unique, timezone-naive, date-only DatetimeIndex is required')


def _date(value):
    if not isinstance(value, (str, date, datetime, pd.Timestamp, np.datetime64)):
        raise ValueError('Date must be a timezone-naive date')
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Date must be a timezone-naive date') from exc
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError('Date must be a timezone-naive date')
    return result


def _integer(value, field, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{field} must be an integer of at least {minimum}')
    return int(value)


def _validate_trend(trend):
    if not isinstance(trend, pd.DataFrame):
        raise ValueError('trend must be the daily trend DataFrame')
    _validate_index(trend.index)
    if not {'state', 'evidence_date'}.issubset(trend.columns):
        raise ValueError('trend requires state and evidence_date columns')
    for day, state, evidence in zip(trend.index, trend['state'], trend['evidence_date']):
        if not isinstance(state, str) or state not in STATES:
            raise ValueError('trend state must be ON, OFF or UNKNOWN')
        if state == 'UNKNOWN':
            if evidence is not None and not pd.isna(evidence):
                raise ValueError('UNKNOWN must not claim a valid evidence_date')
        elif _date(evidence) != day:
            raise ValueError('A known trend must use evidence from its own date')


def build_trend(close: pd.Series) -> pd.DataFrame:
    """Compare each valid close strictly with its last 120 observed closes.

    The current positive finite close is included in the mean. Gaps do not use
    up observations. ``ma120`` can show the latest available trailing mean on
    a gap, but the current ``state`` remains UNKNOWN and has no evidence date.
    Observation count is the number of observed prices so far, capped at 120.
    Evidence dates are ISO dates (or None), suitable for the research ledger.
    """
    if not isinstance(close, pd.Series):
        raise ValueError('close must be a pandas Series')
    _validate_index(close.index)
    try:
        numeric = pd.to_numeric(close, errors='raise')
        if np.iscomplexobj(numeric):
            raise ValueError('complex prices are not valid')
        numeric = numeric.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError('close must contain numeric prices or missing values') from exc
    observed = numeric.where(np.isfinite(numeric) & numeric.gt(0))
    count = observed.notna().cumsum().clip(upper=LOOKBACK).astype(int)
    valid = observed.dropna()
    mean = valid.rolling(LOOKBACK, min_periods=LOOKBACK).mean().reindex(close.index).ffill()
    known = observed.notna() & mean.notna() & count.eq(LOOKBACK)
    state = pd.Series('UNKNOWN', index=close.index, dtype=object)
    state.loc[known] = np.where(observed.loc[known].gt(mean.loc[known]), 'ON', 'OFF')
    evidence = [str(day.date()) if is_known else None for day, is_known in zip(close.index, known)]
    return pd.DataFrame({'close': observed, 'ma120': mean, 'state': state,
                         'observation_count': count,
                         'evidence_date': pd.Series(evidence, index=close.index, dtype=object)},
                        index=close.index)


def execution_controls(trend: pd.DataFrame, delay=1) -> pd.DataFrame:
    """Shift observations by market rows; never use the execution-day close.

    decision_date identifies the earlier observed row, including UNKNOWN rows.
    Only initial rows with no preceding decision have decision_date=None.
    Consumers may retain their last known allocation, but this function does
    not turn missing observations into a known state.
    """
    delay = _integer(delay, 'delay', 1)
    _validate_trend(trend)
    states = trend['state'].shift(delay, fill_value='UNKNOWN').astype(object)
    dates = pd.Series([str(day.date()) for day in trend.index], index=trend.index, dtype=object)
    dates = dates.shift(delay, fill_value=None)
    return pd.DataFrame({'state': states, 'decision_date': dates}, index=trend.index)


def gate_events(events, trend: pd.DataFrame, extra_entry_delay=0) -> tuple[list[dict], list[dict]]:
    """Keep original ON signal events, then delay the supplied next-session entry.

    No signal is retimed or rescored. Invalid/missing dates, OFF and UNKNOWN
    observations remain in the rejection ledger. An original entry must be
    exactly the next supplied market session; the optional delay adds market
    rows, not calendar days. All original event fields, members and priorities
    are preserved, except that accepted entry_date reflects the explicit delay.
    """
    extra_entry_delay = _integer(extra_entry_delay, 'extra_entry_delay', 0)
    if extra_entry_delay > 1:
        raise ValueError('extra_entry_delay must be 0 or 1 for the preregistered comparison')
    _validate_trend(trend)
    days = trend.index
    positions = {day: i for i, day in enumerate(days)}
    accepted, rejected = [], []
    for supplied in events:
        if not isinstance(supplied, dict):
            raise ValueError('Every event must be a dictionary')
        event = deepcopy(supplied)
        reason = None
        signal = entry = None
        try:
            signal = _date(event.get('signal_date'))
        except ValueError:
            reason = 'invalid_signal_date'
        if reason is None:
            if not len(days) or signal < days[0] or signal > days[-1]:
                reason = 'signal_date_outside_calendar'
            elif signal not in positions:
                reason = 'signal_date_not_trading_session'
        if reason is None:
            signal_state = trend.at[signal, 'state']
            event['trend_state'] = signal_state
            event['trend_decision_date'] = str(signal.date())
            try:
                entry = _date(event.get('entry_date'))
            except ValueError:
                reason = 'invalid_entry_date'
        if reason is None:
            if entry <= signal:
                reason = 'entry_date_not_after_signal'
            elif entry < days[0] or entry > days[-1]:
                reason = 'entry_date_outside_calendar'
            elif entry not in positions:
                reason = 'entry_date_not_trading_session'
            elif positions[entry] != positions[signal] + 1:
                reason = 'entry_date_not_next_session'
            elif signal_state == 'UNKNOWN':
                reason = 'trend_unknown'
            elif signal_state == 'OFF':
                reason = 'trend_off'
        if reason is None:
            delayed = positions[entry] + extra_entry_delay
            if delayed >= len(days):
                reason = 'entry_delay_outside_calendar'
            else:
                event['entry_date'] = str(days[delayed].date())
        if reason is not None:
            rejected.append({**event, 'reason': reason})
        else:
            accepted.append(event)
    return accepted, rejected
