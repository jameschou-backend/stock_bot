"""Causal fixed-range volume profiles from authentic regular-board trade ticks.

The caller must normalize exchange/provider units to shares and independently
verify the source, market, complete session downloads and corporate actions.
This module cannot turn daily OHLCV, quote snapshots or a volume allocation
approximation into an authentic transaction-price profile.

Algorithm fixed before outcome research: 40 equal-width bins by default; one
common range for the full window and its chronological halves. The highest
volume bin is POC (ties choose lower price). A contiguous 70% value area grows
from POC toward the adjacent bin with more volume (ties choose lower price).
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd


SOURCE_KIND = "authentic_regular_board_trade_ticks"
DEFAULT_BINS = 40
DEFAULT_VALUE_FRACTION = .70
ALGORITHM_VERSION = "fixed_range_contiguous_value_area_v1"


def _naive_date(value, label):
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(label + " must be a naive Taipei calendar date") from exc
    if pd.isna(stamp) or stamp.tz is not None or stamp != stamp.normalize():
        raise ValueError(label + " must be a naive Taipei calendar date at midnight")
    return stamp


def _unavailable(reason, bin_volumes=None):
    return dict(available=False, status="unavailable", reason=reason,
        total_shares=0., bin_volumes=bin_volumes, occupied_bins=0,
        poc_index=None, poc_price=None, poc_low=None, poc_high=None, poc_shares=None,
        value_area_low_index=None, value_area_high_index=None, val=None, vah=None,
        value_area_shares=None, value_area_fraction=None, expansion_order=[])


def _profile(edges, volumes, value_fraction):
    """Histogram calculation only; never assigns daily volume to prices."""
    try:
        total = math.fsum(volumes)
    except OverflowError as exc:
        raise ValueError("Aggregated trade shares must remain finite") from exc
    if not math.isfinite(total):
        raise ValueError("Aggregated trade shares must remain finite")
    if total == 0:
        return _unavailable("zero_total_shares", list(volumes))
    maximum = max(volumes)
    poc = volumes.index(maximum)  # First maximum is the lower-price bin.
    low = high = poc
    order = [poc]
    covered = volumes[poc]
    target = total * value_fraction
    while covered < target and (low > 0 or high < len(volumes) - 1):
        if low == 0:
            high += 1
            order.append(high)
        elif high == len(volumes) - 1:
            low -= 1
            order.append(low)
        elif volumes[low - 1] >= volumes[high + 1]:
            low -= 1
            order.append(low)
        else:
            high += 1
            order.append(high)
        covered = math.fsum(volumes[low:high + 1])
    return dict(available=True, status="available", reason=None,
        total_shares=total, bin_volumes=list(volumes),
        occupied_bins=sum(v > 0 for v in volumes), poc_index=poc,
        poc_price=edges[poc] + (edges[poc + 1] - edges[poc]) / 2,
        poc_low=edges[poc], poc_high=edges[poc + 1], poc_shares=maximum,
        value_area_low_index=low, value_area_high_index=high,
        val=edges[low], vah=edges[high + 1], value_area_shares=covered,
        value_area_fraction=covered / total, expansion_order=order)


def _histogram(indices, shares, bins):
    """Order-invariant sums preserve repeated executions and fractional shares."""
    if not len(indices):
        return [0.] * bins
    order = np.argsort(indices, kind="stable")
    sorted_indices, sorted_shares = indices[order], shares[order]
    boundaries = np.searchsorted(sorted_indices, np.arange(bins + 1), side="left")
    try:
        result = [math.fsum(sorted_shares[boundaries[i]:boundaries[i + 1]]) for i in range(bins)]
    except OverflowError as exc:
        raise ValueError("Aggregated trade shares must remain finite") from exc
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Aggregated trade shares must remain finite")
    return result


def build_volume_profile(ticks, *, signal_date, session_dates, source_kind,
                         bins=DEFAULT_BINS, value_fraction=DEFAULT_VALUE_FRACTION):
    """Build a JSON-serializable full/first-half/second-half volume profile.

    ``ticks`` is a DataFrame with ``timestamp``, ``price`` and ``shares``.
    Timestamps must be timezone-naive Taipei exchange times. All rows must be
    strictly before ``signal_date`` and inside ``session_dates``. No rows are
    deduplicated: equal execution timestamps and prices may be real trades.

    ``session_dates`` is an ordered, unique even number of market sessions >=2
    (the preregistered caller uses 20). Its two chronological halves share edges
    determined by all *positive-volume* executions in this past-only window.
    Zero-share rows are validated and counted but do not widen the price range.

    Normal bins are [low, high), except the final bin includes the maximum.
    For a single traded price, all edges equal that price and volume is placed
    in bin0: POC=VAL=VAH exactly that observed price, without an invented range.

    Missing trading dates are reported, not inferred to be failed downloads or
    filled with synthetic trades. The caller must distinguish a confirmed
    zero-trade day from an incomplete response. Empty/zero-volume profiles and
    empty halves remain unavailable; invalid data raises an explicit error.
    """
    if source_kind != SOURCE_KIND:
        raise ValueError("Only authentic regular-board trade ticks are accepted; daily OHLCV/proxies are unsupported")
    if not isinstance(bins, Integral) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    bins = int(bins)
    if not isinstance(value_fraction, Real) or isinstance(value_fraction, bool) or not math.isfinite(value_fraction) or not 0 < value_fraction <= 1:
        raise ValueError("value_fraction must be finite and in (0, 1]")
    value_fraction = float(value_fraction)
    signal = _naive_date(signal_date, "signal_date")
    dates = pd.DatetimeIndex([_naive_date(d, "session_dates") for d in session_dates])
    if len(dates) < 2 or len(dates) % 2 or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("session_dates must be unique, ordered and contain an even number of sessions >=2")
    if not (dates < signal).all():
        raise ValueError("Every profile session must be strictly before signal_date")
    if not isinstance(ticks, pd.DataFrame) or not {"timestamp", "price", "shares"}.issubset(ticks.columns):
        raise ValueError("Normalized ticks require timestamp, price and shares columns")
    if not ticks.columns.is_unique:
        raise ValueError("Tick columns must be unique")
    if pd.api.types.is_numeric_dtype(ticks["timestamp"].dtype) and len(ticks):
        raise ValueError("Tick timestamps must be explicit naive Taipei datetimes, not numeric epochs")
    try:
        stamps = pd.DatetimeIndex(ticks["timestamp"])
        prices = ticks["price"].to_numpy(dtype=float)
        shares = ticks["shares"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Normalized timestamps, numeric prices and share volumes are required") from exc
    if stamps.tz is not None or stamps.hasnans:
        raise ValueError("Tick timestamps must be nonmissing timezone-naive Taipei datetimes")
    if not (stamps < signal).all():
        raise ValueError("Trade ticks at or after signal_date are forbidden")
    trade_dates = stamps.normalize()
    if not trade_dates.isin(dates).all():
        raise ValueError("Trade tick date lies outside the declared profile sessions")
    if not (np.isfinite(prices) & (prices > 0)).all():
        raise ValueError("Trade prices must be finite and strictly positive")
    if not (np.isfinite(shares) & (shares >= 0)).all():
        raise ValueError("Trade share volumes must be finite and nonnegative")
    halfway = len(dates) // 2
    first_mask = trade_dates.isin(dates[:halfway])
    positive = shares > 0
    observed = set(trade_dates)
    result = dict(algorithm_version=ALGORITHM_VERSION, source_kind=SOURCE_KIND,
        timezone="Asia/Taipei; naive exchange timestamps", volume_unit="shares",
        signal_date=str(signal.date()), session_dates=[str(d.date()) for d in dates],
        first_half_sessions=[str(d.date()) for d in dates[:halfway]],
        second_half_sessions=[str(d.date()) for d in dates[halfway:]],
        observed_sessions=[str(d.date()) for d in dates if d in observed],
        sessions_without_tick_rows=[str(d.date()) for d in dates if d not in observed],
        source_completeness_verified_by_module=False,
        corporate_action_consistency_verified_by_module=False,
        input_tick_count=len(ticks), positive_volume_tick_count=int(positive.sum()),
        zero_volume_tick_count=int((~positive).sum()), first_half_tick_count=int(first_mask.sum()),
        second_half_tick_count=int((~first_mask).sum()), bins=bins,
        target_value_fraction=value_fraction, bin_edges=None, single_price_range=False,
        poc_shift=None, poc_shift_fraction=None, poc_up=None, halves_comparable=False)
    if not positive.any():
        reason = "empty_ticks" if not len(ticks) else "zero_total_shares"
        result.update(available=False, status="unavailable", reason=reason,
            full=_unavailable(reason), first_half=_unavailable(reason), second_half=_unavailable(reason))
        return result
    minimum, maximum = float(prices[positive].min()), float(prices[positive].max())
    single = minimum == maximum
    edges = np.full(bins + 1, minimum) if single else np.linspace(minimum, maximum, bins + 1)
    if not np.isfinite(edges).all():
        raise ValueError("Trade price range cannot form finite bin edges")
    if not single and not (np.diff(edges) > 0).all():
        raise ValueError("Trade price precision cannot represent the requested distinct bin edges")
    # Zero-volume quotes outside the executed range contribute nothing; clipping
    # their indices cannot add volume and does not create a synthetic execution.
    indices = np.zeros(len(prices), dtype=int) if single else np.clip(np.searchsorted(edges, prices, side="right") - 1, 0, bins - 1)
    full = _profile(edges.tolist(), _histogram(indices, shares, bins), value_fraction)
    first = _profile(edges.tolist(), _histogram(indices[first_mask], shares[first_mask], bins), value_fraction)
    second = _profile(edges.tolist(), _histogram(indices[~first_mask], shares[~first_mask], bins), value_fraction)
    comparable = first["available"] and second["available"]
    result.update(available=True, status="available", reason=None, bin_edges=edges.tolist(),
                  single_price_range=single, full=full, first_half=first, second_half=second,
                  halves_comparable=comparable)
    if comparable:
        shift = second["poc_price"] - first["poc_price"]
        result.update(poc_shift=shift, poc_shift_fraction=second["poc_price"] / first["poc_price"] - 1,
                      poc_up=shift > 0)
    return result
