"""Leave-one-stock-out sector diagnostics from a current membership snapshot.

These features describe relative traded turnover, not net money inflows. The
membership snapshot is explicitly not a historical point-in-time classification.
All price, listing, turnover and observation checks use signal-date history only.
"""
from __future__ import annotations

from collections.abc import Iterable
import re

import numpy as np
import pandas as pd


_RULES = ("breadth_confirmed", "turnover_confirmed", "sector_confirmed")
_COLUMNS = [
    "signal_date", "stock_id", "industry", "membership_point_in_time",
    "expected_peers", "observed_peers", "peer_coverage", "peer_breadth",
    "peer_median_excess20", "share5", "share_previous20", "share_multiple",
    *_RULES, "peer_ids", "market_min_coverage25",
]


def _calendar(values, label: str) -> pd.DatetimeIndex:
    try:
        days = pd.DatetimeIndex(pd.to_datetime(values))
    except (TypeError, ValueError) as exc:
        raise ValueError(label + " must contain valid market dates") from exc
    if days.hasnans or days.tz is not None or not days.equals(days.normalize()):
        raise ValueError(label + " must contain timezone-naive dates without times")
    return days


def _prepare(close, raw, volume, companies, members, signal_dates):
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("Price index must be a DatetimeIndex")
    days = _calendar(close.index, "Price index")
    if not days.is_unique or not days.is_monotonic_increasing:
        raise ValueError("Market calendar must be unique and increasing")
    if not close.columns.is_unique or "0050" not in close:
        raise ValueError("Prices require unique stock columns and the 0050 benchmark")
    if any(not isinstance(sid, str) or not re.fullmatch(r"\d{4}", sid) for sid in close.columns):
        raise ValueError("Price columns must be four-digit stock-id strings")
    for frame in (raw, volume):
        if not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError("Price/volume dates and stock columns must align exactly")
    if not {"stock_id", "listed_date"}.issubset(companies.columns):
        raise ValueError("Companies require stock_id and listed_date")
    if not {"stock_id", "industry"}.issubset(members.columns):
        raise ValueError("Members require stock_id and industry")
    if companies.stock_id.duplicated().any():
        raise ValueError("Company identities must be unique")
    for values in (companies.stock_id, members.stock_id):
        if any(not isinstance(sid, str) or not re.fullmatch(r"\d{4}", sid) for sid in values):
            raise ValueError("Company/member ids must be four-digit stock-id strings")
    if any(not isinstance(group, str) or not group.strip() for group in members.industry):
        raise ValueError("Industry identifiers must be nonempty strings")
    listed_dates = _calendar(companies.listed_date, "Listing dates")
    cohort = companies.assign(listed_date=listed_dates).loc[companies.stock_id.ne("0050")]
    cohort = cohort.set_index("stock_id").sort_index()
    membership = members.loc[members.stock_id.ne("0050"), ["stock_id", "industry"]].drop_duplicates()
    missing = sorted(set(membership.stock_id) - set(cohort.index))
    if missing:
        raise ValueError("Membership has no known company/listing date: " + ", ".join(missing[:10]))
    selected = _calendar(list(signal_dates), "Signal dates").drop_duplicates().sort_values()
    positions = days.get_indexer(selected)
    if (positions < 0).any():
        raise ValueError("Every signal date must be in the supplied market calendar")
    return days, cohort, membership, selected, positions


def _complete_windows(valid: np.ndarray, window: int) -> np.ndarray:
    """Compute all stock/date completeness checks in one cumulative pass."""
    cumulative = np.empty((len(valid) + 1, valid.shape[1]), dtype=np.int32)
    cumulative[0] = 0
    np.cumsum(valid, axis=0, dtype=np.int32, out=cumulative[1:])
    complete = np.zeros(valid.shape, dtype=bool)
    if len(valid) >= window:
        complete[window - 1:] = (cumulative[window:] - cumulative[:-window]) == window
    return complete


def _exclusive_sums(values: np.ndarray) -> np.ndarray:
    """Sum each row except one column, without subtracting its target value.

    Prefix/suffix sums prevent the target's extreme volume from changing its own
    peers through floating-point cancellation in ``total - target``.
    """
    prefix = np.zeros((values.shape[0], values.shape[1] + 1), dtype=float)
    suffix = np.zeros_like(prefix)
    with np.errstate(over="ignore", invalid="ignore"):
        np.cumsum(values, axis=1, out=prefix[:, 1:])
        suffix[:, :-1] = np.cumsum(values[:, ::-1], axis=1)[:, ::-1]
        return prefix[:, :-1] + suffix[:, 1:]


def _peer_medians(excess: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """All leave-one-out medians, sorting each group's observed values once."""
    result = np.full(len(excess), np.nan)
    indices = np.flatnonzero(observed)
    if not len(indices):
        return result
    ranked = indices[np.argsort(excess[indices], kind="stable")]
    values = excess[ranked]
    result[:] = np.median(values)
    n = len(values) - 1
    if not n:
        result[ranked] = np.nan
        return result
    low, high = (n - 1) // 2, n // 2
    removed = np.arange(len(values))
    result[ranked] = (values[low + (removed <= low)] + values[high + (removed <= high)]) / 2
    return result


def sector_features(
    close: pd.DataFrame,
    raw: pd.DataFrame,
    volume: pd.DataFrame,
    companies: pd.DataFrame,
    members: pd.DataFrame,
    signal_dates: Iterable,
) -> pd.DataFrame:
    """Return one row per listed member target, sector, and requested signal date.

    Expected peers include listed members with missing price columns. Observed
    peers require 61 complete positive price observations, 25 complete positive
    raw-price/share-volume observations, and no absolute daily move above 15%
    over the last 60 returns. The eligible peer set at T stays fixed throughout
    the 25-day turnover calculation. The target is excluded everywhere.

    Peer coverage must be at least 80%, with at least three observed peers, for
    either rule. Turnover also needs 95% market coverage on each of 25 dates;
    the market universe is every listed company, excluding 0050 and the target.
    Insufficient evidence gives nullable unknowns, never a negative observation.
    """
    days, cohort, membership, selected, positions = _prepare(
        close, raw, volume, companies, members, signal_dates,
    )
    ids = list(cohort.index)
    id_positions = {sid: i for i, sid in enumerate(ids)}
    listed = days.to_numpy()[:, None] >= cohort.listed_date.to_numpy()[None, :]
    try:
        prices = close.reindex(columns=ids).to_numpy(dtype=float)
        raw_prices = raw.reindex(columns=ids).to_numpy(dtype=float)
        shares = volume.reindex(columns=ids).to_numpy(dtype=float)
        benchmark = close["0050"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prices and volume must be numeric") from exc
    price_valid = np.isfinite(prices) & (prices > 0) & listed
    with np.errstate(over="ignore", invalid="ignore"):
        amounts = raw_prices * shares
    amount_valid = (np.isfinite(raw_prices) & (raw_prices > 0) & np.isfinite(shares)
                    & (shares > 0) & np.isfinite(amounts) & (amounts > 0) & listed)
    amounts = np.where(amount_valid, amounts, 0.)
    complete_prices = _complete_windows(price_valid, 61)
    complete_amounts = _complete_windows(amount_valid, 25)
    acceptable_return = np.zeros_like(price_valid)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        daily_returns = prices[1:] / prices[:-1] - 1
    acceptable_return[1:] = (price_valid[1:] & price_valid[:-1]
                             & np.isfinite(daily_returns) & (np.abs(daily_returns) <= .15))
    observed = complete_prices & complete_amounts & _complete_windows(acceptable_return, 60)
    groups = [(industry, np.array([id_positions[sid] for sid in sorted(rows.stock_id)], dtype=int))
              for industry, rows in membership.groupby("industry", sort=True)]
    output = []
    for signal, pos in zip(selected, positions):
        active = listed[pos]
        benchmark_known = (pos >= 20 and np.isfinite(benchmark[[pos - 20, pos]]).all()
                           and (benchmark[[pos - 20, pos]] > 0).all())
        benchmark_return = np.nan
        momentum = np.full(len(ids), np.nan)
        if pos >= 20:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                momentum = prices[pos] / prices[pos - 20] - 1
                if benchmark_known:
                    benchmark_return = benchmark[pos] / benchmark[pos - 20] - 1
            benchmark_known = benchmark_known and np.isfinite(benchmark_return)
        market_coverage = np.full(len(ids), np.nan)
        market_ok = np.zeros(len(ids), dtype=bool)
        denominators = None
        window = amounts[max(0, pos - 24):pos + 1]
        if pos >= 24:
            valid_window = amount_valid[pos - 24:pos + 1]
            listed_window = listed[pos - 24:pos + 1]
            expected_market = listed_window.sum(axis=1)[:, None] - listed_window
            observed_market = valid_window.sum(axis=1)[:, None] - valid_window
            coverage = np.divide(observed_market, expected_market,
                                 out=np.full(expected_market.shape, np.nan), where=expected_market > 0)
            market_coverage = np.min(coverage, axis=0)
            denominators = _exclusive_sums(window)
            market_ok = ((coverage >= .95).all(axis=0) & np.isfinite(denominators).all(axis=0)
                         & (denominators > 0).all(axis=0))
        for industry, all_indices in groups:
            indices = all_indices[active[all_indices]]
            if not len(indices):
                continue
            good = observed[pos, indices]
            observed_count = int(good.sum())
            expected_count = len(indices) - 1
            counts = observed_count - good.astype(int)
            peer_coverage = counts / expected_count if expected_count else np.full(len(indices), np.nan)
            sufficient = (counts >= 3) & (peer_coverage >= .8)
            breadth = np.full(len(indices), np.nan)
            median_excess = np.full(len(indices), np.nan)
            if benchmark_known:
                excess = momentum[indices] - benchmark_return
                positive = good & (momentum[indices] > 0) & (excess > 0)
                breadth = np.divide(int(positive.sum()) - positive.astype(int), counts,
                                    out=np.full(len(indices), np.nan), where=counts > 0)
                median_excess = _peer_medians(excess, good)
            share5 = np.full(len(indices), np.nan)
            previous20 = np.full(len(indices), np.nan)
            multiple = np.full(len(indices), np.nan)
            turnover_known = sufficient & market_ok[indices]
            if pos >= 24:
                peer_amounts = np.where(good[None, :], window[:, indices], 0.)
                numerator = _exclusive_sums(peer_amounts)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    daily_share = numerator / denominators[:, indices]
                    current_share = daily_share[-5:].mean(axis=0)
                    earlier_share = daily_share[:20].mean(axis=0)
                    share_ratio = current_share / earlier_share
                values_known = (market_ok[indices] & (counts > 0) & np.isfinite(daily_share).all(axis=0)
                                & np.isfinite(share_ratio) & (earlier_share > 0))
                share5[values_known] = current_share[values_known]
                previous20[values_known] = earlier_share[values_known]
                multiple[values_known] = share_ratio[values_known]
                turnover_known &= values_known
            for j, target in enumerate(indices):
                peer_indices = indices[good & (indices != target)]
                breadth_rule = bool(breadth[j] >= .6) if sufficient[j] and benchmark_known else pd.NA
                turnover_rule = bool(multiple[j] >= 1.2) if turnover_known[j] else pd.NA
                output.append({
                    "signal_date": signal.strftime("%Y-%m-%d"), "stock_id": ids[target], "industry": industry,
                    "membership_point_in_time": False, "expected_peers": int(expected_count),
                    "observed_peers": int(counts[j]), "peer_coverage": float(peer_coverage[j]),
                    "peer_breadth": float(breadth[j]), "peer_median_excess20": float(median_excess[j]),
                    "share5": float(share5[j]), "share_previous20": float(previous20[j]),
                    "share_multiple": float(multiple[j]), "breadth_confirmed": breadth_rule,
                    "turnover_confirmed": turnover_rule,
                    "sector_confirmed": pd.NA,
                    "peer_ids": [ids[index] for index in peer_indices],
                    "market_min_coverage25": float(market_coverage[target]),
                })
    result = pd.DataFrame(output, columns=_COLUMNS)
    for rule in _RULES:
        result[rule] = result[rule].astype("boolean")
    result["sector_confirmed"] = result["breadth_confirmed"] & result["turnover_confirmed"]
    return result
