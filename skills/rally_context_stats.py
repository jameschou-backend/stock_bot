"""Descriptive, point-in-time filter comparisons for fixed-horizon events.

This module never builds a trading account or searches for a profitable filter.
Returns and rates use fractions (0.30 means 30%).  The input's full daily path
must have been checked by its producer before setting ``complete=True``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


_EVENT_KEYS = ("cohort", "horizon", "event_id")
_GROUP_KEYS = ("cohort", "horizon", "stock_id")
_BASE_COLUMNS = (*_EVENT_KEYS, "stock_id", "signal_date", "signal_index")
_OUTCOME_COLUMNS = (
    "entry_date", "exit_date", "mature", "complete", "gross_return",
    "net_return", "benchmark_net_return", "mfe", "mae", "threshold",
)


def _require_columns(frame: pd.DataFrame, names: Sequence[str]) -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing event columns: {', '.join(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate DataFrame column names are not supported")


def _dates(series: pd.Series, name: str, *, nullable: bool) -> pd.Series:
    nonmissing = series.notna()
    text = series.loc[nonmissing].astype(str)
    if not text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError(f"{name} must contain ISO YYYY-MM-DD dates")
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    if (nonmissing & parsed.isna()).any() or (not nullable and parsed.isna().any()):
        raise ValueError(f"{name} contains invalid or missing dates")
    return parsed


def _boolean(series: pd.Series, name: str, *, nullable: bool) -> pd.Series:
    # Do not accept strings, numbers, or fill unknown values with False.
    known = series.loc[series.notna()]
    if not known.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{name} must contain booleans, not strings or numbers")
    if not nullable and series.isna().any():
        raise ValueError(f"{name} cannot contain unknown values")
    return series.astype("boolean")


def _validated_events(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _BASE_COLUMNS)
    data = frame.copy()
    for name in ("cohort", "event_id", "stock_id"):
        if data[name].isna().any() or not data[name].map(
            lambda value: isinstance(value, str) and bool(value)
        ).all():
            raise ValueError(f"{name} must contain nonempty strings")
    for name in ("horizon", "signal_index"):
        values = pd.to_numeric(data[name], errors="coerce")
        if not (np.isfinite(values) & (values == np.floor(values))).all():
            raise ValueError(f"{name} must contain finite integer values")
        if (values < (1 if name == "horizon" else 0)).any():
            raise ValueError(f"{name} is outside the valid range")
        data[name] = values.astype("int64")
    if data.duplicated(list(_EVENT_KEYS)).any():
        raise ValueError("Duplicate (cohort, horizon, event_id) event identities")
    signal_dates = _dates(data["signal_date"], "signal_date", nullable=False)
    data["signal_date"] = signal_dates.dt.strftime("%Y-%m-%d")
    # The event-id tie break makes selection independent of input row order.
    data = data.sort_values(
        [*_GROUP_KEYS, "signal_index", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    for _, group in data.groupby(list(_GROUP_KEYS), sort=False, observed=True):
        if not group["signal_date"].is_monotonic_increasing:
            raise ValueError("signal_index order disagrees with signal_date order")
        if group.groupby("signal_index")["signal_date"].nunique().gt(1).any():
            raise ValueError("One signal_index maps to multiple signal dates")
        if group.groupby("signal_date")["signal_index"].nunique().gt(1).any():
            raise ValueError("One signal_date maps to multiple signal indices")
    return data


def nonoverlapping_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the first event, then one every >= horizon sessions per stock.

    Selection uses only cohort, horizon, stock, signal index, and event identity.
    An event with an unknown filter, missing outcome, or immature future still
    consumes its cooldown; changing labels cannot change the selected sample.
    """
    data = _validated_events(frame)
    keep: list[int] = []
    for (_, horizon, _), group in data.groupby(
        list(_GROUP_KEYS), sort=False, observed=True
    ):
        last: int | None = None
        for row_index, signal_index in zip(group.index, group["signal_index"]):
            current = int(signal_index)
            if last is None or current - last >= int(horizon):
                keep.append(int(row_index))
                last = current
    return data.loc[keep].reset_index(drop=True)


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _mean(series: pd.Series) -> float | None:
    return float(series.mean()) if series.notna().any() else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _statistics(frame: pd.DataFrame) -> dict[str, Any]:
    net = frame["net_return"]
    gross = frame["gross_return"]
    benchmark_known = frame["benchmark_net_return"].notna()
    paired = frame.loc[benchmark_known]
    rally_count = int(gross.ge(frame["threshold"]).sum())
    return {
        "n": int(len(frame)),
        "win": int(net.gt(0).sum()),
        "loss": int(net.lt(0).sum()),
        "breakeven": int(net.eq(0).sum()),
        "rally_count": rally_count,
        "rally_rate": _ratio(rally_count, len(frame)),
        "win_rate": _ratio(int(net.gt(0).sum()), len(frame)),
        "mean_net": _mean(net),
        "median_net": float(net.median()) if len(net) else None,
        "benchmark_paired_n": int(benchmark_known.sum()),
        "mean_excess": _mean(paired["net_return"] - paired["benchmark_net_return"]),
        "mae_n": int(frame["mae"].notna().sum()),
        "mean_mae": _mean(frame["mae"]),
        "mfe_n": int(frame["mfe"].notna().sum()),
        "mean_mfe": _mean(frame["mfe"]),
    }


def _same_date(pass_events: pd.DataFrame, reject_events: pd.DataFrame) -> dict[str, Any]:
    passed = pass_events.groupby("signal_date")["net_return"].agg(["mean", "size"])
    rejected = reject_events.groupby("signal_date")["net_return"].agg(["mean", "size"])
    paired = passed.join(rejected, how="inner", lsuffix="_pass", rsuffix="_reject")
    return {
        "interpretation": "descriptive_noncausal",
        "weighting": "equal_signal_date_weight",
        "reason": None if len(paired) else "no_dates_with_both_groups",
        "n_dates": int(len(paired)),
        "pass_event_n": int(paired["size_pass"].sum()),
        "reject_event_n": int(paired["size_reject"].sum()),
        "mean_net_pass": _mean(paired["mean_pass"]),
        "mean_net_reject": _mean(paired["mean_reject"]),
        "delta_mean_net": _mean(paired["mean_pass"] - paired["mean_reject"]),
    }


def _one_filter(
    data: pd.DataFrame, filter_id: str, period: str,
) -> dict[str, Any]:
    if period != "all":
        data = data.loc[data["signal_date"].str[:4].eq(period)]
    mature = data["mature"].astype(bool)
    outcome_known = (
        data["complete"].astype(bool)
        & data["entry_date"].notna() & data["exit_date"].notna()
        & data["net_return"].notna() & data["gross_return"].notna()
    )
    complete = mature & outcome_known
    boundary = pd.Series(False, index=data.index)
    if period != "all":
        boundary = complete & (
            data["entry_date"].str[:4].ne(period)
            | data["exit_date"].str[:4].ne(period)
        )
    eligible = complete & ~boundary
    known_filter = data[filter_id].notna()
    missing_future = mature & ~outcome_known
    paired_data = data.loc[eligible & known_filter]
    passed = paired_data.loc[paired_data[filter_id].eq(True)]  # noqa: E712
    rejected = paired_data.loc[paired_data[filter_id].eq(False)]  # noqa: E712
    baseline = _statistics(paired_data)
    pass_stats, reject_stats = _statistics(passed), _statistics(rejected)
    return {
        "filter_id": filter_id,
        "period": period,
        "counts": {
            "candidate_n": int(len(data)),
            "immature_n": int((~mature).sum()),
            "missing_future_n": int((mature & ~outcome_known).sum()),
            "boundary_n": int(boundary.sum()),
            "complete_in_period_n": int(eligible.sum()),
            "filter_unknown_n": int((eligible & ~known_filter).sum()),
            "filter_known_n": int((eligible & known_filter).sum()),
            "filter_known_candidate_n": int(known_filter.sum()),
            "filter_unknown_candidate_n": int((~known_filter).sum()),
            "missing_future_by_filter": {
                "pass": int((missing_future & data[filter_id].eq(True)).sum()),
                "reject": int((missing_future & data[filter_id].eq(False)).sum()),
                "unknown": int((missing_future & ~known_filter).sum()),
            },
        },
        "baseline": baseline,
        "pass": pass_stats,
        "reject": reject_stats,
        "rally_retention": _ratio(pass_stats["rally_count"], baseline["rally_count"]),
        "loss_removal": _ratio(reject_stats["loss"], baseline["loss"]),
        "positive_return_removal": _ratio(reject_stats["win"], baseline["win"]),
        "same_signal_date": _same_date(passed, rejected),
    }


def summarize_filters(events: pd.DataFrame, filter_ids: Sequence[str]) -> dict[str, Any]:
    """Compare each nullable signal filter against its own known-event baseline.

    Produces flat ``records`` for raw/nonoverlapping populations, every observed
    cohort/horizon, each requested filter, and all/2024/2025/2026.  Yearly rows
    use signals from that year and require both entry and exit in that year;
    boundary events remain in the all-period row.  This is descriptive evidence,
    not unseen validation, an independence claim, or an account return.
    """
    filters = list(filter_ids)
    if len(filters) != len(set(filters)):
        raise ValueError("filter_ids must not contain duplicates")
    _require_columns(events, (*_BASE_COLUMNS, *_OUTCOME_COLUMNS, *filters))
    data = _validated_events(events)
    for name in ("mature", "complete"):
        data[name] = _boolean(data[name], name, nullable=False)
    for name in filters:
        data[name] = _boolean(data[name], name, nullable=True)
    for name in ("entry_date", "exit_date"):
        dates = _dates(data[name], name, nullable=True)
        data[name] = dates.dt.strftime("%Y-%m-%d").astype("string")
    known_entry, known_exit = data["entry_date"].notna(), data["exit_date"].notna()
    if (known_entry & data["entry_date"].le(data["signal_date"])).any():
        raise ValueError("entry_date must follow the signal close date")
    if (known_entry & known_exit & data["exit_date"].lt(data["entry_date"])).any():
        raise ValueError("exit_date cannot precede entry_date")
    for name in (
        "gross_return", "net_return", "benchmark_net_return", "mfe", "mae", "threshold"
    ):
        original = data[name]
        numeric = _finite(original)
        if name == "threshold":
            if numeric.isna().any() or numeric.le(0).any():
                raise ValueError("threshold must contain positive finite numbers")
        elif (original.notna() & pd.to_numeric(original, errors="coerce").isna()).any():
            raise ValueError(f"{name} contains nonnumeric values")
        data[name] = numeric
    records: list[dict[str, Any]] = []
    deduplicated = nonoverlapping_events(data)
    for population, sample in (("raw", data), ("nonoverlapping", deduplicated)):
        for (cohort, horizon), group in sample.groupby(
            ["cohort", "horizon"], sort=True, observed=True
        ):
            for filter_id in filters:
                for period in ("all", "2024", "2025", "2026"):
                    record = _one_filter(group, filter_id, period)
                    record.update(population=population, cohort=str(cohort), horizon=int(horizon))
                    records.append(record)
    return {
        "schema_version": 1,
        "interpretation": "descriptive_noncausal",
        "unseen_validation": False,
        "is_account_backtest": False,
        "independent_samples_claimed": False,
        "rates_and_returns_unit": "fraction",
        "rally_definition": "gross_return >= event threshold at the fixed horizon",
        "win_loss_definition": "net_return > 0 / net_return < 0; zero is breakeven",
        "mfe_mae_definition": "gross observed path extrema, not executable exits",
        "baseline_definition": "same eligible events with this specific filter known",
        "nonoverlap_definition": "first then signal_index gap >= horizon per cohort/horizon/stock",
        "year_definition": "signal, entry, and exit in the same calendar year",
        "records": records,
    }
