"""Frozen, descriptive entry-filter comparisons; never a trading account.

Conditions only read point-in-time boolean features.  Forward labels are used
after condition construction to evaluate fixed-horizon net profits.  Missing
evidence stays unknown, and a zero net return is not a winning entry.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from skills.rally_context_stats import nonoverlapping_events, summarize_filters


BASE_ATOMS = (
    "not_extended", "contraction", "moderate_volume", "prior_quiet",
    "close_strong", "rs20_positive", "market_above60", "market_breadth",
    "peer_breadth", "peer_turnover", "flow_positive_lag1",
    "flow_positive_lag3", "poc_up",
)
COMPLEMENT_ATOMS = {
    "market_not_above60": "market_above60",
    "market_narrow": "market_breadth",
}
ATOMS = (*BASE_ATOMS, *COMPLEMENT_ATOMS)
EXCLUDED_PAIRS = frozenset({
    frozenset({"market_above60", "market_not_above60"}),
    frozenset({"market_breadth", "market_narrow"}),
    frozenset({"flow_positive_lag1", "flow_positive_lag3"}),
})
PERIODS = ("all", "2024", "2025", "2026")


def condition_definitions() -> dict[str, list[str]]:
    """Return the baseline and all 117 predeclared single/pair conditions."""
    definitions: dict[str, list[str]] = {"all_entries": []}
    for atom in ATOMS:
        definitions[f"ew__{atom}"] = [atom]
    for pair in combinations(ATOMS, 2):
        if frozenset(pair) not in EXCLUDED_PAIRS:
            definitions["ew__" + "__and__".join(pair)] = list(pair)
    return definitions


def _nullable_boolean(values: pd.Series, name: str) -> pd.Series:
    known = values.loc[values.notna()]
    if not known.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"Entry atom must contain booleans: {name}")
    return values.astype("boolean")


def build_conditions(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Preserve all feature rows and add frozen nullable entry conditions.

    No outcome, holding capacity, or future availability is inspected here.
    The two market complements are computed only from their original atoms.
    A pair with either atom unknown stays unknown even if the other is false.
    Existing derived columns are rejected rather than overwritten.
    """
    if not features.columns.is_unique:
        raise ValueError("Entry feature columns must be unique")
    missing = sorted(set(BASE_ATOMS).difference(features.columns))
    if missing:
        raise ValueError("Missing entry atoms: " + ", ".join(missing))
    definitions = condition_definitions()
    conflicting = sorted(set(definitions).intersection(features.columns))
    if conflicting:
        raise ValueError("Derived entry columns already exist: " + ", ".join(conflicting))
    atoms = {name: _nullable_boolean(features[name], name) for name in BASE_ATOMS}
    atoms.update({name: ~atoms[source] for name, source in COMPLEMENT_ATOMS.items()})
    derived: dict[str, pd.Series] = {
        "all_entries": pd.Series(True, index=features.index, dtype="boolean"),
    }
    for condition_id, names in definitions.items():
        if not names:
            continue
        value = atoms[names[0]].copy()
        known = value.notna()
        for name in names[1:]:
            value = value & atoms[name]
            known = known & atoms[name].notna()
        derived[condition_id] = value.where(known, pd.NA).astype("boolean")
    return pd.concat([features.copy(), pd.DataFrame(derived, index=features.index)], axis=1), definitions


def _extended_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    net = pd.to_numeric(frame["net_return"], errors="raise").astype(float)
    positive, negative = net.loc[net.gt(0)], net.loc[net.lt(0)]
    avg_win = float(positive.mean()) if len(positive) else None
    avg_loss = float(negative.mean()) if len(negative) else None
    factor = float(positive.sum() / -negative.sum()) if len(negative) else None
    if not len(net):
        status = "no_observations"
    elif not len(negative):
        status = "no_losses"
    elif not len(positive):
        status = "no_wins"
    else:
        status = "finite"
    return {
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": avg_win / -avg_loss if avg_win is not None and avg_loss is not None else None,
        "unit_profit_factor": factor,
        "profit_factor_status": status,
        "p05": float(net.quantile(.05)) if len(net) else None,
        "worst": float(net.min()) if len(net) else None,
        "unique_stocks": int(frame["stock_id"].nunique()),
        "unique_dates": int(frame["signal_date"].nunique()),
    }


def _complete_sample(data: pd.DataFrame, period: str) -> pd.DataFrame:
    if period != "all":
        data = data.loc[data["signal_date"].astype(str).str[:4].eq(period)]
    net = pd.to_numeric(data["net_return"], errors="raise")
    gross = pd.to_numeric(data["gross_return"], errors="raise")
    eligible = (
        data["mature"].astype(bool) & data["complete"].astype(bool)
        & data["entry_date"].notna() & data["exit_date"].notna()
        & np.isfinite(net) & np.isfinite(gross)
    )
    if period != "all":
        eligible = (
            eligible & data["entry_date"].astype(str).str[:4].eq(period)
            & data["exit_date"].astype(str).str[:4].eq(period)
        )
    return data.loc[eligible]


def summarize_entry_winrate(
    labelled: pd.DataFrame, definitions: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Summarize every condition with its own known-event baseline.

    The existing validated summary controls maturity, missing outcomes, annual
    boundaries, benchmark pairing and outcome-independent cooldown.  Extended
    payoff statistics use precisely the same complete and known event sets.
    """
    expected = condition_definitions()
    supplied = {str(key): list(value) for key, value in definitions.items()}
    if supplied != expected:
        raise ValueError("Entry condition definitions differ from the frozen 117-condition specification")
    result = summarize_filters(labelled, list(expected))
    populations = {"raw": labelled, "nonoverlapping": nonoverlapping_events(labelled)}
    samples: dict[tuple[str, str, int, str], pd.DataFrame] = {}
    for population, frame in populations.items():
        for (cohort, horizon), group in frame.groupby(["cohort", "horizon"], sort=True, observed=True):
            for period in PERIODS:
                samples[(population, str(cohort), int(horizon), period)] = _complete_sample(group, period)
    for record in result["records"]:
        condition_id = record["filter_id"]
        sample = samples[(record["population"], record["cohort"], record["horizon"], record["period"])]
        known = sample.loc[sample[condition_id].notna()]
        groups = {
            "baseline": known,
            "pass": known.loc[known[condition_id].eq(True)],  # noqa: E712
            "reject": known.loc[known[condition_id].eq(False)],  # noqa: E712
        }
        for name, group in groups.items():
            if len(group) != record[name]["n"]:
                raise AssertionError("Extended payoff sample differs from validated filter summary")
            record[name].update(_extended_statistics(group))
        record["condition_id"] = condition_id
        record["atoms"] = list(expected[condition_id])
    result.update({
        "schema": "entry_winrate_v1",
        "condition_definitions": expected,
        "condition_count_excluding_baseline": len(expected) - 1,
        "hypothesis_count": (len(expected) - 1) * labelled[["cohort", "horizon"]].drop_duplicates().shape[0],
        "live_qualified": False,
        "multiple_testing_corrected": False,
        "pair_unknown_definition": "any constituent unknown makes the whole condition unknown",
        "profit_factor_definition": "sum positive net returns / absolute sum negative net returns; equal unit entries",
        "payoff_definition": "mean winning net return / absolute mean losing net return",
        "no_loss_profit_factor": "null with explicit status; no infinite JSON values",
    })
    return result


def _at_least_sixty(stats: Mapping[str, Any], minimum: int) -> bool:
    rate = stats.get("win_rate")
    return stats.get("n", 0) >= minimum and rate is not None and float(rate) >= .60


def summarize_candidates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Label primary full-period versions without hiding unsuccessful versions.

    The yearly consistency flag is separate from full-period sample eligibility.
    All comparisons are on already-seen history and are descriptive, not proof
    of a future 60 percent hit rate or of beating a benchmark account.
    """
    primary: dict[tuple[str, int, str], dict[str, Mapping[str, Any]]] = {}
    for record in records:
        if record["population"] != "nonoverlapping":
            continue
        key = (str(record["cohort"]), int(record["horizon"]), str(record["filter_id"]))
        period = str(record["period"])
        group = primary.setdefault(key, {})
        if period in group:
            raise ValueError("Duplicate primary entry summary period")
        group[period] = record
    versions: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for key, periods in sorted(primary.items()):
        if "all" not in periods:
            raise ValueError("Missing full-period primary entry summary")
        record = dict(periods["all"])
        stats = record["pass"]
        sample_sufficient = (
            stats["n"] >= 100 and stats["unique_stocks"] >= 25
            and stats["unique_dates"] >= 30
        )
        win_rate_60 = _at_least_sixty(stats, 1)
        qualified = win_rate_60 and sample_sufficient
        yearly = {year: dict(periods[year]) for year in PERIODS[1:] if year in periods}
        cross_year = all(
            year in yearly and _at_least_sixty(yearly[year]["pass"], 30)
            for year in PERIODS[1:]
        )
        record.update({
            "win_rate_60": win_rate_60,
            "sample_sufficient": sample_sufficient,
            "qualified_60": qualified,
            "cross_year_60": cross_year,
            "cross_year_qualified_60": qualified and cross_year,
            "small_sample_60": win_rate_60 and not sample_sufficient,
            "yearly": yearly,
        })
        (baselines if key[2] == "all_entries" else versions).append(record)
    return {
        "schema": "entry_winrate_candidates_v1",
        "primary_population": "nonoverlapping",
        "qualification": {"minimum_win_rate": .60, "minimum_n": 100,
                          "minimum_stocks": 25, "minimum_signal_dates": 30},
        "yearly_qualification": {"years": list(PERIODS[1:]), "minimum_n_per_year": 30,
                                 "minimum_win_rate_per_year": .60},
        "primary_versions": versions,
        "primary_baselines": baselines,
        "qualified_60": [row for row in versions if row["qualified_60"]],
        "cross_year_60": [row for row in versions if row["cross_year_60"]],
        "cross_year_qualified_60": [row for row in versions if row["cross_year_qualified_60"]],
        "small_sample_60": [row for row in versions if row["small_sample_60"]],
        "hypothesis_count": len(versions),
        "live_qualified": False,
        "unseen_validation": False,
        "multiple_testing_corrected": False,
    }
