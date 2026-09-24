"""Descriptive surge statistics and exact-stratum retrospective controls.

Unknown labels and unknown rule values remain separate. Bootstrap samples are
whole signal dates, never individual stock rows. Boundary observations do not
enter either temporal phase, pooled estimates, or annual estimates.
"""
from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260925
MIN_BOOTSTRAP_DATES = 5
_PHASES = ("discovery", "replication")
_STRATUM = ["signal_date", "market", "industry", "liquidity_bin"]


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _nullable_boolean(values: pd.Series, name: str) -> pd.Series:
    for value in values:
        if pd.isna(value):
            continue
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} requires boolean or missing values")
    return values.astype("boolean")


def _dates(values: pd.Series, name: str, *, optional: bool = False) -> pd.Series:
    for value in values:
        if optional and pd.isna(value):
            continue
        if not isinstance(value, str):
            raise ValueError(f"{name} requires ISO date strings")
        try:
            parsed = pd.Timestamp(value)
        except ValueError as exc:
            raise ValueError(f"{name} requires ISO date strings") from exc
        if pd.isna(parsed) or parsed.tz is not None or parsed.strftime("%Y-%m-%d") != value:
            raise ValueError(f"{name} requires ISO date strings")
    return pd.to_datetime(values, errors="raise")


def _prepare(table: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "entry_date", "exit_date", "stock_id", "phase", "event", "forward_return"}
    if not required.issubset(table.columns):
        raise ValueError("Missing study columns: " + ", ".join(sorted(required - set(table.columns))))
    if not table.columns.is_unique:
        raise ValueError("Study column names must be unique")
    data = table.copy()
    if not data["phase"].isin((*_PHASES, "boundary")).all():
        raise ValueError("Unknown temporal phase")
    if any(not isinstance(sid, str) or not sid for sid in data["stock_id"]):
        raise ValueError("Stock ids must be nonempty strings")
    if data.duplicated(["signal_date", "stock_id"]).any():
        raise ValueError("Duplicate signal-date/stock observation")
    signal = _dates(data["signal_date"], "signal_date")
    entry = _dates(data["entry_date"], "entry_date", optional=True)
    exit_day = _dates(data["exit_date"], "exit_date", optional=True)
    if ((entry.notna() & (entry <= signal)) | (exit_day.notna() & (exit_day < entry))).any():
        raise ValueError("Entry must follow the signal and exit must not precede entry")
    discovery = data["phase"].eq("discovery")
    replication = data["phase"].eq("replication")
    if (discovery & (exit_day.isna() | (exit_day > pd.Timestamp("2024-12-31")))).any():
        raise ValueError("Discovery outcomes must finish by 2024-12-31")
    if (replication & (signal < pd.Timestamp("2025-01-01"))).any():
        raise ValueError("Replication signals must begin on or after 2025-01-01")
    data["event"] = _nullable_boolean(data["event"], "event")
    returns = data["forward_return"]
    for value in returns.dropna():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("Forward returns must be finite numbers or missing")
    if (data["event"].notna() & (returns.isna() | entry.isna() | exit_day.isna())).any():
        raise ValueError("Known outcomes require finite returns and entry/exit dates")
    data["_exit_year"] = exit_day.dt.year
    return data


def _scopes(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    included = data.loc[data["phase"].isin(_PHASES)]
    scopes = [(phase, included.loc[included["phase"].eq(phase)]) for phase in _PHASES]
    scopes.append(("all", included))
    for year in sorted(included["_exit_year"].dropna().unique()):
        scopes.append((str(int(year)), included.loc[included["_exit_year"].eq(year)]))
    return scopes


def _lift_interval(blocks: np.ndarray, weights: np.ndarray | None) -> dict[str, Any]:
    info = {
        "lift_ci_low": None, "lift_ci_high": None,
        "bootstrap_unit": "signal_date", "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_valid_replicates": 0,
        "bootstrap_dates": int(len(blocks)),
    }
    if weights is None:
        return info
    # Columns: true-positive triggers, known triggers, all events, known labels.
    sampled = weights @ blocks
    tp, triggers, events, observed = sampled.T
    usable = (triggers > 0) & (events > 0) & (observed > 0)
    estimates = (tp[usable] / triggers[usable]) / (events[usable] / observed[usable])
    info["bootstrap_valid_replicates"] = int(usable.sum())
    # An undefined replicate is not a zero lift. Report its count explicitly;
    # too many undefined draws cannot support a nominal 95% interval.
    if len(estimates) >= .95 * BOOTSTRAP_REPLICATES:
        low, high = np.quantile(estimates, [.025, .975])
        info.update(lift_ci_low=float(low), lift_ci_high=float(high))
    return info


def rule_statistics(table: pd.DataFrame, rule_columns: list[str]) -> list[dict[str, Any]]:
    """Return JSON-safe phase/pooled/exit-year metrics for fixed boolean rules.

    ``tp/fp/fn/tn`` and recall/FPR use only jointly known predictions and labels.
    Baseline event rate uses every known label in the scope. ``prediction_unknown``
    and its event/non-event breakdown show the omitted classification coverage.
    Precision bounds assign every unknown outcome among positive triggers to
    failure/success; unknown predictions never become negative predictions.
    """
    data = _prepare(table)
    if len(rule_columns) != len(set(rule_columns)):
        raise ValueError("Rule column names must be unique")
    for rule in rule_columns:
        if rule not in data:
            raise ValueError("Missing rule column: " + rule)
        data[rule] = _nullable_boolean(data[rule], rule)
    output = []
    for scope, rows in _scopes(data):
        dates, codes = np.unique(rows["signal_date"].to_numpy(), return_inverse=True)
        weights = None
        if len(dates) >= MIN_BOOTSTRAP_DATES:
            weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
                len(dates), np.full(len(dates), 1 / len(dates)), size=BOOTSTRAP_REPLICATES,
            )
        observed = rows["event"].notna().to_numpy()
        event = rows["event"].fillna(False).to_numpy(dtype=bool)
        observed_labels = int(observed.sum())
        baseline = _ratio(int(event.sum()), observed_labels)
        for rule in rule_columns:
            prediction_known = rows[rule].notna().to_numpy()
            positive = rows[rule].fillna(False).to_numpy(dtype=bool)
            negative = prediction_known & ~positive
            tp_mask = observed & positive & event
            fp_mask = observed & positive & ~event
            tp, fp = int(tp_mask.sum()), int(fp_mask.sum())
            fn = int((observed & negative & event).sum())
            tn = int((observed & negative & ~event).sum())
            triggers = int(positive.sum())
            unknown_triggers = int((positive & ~observed).sum())
            precision = _ratio(tp, tp + fp)
            blocks = np.column_stack([
                np.bincount(codes, weights=mask.astype(float), minlength=len(dates))
                for mask in (tp_mask, positive & observed, event, observed)
            ])
            item = {
                "scope": scope, "rule": rule,
                "total_eligible": int(len(rows)), "observed_labels": observed_labels,
                "unknown_labels": int((~observed).sum()), "triggers": triggers,
                "outcome_unknown_triggers": unknown_triggers,
                "prediction_unknown": int((~prediction_known).sum()),
                "known_event_prediction_unknown": int((observed & event & ~prediction_known).sum()),
                "known_nonevent_prediction_unknown": int((observed & ~event & ~prediction_known).sum()),
                "evaluated_labels": tp + fp + fn + tn,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": _ratio(tp, tp + fn),
                "false_positive_rate": _ratio(fp, fp + tn),
                "false_discovery_rate": _ratio(fp, tp + fp),
                "baseline_rate": baseline,
                "lift": _ratio(precision, baseline) if precision is not None and baseline else None,
                "precision_lower_bound": _ratio(tp, triggers),
                "precision_upper_bound": _ratio(tp + unknown_triggers, triggers),
                "trigger_rate": _ratio(triggers, len(rows)),
                "unknown_label_rate": _ratio(int((~observed).sum()), len(rows)),
                "boundary_rows_excluded": int(data["phase"].eq("boundary").sum()),
                **_lift_interval(blocks, weights),
            }
            output.append(item)
    return output


def _feature(value: Any, name: str) -> float | None:
    if pd.isna(value):
        return None
    if not isinstance(value, (Real, bool, np.bool_)) or not np.isfinite(value):
        raise ValueError("Matched features must be finite numbers, booleans, or missing: " + name)
    return float(value)


def _matching_summary(rows: pd.DataFrame, pairs: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    total_cases = int(rows["event"].fillna(False).sum())
    case_ids = set(rows.loc[rows["event"].fillna(False), "signal_date"] + ":" + rows.loc[rows["event"].fillna(False), "stock_id"])
    selected = pairs.loc[pairs["case_id"].isin(case_ids)]
    matched = int(selected["case_id"].nunique())
    feature_summary = {}
    for feature in features:
        usable = selected.loc[selected["diff_" + feature].notna()]
        case_mean = usable.groupby("case_id")["diff_" + feature].mean()
        feature_summary[feature] = {
            "observed_pairs": int(len(usable)), "observed_cases": int(len(case_mean)),
            "case_mean": float(usable["case_" + feature].mean()) if len(usable) else None,
            "control_mean": float(usable["control_" + feature].mean()) if len(usable) else None,
            "mean_pair_difference": float(usable["diff_" + feature].mean()) if len(usable) else None,
            "mean_case_difference": float(case_mean.mean()) if len(case_mean) else None,
        }
    return {
        "total_cases": total_cases, "matched_cases": matched, "unmatched_cases": total_cases - matched,
        "matched_fraction": _ratio(matched, total_cases), "pair_count": int(len(selected)),
        "unique_control_observations": int(selected["control_id"].nunique()),
        "reused_control_observations": int(selected.loc[selected["control_id"].duplicated(keep=False), "control_id"].nunique()),
        "feature_differences": feature_summary,
    }


def matched_controls(table: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match each known event to at most three known non-events, without fallback.

    Matching is exact on signal date, market, industry and liquidity quintile,
    then nearest by absolute log-ADV20 distance, breaking ties by stock id.
    Controls may be reused for different cases. Paired feature differences are
    descriptive and are not assigned an independent-pair confidence interval.
    """
    data = _prepare(table)
    required = {*_STRATUM, "adv20", *feature_columns}
    if not required.issubset(data.columns):
        raise ValueError("Missing matching columns: " + ", ".join(sorted(required - set(data.columns))))
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Feature names must be unique")
    for value in data["adv20"]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError("ADV20 must be finite and strictly positive")
    for value in data["liquidity_bin"].dropna():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or value not in range(5):
            raise ValueError("Liquidity bins must be integers from 0 to 4 or missing")
    for feature in feature_columns:
        data[feature] = data[feature].map(lambda value: _feature(value, feature))
    included = data.loc[data["phase"].isin(_PHASES)].copy()
    matchable = included.loc[included[_STRATUM].notna().all(axis=1)]
    controls = matchable.loc[matchable["event"].eq(False).fillna(False)]
    grouped = {key: group for key, group in controls.groupby(_STRATUM, sort=True)}
    cases = matchable.loc[matchable["event"].fillna(False)].sort_values(["signal_date", "stock_id"])
    output = []
    for case in cases.to_dict("records"):
        key = tuple(case[field] for field in _STRATUM)
        candidates = grouped.get(key)
        if candidates is None:
            continue
        ranked = candidates.assign(_distance=np.abs(np.log(candidates["adv20"].astype(float)) - np.log(case["adv20"])))
        ranked = ranked.sort_values(["_distance", "stock_id"], kind="stable").head(3)
        for control in ranked.to_dict("records"):
            row = {
                "case_id": case["signal_date"] + ":" + case["stock_id"],
                "control_id": control["signal_date"] + ":" + control["stock_id"],
                "case_stock_id": case["stock_id"], "control_stock_id": control["stock_id"],
                "signal_date": case["signal_date"], "phase": case["phase"],
                "market": case["market"], "industry": case["industry"],
                "liquidity_bin": int(case["liquidity_bin"]),
                "case_adv20": float(case["adv20"]), "control_adv20": float(control["adv20"]),
                "log_adv_distance": float(control["_distance"]),
            }
            for feature in feature_columns:
                left, right = _feature(case[feature], feature), _feature(control[feature], feature)
                row["case_" + feature], row["control_" + feature] = left, right
                row["diff_" + feature] = left - right if left is not None and right is not None else None
            output.append(row)
    columns = ["case_id", "control_id", "case_stock_id", "control_stock_id", "signal_date", "phase",
               "market", "industry", "liquidity_bin", "case_adv20", "control_adv20", "log_adv_distance"]
    columns += [prefix + name for name in feature_columns for prefix in ("case_", "control_", "diff_")]
    columns = list(dict.fromkeys(columns))
    # Object dtype preserves JSON nulls in pair records, rather than NaN coercion.
    pairs = pd.DataFrame(output, columns=columns, dtype=object)
    summary = {
        "method": "same_signal_date_market_industry_liquidity_bin_then_nearest_log_adv20",
        "maximum_controls_per_case": 3, "matching_with_replacement": True,
        "independent_pairs": False, "fallback_used": False,
        "boundary_rows_excluded": int(data["phase"].eq("boundary").sum()),
        "by_scope": {scope: _matching_summary(rows, pairs, feature_columns) for scope, rows in _scopes(data)},
    }
    return pairs, summary
