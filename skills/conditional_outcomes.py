"""Compare fixed entry conditions on the same pre-existing candidate cohort.

Signals and feature timing belong to upstream causal modules. This module only
accepts condition booleans, joins them without dropping unknown candidates, and
reports descriptive forward outcomes. Cash replaces known excluded entries;
missing features and unfinished/missing outcomes are never zero-return trades.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral

import numpy as np
import pandas as pd

from skills.strategy_scanner.engine import _day

HORIZONS = (5, 20, 60)
RALLY_THRESHOLDS = {20: .30, 60: .50}
OUTCOME_STATES = {'evaluated', 'immature', 'stock_path_missing', 'benchmark_path_missing'}
CONDITION_COLUMNS = {'signal_date', 'stock_id', 'filter_id', 'known', 'matched'}
EVENT_COLUMNS = {'signal_date', 'stock_id', 'horizon', 'status', 'gross_return',
                 'net_return', 'benchmark_net_return', 'excess_vs0050'}
KEYS = ['signal_date', 'stock_id']
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_SEED = 20261006


def _dates(frame):
    dates = pd.DatetimeIndex(pd.to_datetime(frame['signal_date'], errors='raise'))
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError('Signal dates must be observed naive market dates without clock times')
    frame['signal_date'] = dates.strftime('%Y-%m-%d')


def _validate(events, conditions, start, end, filter_ids=None):
    if not isinstance(events, pd.DataFrame) or not EVENT_COLUMNS.issubset(events.columns):
        raise ValueError('Events require the complete measure_events outcome schema')
    if not isinstance(conditions, pd.DataFrame) or set(conditions.columns) != CONDITION_COLUMNS:
        raise ValueError('Conditions require only signal_date, stock_id, filter_id, known, matched; outcome columns forbidden')
    declared = None
    if filter_ids is not None:
        if isinstance(filter_ids, (str, bytes)):
            raise ValueError('Declared filter IDs require a nonempty unique sequence')
        declared = list(filter_ids)
        if (not declared or any(not isinstance(x, str) or not x.strip() for x in declared)
                or len(declared) != len(set(declared))):
            raise ValueError('Declared filter IDs require nonempty unique strings')
    if (events.empty != conditions.empty) or (events.empty and declared is None):
        raise ValueError('Complete nonempty sources are required unless both are empty with explicit filter_ids')
    start, end = _day(start), _day(end)
    if start > end:
        raise ValueError('Study start exceeds end')
    events, conditions = events.copy(), conditions.copy()
    for frame in (events, conditions):
        _dates(frame)
        if not frame.stock_id.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError('Candidate stock identifiers must be nonempty strings')
    if not events.horizon.map(lambda x: isinstance(x, Integral) and not isinstance(x, (bool, np.bool_)) and x in HORIZONS).all():
        raise ValueError('Events require fixed integer 5/20/60-session horizons')
    if events.duplicated(KEYS+['horizon']).any():
        raise ValueError('Duplicate candidate/horizon event rows')
    if not events.status.isin(OUTCOME_STATES).all():
        raise ValueError('Unknown outcome status; do not silently classify missing results')
    if not conditions.filter_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Filter identifiers must be nonempty strings')
    if conditions.duplicated(KEYS+['filter_id']).any():
        raise ValueError('Duplicate candidate/filter condition rows')
    for name in ('known', 'matched'):
        if not conditions[name].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError('Condition known/matched must be explicit booleans, never null or numeric')
        conditions[name] = conditions[name].astype(bool)
    if (conditions.matched & ~conditions.known).any():
        raise ValueError('Matched conditions cannot be true when feature availability is unknown')
    observed_filters = set(conditions.filter_id.unique())
    if declared is not None and observed_filters and set(declared) != observed_filters:
        raise ValueError('Declared filter IDs must equal all supplied feature filters')
    filters = sorted(declared if declared is not None else observed_filters)
    start_text, end_text = str(start.date()), str(end.date())
    events = events[events.signal_date.between(start_text, end_text)].copy()
    conditions = conditions[conditions.signal_date.between(start_text, end_text)].copy()
    if (events.empty != conditions.empty) or (events.empty and declared is None):
        raise ValueError('Requested study interval has no complete candidates and filters')
    per_candidate = events.groupby(KEYS, sort=False).horizon.agg(set)
    if not per_candidate.map(lambda values: values == set(HORIZONS)).all():
        raise ValueError('Every candidate must retain all three fixed 5/20/60 outcome rows')
    expected = set(map(tuple, events[KEYS].drop_duplicates().to_numpy()))
    for identifier in filters:
        actual = set(map(tuple, conditions.loc[conditions.filter_id.eq(identifier), KEYS].to_numpy()))
        if actual != expected:
            raise ValueError(f'Filter {identifier} must cover exactly every candidate; missing={len(expected-actual)}, extra={len(actual-expected)}')
    for field in ('gross_return', 'net_return', 'benchmark_net_return', 'excess_vs0050'):
        events[field] = pd.to_numeric(events[field], errors='raise')
    evaluated = events[events.status.eq('evaluated')]
    numeric = evaluated[['gross_return', 'net_return', 'benchmark_net_return', 'excess_vs0050']].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError('Evaluated outcomes require finite stock and paired benchmark returns')
    if not np.allclose(evaluated.excess_vs0050, evaluated.net_return-evaluated.benchmark_net_return, rtol=1e-12, atol=1e-12):
        raise ValueError('Evaluated excess return disagrees with its paired benchmark')
    return events, conditions, filters, start, end


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, median=None, worst5pct_mean=None, worst5pct_count=0)
    tail_count = max(1, int(np.ceil(.05*len(values))))
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                worst5pct_mean=float(np.sort(values)[:tail_count].mean()),
                worst5pct_count=tail_count)


def _interval(rows):
    clusters = rows.groupby('signal_date', sort=True).opportunity_delta.agg(['sum', 'count'])
    if len(clusters) < 2:
        return None
    totals, counts = clusters['sum'].to_numpy(), clusters['count'].to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        picked = rng.integers(0, len(clusters), len(clusters))
        draws[i] = totals[picked].sum()/counts[picked].sum()
    low, high = np.quantile(draws, [.025, .975])
    return dict(low=float(low), high=float(high), confidence=.95, samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED, cluster='signal_date', clusters=len(clusters),
        event_weighted=True, descriptive_only=True, multiple_testing_adjusted=False)


def _same_date(rows):
    contrasts, counts = [], []
    for _, group in rows.groupby('signal_date', sort=True):
        kept, excluded = group[group.matched], group[~group.matched]
        if kept.empty or excluded.empty:
            continue
        contrasts.append((float(kept.net_return.mean()-excluded.net_return.mean()),
                          float(kept.net_return.gt(0).mean()-excluded.net_return.gt(0).mean())))
        counts.append((len(kept), len(excluded)))
    return dict(dates=len(contrasts), kept_events=sum(pair[0] for pair in counts),
        excluded_events=sum(pair[1] for pair in counts),
        mean_net_contrast=float(np.mean([pair[0] for pair in contrasts])) if contrasts else None,
        win_rate_contrast=float(np.mean([pair[1] for pair in contrasts])) if contrasts else None,
        weighting='equal_signal_date_after_within_date_arm_means',
        interpretation='descriptive_association_not_causal_or_randomly_matched')


def _summarize(frame, identifier, h, year, scope):
    known_mask = frame.known if scope == 'filter_known' else frame.all_filters_known
    known = frame[known_mask]
    evaluated = known[known.status.eq('evaluated')].copy()
    kept, excluded = evaluated[evaluated.matched], evaluated[~evaluated.matched]
    evaluated['filtered_cash_return'] = np.where(evaluated.matched, evaluated.net_return, 0.)
    evaluated['opportunity_delta'] = evaluated.filtered_cash_return-evaluated.net_return
    baseline = _distribution(evaluated.net_return)
    kept_dist, excluded_dist = _distribution(kept.net_return), _distribution(excluded.net_return)
    filtered = _distribution(evaluated.filtered_cash_return)
    count = len(evaluated)
    positives = evaluated.net_return.gt(0)
    losses = evaluated.net_return.lt(0)
    retained_profit = int(kept.net_return.gt(0).sum())
    avoided_loss = int(excluded.net_return.lt(0).sum())
    outcome_counts = dict(Counter(frame.status))
    known_counts = dict(Counter(known.status))
    threshold = RALLY_THRESHOLDS.get(h)
    rally_total = int(evaluated.gross_return.ge(threshold).sum()) if threshold is not None else None
    rally_kept = int(kept.gross_return.ge(threshold).sum()) if threshold is not None else None
    return dict(scope=scope, filter_id=identifier, horizon=h, year=year,
        all_candidates=len(frame), all_evaluated=outcome_counts.get('evaluated', 0),
        all_outcome_counts=outcome_counts, feature_known=len(known),
        feature_unknown=len(frame)-len(known), this_filter_unknown=int((~frame.known).sum()),
        coverage=len(known)/len(frame) if len(frame) else None,
        known_evaluated=count, known_outcome_counts=known_counts,
        immature=outcome_counts.get('immature', 0),
        stock_path_missing=outcome_counts.get('stock_path_missing', 0),
        benchmark_path_missing=outcome_counts.get('benchmark_path_missing', 0),
        known_immature=known_counts.get('immature', 0),
        known_stock_path_missing=known_counts.get('stock_path_missing', 0),
        known_benchmark_path_missing=known_counts.get('benchmark_path_missing', 0),
        unknown_evaluated=outcome_counts.get('evaluated', 0)-count,
        kept_candidates=int(known.matched.sum()), excluded_candidates=int((~known.matched).sum()),
        kept_evaluated=len(kept), excluded_evaluated=len(excluded),
        baseline_known=baseline, kept=kept_dist, excluded=excluded_dist, filtered_cash=filtered,
        baseline_mean_net=baseline['mean'], kept_mean_net=kept_dist['mean'],
        excluded_mean_net=excluded_dist['mean'], filtered_cash_mean_net=filtered['mean'],
        paired_mean_delta=float(evaluated.opportunity_delta.mean()) if count else None,
        benchmark_mean_net=float(evaluated.benchmark_net_return.mean()) if count else None,
        kept_benchmark_mean_net=float(kept.benchmark_net_return.mean()) if len(kept) else None,
        excluded_benchmark_mean_net=float(excluded.benchmark_net_return.mean()) if len(excluded) else None,
        kept_mean_excess=float(kept.excess_vs0050.mean()) if len(kept) else None,
        excluded_mean_excess=float(excluded.excess_vs0050.mean()) if len(excluded) else None,
        kept_beat_benchmark_rate=float(kept.net_return.gt(kept.benchmark_net_return).mean()) if len(kept) else None,
        excluded_beat_benchmark_rate=float(excluded.net_return.gt(excluded.benchmark_net_return).mean()) if len(excluded) else None,
        baseline_mean_excess=float(evaluated.excess_vs0050.mean()) if count else None,
        filtered_cash_mean_excess=float((evaluated.filtered_cash_return-evaluated.benchmark_net_return).mean()) if count else None,
        baseline_profit_win_rate=float(positives.mean()) if count else None,
        kept_profit_win_rate=float(kept.net_return.gt(0).mean()) if len(kept) else None,
        excluded_profit_win_rate=float(excluded.net_return.gt(0).mean()) if len(excluded) else None,
        filtered_cash_profit_win_rate=float(evaluated.filtered_cash_return.gt(0).mean()) if count else None,
        cash_is_win=False, positive_profit_count=int(positives.sum()), loss_count=int(losses.sum()),
        retained_positive_profit_count=retained_profit,
        retained_positive_profit_fraction=retained_profit/int(positives.sum()) if positives.any() else None,
        avoided_loss_count=avoided_loss,
        avoided_loss_rate=avoided_loss/int(losses.sum()) if losses.any() else None,
        rally_gross_threshold=threshold, rally_total_count=rally_total, rally_kept_count=rally_kept,
        rally_retention=rally_kept/rally_total if rally_total else None,
        rally_kept_precision=rally_kept/len(kept) if threshold is not None and len(kept) else None,
        rally_baseline_precision=rally_total/count if threshold is not None and count else None,
        same_date_contrast=_same_date(evaluated),
        paired_delta_cluster_bootstrap=_interval(evaluated),
        denominator='scope_known_and_paired_outcome_evaluated',
        tail_definition='ceil_5pct_worst_per_event_returns_including_cash_not_account_drawdown',
        live_qualified=False)


def analyze_conditions(events, conditions_frame, start, end, *, filter_ids=None):
    """Describe seven or any predeclared nonempty filter set without tuning.

    Each candidate must retain all three horizons. Each candidate/filter pair
    must be present explicitly, including ``known=False``. Inputs must already
    encode features as known at signal close; this function cannot certify the
    provenance of arbitrary supplied booleans. Explicit ``filter_ids`` preserve
    a declared filter set for empty candidate cohorts. It never trains or selects rules.
    """
    events, conditions, filters, start, end = _validate(events, conditions_frame, start, end, filter_ids)
    common = conditions.groupby(KEYS, sort=False).known.all().rename('all_filters_known').reset_index()
    base = events.merge(common, on=KEYS, how='left', validate='many_to_one')
    primary, secondary = [], []
    for identifier in filters:
        feature = conditions[conditions.filter_id.eq(identifier)][KEYS+['known', 'matched']]
        frame = base.merge(feature, on=KEYS, how='left', validate='many_to_one')
        if frame[['known', 'matched', 'all_filters_known']].isna().any().any() or len(frame) != len(events):
            raise ValueError('Feature join lost or duplicated candidates')
        for h in HORIZONS:
            window = frame[frame.horizon.eq(h)]
            for year in ['all']+[str(y) for y in range(start.year, end.year+1)]:
                group = window if year == 'all' else window[window.signal_date.str.startswith(year)]
                primary.append(_summarize(group, identifier, h, year, 'filter_known'))
                secondary.append(_summarize(group, identifier, h, year, 'all_filters_known'))
    return dict(schema='conditional_entry_outcomes_v1', start=str(start.date()), end=str(end.date()),
        filters=filters, horizons=list(HORIZONS), candidate_count=len(events[KEYS].drop_duplicates()),
        candidate_horizon_rows=len(events), condition_rows=len(conditions), summary=primary,
        common_pool_summary=secondary, returns_are_fractions=True,
        definitions=dict(primary_scope='Each filter compared on its own known paired-evaluated candidate cohort.',
            common_scope='Every filter compared on the same all-filters-known paired-evaluated cohort.',
            feature_unknown='Unknown in the selected scope; this_filter_unknown separately counts that filter alone.',
            cash='Known excluded opportunities return exactly zero with no hypothetical trading costs.',
            costs='Already included in supplied net_return; no second fee deduction.',
            paired_delta='Kept net return or excluded cash zero, minus same-candidate baseline net return.',
            kept_only='Conditional descriptive distribution, never substituted for whole opportunity denominator.',
            rally='Gross endpoint return >=30% at20 sessions or >=50% at60; no5-session rally definition.',
            tail='Mean of worst ceil(0.05*n) per-event returns; not account maximum drawdown.',
            timing='Feature availability must be verified upstream at signal close; no future outcomes used to classify.',
            same_date='Equal-date mean of within-date kept-minus-excluded means; descriptive association only.'),
        limitations=['Historical periods were previously researched, not unseen validation.',
            'Same-date controls do not remove stock selection confounding or demonstrate causality.',
            'Date-cluster bootstrap retains repeated-stock and overlapping-holding dependence.',
            'No multiple-testing correction, portfolio capacity or executable-fill validation.',
            'An aggregate input schema cannot prove upstream booleans are free of look-ahead.'],
        study_type='descriptive_fixed_candidate_opportunity_comparison_not_portfolio_backtest',
        unseen_validation=False, multiple_testing_adjusted=False, account_independent=True,
        live_qualified=False, cumulative_return=None, max_drawdown=None)
