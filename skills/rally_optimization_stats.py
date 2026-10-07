"""Matched descriptive evaluations; rankings/checkpoint decisions already frozen."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skills.rally_context_stats import nonoverlapping_events, _statistics
from skills.rally_checkpoint import CHECKPOINT_ACTIONS
from skills.rally_ranking import RANKING_VARIANTS
from skills.strategy_scanner.outcomes import COSTS, _net


def rank_selections(ranked):
    result = ranked.copy()
    conditions = []
    for variant in RANKING_VARIANTS:
        rank = result[f'rank_{variant}']
        for top in (3, 5):
            name = f'{variant}_top{top}'
            result[name] = rank.le(top).astype('boolean')
            conditions.append(name)
        quintile = ((rank-1)*5//result.ranking_day_n.where(result.ranking_day_n.gt(0)))+1
        for q in range(1, 6):
            name = f'{variant}_q{q}'
            result[name] = quintile.eq(q).where(result.ranking_day_n.ge(10), pd.NA).astype('boolean')
            conditions.append(name)
    return result, conditions


def complete_in_period(data, period):
    sample = data if period == 'all' else data.loc[data.signal_date.str[:4].eq(period)]
    complete = sample.mature & sample.complete & np.isfinite(sample.net_return) & np.isfinite(sample.gross_return)
    if period != 'all':
        complete &= sample.entry_date.str[:4].eq(period) & sample.exit_date.str[:4].eq(period)
    return sample, complete.fillna(False)


def ranking_day_pairs(events):
    """Compare whole selected groups on dates with both fully observed outcomes.

    This ex-post completeness condition is diagnostic only. It never reranks,
    backfills vacant ranks, or changes the separately stored candidate selection.
    """
    records = []
    for population, frame in [('raw', events), ('nonoverlapping', nonoverlapping_events(events))]:
        for (cohort, horizon), group in frame.groupby(['cohort', 'horizon']):
            for period in ('all', '2024', '2025', '2026'):
                part, complete = complete_in_period(group, period)
                for top in (3, 5):
                    for variant in ('context', 'blend'):
                        pairs, absent, incomplete = [], 0, 0
                        for day, daily in part.groupby('signal_date'):
                            a = daily.loc[daily[f'{variant}_top{top}'].eq(True)]
                            b = daily.loc[daily[f'rs_top{top}'].eq(True)]
                            if a.empty or b.empty:
                                absent += 1
                            elif not complete.loc[a.index].all() or not complete.loc[b.index].all():
                                incomplete += 1
                            else:
                                pairs.append(float(a.net_return.mean()-b.net_return.mean()))
                        records.append(dict(population=population, cohort=cohort, horizon=int(horizon),
                            period=period, variant=variant, top=top, paired_dates=len(pairs),
                            excluded_no_group_dates=absent, excluded_incomplete_or_boundary_dates=incomplete,
                            mean_delta_net=float(np.mean(pairs)) if pairs else None,
                            median_delta_net=float(np.median(pairs)) if pairs else None,
                            outperform_date_fraction=float(np.mean(np.array(pairs)>0)) if pairs else None))
    return records


def build_exit_policies(labelled, checkpoints, f, days, ids):
    """Match next-open exits and baseline to their SAME originally fixed horizon."""
    keys = ['cohort', 'event_id', 'stock_id', 'signal_date', 'signal_index']
    if checkpoints.duplicated(keys+['checkpoint_age']).any():
        raise ValueError('Duplicate checkpoint identity')
    expected_keys = set(labelled[keys].itertuples(index=False, name=None))
    actual_keys = set(checkpoints[keys].itertuples(index=False, name=None))
    if expected_keys != actual_keys or not checkpoints.groupby(keys).checkpoint_age.agg(
            lambda ages:set(ages)=={3,5}).all():
        raise ValueError('Each event must have exactly checkpoint ages 3 and 5')
    if not checkpoints.checkpoint_index.eq(checkpoints.signal_index+checkpoints.checkpoint_age).all():
        raise ValueError('Checkpoint index must equal signal index plus age')
    dates = [str(pd.Timestamp(day).date()) for day in days]
    for checkpoint in checkpoints.itertuples(index=False):
        for key, index in [('checkpoint_date',checkpoint.checkpoint_index),
                           ('next_exit_date',checkpoint.checkpoint_index+1)]:
            expected = dates[index] if 0<=index<len(dates) else None
            value = getattr(checkpoint,key)
            if (expected is None and pd.notna(value)) or (expected is not None and value!=expected):
                raise ValueError('Checkpoint calendar date differs from declared index: '+key)
    expanded = labelled.merge(checkpoints, on=keys, how='left', validate='many_to_many')
    if len(expanded) != len(labelled)*2 or expanded.checkpoint_age.isna().any():
        raise ValueError('Each labelled event must have exactly two checkpoints')
    frames = []
    columns = {sid:j for j,sid in enumerate(ids)}
    exit_index = expanded.checkpoint_index.to_numpy(int)+1
    column = expanded.stock_id.map(columns)
    if column.isna().any():
        raise ValueError('Policy stock missing from price matrix')
    column = column.to_numpy(int)
    within = exit_index < len(days)
    safe = np.minimum(exit_index, len(days)-1)
    good = (f['valid'] & f['eligible'].eq(True).fillna(False) & f['volume'].gt(0)).to_numpy(bool)
    open_adj = (f['open']*f['c']/f['close']).to_numpy(float)
    exit_price = open_adj[safe, column]
    exit_known = within & good[safe, column] & np.isfinite(exit_price) & (exit_price > 0)
    early_gross = exit_price/expanded.entry_price_adj-1
    early_net = _net(expanded.entry_price_adj, exit_price, COSTS['stock_sell_tax'])
    baseline_known = expanded.mature & expanded.complete & np.isfinite(expanded.net_return) & np.isfinite(expanded.gross_return)
    for policy in CHECKPOINT_ACTIONS:
        part = expanded.copy()
        known_action = part[policy].notna() & part.checkpoint_known
        triggered = part[policy].eq(True).fillna(False).to_numpy(bool)
        paired = baseline_known & known_action & (~triggered | exit_known)
        part['policy'] = policy
        part['triggered'] = part[policy]
        part['paired'] = paired
        part['policy_net_return'] = np.where(paired, np.where(triggered, early_net, part.net_return), np.nan)
        part['policy_gross_return'] = np.where(paired, np.where(triggered, early_gross, part.gross_return), np.nan)
        part['policy_exit_date'] = np.where(paired, np.where(triggered, part.next_exit_date, part.exit_date), None)
        part['policy_issue'] = np.select([~baseline_known, ~known_action, triggered & ~exit_known],
            ['baseline_future_unavailable','checkpoint_unknown','next_exit_open_unavailable'], default='')
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def summarize_exit_policies(policies, labelled):
    dedup = nonoverlapping_events(labelled)[['cohort', 'event_id', 'horizon']].assign(dedup=True)
    marked = policies.merge(dedup, on=['cohort','event_id','horizon'], how='left', validate='many_to_one')
    results = []
    for population, frame in [('raw', marked), ('nonoverlapping', marked.loc[marked.dedup.eq(True)])]:
        for (cohort, horizon, age, policy), group in frame.groupby(['cohort','horizon','checkpoint_age','policy']):
            for period in ('all','2024','2025','2026'):
                part, complete = complete_in_period(group, period)
                evaluated = part.loc[complete & part.paired].copy()
                alt = evaluated.copy()
                alt['net_return'] = alt.policy_net_return
                alt['gross_return'] = alt.policy_gross_return
                # Baseline extrema cannot stand in for the new exit path.
                alt['mfe'] = np.nan;alt['mae'] = np.nan
                base_stats, policy_stats = _statistics(evaluated), _statistics(alt)
                original_rally = evaluated.gross_return.ge(evaluated.threshold)
                original_win = evaluated.net_return.gt(0)
                original_loss = evaluated.net_return.lt(0)
                changed = evaluated.triggered.eq(True)
                delta = evaluated.policy_net_return-evaluated.net_return
                results.append(dict(population=population,cohort=cohort,horizon=int(horizon),checkpoint_age=int(age),
                    policy=policy,period=period,candidates=len(part),baseline_complete_in_period=int(complete.sum()),
                    checkpoint_unknown=int((complete & ~part.checkpoint_known).sum()),
                    policy_unknown=int((complete & ~part.paired).sum()),paired_n=len(evaluated),
                    triggered_n=int(changed.sum()),baseline=base_stats,modified=policy_stats,
                    mean_delta_net=float(delta.mean()) if len(delta) else None,
                    mean_delta_among_triggered=float(delta.loc[changed].mean()) if changed.any() else None,
                    original_rallies=int(original_rally.sum()),
                    original_rallies_triggered=int((original_rally & changed).sum()),
                    original_rallies_still_above_target=int((original_rally & evaluated.policy_gross_return.ge(evaluated.threshold)).sum()),
                    losses_rescued=int((original_loss & evaluated.policy_net_return.ge(0)).sum()),
                    winners_turned_loss=int((original_win & evaluated.policy_net_return.lt(0)).sum()),
                    improved_n=int(delta.gt(0).sum()),worsened_n=int(delta.lt(0).sum())))
    return results
