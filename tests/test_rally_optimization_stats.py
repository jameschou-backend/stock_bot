"""Frozen selections and same-horizon, next-open exit comparisons."""
import numpy as np
import pandas as pd
import pytest

from skills.rally_checkpoint import CHECKPOINT_ACTIONS, build_checkpoints
from skills.rally_optimization_stats import (
    build_exit_policies, complete_in_period, rank_selections, ranking_day_pairs,
    summarize_exit_policies,
)
from skills.rally_ranking import build_rankings
from skills.strategy_scanner.outcomes import COSTS, _net


def ranked_events(count=6):
    source = pd.DataFrame([
        dict(cohort='original_red', stock_id=str(2300+i), event_id=f'e-{i}',
             signal_date='2024-01-02', signal_index=0, relative_return20=count-i,
             peer_breadth_value=i/count, peer_turnover_multiple=i+1.,
             flow_ratio5_lag1=float(i), horizon=20, entry_date='2024-01-03',
             exit_date='2024-01-30', mature=True, complete=True,
             net_return=i/100., gross_return=i/100.+.008,
             benchmark_net_return=.01, mfe=.1, mae=-.02, threshold=.3)
        for i in range(count)
    ])
    return build_rankings(source)


def get_pair(rows, **changes):
    wanted = dict(population='raw', cohort='original_red', horizon=20,
                  period='all', variant='context', top=3)
    wanted.update(changes)
    return next(row for row in rows if all(row[key] == value for key, value in wanted.items()))


def policy_inputs(stock_ids=('2330',), signal_indexes=(0,), horizons=(20, 60), length=90):
    days = pd.bdate_range('2024-01-02', periods=length)
    ids = ['0050', *stock_ids]
    f = {key: pd.DataFrame(value, index=days, columns=ids)
         for key, value in [('c', 100.), ('l', 95.), ('open', 100.),
                           ('close', 100.), ('volume', 1000.),
                           ('valid', True), ('eligible', True)]}
    events = pd.DataFrame([
        dict(cohort='original_red', event_id=f'e-{sid}-{index}', stock_id=sid,
             signal_date=str(days[index].date()), signal_index=index)
        for sid in stock_ids for index in signal_indexes
    ])
    labels = []
    for event in events.to_dict('records'):
        for horizon in horizons:
            end = event['signal_index'] + horizon
            labels.append(dict(event, horizon=horizon,
                entry_date=str(days[event['signal_index']+1].date()),
                exit_date=str(days[end].date()) if end < len(days) else None,
                mature=end < len(days), complete=end < len(days),
                net_return=_net(100., 120., COSTS['stock_sell_tax']) if end < len(days) else np.nan,
                gross_return=.2 if end < len(days) else np.nan,
                benchmark_net_return=_net(100., 110., COSTS['benchmark_sell_tax']),
                mfe=.3, mae=-.1, threshold=.3 if horizon == 20 else .5))
    return pd.DataFrame(labels), events, f, days, ids


def trigger(f, days, sid='2330', index=0):
    for age in (3, 5):
        f['c'].loc[days[index+age], sid] = 90.
        f['close'].loc[days[index+age], sid] = 90.
        f['volume'].loc[days[index+age-1:index+age+1], sid] = 200.


def get_policy_summary(rows, **changes):
    wanted = dict(population='raw', cohort='original_red', horizon=20,
                  checkpoint_age=3, policy='support_failed', period='all')
    wanted.update(changes)
    return next(row for row in rows if all(row[key] == value for key, value in wanted.items()))


def test_top_selection_uses_frozen_rank_and_never_promotes_missing_outcomes():
    ranked = ranked_events()
    selected, conditions = rank_selections(ranked)
    assert len(conditions) == 21
    assert selected.loc[selected.rs_top3.eq(True), 'stock_id'].tolist() == ['2300', '2301', '2302']
    changed = ranked.copy()
    changed.loc[0, 'complete'] = False
    changed.loc[0, ['net_return', 'gross_return']] = np.nan
    again, _ = rank_selections(changed)
    pd.testing.assert_frame_equal(selected[conditions], again[conditions])
    assert again.loc[0, 'rs_top3']
    assert not again.loc[3, 'rs_top3']


def test_daily_pair_excludes_entire_day_if_any_selected_future_is_missing():
    selected, _ = rank_selections(ranked_events())
    assert get_pair(ranking_day_pairs(selected))['paired_dates'] == 1
    selected.loc[0, 'complete'] = False
    selected.loc[0, ['net_return', 'gross_return']] = np.nan
    row = get_pair(ranking_day_pairs(selected))
    assert row['paired_dates'] == 0
    assert row['excluded_incomplete_or_boundary_dates'] == 1
    assert row['mean_delta_net'] is None


def test_missing_nonselected_future_does_not_drop_top3_pair():
    selected, _ = rank_selections(ranked_events(10))
    selected.loc[4, 'complete'] = False
    selected.loc[4, ['net_return', 'gross_return']] = np.nan
    row = get_pair(ranking_day_pairs(selected))
    assert row['paired_dates'] == 1
    assert row['excluded_incomplete_or_boundary_dates'] == 0
    assert row['mean_delta_net'] == pytest.approx(.07)


def test_pair_differences_are_equal_date_weighted_not_pooled_event_weighted():
    first, _ = rank_selections(ranked_events(6))
    second, _ = rank_selections(ranked_events(2))
    second['signal_date'] = '2024-02-06'
    second['signal_index'] = 25
    second['entry_date'] = '2024-02-07'
    second['exit_date'] = '2024-03-05'
    second['event_id'] += '-second'
    combined = pd.concat([first, second], ignore_index=True)
    row = get_pair(ranking_day_pairs(combined))
    assert row['paired_dates'] == 2
    assert row['mean_delta_net'] == pytest.approx(.03 / 2)


@pytest.mark.parametrize('count', [1, 9, 10, 12])
def test_quintiles_require_ten_known_daily_candidates(count):
    ranked = ranked_events(count)
    selected, _ = rank_selections(ranked)
    for variant in ('rs', 'context', 'blend'):
        columns = [f'{variant}_q{q}' for q in range(1, 6)]
        if count < 10:
            assert selected[columns].isna().all().all()
        else:
            assert selected[columns].notna().all().all()
            assert selected[columns].sum(axis=1).eq(1).all()
            assert selected[columns].sum().sum() == count


def test_unknown_rank_stays_unknown_and_does_not_count_towards_quintile_minimum():
    source = ranked_events(10)
    source.loc[0, 'peer_breadth_value'] = np.nan
    selected, _ = rank_selections(build_rankings(source))
    assert selected.ranking_day_n.eq(9).all()
    for variant in ('rs', 'context', 'blend'):
        assert pd.isna(selected.loc[0, f'{variant}_top3'])
        assert selected[f'{variant}_q1'].isna().all()


def test_annual_period_requires_original_entry_and_fixed_exit_in_same_year():
    selected, _ = rank_selections(ranked_events())
    selected['signal_date'] = '2024-12-30'
    selected['entry_date'] = '2024-12-31'
    selected['exit_date'] = '2025-01-27'
    _, annual = complete_in_period(selected, '2024')
    _, all_period = complete_in_period(selected, 'all')
    assert not annual.any()
    assert all_period.all()
    row = get_pair(ranking_day_pairs(selected), period='2024')
    assert row['paired_dates'] == 0
    assert row['excluded_incomplete_or_boundary_dates'] == 1


@pytest.mark.parametrize('age,next_index', [(3, 4), (5, 6)])
def test_true_policy_exits_next_open_not_checkpoint_close(age, next_index):
    labelled, events, f, days, ids = policy_inputs()
    trigger(f, days)
    f['open'].loc[days[next_index], '2330'] = 83.
    checkpoints = build_checkpoints(events, f, days, ids)
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    rows = actual.loc[actual.checkpoint_age.eq(age)]
    assert rows.paired.all()
    assert rows.triggered.all()
    assert rows.policy_exit_date.eq(str(days[next_index].date())).all()
    assert rows.policy_gross_return.eq(83/100.-1).all()
    np.testing.assert_allclose(rows.policy_net_return, _net(100., 83., COSTS['stock_sell_tax']))
    # Full-horizon benchmark includes the time spent in cash after an early exit.
    np.testing.assert_allclose(rows.benchmark_net_return, _net(100., 110., COSTS['benchmark_sell_tax']))


def test_false_policy_keeps_original_fixed_exit_even_when_next_open_missing():
    labelled, events, f, days, ids = policy_inputs()
    checkpoints = build_checkpoints(events, f, days, ids)
    assert not checkpoints[list(CHECKPOINT_ACTIONS)].any().any()
    f['open'].loc[days[4], '2330'] = np.nan
    f['valid'].loc[days[6], '2330'] = False
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    assert actual.paired.all()
    assert actual.policy_exit_date.eq(actual.exit_date).all()
    np.testing.assert_allclose(actual.policy_net_return, actual.net_return)


@pytest.mark.parametrize('field,bad', [('open', np.nan), ('valid', False),
                                    ('eligible', False), ('volume', 0.)])
def test_true_policy_requires_known_tradeable_next_open(field, bad):
    labelled, events, f, days, ids = policy_inputs()
    trigger(f, days)
    checkpoints = build_checkpoints(events, f, days, ids)
    f[field].loc[days[4], '2330'] = bad
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    rows = actual.loc[actual.checkpoint_age.eq(3)]
    assert not rows.paired.any()
    assert rows.policy_net_return.isna().all()
    assert rows.policy_issue.eq('next_exit_open_unavailable').all()
    assert actual.loc[actual.checkpoint_age.eq(5), 'paired'].all()


def test_incomplete_baseline_never_becomes_known_because_early_exit_is_observed():
    labelled, events, f, days, ids = policy_inputs()
    trigger(f, days)
    checkpoints = build_checkpoints(events, f, days, ids)
    labelled.loc[labelled.horizon.eq(60), 'complete'] = False
    labelled.loc[labelled.horizon.eq(60), ['net_return', 'gross_return']] = np.nan
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    unknown = actual.loc[actual.horizon.eq(60)]
    assert not unknown.paired.any()
    assert unknown.policy_issue.eq('baseline_future_unavailable').all()
    assert unknown.policy_net_return.isna().all()
    assert actual.loc[actual.horizon.eq(20), 'paired'].all()


def test_unknown_checkpoint_never_silently_becomes_hold_or_exit():
    labelled, events, f, days, ids = policy_inputs()
    f['valid'].loc[days[2], '2330'] = False
    checkpoints = build_checkpoints(events, f, days, ids)
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    assert not actual.paired.any()
    assert actual.policy_net_return.isna().all()
    assert actual.policy_issue.eq('checkpoint_unknown').all()


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'wrong_age', 'uneven_count'])
def test_every_event_requires_exactly_one_age3_and_one_age5_checkpoint(mutation):
    labelled, events, f, days, ids = policy_inputs(stock_ids=('2330', '2317'))
    checkpoints = build_checkpoints(events, f, days, ids)
    if mutation == 'duplicate':
        checkpoints = pd.concat([checkpoints, checkpoints.iloc[[0]]], ignore_index=True)
    elif mutation == 'missing':
        checkpoints = checkpoints.iloc[1:]
    elif mutation == 'wrong_age':
        checkpoints.loc[0, 'checkpoint_age'] = 7
    else:
        checkpoints = checkpoints.iloc[1:].copy()
        extra = checkpoints.iloc[[-1]].copy()
        extra['checkpoint_age'] = 7
        checkpoints = pd.concat([checkpoints, extra], ignore_index=True)
    with pytest.raises(ValueError, match='checkpoint|Checkpoint'):
        build_exit_policies(labelled, checkpoints, f, days, ids)


@pytest.mark.parametrize('column,value', [('checkpoint_index', 2),
                                       ('next_exit_date', '2024-01-05')])
def test_checkpoint_execution_coordinate_cannot_silently_shift(column, value):
    labelled, events, f, days, ids = policy_inputs()
    checkpoints = build_checkpoints(events, f, days, ids)
    checkpoints.loc[0, column] = value
    with pytest.raises(ValueError, match='checkpoint|Checkpoint|exit|calendar'):
        build_exit_policies(labelled, checkpoints, f, days, ids)


def test_summary_preserves_fixed_horizon_benchmark_and_counts_rally_damage():
    labelled, events, f, days, ids = policy_inputs(stock_ids=('2330', '2317'), horizons=(20,))
    trigger(f, days, sid='2330')
    trigger(f, days, sid='2317')
    labelled.loc[labelled.stock_id.eq('2330'), ['gross_return', 'net_return']] = [.5, .49]
    labelled.loc[labelled.stock_id.eq('2317'), ['gross_return', 'net_return']] = [-.2, -.21]
    f['open'].loc[days[4], '2330'] = 90.
    f['open'].loc[days[4], '2317'] = 110.
    checkpoints = build_checkpoints(events, f, days, ids)
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    row = get_policy_summary(summarize_exit_policies(actual, labelled))
    assert row['paired_n'] == row['triggered_n'] == 2
    assert row['original_rallies'] == row['original_rallies_triggered'] == 1
    assert row['original_rallies_still_above_target'] == 0
    assert row['winners_turned_loss'] == row['losses_rescued'] == 1
    assert row['improved_n'] == row['worsened_n'] == 1
    expected_net = (_net(100., 90., COSTS['stock_sell_tax'])
                    + _net(100., 110., COSTS['stock_sell_tax'])) / 2
    expected_benchmark = _net(100., 110., COSTS['benchmark_sell_tax'])
    assert row['modified']['mean_net'] == pytest.approx(expected_net)
    assert row['modified']['mean_excess'] == pytest.approx(expected_net-expected_benchmark)
    assert row['mean_delta_net'] == pytest.approx(expected_net-(.49-.21)/2)
    assert row['modified']['mfe_n'] == row['modified']['mae_n'] == 0
    assert row['modified']['mean_mfe'] is None


def test_nonoverlap_cooldown_keeps_original_horizon_after_early_exit_or_missing_baseline():
    labelled, events, f, days, ids = policy_inputs(signal_indexes=(0, 10, 20), horizons=(20,))
    for index in (0, 10, 20):
        trigger(f, days, index=index)
    checkpoints = build_checkpoints(events, f, days, ids)
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    rows = summarize_exit_policies(actual, labelled)
    assert get_policy_summary(rows)['paired_n'] == 3
    assert get_policy_summary(rows, population='nonoverlapping')['paired_n'] == 2
    # The first event still consumes 20 sessions even when its outcome is missing.
    labelled.loc[labelled.signal_index.eq(0), 'complete'] = False
    labelled.loc[labelled.signal_index.eq(0), ['net_return', 'gross_return']] = np.nan
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    row = get_policy_summary(summarize_exit_policies(actual, labelled), population='nonoverlapping')
    assert row['candidates'] == 2
    assert row['paired_n'] == 1


def test_policy_year_membership_uses_original_horizon_not_early_exit_year():
    labelled, events, f, days, ids = policy_inputs(horizons=(20,))
    trigger(f, days)
    checkpoints = build_checkpoints(events, f, days, ids)
    actual = build_exit_policies(labelled, checkpoints, f, days, ids)
    # Isolate the summary's period contract while retaining matching event IDs.
    for frame in (labelled, actual):
        frame['signal_date'] = '2024-12-20'
        frame['entry_date'] = '2024-12-23'
        frame['exit_date'] = '2025-01-21'
    actual['policy_exit_date'] = '2024-12-26'
    rows = summarize_exit_policies(actual, labelled)
    assert get_policy_summary(rows)['paired_n'] == 1
    assert get_policy_summary(rows, period='2024')['paired_n'] == 0
