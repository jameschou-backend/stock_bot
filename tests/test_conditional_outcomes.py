import json

import numpy as np
import pandas as pd
import pytest

from skills.conditional_outcomes import analyze_conditions


def sources():
    candidates = [
        ('2024-01-02', '1101', .4, True, True, True, True),
        ('2024-01-02', '1102', -.2, True, False, True, True),
        ('2024-01-02', '1103', .8, False, False, True, True),
        ('2024-01-03', '1104', .1, True, True, True, False),
        ('2024-01-03', '1105', -.4, True, False, False, False),
        ('2025-01-02', '1106', .7, True, False, True, True),
        ('2026-01-02', '1107', -.1, True, True, True, True),
    ]
    events, conditions = [], []
    for date, stock, net, known_a, matched_a, known_b, matched_b in candidates:
        for horizon in (5, 20, 60):
            events.append(dict(signal_date=date, stock_id=stock, horizon=horizon,
                status='evaluated', gross_return=net+.02, net_return=net,
                benchmark_net_return=.02, excess_vs0050=net-.02))
        for identifier, known, matched in [('a', known_a, matched_a), ('b', known_b, matched_b)]:
            conditions.append(dict(signal_date=date, stock_id=stock, filter_id=identifier,
                                   known=known, matched=matched))
    return pd.DataFrame(events), pd.DataFrame(conditions)


def run(events=None, conditions=None):
    default_events, default_conditions = sources()
    return analyze_conditions(default_events if events is None else events,
                              default_conditions if conditions is None else conditions,
                              '2024-01-02', '2026-10-05')


def row(report, identifier='a', horizon=20, year='all', *, common=False):
    key = 'common_pool_summary' if common else 'summary'
    return next(r for r in report[key] if r['filter_id'] == identifier and r['horizon'] == horizon and r['year'] == year)


def test_full_known_cohort_cash_denominator_costs_and_positive_negative_retention():
    report = run()
    r = row(report)
    assert r['all_candidates'] == 7 and r['feature_known'] == 6 and r['feature_unknown'] == 1
    assert r['known_evaluated'] == 6 and r['kept_evaluated'] == 3 and r['excluded_evaluated'] == 3
    assert r['coverage'] == pytest.approx(6/7)
    assert r['baseline_mean_net'] == pytest.approx(.5/6)
    assert r['filtered_cash_mean_net'] == pytest.approx(.4/6)
    assert r['paired_mean_delta'] == pytest.approx(-.1/6)
    assert r['kept_mean_net'] == pytest.approx(.4/3)
    assert r['excluded_mean_net'] == pytest.approx(.1/3)
    assert r['benchmark_mean_net'] == .02
    assert r['filtered_cash_mean_excess'] == pytest.approx(.4/6-.02)
    assert r['kept_profit_win_rate'] == pytest.approx(2/3)
    assert r['filtered_cash_profit_win_rate'] == pytest.approx(2/6)
    assert not r['cash_is_win']
    assert r['retained_positive_profit_count'] == 2
    assert r['retained_positive_profit_fraction'] == pytest.approx(2/3)
    assert r['avoided_loss_count'] == 2 and r['avoided_loss_rate'] == pytest.approx(2/3)
    assert report['definitions']['costs'].startswith('Already included')
    json.dumps(report, allow_nan=False)


def test_unknown_is_neither_false_nor_cash_and_all_unknown_means_remain_null():
    events, conditions = sources()
    assert row(run())['kept_evaluated'] == 3
    conditions['known'] = False
    conditions['matched'] = False
    report = run(events, conditions)
    for r in report['summary'] + report['common_pool_summary']:
        assert r['known_evaluated'] == 0 and r['kept_evaluated'] == 0
        assert r['feature_unknown'] == r['all_candidates']
        assert r['filtered_cash_mean_net'] is None and r['paired_mean_delta'] is None
        assert r['kept_profit_win_rate'] is None and r['avoided_loss_rate'] is None
        assert r['baseline_known']['median'] is None
        assert r['filtered_cash']['worst5pct_mean'] is None
    assert row(report)['coverage'] == 0


def test_all_filters_known_secondary_has_identical_comparison_population():
    report = run()
    a, b = row(report, common=True), row(report, 'b', common=True)
    assert a['scope'] == 'all_filters_known'
    assert a['all_candidates'] == b['all_candidates'] == 7
    assert a['feature_known'] == b['feature_known'] == 5
    assert a['feature_unknown'] == b['feature_unknown'] == 2
    assert a['this_filter_unknown'] == b['this_filter_unknown'] == 1
    assert a['baseline_mean_net'] == b['baseline_mean_net'] == pytest.approx(.9/5)
    assert a['filtered_cash_mean_net'] == pytest.approx(.4/5)
    assert b['filtered_cash_mean_net'] == pytest.approx(.8/5)
    assert a['same_date_contrast']['dates'] == 1
    assert a['same_date_contrast']['mean_net_contrast'] == pytest.approx(.6)


def test_immature_and_missing_outcomes_counted_without_zero_imputation_or_inner_join():
    events, conditions = sources()
    for stock, status in [('1102', 'stock_path_missing'), ('1104', 'benchmark_path_missing'), ('1106', 'immature')]:
        events.loc[events.stock_id.eq(stock) & events.horizon.eq(20), 'status'] = status
    report = run(events, conditions)
    r = row(report)
    assert r['all_candidates'] == 7 and r['all_evaluated'] == 4
    assert r['feature_known'] == 6 and r['known_evaluated'] == 3 and r['unknown_evaluated'] == 1
    assert r['immature'] == r['stock_path_missing'] == r['benchmark_path_missing'] == 1
    assert r['known_immature'] == r['known_stock_path_missing'] == r['known_benchmark_path_missing'] == 1
    assert r['baseline_mean_net'] == pytest.approx(-.1/3)
    assert r['filtered_cash_mean_net'] == pytest.approx(.3/3)
    # Partial return fields carried by unavailable rows never enter means.
    assert r['kept_evaluated'] == 2 and r['excluded_evaluated'] == 1


def test_same_date_contrast_is_equal_date_weighted_and_not_kept_vs_excluded_pool():
    r = row(run())
    same = r['same_date_contrast']
    assert same['dates'] == 2 and same['kept_events'] == same['excluded_events'] == 2
    assert same['mean_net_contrast'] == pytest.approx((.6+.5)/2)
    assert same['win_rate_contrast'] == 1
    assert same['mean_net_contrast'] != pytest.approx(r['kept_mean_net']-r['excluded_mean_net'])
    assert 'not_causal' in same['interpretation']
    no_pairs = row(run(), 'b')['same_date_contrast']
    assert no_pairs['dates'] == 0 and no_pairs['mean_net_contrast'] is None


def test_rally_thresholds_retention_precision_and_five_day_undefined():
    report = run()
    twenty, sixty, five = (row(report, horizon=h) for h in (20, 60, 5))
    assert twenty['rally_total_count'] == 2 and twenty['rally_kept_count'] == 1
    assert twenty['rally_retention'] == .5
    assert twenty['rally_kept_precision'] == pytest.approx(1/3)
    assert sixty['rally_total_count'] == 1 and sixty['rally_kept_count'] == 0
    assert sixty['rally_retention'] == 0 and sixty['rally_kept_precision'] == 0
    assert five['rally_gross_threshold'] is None and five['rally_total_count'] is None
    assert five['rally_kept_count'] is None and five['rally_retention'] is None


def test_annual_rows_conserve_candidates_coverage_outcomes_and_weighted_returns():
    report = run()
    assert len(report['summary']) == len(report['common_pool_summary']) == 2*3*4
    for key in ('summary', 'common_pool_summary'):
        for total in [r for r in report[key] if r['year'] == 'all']:
            annual = [r for r in report[key] if r['filter_id'] == total['filter_id']
                      and r['horizon'] == total['horizon'] and r['year'] != 'all']
            for field in ['all_candidates', 'all_evaluated', 'feature_known', 'feature_unknown',
                          'known_evaluated', 'kept_evaluated', 'excluded_evaluated',
                          'positive_profit_count', 'loss_count', 'retained_positive_profit_count', 'avoided_loss_count']:
                assert sum(r[field] for r in annual) == total[field]
            weighted = sum(r['known_evaluated']*r['filtered_cash_mean_net'] for r in annual if r['known_evaluated'])
            assert weighted/total['known_evaluated'] == pytest.approx(total['filtered_cash_mean_net'])


def test_tail_uses_ceil_five_percent_including_cash_and_is_not_account_drawdown():
    events, conditions = [], []
    for i, net in enumerate([-.9, -.4]+[.05]*19):
        sid = str(1100+i)
        for h in (5, 20, 60):
            events.append(dict(signal_date='2024-01-02', stock_id=sid, horizon=h,
                status='evaluated', gross_return=net+.02, net_return=net,
                benchmark_net_return=.02, excess_vs0050=net-.02))
        conditions.append(dict(signal_date='2024-01-02', stock_id=sid,
                               filter_id='a', known=True, matched=i == 0))
    report = run(pd.DataFrame(events), pd.DataFrame(conditions))
    r = row(report)
    assert r['baseline_known']['worst5pct_count'] == 2
    assert r['baseline_known']['worst5pct_mean'] == pytest.approx(-.65)
    assert r['filtered_cash']['worst5pct_count'] == 2
    assert r['filtered_cash']['worst5pct_mean'] == pytest.approx(-.45)
    assert r['filtered_cash']['median'] == 0
    assert r['kept']['worst5pct_mean'] == -.9
    assert r['paired_delta_cluster_bootstrap'] is None  # Only one signal date.
    assert 'not_account_drawdown' in r['tail_definition']


def test_bootstrap_reproducible_date_clustered_and_does_not_make_promotion_claim():
    report, again = run(), run()
    interval = row(report)['paired_delta_cluster_bootstrap']
    assert interval == row(again)['paired_delta_cluster_bootstrap']
    assert interval['clusters'] == 4 and interval['samples'] == 500
    assert interval['seed'] == 20261006 and interval['descriptive_only'] is True
    assert interval['low'] <= row(report)['paired_mean_delta'] <= interval['high']
    assert report['live_qualified'] is False and report['cumulative_return'] is None


def test_changing_outcomes_cannot_reclassify_candidate_feature_availability_or_selection():
    events, conditions = sources()
    before = run(events, conditions)
    events['gross_return'] += 3
    events['net_return'] += 3
    events['excess_vs0050'] += 3
    after = run(events, conditions)
    for key in ('summary', 'common_pool_summary'):
        for a, b in zip(before[key], after[key]):
            for field in ['filter_id', 'horizon', 'year', 'all_candidates', 'feature_known',
                          'feature_unknown', 'coverage', 'kept_candidates', 'excluded_candidates']:
                assert a[field] == b[field]
    bad = conditions.assign(future_return=1.)
    with pytest.raises(ValueError, match='outcome columns forbidden'):
        run(events, bad)


@pytest.mark.parametrize('mutation,match', [
    ('duplicate_event', 'Duplicate candidate/horizon'),
    ('duplicate_condition', 'Duplicate candidate/filter'),
    ('missing_condition', 'exactly every candidate'),
    ('extra_condition', 'exactly every candidate'),
    ('missing_horizon', 'all three'),
    ('bad_boolean', 'explicit booleans'),
    ('bad_status', 'Unknown outcome status'),
    ('infinite_evaluated', 'finite'),
    ('wrong_excess', 'paired benchmark'),
    ('empty_filter_name', 'Filter identifiers'),
])
def test_bad_sources_fail_instead_of_silent_join_loss(mutation, match):
    events, conditions = sources()
    if mutation == 'duplicate_event': events = pd.concat([events, events.iloc[:1]], ignore_index=True)
    elif mutation == 'duplicate_condition': conditions = pd.concat([conditions, conditions.iloc[:1]], ignore_index=True)
    elif mutation == 'missing_condition': conditions = conditions.iloc[1:].copy()
    elif mutation == 'extra_condition':
        extra = conditions.iloc[:1].copy(); extra['stock_id'] = '9999'
        conditions = pd.concat([conditions, extra], ignore_index=True)
    elif mutation == 'missing_horizon': events = events.iloc[1:].copy()
    elif mutation == 'bad_boolean': conditions['known'] = conditions.known.astype(int)
    elif mutation == 'bad_status': events.loc[0, 'status'] = 'unknown_filled_zero'
    elif mutation == 'infinite_evaluated': events.loc[0, 'net_return'] = np.inf
    elif mutation == 'wrong_excess': events.loc[0, 'excess_vs0050'] = 100
    elif mutation == 'empty_filter_name': conditions.loc[0, 'filter_id'] = ''
    with pytest.raises(ValueError, match=match):
        run(events, conditions)


def test_empty_source_and_empty_requested_interval_raise_explicit_errors():
    events, conditions = sources()
    for e, c in [(events.iloc[:0], conditions), (events, conditions.iloc[:0]), (events.iloc[:0], conditions.iloc[:0])]:
        with pytest.raises(ValueError, match='nonempty'):
            run(e, c)
    with pytest.raises(ValueError, match='no complete candidates'):
        analyze_conditions(events, conditions, '2023-01-02', '2023-12-29')


def test_only_requested_period_is_compared_and_input_frames_remain_unchanged():
    events, conditions = sources()
    ecopy, ccopy = events.copy(deep=True), conditions.copy(deep=True)
    report = analyze_conditions(events, conditions, '2025-01-02', '2026-10-05')
    assert report['candidate_count'] == 2 and report['candidate_horizon_rows'] == 6
    assert set(r['year'] for r in report['summary']) == {'all', '2025', '2026'}
    pd.testing.assert_frame_equal(events, ecopy)
    pd.testing.assert_frame_equal(conditions, ccopy)


def test_explicit_filter_ids_allow_empty_candidate_report_without_fabricated_zero_metrics():
    events, conditions = sources()
    report = analyze_conditions(events.iloc[:0], conditions.iloc[:0], '2024-01-02', '2026-10-05', filter_ids=['a', 'b'])
    assert report['candidate_count'] == report['candidate_horizon_rows'] == report['condition_rows'] == 0
    assert report['filters'] == ['a', 'b']
    assert len(report['summary']) == len(report['common_pool_summary']) == 2*3*4
    for r in report['summary']+report['common_pool_summary']:
        assert r['all_candidates'] == r['feature_unknown'] == r['known_evaluated'] == 0
        assert r['coverage'] is None and r['filtered_cash_mean_net'] is None
        assert r['paired_mean_delta'] is None and r['filtered_cash_profit_win_rate'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('identifiers', [[], ['a', 'a'], [''], 'a', ['a'], ['a', 'b', 'c']])
def test_declared_filter_ids_validate_unique_nonempty_and_match_feature_source(identifiers):
    events, conditions = sources()
    with pytest.raises(ValueError, match='filter IDs'):
        analyze_conditions(events, conditions, '2024-01-02', '2026-10-05', filter_ids=identifiers)


def test_unknown_feature_cannot_be_labeled_matched():
    events, conditions = sources()
    conditions.loc[~conditions.known, 'matched'] = True
    with pytest.raises(ValueError, match='cannot be true'):
        run(events, conditions)


def test_kept_and_excluded_benchmark_comparisons_use_their_own_dates_not_full_pool():
    events, conditions = sources()
    # Distinct signal dates give these stocks genuinely different benchmark windows.
    dates = {'1101': '2024-01-02', '1102': '2024-01-03', '1103': '2024-01-04',
             '1104': '2024-01-05', '1105': '2024-01-08',
             '1106': '2025-01-02', '1107': '2026-01-02'}
    events['signal_date'] = events.stock_id.map(dates)
    conditions['signal_date'] = conditions.stock_id.map(dates)
    benchmark_by_stock = {'1101': .5, '1102': -.5, '1103': 99., '1104': -.2,
                          '1105': .2, '1106': 1., '1107': -.2}
    events['benchmark_net_return'] = events.stock_id.map(benchmark_by_stock)
    events['excess_vs0050'] = events.net_return-events.benchmark_net_return
    report = run(events, conditions)
    r = row(report)
    assert r['benchmark_mean_net'] == pytest.approx(.8/6)
    assert r['kept_benchmark_mean_net'] == pytest.approx(.1/3)
    assert r['excluded_benchmark_mean_net'] == pytest.approx(.7/3)
    assert r['kept_mean_excess'] == pytest.approx(.1)
    assert r['excluded_mean_excess'] == pytest.approx(-.2)
    assert r['kept_mean_excess'] == pytest.approx(r['kept_mean_net']-r['kept_benchmark_mean_net'])
    assert r['kept_mean_excess'] != pytest.approx(r['kept_mean_net']-r['benchmark_mean_net'])
    assert r['kept_beat_benchmark_rate'] == pytest.approx(2/3)
    assert r['excluded_beat_benchmark_rate'] == pytest.approx(1/3)
    common = row(report, common=True)
    assert common['kept_benchmark_mean_net'] == pytest.approx(.1/3)
    assert common['excluded_benchmark_mean_net'] == pytest.approx(.5/2)
    assert common['kept_mean_excess'] == pytest.approx(.1)
    assert common['excluded_mean_excess'] == pytest.approx(0.)
    # No selected events in this annual slice: do not invent benchmark zero.
    annual = row(report, year='2025')
    for key in ['kept_benchmark_mean_net', 'kept_mean_excess', 'kept_beat_benchmark_rate']:
        assert annual[key] is None


def test_benchmark_tie_is_not_a_win_and_empty_excluded_group_has_null_comparisons():
    events, conditions = sources()
    events['benchmark_net_return'] = events.net_return
    events['excess_vs0050'] = 0.
    conditions['known'] = True
    conditions['matched'] = True
    r = row(run(events, conditions))
    assert r['kept_beat_benchmark_rate'] == 0 and r['kept_mean_excess'] == 0
    for key in ['excluded_benchmark_mean_net', 'excluded_mean_excess', 'excluded_beat_benchmark_rate']:
        assert r[key] is None
