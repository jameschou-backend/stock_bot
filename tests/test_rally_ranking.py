"""Ranking evidence, tie handling, and temporal independence contracts."""
import numpy as np
import pandas as pd
import pytest

from skills.rally_ranking import (
    RANKING_DEFINITIONS, RANKING_FEATURES, RANKING_VARIANTS, build_rankings,
)


def candidates():
    return pd.DataFrame([
        dict(cohort='original_red', signal_date='2024-01-02', stock_id=sid,
             event_id=f'event-{sid}', relative_return20=rs,
             peer_breadth_value=breadth, peer_turnover_multiple=turnover,
             flow_ratio5_lag1=flow)
        for sid, rs, breadth, turnover, flow in [
            ('2330', .3, .25, 1.0, -.1),
            ('2317', .2, .5, 2.0, 0.),
            ('2492', .1, .75, 3.0, .1),
        ]
    ])


def derived_columns(frame):
    return [column for column in frame if column.startswith(
        ('ranking_', 'percentile_', 'score_', 'rank_'))]


def test_exact_fixed_scores_and_ranks_use_common_known_population():
    actual = build_rankings(candidates())
    assert actual.ranking_known.all()
    assert actual.ranking_day_n.tolist() == [3, 3, 3]
    np.testing.assert_allclose(actual.score_rs, [1., 2/3, 1/3])
    np.testing.assert_allclose(actual.score_context, [1/3, 2/3, 1.])
    np.testing.assert_allclose(actual.score_blend, [2/3, 2/3, 2/3])
    assert actual.rank_rs.tolist() == [1, 2, 3]
    assert actual.rank_context.tolist() == [3, 2, 1]
    assert actual.rank_blend.tolist() == [2, 1, 3]
    assert RANKING_DEFINITIONS['outcomes_used'] is False
    assert RANKING_DEFINITIONS['rerank_after_missing_future'] is False


@pytest.mark.parametrize('feature', RANKING_FEATURES)
@pytest.mark.parametrize('missing', [np.nan, np.inf, -np.inf, None, pd.NA])
def test_nonfinite_component_excludes_row_from_all_three_comparisons(feature, missing):
    source = candidates().astype({feature: object})
    source.loc[1, feature] = missing
    actual = build_rankings(source)
    assert actual.ranking_known.tolist() == [True, False, True]
    assert actual.ranking_day_n.tolist() == [2, 2, 2]
    for column in derived_columns(actual):
        if not column.startswith('ranking_'):
            assert pd.isna(actual.loc[1, column])
    assert actual.loc[0, 'score_rs'] == 1.
    assert actual.loc[2, 'score_rs'] == .5


def test_zero_and_negative_flow_are_observed_evidence():
    actual = build_rankings(candidates())
    assert actual.loc[:1, 'ranking_known'].all()
    assert actual.loc[:1, 'rank_context'].notna().all()


def test_days_and_cohorts_rank_independently():
    source = candidates()
    more = source.iloc[[2]].copy()
    more['signal_date'] = '2024-01-03'
    more['event_id'] = 'next-date'
    other = source.iloc[[1]].copy()
    other['cohort'] = 'legacy_course_breakout'
    other['event_id'] = 'other-cohort'
    actual = build_rankings(pd.concat([source, more, other], ignore_index=True))
    pd.testing.assert_frame_equal(actual.iloc[:3].reset_index(drop=True), build_rankings(source))
    assert actual.ranking_day_n.tolist() == [3, 3, 3, 1, 1]
    for variant in RANKING_VARIANTS:
        assert actual.loc[3:, f'rank_{variant}'].tolist() == [1, 1]
        assert actual.loc[3:, f'score_{variant}'].tolist() == [1., 1.]


def test_tied_percentiles_average_then_ordinal_ranks_use_stock_and_event_ids():
    source = candidates()
    source.loc[:, list(RANKING_FEATURES)] = 1.
    duplicate_stock = source.iloc[[0]].copy()
    duplicate_stock['event_id'] = 'aaa-event'
    source = pd.concat([source, duplicate_stock], ignore_index=True)
    actual = build_rankings(source)
    for variant in RANKING_VARIANTS:
        assert actual[f'score_{variant}'].tolist() == [2.5/4] * 4
        assert actual[f'rank_{variant}'].tolist() == [3, 1, 4, 2]


def test_input_order_and_index_do_not_change_ranks_or_mutate_source():
    source = candidates()
    source.index = [7, 7, 2]
    frozen = source.copy(deep=True)
    actual = build_rankings(source)
    shuffled = build_rankings(source.iloc[[2, 0, 1]])
    pd.testing.assert_frame_equal(source, frozen)
    pd.testing.assert_frame_equal(actual[source.columns], source)
    assert actual.index.tolist() == [7, 7, 2]
    pd.testing.assert_frame_equal(
        actual.sort_values('event_id').reset_index(drop=True),
        shuffled.sort_values('event_id').reset_index(drop=True))


def test_future_outcomes_and_missing_labels_do_not_change_any_ranking():
    source = candidates()
    source['net_return'] = [100., -.99, np.nan]
    source['complete'] = [True, True, False]
    source['exit_date'] = ['2024-04-01', '2024-04-01', None]
    original = build_rankings(source)
    changed = source.copy()
    changed['net_return'] = [-.99, 100., 100.]
    changed['complete'] = [False, False, True]
    changed['exit_date'] = None
    actual = build_rankings(changed)
    pd.testing.assert_frame_equal(original[derived_columns(original)], actual[derived_columns(actual)])
    assert len(actual) == len(source)
    assert actual.ranking_day_n.tolist() == [3, 3, 3]


def test_all_unknown_day_has_zero_known_count_and_null_rankings():
    source = candidates()
    source['peer_breadth_value'] = np.nan
    actual = build_rankings(source)
    assert not actual.ranking_known.any()
    assert actual.ranking_day_n.tolist() == [0, 0, 0]
    assert actual[[column for column in derived_columns(actual)
                   if not column.startswith('ranking_')]].isna().all().all()


def test_empty_input_returns_same_schema_and_nullable_ranks():
    source = candidates().iloc[:0]
    actual = build_rankings(source)
    assert actual.empty
    for variant in RANKING_VARIANTS:
        assert str(actual[f'rank_{variant}'].dtype) == 'Int64'
    assert actual.ranking_known.dtype == bool


def test_existing_derived_values_are_explicitly_recomputed():
    source = candidates()
    source['rank_rs'] = 99
    source['score_rs'] = 99.
    source['ranking_known'] = False
    actual = build_rankings(source)
    assert actual.rank_rs.tolist() == [1, 2, 3]
    assert actual.score_rs.max() == 1.
    assert actual.ranking_known.all()


def test_missing_feature_column_raises_instead_of_using_another_strategy():
    with pytest.raises(ValueError, match='Missing ranking columns: peer_breadth_value'):
        build_rankings(candidates().drop(columns='peer_breadth_value'))


def test_malformed_numeric_feature_is_not_silently_unknown():
    source = candidates().astype({'flow_ratio5_lag1': object})
    source.loc[1, 'flow_ratio5_lag1'] = 'not-a-number'
    with pytest.raises(ValueError, match='feature must be numeric: flow_ratio5_lag1'):
        build_rankings(source)


@pytest.mark.parametrize('key', ['stock_id', 'event_id', 'cohort', 'signal_date'])
def test_missing_identity_cannot_silently_disappear_in_groupby(key):
    source = candidates()
    source.loc[1, key] = None
    with pytest.raises(ValueError, match='identifiers must be nonempty'):
        build_rankings(source)


def test_duplicate_event_coordinate_is_rejected():
    source = candidates()
    with pytest.raises(ValueError, match='Duplicate ranking event'):
        build_rankings(pd.concat([source, source.iloc[[0]]], ignore_index=True))
