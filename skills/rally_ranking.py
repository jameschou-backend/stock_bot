"""Fixed, outcome-independent rankings of same-day signal candidates.

All three rankings use the same candidates with complete finite evidence.  The
caller must rank the signal population before inspecting forward outcomes or
applying account capacity constraints.  Unknown evidence is retained as unknown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


RANKING_FEATURES = (
    'relative_return20',
    'peer_breadth_value',
    'peer_turnover_multiple',
    'flow_ratio5_lag1',
)
RANKING_VARIANTS = ('rs', 'context', 'blend')
RANKING_DEFINITIONS = {
    'schema': 'rally_same_day_rankings_v1',
    'group_by': ['cohort', 'signal_date'],
    'known_requires': list(RANKING_FEATURES),
    'known_rule': 'all_four_numeric_features_finite',
    'percentile': 'ascending_average_tie_rank_divided_by_common_known_day_n',
    'score_rs': 'percentile(relative_return20)',
    'score_context': (
        '(percentile(peer_breadth_value) + percentile(peer_turnover_multiple) '
        '+ percentile(flow_ratio5_lag1)) / 3'
    ),
    'score_blend': '0.5 * score_rs + 0.5 * score_context',
    'rank_order': ['score descending', 'stock_id ascending', 'event_id ascending'],
    'unknown_policy': 'retain_all_rows_with_null_scores_percentiles_and_ranks',
    'outcomes_used': False,
    'rerank_after_missing_future': False,
    'single_known_candidate_percentile': 1.0,
}


def build_rankings(events: pd.DataFrame) -> pd.DataFrame:
    """Return every input row, in input order, with three fixed daily rankings.

    ``ranking_known`` requires all four finite numerical features, including
    zero or negative observed flows.  ``ranking_day_n`` counts the common-known
    candidates in that cohort/date and is available even on unknown rows.
    Percentiles and scores are nullable float columns; ranks are nullable Int64.
    The input's index, columns, and values are preserved, except that existing
    derived ranking columns are explicitly recomputed.  Invalid identifiers or
    malformed numerical values raise an error instead of becoming a fallback.
    """
    keys = ['cohort', 'signal_date', 'stock_id', 'event_id']
    required = keys + list(RANKING_FEATURES)
    missing = sorted(set(required).difference(events.columns))
    if missing:
        raise ValueError('Missing ranking columns: ' + ', '.join(missing))
    if not events.columns.is_unique:
        raise ValueError('Ranking input columns must be unique')
    work = events.loc[:, required].reset_index(drop=True).copy()
    for key in keys:
        if work[key].isna().any() or not work[key].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f'Ranking identifiers must be nonempty strings: {key}')
    if work.duplicated(['cohort', 'signal_date', 'stock_id', 'event_id']).any():
        raise ValueError('Duplicate ranking event coordinate')

    values = pd.DataFrame(index=work.index)
    for feature in RANKING_FEATURES:
        try:
            numeric = pd.to_numeric(work[feature], errors='raise')
            values[feature] = numeric.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Ranking feature must be numeric: {feature}') from exc
    known = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=work.index)
    result = events.copy()
    result['ranking_known'] = known.to_numpy(dtype=bool)
    groups = [work['cohort'], work['signal_date']]
    result['ranking_day_n'] = known.groupby(groups, sort=False).transform('sum').to_numpy(dtype=int)
    percentiles = values.where(known, np.nan).groupby(groups, sort=False).rank(
        method='average', ascending=True, pct=True)
    for feature in RANKING_FEATURES:
        result[f'percentile_{feature}'] = percentiles[feature].to_numpy()

    scores = {
        'rs': percentiles['relative_return20'],
        # min_count prevents partial evidence from producing a context score.
        'context': percentiles[list(RANKING_FEATURES[1:])].sum(axis=1, min_count=3) / 3,
    }
    scores['blend'] = .5 * scores['rs'] + .5 * scores['context']
    for variant, score in scores.items():
        result[f'score_{variant}'] = score.to_numpy()
        order = work.loc[known, keys].copy()
        order['_score'] = score.loc[known]
        order = order.sort_values(
            ['cohort', 'signal_date', '_score', 'stock_id', 'event_id'],
            ascending=[True, True, False, True, True], kind='stable')
        ordered_ranks = order.groupby(['cohort', 'signal_date'], sort=False).cumcount() + 1
        ranks = pd.array([pd.NA] * len(work), dtype='Int64')
        ranks[order.index.to_numpy()] = ordered_ranks.to_numpy(dtype=int)
        result[f'rank_{variant}'] = ranks
    return result
