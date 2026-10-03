from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from scripts.research_sector_participation_20261003 import (
    ARM, PREKNOWN, fixed_pairs, matched_outcomes, monthly_peers,
    participation_features, promotion_gate, summarize, turnover_bin,
)


def fixture():
    days = pd.bdate_range('2019-01-01', periods=340)
    ids = ['0050']+[str(2000+i) for i in range(35)]
    rng = np.random.default_rng(120)
    daily = rng.normal(.0003, .014, (len(days), len(ids)))
    common = .003+.002*np.sin(np.arange(len(days)))
    daily[:, 1:10] = common[:, None]
    close = pd.DataFrame(100*np.cumprod(1+daily, axis=0), index=days, columns=ids)
    raw = pd.DataFrame(100., index=days, columns=ids)
    volume = pd.DataFrame(1e6, index=days, columns=ids)
    eligible = pd.DataFrame(True, index=days, columns=ids)
    t = 325
    volume.iloc[t-4:t+1, 2:10] *= 3
    row = dict(signal_id='one', signal_date=str(days[t].date()), stock_id='2000',
               rank_priority=.2, rank_score_bin='score_10_20pp', turnover_mean20=100e6)
    return [close, raw, volume, eligible], [row], t


def features(frames, rows):
    return participation_features(*frames, rows)


def test_causal_feature_combined_arm_excludes_self_and_0050():
    frames, rows, _ = fixture()
    result, groups = features(frames, rows)
    r = result[0]
    assert r['group_participation'] is True
    assert r['peer_above_ma20_fraction'] == 1
    assert r['peer_share_multiple'] >= 1.2
    assert r['feature_issue'] is None
    assert not {'2000', '0050'} & set(r['peer_ids'])
    assert len(r['peer_ids']) == 8
    assert groups[0]['cutoff_date'] < rows[0]['signal_date'][:7]+'-01'


def test_future_prices_and_outcomes_cannot_change_features():
    frames, rows, t = fixture()
    base = features(frames, rows)
    prefix = [f.iloc[:t+1] for f in frames]
    assert features(prefix, rows) == base
    changed = [f.copy() for f in frames]
    for f in changed[:3]:
        f.iloc[t+1:] = np.nan
    changed[-1].iloc[t+1:] = False
    modified_rows = [dict(rows[0], net_return=999, status='closed', exit_date='2100-01-01')]
    assert features(changed, modified_rows) == base


def test_candidate_after_cutoff_cannot_confirm_itself_in_numerator_or_denominator():
    frames, rows, t = fixture()
    base, groups = features(frames, rows)
    altered = [f.copy() for f in frames]
    after = altered[0].index > pd.Timestamp(groups[0]['cutoff_date'])
    altered[0].loc[after, '2000'] *= 900
    altered[1].loc[after, '2000'] *= 1e12
    altered[2].loc[after, '2000'] *= 1e12
    actual, new_groups = features(altered, rows)
    assert new_groups == groups
    own = {'own_return20', 'own_minus_peer_return20', 'leader_description'}
    assert {k:v for k,v in base[0].items() if k not in own} == {k:v for k,v in actual[0].items() if k not in own}


def test_monthly_membership_does_not_refit_after_cutoff():
    frames, rows, t = fixture()
    rows.append(dict(rows[0], signal_id='two', signal_date=str(frames[0].index[t-1].date())))
    result, groups = features(frames, rows)
    assert len(groups) == 1
    assert result[0]['peer_ids'] == result[1]['peer_ids']


def test_common_peer_set_uses_80pct_and_missing_does_not_become_zero():
    frames, rows, t = fixture()
    frames[2].iloc[t-7, frames[2].columns.get_loc('2001')] = np.nan
    result, _ = features(frames, rows)
    assert result[0]['observed_peers'] == 7
    assert '2001' not in result[0]['observed_peer_ids']
    assert result[0]['feature_issue'] is None
    frames[2].iloc[t-8, frames[2].columns.get_loc('2002')] = np.nan
    result, _ = features(frames, rows)
    assert result[0]['observed_peers'] == 6
    assert result[0][ARM] is None
    assert result[0]['feature_issue'] == 'insufficient_common_peer_coverage'


def test_market_coverage_gap_is_unknown_not_false():
    frames, rows, t = fixture()
    frames[1].iloc[t-10, 20:24] = np.nan
    result, _ = features(frames, rows)
    assert result[0][ARM] is None
    assert result[0]['breadth_confirmed'] is True
    assert result[0]['feature_issue'] == 'insufficient_market_amount_coverage'


def test_too_few_peers_and_cutoff_ineligibility_are_unknown():
    frames, rows, _ = fixture()
    first = frames[0].index.to_period('M') == pd.Period(rows[0]['signal_date'][:7])
    cutoff = np.flatnonzero(first)[0]-1
    frames[3].iloc[cutoff, 1] = False
    result, _ = features(frames, rows)
    assert result[0][ARM] is None
    assert result[0]['feature_issue'] == 'insufficient_history_or_ineligible_cutoff'


def test_pairwise_observation_threshold_and_ties_are_fixed():
    days = pd.bdate_range('2020-01-01', periods=180)
    ids = ['0050', '2000', '2001', '2002', '2003', '2004']
    values = np.tile(np.sin(np.arange(len(days)))[:, None], (1, len(ids)))
    eligible = np.ones(values.shape, dtype=bool)
    row = dict(signal_date=str(days[-1].date()), stock_id='2000')
    cutoff = np.flatnonzero(days.to_period('M') == days[-1].to_period('M'))[0]-1
    values[cutoff-119:cutoff-99, 2] = np.nan  # exactly100pairedobservations
    values[cutoff-119:cutoff-98, 3] = np.nan  #99fails
    groups = monthly_peers(values, eligible, days, ids, [row])
    group = groups[0]
    assert '2001' in group['peer_ids'] and '2002' not in group['peer_ids']
    assert dict(zip(group['peer_ids'], group['paired_counts']))['2001'] == 100
    tied = monthly_peers(np.tile(np.sin(np.arange(len(days)))[:, None], (1, len(ids))), eligible, days, ids, [row])[0]
    assert tied['peer_ids'] == ['2001', '2002', '2003', '2004']


def match_rows():
    return [dict(signal_id=sid, signal_date='2024-01-02', rank_score_bin='score_10_20pp',
                 rank_priority=score, turnover_mean20=120e6, group_participation=keep)
            for sid, score, keep in [('p1', .11, True), ('p2', .15, True), ('n1', .12, False), ('n2', .17, False), ('u1', .13, None)]]


def test_matching_is_deterministic_without_replacement_and_outcome_independent():
    rows = match_rows()
    result = fixed_pairs(rows)
    assert [(r['pass_signal_id'], r['fail_signal_id']) for r in result['pairs']] == [('p1', 'n1'), ('p2', 'n2')]
    assert len({r['pass_signal_id'] for r in result['pairs']}) == 2
    modified = [dict(r, status='open' if i%2 else 'closed', net_return=999) for i,r in enumerate(reversed(rows))]
    assert fixed_pairs(modified) == result
    assert result['excluded'] == [dict(signal_id='u1', reason='unknown_group_participation')]


def test_matching_does_not_cross_day_score_or_turnover_cells():
    rows = match_rows()
    rows[2]['signal_date'] = '2024-01-03'
    rows[3]['turnover_mean20'] = 300e6
    assert not fixed_pairs(rows)['pairs']
    assert [turnover_bin(x) for x in [99e6, 100e6, 299e6, 300e6]] == ['below100m', '100m_to300m', '100m_to300m', 'at_least300m']


def test_unknown_outcome_excludes_fixed_pair_without_rematching():
    rows = match_rows()
    pairs = fixed_pairs(rows)
    outcomes = [dict(r, status='closed', net_return=.1) for r in rows]
    outcomes[2]['status'] = 'open'
    out = matched_outcomes(pairs, outcomes)
    assert out['scopes']['all']['fixed_pairs'] == 2
    assert out['scopes']['all']['both_closed_pairs'] == 1
    assert out['scopes']['all']['incomplete_pairs'] == 1
    assert len(out['pairs']) == 2


def test_opportunity_denominator_and_unknowns_are_separate():
    rows = [dict(signal_id=str(i), signal_date='2024-01-02', status='closed', net_return=ret,
                 holding_days=3, exit_reason='test', group_participation=keep, leader_description=None,
                 breadth_confirmed=None, turnover_confirmed=None)
            for i,(ret,keep) in enumerate([(.5, True), (-.1, False), (.2, None)])]
    result = summarize(rows)
    assert result['opportunities']['selected_plus_cash_mean_on_known'] == .25
    assert result['opportunities']['known_original_mean'] == .2
    assert result['opportunities']['known_opportunity_improvement'] == pytest.approx(.05)
    assert result['opportunities']['fully_observed_all_original_mean'] is None
    assert result['unknown']['closed'] == 1
    assert result['return30_retention'] == 1


def test_promotion_requires_coverage_righttail_and_all_periods():
    result = dict(observable_signal_fraction=.9, return30_retention=.85)
    periods = ('2019_2022', '2023_2024', '2025_2026')
    results = dict(all=result, **{p:dict(opportunities=dict(known_opportunity_improvement=.01)) for p in periods})
    assert promotion_gate(results)['passed']
    results['2023_2024']['opportunities']['known_opportunity_improvement'] = 0
    assert not promotion_gate(results)['passed']
    results['all']['observable_signal_fraction'] = .79
    results['all']['return30_retention'] = .79
    assert not any(promotion_gate(results)['checks'].values())
