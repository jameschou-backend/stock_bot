"""Fixed entry filters, known-outcome denominators and honest 60% candidates."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.entry_winrate import (
    build_conditions, summarize_candidates, summarize_entry_winrate,
)


BASE_ATOMS = (
    'not_extended', 'contraction', 'moderate_volume', 'prior_quiet',
    'close_strong', 'rs20_positive', 'market_above60', 'market_breadth',
    'peer_breadth', 'peer_turnover', 'flow_positive_lag1',
    'flow_positive_lag3', 'poc_up',
)
COND = 'ew__not_extended'


def features(count=3):
    return pd.DataFrame([
        dict(cohort='original_red', event_id=f'e-{i}', stock_id=str(2300+i),
             signal_date='2024-01-02', signal_index=0,
             **{name: True for name in BASE_ATOMS})
        for i in range(count)
    ])


def event(event_id='e-0', **changes):
    row = dict(
        cohort='original_red', event_id=event_id, stock_id='2300',
        signal_date='2024-01-02', signal_index=0, horizon=20,
        entry_date='2024-01-03', exit_date='2024-01-30',
        mature=True, complete=True, gross_return=.1, net_return=.09,
        benchmark_net_return=.01, mfe=.4, mae=-.1, threshold=.3,
        all_entries=True, **{COND: True},
    )
    row.update(changes)
    return row


def summarize(rows):
    frame = pd.DataFrame(rows)
    for atom in BASE_ATOMS:
        frame[atom] = True
    frame['not_extended'] = frame.pop(COND)
    frame = frame.drop(columns='all_entries')
    labelled, definitions = build_conditions(frame)
    return summarize_entry_winrate(labelled, definitions)


def get_record(summary, **changes):
    wanted = dict(population='raw', cohort='original_red', horizon=20,
                  condition_id=COND, period='all')
    wanted.update(changes)
    matches = [r for r in summary['records']
               if all(r.get(key) == value for key, value in wanted.items())]
    assert len(matches) == 1
    return matches[0]


def candidate_record(period='all', *, population='nonoverlapping', **changes):
    stats = dict(n=100, win=60, loss=40, breakeven=0, win_rate=.6,
                 unique_stocks=25, unique_dates=30)
    stats.update(changes)
    return dict(population=population, cohort='original_red', horizon=20,
                condition_id=COND, filter_id=COND, atoms=['not_extended'],
                period=period, counts={}, baseline={}, reject={},
                **{'pass': stats})


def candidate_summary(**changes):
    rows = [candidate_record(**changes)]
    rows += [candidate_record(year, n=30, win=18, loss=12)
             for year in ('2024', '2025', '2026')]
    return summarize_candidates(rows)


def test_fixed_grid_contains_fifteen_atoms_102_pairs_and_baseline():
    result, definitions = build_conditions(features())
    assert len(definitions) == 118
    assert definitions['all_entries'] == []
    atoms = [value[0] for value in definitions.values() if len(value) == 1]
    assert atoms == [*BASE_ATOMS, 'market_not_above60', 'market_narrow']
    pairs = [set(value) for value in definitions.values() if len(value) == 2]
    assert len(pairs) == 102
    for excluded in (
        {'market_above60', 'market_not_above60'},
        {'market_breadth', 'market_narrow'},
        {'flow_positive_lag1', 'flow_positive_lag3'},
    ):
        assert excluded not in pairs
    assert all(len(value) <= 2 for value in definitions.values())
    for key, value in definitions.items():
        assert key == ('all_entries' if not value else 'ew__'+'__and__'.join(value))
        assert key in result
    assert result['all_entries'].all()


def test_unknown_atom_survives_even_when_other_atom_is_false():
    source = features(4)
    source['not_extended'] = pd.array([False, True, False, True], dtype='boolean')
    source['contraction'] = pd.array([None, None, False, True], dtype='boolean')
    result, _ = build_conditions(source)
    values = result['ew__not_extended__and__contraction']
    assert values.iloc[:2].isna().all()
    assert not values.iloc[2]
    assert values.iloc[3]
    assert result['ew__contraction'].iloc[:2].isna().all()


def test_market_complements_preserve_unknown_and_never_change_source():
    source = features()
    source['market_above60'] = pd.array([True, False, None], dtype='boolean')
    source['market_breadth'] = pd.array([False, None, True], dtype='boolean')
    original = source.copy(deep=True)
    result, _ = build_conditions(source)
    assert result['ew__market_not_above60'].iloc[:2].tolist() == [False, True]
    assert pd.isna(result['ew__market_not_above60'].iloc[2])
    assert result['ew__market_narrow'].iloc[0]
    assert pd.isna(result['ew__market_narrow'].iloc[1])
    assert not result['ew__market_narrow'].iloc[2]
    pd.testing.assert_frame_equal(source, original)


def test_future_outcomes_cannot_change_any_condition_or_event_order():
    source = features()
    source.index = [50, 10, 90]
    source['net_return'] = [9., -.99, .4]
    source['mfe'] = [10., 0., .6]
    source['complete'] = [True, False, True]
    original, definitions = build_conditions(source)
    altered = source.copy()
    altered[['net_return', 'mfe']] = np.nan
    altered['complete'] = ~altered['complete']
    rerun, next_definitions = build_conditions(altered)
    assert definitions == next_definitions
    pd.testing.assert_frame_equal(original[list(definitions)], rerun[list(definitions)])
    assert original.event_id.tolist() == source.event_id.tolist()
    assert original.index.tolist() == source.index.tolist()


@pytest.mark.parametrize('value', ['False', 'True', 0, 1, .5])
def test_nonboolean_atom_is_rejected_instead_of_truthiness_cast(value):
    source = features()
    source['not_extended'] = [value, True, False]
    with pytest.raises(ValueError):
        build_conditions(source)


def test_missing_atom_is_explicit_error():
    with pytest.raises(ValueError, match='poc_up'):
        build_conditions(features().drop(columns='poc_up'))


def test_net_costs_determine_win_loss_and_breakeven_stays_in_denominator():
    rows = [event(f'e-{i}', stock_id=str(2300+i), net_return=value)
            for i, value in enumerate([.1, .2, .3, -.4, 0.])]
    stats = get_record(summarize(rows))['pass']
    assert stats['n'] == 5
    assert (stats['win'], stats['loss'], stats['breakeven']) == (3, 1, 1)
    assert stats['win_rate'] == .6
    assert stats['avg_win'] == pytest.approx(.2)
    assert stats['avg_loss'] == -.4
    assert stats['payoff_ratio'] == pytest.approx(.5)
    assert stats['unit_profit_factor'] == pytest.approx(1.5)
    assert stats['worst'] == -.4
    assert stats['p05'] == pytest.approx(-.32)
    assert stats['unique_stocks'] == 5
    assert stats['unique_dates'] == 1


def test_no_losses_is_labeled_not_infinite_profit_factor():
    stats = get_record(summarize([event()]))['pass']
    assert stats['unit_profit_factor'] is None
    assert stats['profit_factor_status'] == 'no_losses'


def test_missing_benchmark_does_not_drop_stock_return():
    rows = [event('a', stock_id='2301', net_return=.1, benchmark_net_return=.02),
            event('b', stock_id='2302', net_return=-.2, benchmark_net_return=np.nan)]
    stats = get_record(summarize(rows))['pass']
    assert stats['n'] == 2
    assert stats['win_rate'] == .5
    assert stats['mean_net'] == pytest.approx(-.05)
    assert stats['benchmark_paired_n'] == 1
    assert stats['mean_excess'] == pytest.approx(.08)


def test_unknown_filter_has_own_baseline_and_missing_future_is_not_loss():
    rows = [
        event('a', stock_id='2301'),
        event('b', stock_id='2302', **{COND: None}),
        event('c', stock_id='2303', **{COND: False}),
        event('d', stock_id='2304', complete=False, net_return=np.nan),
        event('e', stock_id='2305', complete=False, mature=False, net_return=np.nan),
    ]
    summary = summarize(rows)
    row = get_record(summary)
    assert row['baseline']['n'] == 2
    assert row['pass']['n'] == row['reject']['n'] == 1
    assert row['counts']['filter_unknown_n'] == 1
    assert row['counts']['missing_future_n'] == 1
    assert row['counts']['immature_n'] == 1
    assert get_record(summary, condition_id='all_entries')['pass']['n'] == 3


def test_cooldown_is_before_condition_future_and_does_not_restart_per_year():
    rows = [
        event('first', signal_date='2024-12-27', signal_index=240,
              entry_date='2024-12-30', exit_date='2025-01-27',
              complete=False, net_return=np.nan, **{COND: None}),
        event('too-soon', signal_date='2025-01-02', signal_index=243,
              entry_date='2025-01-03', exit_date='2025-01-30', net_return=3.),
        event('later', signal_date='2025-01-24', signal_index=260,
              entry_date='2025-01-27', exit_date='2025-02-24', net_return=-.1),
    ]
    summary = summarize(rows)
    raw = get_record(summary)
    reduced = get_record(summary, population='nonoverlapping')
    assert raw['pass']['n'] == 2
    assert reduced['pass']['n'] == 1
    assert reduced['pass']['win_rate'] == 0
    assert reduced['counts']['missing_future_n'] == 1
    assert get_record(summary, population='nonoverlapping', period='2025')['pass']['n'] == 1


def test_cross_year_exit_only_enters_all_period_and_not_wrong_year():
    rows = [event('cross', signal_date='2024-12-27', signal_index=240,
                  entry_date='2024-12-30', exit_date='2025-01-27')]
    summary = summarize(rows)
    assert get_record(summary)['pass']['n'] == 1
    year = get_record(summary, period='2024')
    assert year['counts']['boundary_n'] == 1
    assert year['pass']['n'] == 0
    assert get_record(summary, period='2025')['counts']['candidate_n'] == 0


@pytest.mark.parametrize('changes', [
    {'n': 99}, {'unique_stocks': 24}, {'unique_dates': 29},
    {'win_rate': .59999999999},
])
def test_qualification_uses_every_unrounded_sample_boundary(changes):
    result = candidate_summary(**changes)
    assert not result['qualified_60']
    assert not result['primary_versions'][0]['qualified_60']
    assert not result['cross_year_qualified_60']


def test_qualification_accepts_exact_threshold_but_not_as_live_proof():
    result = candidate_summary()
    assert len(result['qualified_60']) == 1
    assert len(result['cross_year_60']) == 1
    assert result['primary_versions'][0]['qualified_60']
    assert not result['live_qualified']
    assert not result['unseen_validation']
    assert not result['multiple_testing_corrected']


def test_high_win_rate_with_too_little_data_is_separately_labeled():
    result = candidate_summary(n=99)
    assert not result['qualified_60']
    assert len(result['small_sample_60']) == 1
    assert result['small_sample_60'][0]['win_rate_60']
    assert not result['small_sample_60'][0]['sample_sufficient']
    # Annual consistency is reported independently; it never overrides the
    # full-period minimum sample in the combined qualification.
    assert len(result['cross_year_60']) == 1
    assert not result['cross_year_qualified_60']


@pytest.mark.parametrize('changes', [{'n': 29}, {'win_rate': .59999999}])
def test_yearly_consistency_requires_every_year_known_at_full_precision(changes):
    rows = [candidate_record()]
    rows += [candidate_record(year, n=30, win=18, loss=12)
             for year in ('2024', '2025', '2026')]
    rows[2]['pass'].update(changes)
    result = summarize_candidates(rows)
    assert len(result['qualified_60']) == 1
    assert not result['cross_year_60']


def test_raw_winners_and_baseline_are_not_promoted_to_new_candidates():
    baseline = candidate_record()
    baseline.update(condition_id='all_entries', filter_id='all_entries', atoms=[])
    result = summarize_candidates([
        candidate_record(win_rate=.5), candidate_record(population='raw', win_rate=.9),
        baseline,
    ])
    assert not result['qualified_60']
    assert len(result['primary_baselines']) == 1
    assert len(result['primary_versions']) == 1


def test_failed_conditions_remain_reported_for_all_years_populations():
    frame, definitions = build_conditions(features(1))
    labels = event(net_return=-.1)
    for key in ('cohort', 'event_id', 'stock_id', 'signal_date', 'signal_index'):
        labels.pop(key)
    frame = frame.assign(**labels)
    result = summarize_entry_winrate(frame, definitions)
    assert len(result['records']) == 118 * 2 * 4
    selected = summarize_candidates(result['records'])
    assert len(selected['primary_versions']) == 117
    assert len(selected['primary_baselines']) == 1
    assert not selected['qualified_60']
    assert not selected['small_sample_60']


def test_candidate_classification_does_not_mutate_summary_records():
    rows = [candidate_record()]
    before = deepcopy(rows)
    summarize_candidates(rows)
    assert rows == before
