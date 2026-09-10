"""Fixed comparison design helpers; no historical returns or external data."""
from copy import deepcopy

import pytest

from scripts import research_capacity as driver


def event(eid, priority, day='2025-01-03'):
    return dict(event_id=eid, priority=priority, signal_date='2025-01-02',
                entry_date=day, members=['2330'])


def test_same_day_order_uses_score_then_id_without_dropping_negative_scores():
    old = [event('b', .1), event('a', .2), event('c', .3, '2025-01-06')]
    new = [dict(e, priority=-.1) for e in old]
    new[0]['priority'] = -.05
    original = deepcopy(old)
    result = driver.ranking_stats(old, old, new, 4, [])
    assert old == original
    assert result['multiple_event_days'] == result['reordered_days'] == 1
    assert result['reorder_examples'] == [dict(date='2025-01-03', original_order=['a', 'b'],
                                               residual_order=['b', 'a'])]
    # Equal scores reproduce deterministic ID order, independent of list order.
    new[0]['priority'] = -.1
    assert driver.ranking_stats(old, old, new, 4, [])['reordered_days'] == 0


@pytest.mark.parametrize('mutation', ['event_id', 'entry_date', 'signal_date', 'members'])
def test_priority_contrast_rejects_changed_pool_or_event(mutation):
    old = [event('a', .1)]
    new = deepcopy(old)
    new[0][mutation] = ['2491'] if mutation == 'members' else 'changed'
    with pytest.raises(ValueError):
        driver.ranking_stats(old, old, new, 1, [])


@pytest.mark.parametrize('field', ['summary', 'curve', 'executions', 'cohorts', 'rejections'])
def test_control_reproduction_checks_ledgers_not_only_returns(field):
    sim = {key: [] for key in ('summary', 'curve', 'executions', 'cohorts', 'rejections')}
    ref = dict(deepcopy(sim), available=True)
    driver.verify_control(sim, ref)
    ref[field] = ['changed']
    with pytest.raises(ValueError, match='exactly'):
        driver.verify_control(sim, ref)


def test_cohort_comparison_uses_matched_pool_for_new_ranking():
    sims = {rule: {'cohorts': [{'event_id': eid} for eid in ids]}
            for rule, ids in {'control3': ['old'], 'capacity6': ['old', 'extra'],
                              'matched3': ['matched'], 'residual3': ['ranked']}.items()}
    rows = driver.compare_cohorts(sims, 'official', 'stress', 0)
    assert rows[0]['shared'] == 1 and rows[0]['only_rule'] == ['extra']
    assert rows[2]['reference'] == 'matched3'
    assert rows[2]['only_reference'] == ['matched']
    assert rows[2]['only_rule'] == ['ranked']
