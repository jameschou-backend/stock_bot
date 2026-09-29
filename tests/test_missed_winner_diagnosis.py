import pytest
from scripts.diagnose_2024_missed_winners import group_block, monthly_block


def test_group_already_consumed_precedes_new_breadth_check():
    event = dict(leader_id='2330', leader_date='2024-05-02')
    assert group_block('2024-05-03', '6442', event, .9) == 'group_already_used'
    assert group_block('2024-05-02', '6442', event, .9) == 'already_broad'
    assert group_block('2024-05-02', '6442', event, .4) == 'lower_priority_same_day'
    assert group_block('2024-05-02', '2330', event, .4) == 'selected_leader'


def test_unexplained_candidate_is_not_silently_called_rejected():
    with pytest.raises(ValueError, match='missing'):
        group_block('2024-05-02', '6442', None, .2)


def test_liquidity_top300_and_cluster_size_are_separate():
    month = dict(exclusions={'turnover_below_50000000': ['1111']},
        selected_ids=['3333', '4444'], discarded_clusters=[{'members': ['3333']}],
        clusters=[dict(group_id='g1', members=['4444'])])
    assert monthly_block(month, '1111')[0] == 'monthly_quality_or_liquidity'
    assert monthly_block(month, '2222')[0] == 'outside_top300'
    assert monthly_block(month, '3333')[:2] == ('cluster_size', [1])
    assert monthly_block(month, '4444')[0] is None
