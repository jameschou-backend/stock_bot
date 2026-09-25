import pandas as pd
import pytest

from scripts.research_surge_sector import attach_groups, nullable_any, incremental_comparisons, RULES


def test_three_valued_or():
    assert nullable_any(pd.Series([False, pd.NA], dtype='boolean')) is pd.NA
    assert nullable_any(pd.Series([False, True, pd.NA], dtype='boolean')) is True
    assert nullable_any(pd.Series([False, False], dtype='boolean')) is False


def test_group_confirmation_cannot_be_assembled_from_different_groups():
    observation = pd.DataFrame([dict(signal_date='2026-05-11', stock_id='2492', relative_strength=True),
                                dict(signal_date='2026-05-11', stock_id='1101', relative_strength=True)])
    sectors = pd.DataFrame([
        dict(signal_date='2026-05-11', stock_id='2492', industry='a', breadth_confirmed=True,
             turnover_confirmed=False, sector_confirmed=False),
        dict(signal_date='2026-05-11', stock_id='2492', industry='b', breadth_confirmed=False,
             turnover_confirmed=True, sector_confirmed=False)])
    result = attach_groups(observation, sectors).set_index('stock_id')
    assert result.loc['2492', 'strength_with_breadth']
    assert result.loc['2492', 'strength_with_turnover']
    assert not result.loc['2492', 'strength_with_sector']
    assert result.loc['2492', 'common_peer_observation']
    assert not result.loc['1101', 'has_membership']
    assert not result.loc['1101', 'common_peer_observation']
    assert pd.isna(result.loc['1101', 'strength_with_sector'])
    with pytest.raises(ValueError, match='Duplicate'):
        attach_groups(observation, pd.concat([sectors, sectors.iloc[:1]]))


def test_unknown_group_is_not_a_failed_confirmation():
    observation = pd.DataFrame([dict(signal_date='2026-05-11', stock_id='2492', relative_strength=True)])
    sectors = pd.DataFrame([dict(signal_date='2026-05-11', stock_id='2492', industry='a',
                                breadth_confirmed=True, turnover_confirmed=pd.NA, sector_confirmed=pd.NA)])
    result = attach_groups(observation, sectors).iloc[0]
    assert not result.common_peer_observation
    assert pd.isna(result.strength_with_sector)


def test_incremental_gain_is_paired_and_undefined_is_not_zero():
    records = []
    for day in pd.date_range('2025-01-01', periods=6).strftime('%Y-%m-%d'):
        for event in (True, False):
            records.append(dict(phase='replication', signal_date=day, event=event,
                                relative_strength=True, **{key: event for key in RULES[-3:]}))
    data = pd.DataFrame(records)
    result = incremental_comparisons(data)
    row = next(r for r in result if r['phase'] == 'replication')
    assert row['paired_known_observations'] == 12
    assert row['precision_difference'] == .5
    assert row['difference_ci_low'] == row['difference_ci_high'] == .5
    assert next(r for r in result if r['phase'] == 'discovery')['precision_difference'] is None
    data.loc[0, 'relative_strength'] = False
    with pytest.raises(ValueError, match='subset'):
        incremental_comparisons(data)
