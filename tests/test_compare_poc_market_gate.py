from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.compare_poc_market_gate import candidate_arms, write_new
from skills.candidate_quality import candidate_features


def inputs():
    days = pd.bdate_range(end='2026-10-30', periods=260)
    close = pd.DataFrame({'0050': np.linspace(110, 100, len(days)),
                          '2330': np.linspace(100, 160, len(days)),
                          '2317': np.linspace(100, 110, len(days))}, index=days)
    volume = pd.DataFrame(1_000_000., index=days, columns=close.columns)
    volume.loc[days[-1], ['2330', '2317']] = 2_000_000.
    frames = {'close-official': close, 'close-quality': close.copy(), 'raw-close': close.copy(),
              'raw-volume': volume, 'eligibility': pd.DataFrame(True, index=days, columns=close.columns)}
    companies = pd.DataFrame([dict(stock_id=sid, listed_date=pd.Timestamp('2000-01-01')) for sid in ['2330', '2317']])
    identity = dict(coverage_start=str(days[0].date()), coverage_end='2026-10-30', identity_repair={},
                    episodes=[dict(stock_id=sid, start='2000-01-01', end=None, market='TWSE',
                                   category='股票', snapshot_date='2026-10-30') for sid in ['2330', '2317']],
                    trading_exclusions=[])
    return frames, companies, identity


def test_removing_off_gate_retains_original_inputs_and_relative_strength():
    frames, companies, identity = inputs()
    features = candidate_features(frames, companies)
    features['relative20'].loc[pd.Timestamp('2026-10-30'), '2317'] = -0.01
    before = deepcopy((frames, features, identity))
    baseline, variant, rejected = candidate_arms(frames, companies, identity, '2026-10-30', features=features)
    assert baseline == []
    assert [r['members'] for r in variant] == [['2330']]
    assert variant[0]['entry_date'] is None
    assert rejected == {'baseline': [], 'without_market_gate': []}
    for name in frames:
        pd.testing.assert_frame_equal(frames[name], before[0][name])
    for name in features:
        if isinstance(features[name], pd.DataFrame):
            pd.testing.assert_frame_equal(features[name], before[1][name])
        else:
            pd.testing.assert_series_equal(features[name], before[1][name])
    assert identity == before[2]


@pytest.mark.parametrize('condition', ['volume', 'breakout', 'liquidity', 'quality'])
def test_bypass_does_not_bypass_other_conditions(condition):
    frames, companies, identity = inputs()
    day = pd.Timestamp('2026-10-30')
    if condition == 'volume':
        frames['raw-volume'].at[day, '2330'] = 1_000_000.
    elif condition == 'breakout':
        frames['close-official'].at[day, '2330'] = 155.
        frames['close-quality'].at[day, '2330'] = 155.
    elif condition == 'liquidity':
        frames['raw-close'].loc[:, '2330'] = 1.
    features = candidate_features(frames, companies)
    if condition == 'quality':
        features['quality'].at[day, '2330'] = False
    _, variant, _ = candidate_arms(frames, companies, identity, '2026-10-30', features=features)
    assert '2330' not in {r['members'][0] for r in variant}


def test_on_gate_is_identical_unknown_gate_rejected_and_future_inputs_rejected():
    frames, companies, identity = inputs()
    features = candidate_features(frames, companies)
    features['trend'].iat[-1] = 'ON'
    baseline, variant, _ = candidate_arms(frames, companies, identity, '2026-10-30', features=features)
    assert baseline and baseline == variant
    features['trend'].iat[-1] = 'UNKNOWN'
    with pytest.raises(ValueError, match='known original market gate'):
        candidate_arms(frames, companies, identity, '2026-10-30', features=features)
    with pytest.raises(ValueError, match='stop at the signal date'):
        candidate_arms(frames, companies, identity, '2026-10-29')


def test_existing_experiment_file_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'comparison.json'
    write_new(path, {'original': True})
    with pytest.raises(FileExistsError):
        write_new(path, {'original': False})
    assert 'true' in path.read_text()
