import numpy as np
import pandas as pd
import pytest

from skills.liquidity_diagnostics import liquidity_features, market_breadth


def inputs():
    days = pd.bdate_range('2024-01-01', periods=40)
    c = pd.DataFrame({'1101': np.arange(40)+100., '1102': 100.-np.arange(40),
                      '0050': np.arange(40)+100.}, index=days)
    v = pd.DataFrame(1000., index=days, columns=c.columns)
    e = pd.DataFrame(True, index=days, columns=c.columns)
    return c, v, e


def test_single_spike_does_not_fake_persistent_liquidity():
    c, v, _ = inputs()
    c[:] = 1.
    v[:] = 10_000_000.
    v.iloc[20, 0] = 1_000_000_000.
    f = liquidity_features(c, v)
    assert f['mean20'].iloc[20, 0] > 50_000_000
    assert f['median20'].iloc[20, 0] == 10_000_000
    assert f['prior_mean20'].iloc[20, 0] == 10_000_000


def test_missing_volume_remains_unknown_and_calendar_not_compressed():
    c, v, _ = inputs()
    v.iloc[15, 0] = np.nan
    f = liquidity_features(c, v)
    assert np.isnan(f['median20'].iloc[34, 0])
    assert np.isfinite(f['median20'].iloc[35, 0])
    assert np.isnan(f['prior_mean20'].iloc[35, 0])


def test_breadth_excludes_benchmark_and_ineligible_stocks():
    c, v, e = inputs()
    f = market_breadth(c, c, v, e)
    assert f.iloc[-1].fraction == .5
    assert f.iloc[-1].eligible_count == 2
    e.iloc[-1, 1] = False
    assert market_breadth(c, c, v, e).iloc[-1].fraction == 1.


def test_future_mutation_and_truncation_do_not_change_past():
    c, v, e = inputs()
    cut = c.index[29]
    original = liquidity_features(c, v)
    breadth = market_breadth(c, c, v, e)
    mutated_c, mutated_v, mutated_e = c.copy(), v.copy(), e.copy()
    mutated_c.loc[mutated_c.index > cut] *= 50
    mutated_v.loc[mutated_v.index > cut] *= 100
    mutated_e.loc[mutated_e.index > cut] = False
    for candidate in (liquidity_features(c.loc[:cut], v.loc[:cut]),
                      liquidity_features(mutated_c, mutated_v)):
        for key in original:
            pd.testing.assert_frame_equal(original[key].loc[:cut], candidate[key].loc[:cut])
    for candidate in (market_breadth(c.loc[:cut], c.loc[:cut], v.loc[:cut], e.loc[:cut]),
                      market_breadth(mutated_c, mutated_c, mutated_v, mutated_e)):
        pd.testing.assert_frame_equal(breadth.loc[:cut], candidate.loc[:cut])
    with pytest.raises(ValueError, match='axes'):
        liquidity_features(c, v.iloc[:-1])
