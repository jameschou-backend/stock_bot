"""Causal price/volume features and explicitly future-labelled surge cohorts."""
import numpy as np
import pandas as pd


SINGLE_RULES = ['breakout60', 'volume_expansion', 'relative_strength', 'compression',
                'quiet_base', 'rising_trend', 'near_high']
RULES = ['all_eligible', *SINGLE_RULES, 'A_breakout_volume_strength',
         'B_breakout_volume_compression', 'C_strength_pullback']
NUMERIC_FEATURES = ['momentum20', 'momentum60', 'relative20', 'volume_ratio',
                    'volatility_ratio', 'distance_high60']


def _known_comparison(result, *inputs):
    """A missing feature is unknown, never an observed negative signal."""
    known = pd.DataFrame(True, index=result.index, columns=result.columns)
    for value in inputs:
        known &= np.isfinite(value)
    return result.astype('boolean').where(known, pd.NA)


def features(close, raw, volume, companies):
    """No shifts into the future, fill-forward, or outcome-based eligibility."""
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError('Expected a unique increasing market calendar')
    if '0050' not in close or not close.columns.is_unique:
        raise ValueError('Benchmark missing or duplicate asset columns')
    for frame in (raw, volume):
        if not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError('Price/volume matrices are not aligned')
    if companies.stock_id.duplicated().any():
        raise ValueError('Duplicate company identity')
    ids = sorted(set(companies.stock_id) & set(close.columns) - {'0050'})
    cohort = companies.set_index('stock_id').loc[ids]
    if cohort.listed_date.isna().any():
        raise ValueError('Listing dates missing from the fixed cohort')
    c = close[ids].where((close[ids] > 0) & np.isfinite(close[ids]))
    v, p = volume[ids].where(np.isfinite(volume[ids])), raw[ids].where(np.isfinite(raw[ids]))
    returns = c / c.shift(1) - 1
    mom20, mom60 = c / c.shift(20) - 1, c / c.shift(60) - 1
    benchmark = close['0050'].where((close['0050'] > 0) & np.isfinite(close['0050']))
    rs = mom20.sub(benchmark / benchmark.shift(20) - 1, axis=0)
    adv20 = (p * v).where((p > 0) & (v > 0)).rolling(20, min_periods=20).mean()
    traded_volume = v.where(v > 0)
    volume_ratio = traded_volume / traded_volume.shift(1).rolling(20, min_periods=20).mean()
    std60 = returns.rolling(60, min_periods=60).std()
    compression = returns.rolling(10, min_periods=10).std() / std60.where(std60 > 0)
    high60 = c.rolling(60, min_periods=60).max()
    ma60 = c.rolling(60, min_periods=60).mean()
    listed = pd.DataFrame(c.index.to_numpy()[:, None] >= pd.to_datetime(cohort.listed_date).to_numpy()[None, :],
                          index=c.index, columns=ids)
    complete = c.notna().rolling(120, min_periods=120).sum().eq(120)
    known_anomaly = returns.abs().gt(.15).rolling(120, min_periods=120).max().eq(1)
    base_eligible = listed & complete & adv20.ge(50e6)
    prior_high = c.shift(1).rolling(60, min_periods=60).max()
    masks = dict(
        breakout60=_known_comparison(c.gt(prior_high), c, prior_high),
        volume_expansion=_known_comparison(volume_ratio.ge(2), volume_ratio),
        relative_strength=_known_comparison(rs.ge(.10), rs),
        compression=_known_comparison(compression.le(.65), compression),
        quiet_base=_known_comparison(mom60.ge(-.20) & mom60.le(.10), mom60),
        rising_trend=_known_comparison(c.gt(ma60) & ma60.gt(ma60.shift(20)), c, ma60, ma60.shift(20)),
        near_high=_known_comparison((c / high60).ge(.95), c, high60),
    )
    masks['A_breakout_volume_strength'] = masks['breakout60'] & masks['volume_expansion'] & masks['relative_strength']
    masks['B_breakout_volume_compression'] = masks['breakout60'] & masks['volume_expansion'] & masks['compression']
    ret5 = c / c.shift(5) - 1
    masks['C_strength_pullback'] = (masks['relative_strength'] & masks['near_high'] &
                                    _known_comparison(ret5.ge(-.03) & ret5.le(.05), ret5))
    masks['all_eligible'] = pd.DataFrame(True, index=c.index, columns=ids)
    numeric = dict(momentum20=mom20, momentum60=mom60, relative20=rs, volume_ratio=volume_ratio,
                   volatility_ratio=compression, distance_high60=c / high60 - 1)
    return dict(rules=masks, numeric=numeric, eligible=base_eligible & ~known_anomaly,
                base_eligible=base_eligible, known_anomaly=known_anomaly, adv20=adv20, companies=cohort)


def cohort_table(close, quality, raw, volume, companies, *, start='2022-01-03', step=21, horizon=20):
    if step != 21 or horizon != 20:
        raise ValueError('Use the preregistered non-overlapping 21-session anchor and 20-return label')
    if not quality.index.equals(close.index) or not quality.columns.equals(close.columns):
        raise ValueError('Alternate price matrix is not aligned')
    computed = features(close, raw, volume, companies)
    positions = np.flatnonzero(close.index >= pd.Timestamp(start))[::step]
    rows, coverage = [], []
    for pos in positions:
        day = close.index[pos]
        ids = computed['eligible'].columns[computed['eligible'].iloc[pos]]
        missing_future = pos + horizon + 1 >= len(close)
        record = dict(signal_date=str(day.date()), eligible=int(len(ids)),
                      past_anomaly_excluded=int((computed['base_eligible'].iloc[pos] & computed['known_anomaly'].iloc[pos]).sum()),
                      label_mature=not missing_future)
        coverage.append(record)
        # Immature dates are inventoried, not counted as non-surges.
        if missing_future or not len(ids):
            continue
        entry, end = close.index[pos+1], close.index[pos+horizon+1]
        official = close.iloc[pos+1:pos+horizon+2][[*ids, '0050']]
        alternate = quality.iloc[pos+1:pos+horizon+2][[*ids, '0050']]
        result = official.iloc[-1] / official.iloc[0] - 1
        alternative_result = alternate.iloc[-1] / alternate.iloc[0] - 1
        incomplete = (~((official > 0) & np.isfinite(official))).any() | (~((alternate > 0) & np.isfinite(alternate))).any()
        anomaly = ((official / official.shift(1) - 1).abs() > .15).any() | ((alternate / alternate.shift(1) - 1).abs() > .15).any()
        disagreement = (result - alternative_result).abs() > .05
        known = ~(incomplete | anomaly | disagreement)
        known = known & bool(known['0050'])
        sample = computed['companies'].loc[ids, ['name', 'industry', 'market']].copy()
        sample['stock_id'] = sample.index
        sample['signal_date'], sample['entry_date'], sample['exit_date'] = str(day.date()), str(entry.date()), str(end.date())
        sample['phase'] = ('discovery' if end <= pd.Timestamp('2024-12-31') else
                           'replication' if day >= pd.Timestamp('2025-01-01') else 'boundary')
        sample['adv20'] = computed['adv20'].iloc[pos][ids]
        ranks = sample.adv20.rank(method='average', pct=True)
        sample['liquidity_bin'] = np.minimum((ranks * 5).astype(int), 4)
        sample['forward_return'] = result[ids].where(known[ids])
        sample['benchmark_return'] = result['0050'] if known['0050'] else np.nan
        sample['excess_return'] = (result[ids] - result['0050']).where(known[ids])
        sample['alternate_return'] = alternative_result[ids].where(known[ids])
        sample['label_known'] = known[ids]
        sample['event'] = ((result[ids] >= .30) & (sample.excess_return >= .20)).astype(object).where(known[ids], None)
        sample['label_reason'] = [','.join(key for key, flag in (
            ('missing_price', bool(incomplete[sid])), ('large_daily_move', bool(anomaly[sid])),
            ('price_version_disagreement', bool(disagreement[sid])), ('benchmark_unresolved', not bool(known['0050']))) if flag)
            for sid in ids]
        for key, frame in computed['rules'].items():
            sample[key] = frame.iloc[pos][ids]
        for key, frame in computed['numeric'].items():
            sample[key] = frame.iloc[pos][ids]
        rows.append(sample.reset_index(drop=True))
    if not rows:
        raise ValueError('No mature eligible observations')
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coverage)


def causal_check(close, raw, volume, companies, cutoffs):
    baseline = features(close, raw, volume, companies)
    checks = []
    for cutoff in cutoffs:
        day = pd.Timestamp(cutoff)
        if day not in close.index:
            raise ValueError('Causality cutoff is not a market session')
        for mode in ('truncate', 'mutate'):
            modified = []
            for frame in (close, raw, volume):
                copy = frame.loc[:day].copy() if mode == 'truncate' else frame.copy()
                if mode == 'mutate':
                    copy.loc[copy.index > day] *= 9.0
                modified.append(copy)
            actual = features(*modified, companies)
            for key in ('eligible', 'base_eligible', 'known_anomaly', 'adv20'):
                pd.testing.assert_frame_equal(baseline[key].loc[:day], actual[key].loc[:day])
            for group in ('rules', 'numeric'):
                for key in baseline[group]:
                    pd.testing.assert_frame_equal(baseline[group][key].loc[:day], actual[group][key].loc[:day])
            checks.append(dict(cutoff=cutoff, mode=mode, passed=True))
    return checks
