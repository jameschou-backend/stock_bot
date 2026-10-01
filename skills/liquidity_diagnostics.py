"""Past-only features for the fixed liquidity diagnostic; no trade execution."""
import numpy as np
import pandas as pd


def liquidity_features(raw_close, volume):
    if not raw_close.index.equals(volume.index) or not raw_close.columns.equals(volume.columns):
        raise ValueError('Price and volume axes must match')
    if not raw_close.index.is_unique or not raw_close.index.is_monotonic_increasing:
        raise ValueError('An ordered unique market calendar is required')
    valid = np.isfinite(raw_close) & np.isfinite(volume) & raw_close.gt(0) & volume.gt(0)
    amount = (raw_close * volume).where(valid)
    return dict(mean20=amount.rolling(20, min_periods=20).mean(),
                median20=amount.rolling(20, min_periods=20).median(),
                prior_mean20=amount.shift(1).rolling(20, min_periods=20).mean())


def market_breadth(adjusted_close, raw_close, volume, eligibility):
    for frame in (raw_close, volume, eligibility):
        if not frame.index.equals(adjusted_close.index) or not frame.columns.equals(adjusted_close.columns):
            raise ValueError('Breadth inputs must have identical axes')
    if eligibility.isna().any().any():
        raise ValueError('Unknown historical eligibility')
    c = adjusted_close.where(np.isfinite(adjusted_close) & adjusted_close.gt(0))
    mean = c.rolling(20, min_periods=20).mean()
    valid = (eligibility.astype(bool) & c.notna() & mean.notna()
             & np.isfinite(raw_close) & raw_close.gt(0) & np.isfinite(volume) & volume.gt(0))
    if '0050' in valid:
        valid['0050'] = False
    denominator = valid.sum(axis=1)
    fraction = (valid & c.gt(mean)).sum(axis=1) / denominator.replace(0, np.nan)
    return pd.DataFrame(dict(fraction=fraction, change5=fraction-fraction.shift(5),
                             eligible_count=denominator))
