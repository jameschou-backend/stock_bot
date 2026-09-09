"""Calendar-aligned, explicitly lagged revenue research; no model or I/O."""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = {
    'price': '價格強勢',
    'covered': '價格＋營收齊全',
    'growth': '價格＋營收加速',
    'growth_trust': '價格＋營收加速＋投信',
    'revenue_first': '營收優先',
}


def revenue_features(rows, days, ids, lag_days):
    """Lag is an availability assumption, NOT an observed publication timestamp.

    Reindex entire monthly rows (including NaN) as of each day. Elementwise
    ffill would incorrectly keep old valid growth when a new month is missing.
    """
    if lag_days not in (45, 60):
        raise ValueError('Only preregistered 45/60-day availability assumptions are allowed')
    frame = rows[['stock_id', 'trading_date', 'revenue_current_month']].copy()
    frame['trading_date'] = pd.to_datetime(frame.trading_date)
    if (frame.empty or frame.duplicated(['stock_id', 'trading_date']).any()
            or frame.trading_date.isna().any() or not frame.trading_date.dt.is_month_start.all()):
        raise ValueError('Revenue dates must be unique provider month-start dates')
    if not frame.stock_id.str.fullmatch(r'[0-9]{4}').all():
        raise ValueError('Only four-digit stock identities are supported')
    monthly = frame.pivot(index='trading_date', columns='stock_id', values='revenue_current_month')
    calendar = pd.date_range(monthly.index.min(), days.max().to_period('M').start_time, freq='MS')
    monthly = monthly.reindex(index=calendar, columns=ids).astype(float)
    monthly = monthly.where(np.isfinite(monthly) & monthly.ge(0))
    trailing = monthly.rolling(3, min_periods=3).sum()
    prior = trailing.shift(12)
    growth = trailing / prior.where(prior > 0) - 1
    acceleration = growth - growth.shift(3)
    result = {}
    for name, values in [('growth', growth), ('acceleration', acceleration)]:
        values.index = calendar + pd.Timedelta(days=lag_days)
        result[name] = values.reindex(days, method='ffill')
    return result


def revenue_scores(fields, price_scores, trust_scores, features):
    close = fields['adj_close']
    growth, acceleration = features['growth'], features['acceleration']
    frames = (price_scores, trust_scores, growth, acceleration)
    if any(not x.index.equals(close.index) or not x.columns.equals(close.columns) for x in frames):
        raise ValueError('All research matrices must align')
    covered = growth.notna() & acceleration.notna()
    growing = growth.ge(.2) & acceleration.gt(0)
    turnover = fields['raw_close'] * fields['raw_volume']
    liquid = turnover.rolling(20, min_periods=20).mean().ge(50_000_000)
    trend = close.gt(close.rolling(120, min_periods=120).mean())
    result = {'price': price_scores.copy(), 'covered': price_scores.where(covered),
              'growth': price_scores.where(growing),
              'growth_trust': trust_scores.where(growing),
              'revenue_first': growth.where(growing & liquid & trend)}
    for values in result.values():
        # Explicit benchmark exception; never an ordinary-stock candidate.
        if '0050' in values:
            values.loc[:, '0050'] = np.nan
    return result
