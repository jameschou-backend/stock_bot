"""Explicit provider-total and independent-adjusted enrichment of official OHLC.

These helpers do not fetch data, modify prices in place, or upgrade trading
qualification. A returned independent price is a revised research snapshot.
"""
from datetime import date
import math

from skills.market_input_validation import numeric, require
from skills.official_quote_repair import PRICE_FIELDS


def provider_rows(receipt, query):
    require(receipt.get('query') == query, 'Provider receipt query mismatch')
    require(set(query) == {'dataset','data_id','start_date','end_date'}
            and query['dataset'] in ('TaiwanStockPrice','TaiwanStockPriceAdj'), 'Unsupported provider query')
    require(date.fromisoformat(query['start_date']).isoformat() == query['start_date']
            and date.fromisoformat(query['end_date']).isoformat() == query['end_date']
            and query['start_date'] <= query['end_date'], 'Invalid provider date range')
    require(isinstance(receipt.get('data'), list), 'Provider receipt lacks rows')
    result = {}
    for row in receipt['data']:
        require(isinstance(row, dict) and row.get('stock_id') == query['data_id']
                and isinstance(row.get('date'), str)
                and date.fromisoformat(row['date']).isoformat() == row['date']
                and query['start_date'] <= row['date'] <= query['end_date'], 'Provider row outside requested scope')
        require(row['date'] not in result, 'Duplicate provider stock/date')
        result[row['date']] = row
    return result


def with_provider_total(official, provider, *, source_path):
    require(provider['stock_id'] == official['stock_id'] and provider['date'] == official['date'],
            'Provider total stock/date mismatch')
    names = dict(open='open', high='max', low='min', close='close')
    require(all(numeric(provider[names[k]]) == numeric(official[k]) for k in PRICE_FIELDS),
            'Provider raw OHLC disagrees with official')
    total = numeric(provider['Trading_Volume'], integral=True)
    ordinary = official['ordinary_session_volume']
    require(ordinary is None or total >= ordinary, 'Provider daily total below official ordinary volume')
    require(official['total_daily_volume'] is None or total == official['total_daily_volume'],
            'Provider daily total disagrees with official daily total')
    return dict(official, total_daily_volume=total, volume=total,
        total_volume_source_path=source_path, total_volume_evidence='provider_total_with_official_ohlc_match',
        # This resolves the local input field without claiming an independent
        # official verification of the sum of every trading session.
        total_daily_volume_verified=official['total_daily_volume_verified'],
        ready_for_raw_quote_insert=True)


def align_independent_close(series, existing, stamp, *, minimum_prefix_overlap=20,
                            allow_empty_existing=False):
    import numpy as np
    import pandas as pd
    day = pd.Timestamp(stamp)
    require(series.index.is_unique and existing.index.is_unique and day in series.index,
            'Independent adjusted stock/date missing or duplicated')
    require(pd.api.types.is_datetime64_any_dtype(series.index)
            and pd.api.types.is_datetime64_any_dtype(existing.index), 'Adjusted dates must be normalized')
    require(series.index.is_monotonic_increasing and existing.index.is_monotonic_increasing,
            'Adjusted date order differs')
    price = float(series.at[day])
    require(math.isfinite(price) and price > 0, 'Independent adjusted quote is not positive')
    valid_existing = np.isfinite(existing) & existing.gt(0)
    if not valid_existing.any():
        require(allow_empty_existing, 'No existing adjusted basis; full-series initialization required')
        return dict(value=price, scale=1.0, anchor_dates=[], overlap_count=0,
                    method='independent_full_series_initialization', source_is_revised_snapshot=True)
    common = series.reindex(existing.index)
    good = valid_existing & np.isfinite(common) & common.gt(0)
    before, after = existing.index[good & (existing.index < day)], existing.index[good & (existing.index > day)]
    if len(before) and len(after):
        anchors = [before[-1], after[0]]
        ratios = existing.loc[anchors] / common.loc[anchors]
        method = 'two_sided_nearest_scale_alignment'
    else:
        # A missing prefix has no prior frozen observation. It is still an
        # actual independent adjusted series, not a raw-close proxy. Demand
        # broad, constant-scale overlap with every available later observation.
        require(not len(before) and len(after) >= minimum_prefix_overlap
                and day < existing.index[valid_existing].min(),
                'Insufficient independent prefix overlap')
        anchors = list(existing.index[good])
        ratios = existing.loc[good] / common.loc[good]
        method = 'independent_prefix_all_overlap_scale_alignment'
    scale = float(ratios.mean())
    require(math.isfinite(scale) and scale > 0 and np.isfinite(ratios).all()
            and (abs(ratios / scale - 1) < 1e-6).all(), 'Independent adjusted basis conflicts')
    return dict(value=price * scale, scale=scale, anchor_dates=[str(d.date()) for d in anchors],
                overlap_count=len(anchors), method=method, source_is_revised_snapshot=True)


def ready_columns(row, adjustment=None):
    """Stable integration columns; nulls remain explicit and block required inputs."""
    return dict(row, total_volume=row['total_daily_volume'], ordinary_volume=row['ordinary_session_volume'],
        quality_adjusted_close=adjustment['value'] if adjustment else None,
        quality_alignment_method=adjustment['method'] if adjustment else 'independent_adjusted_missing',
        quality_adjusted_verified=bool(adjustment),
        ready_for_signal_rebuild=row['total_daily_volume'] is not None and bool(adjustment))
