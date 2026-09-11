"""Strict FinMind holder mapping. Ratios are fractions, never percentages.

The highest reported tier is >1,000 lots. It cannot isolate exactly 1,000 lots.
Availability date is a conservative seven-calendar-day lag followed by the next
calendar day; it is an assumption, not a verified historical release timestamp.
"""
from datetime import timedelta
import math
import re
import pandas as pd
from app.finmind import FinMindError

TIERS = ('1-999', '1000-5000', '5001-10000', '10001-15000',
         '15001-20000', '20001-30000', '30001-40000', '40001-50000',
         '50001-100000', '100001-200000', '200001-400000',
         '400001-600000', '600001-800000', '800001-1000000', '1000001+')


def aggregate(raw, allowed_stock_ids=None):
    if raw.empty:
        return pd.DataFrame()
    needed = {'date', 'stock_id', 'HoldingSharesLevel', 'people', 'unit', 'percent'}
    if not needed.issubset(raw):
        raise FinMindError('Holder schema missing: ' + ', '.join(sorted(needed-set(raw))))
    raw = raw.copy()
    raw['stock_id'] = raw.stock_id.astype(str)
    raw = raw[raw.stock_id.str.fullmatch(r'\d{4}')]
    if allowed_stock_ids is not None:
        raw = raw[raw.stock_id.isin(allowed_stock_ids)]
    records = []
    for (sid, day), group in raw.groupby(['stock_id', 'date']):
        levels, total, adjustment = {}, None, 0
        for row in group.to_dict('records'):
            label = str(row['HoldingSharesLevel']).replace(',', '').strip().lower()
            if label.startswith('差異數調整'):
                value = float(row['unit'])
                if not math.isfinite(value) or value != int(value):
                    raise FinMindError(f'Invalid holder adjustment: {sid} {day}')
                adjustment += int(value)
                continue
            if label in ('more than 1000001', 'over 1000001'):
                label = '1000001+'
            if label not in (*TIERS, 'total') or label in levels:
                raise FinMindError(f'Unknown/duplicate holder tier: {sid} {day} {label}')
            values = {k: float(row[k]) for k in ('unit', 'people', 'percent')}
            if (not all(math.isfinite(v) and v >= 0 for v in values.values())
                    or values['percent'] > 100 or any(values[k] != int(values[k]) for k in ('unit', 'people'))):
                raise FinMindError(f'Invalid holder value: {sid} {day}')
            levels[label] = values
        total = levels.pop('total', None)
        if set(levels) != set(TIERS) or total is None or total['unit'] <= 0:
            raise FinMindError(f'Incomplete holder tiers/total: {sid} {day}')
        units = total['unit']
        if (sum(r['people'] for r in levels.values()) != total['people']
                or sum(r['unit'] for r in levels.values()) + adjustment != units
                or any(abs(r['percent'] - r['unit']/units*100) > .011 for r in levels.values())):
            raise FinMindError(f'Holder totals do not reconcile: {sid} {day}')
        # Difference adjustments are not investors; denominator remains the
        # reported depository inventory, consistent with source percentages.
        tier_units = sum(r['unit'] for r in levels.values())
        if tier_units <= 0 or abs(adjustment)/units > .005:
            raise FinMindError(f'Excessive holder adjustment: {sid} {day}')
        large = levels['1000001+']['unit']/units
        observed = pd.Timestamp(day).date()
        records.append(dict(stock_id=sid, trading_date=observed,
            available_date=observed+timedelta(days=8), large_holder_pct=round(large, 4),
            small_holder_pct=round((tier_units-levels['1000001+']['unit'])/units, 4), top_level_pct=round(large, 4),
            holder_count=int(total['people'])))
    return pd.DataFrame(records)
