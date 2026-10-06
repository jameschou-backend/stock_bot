"""Fixed, causal daily SMC research assumptions, separate from live scanning.

No price zone identifies an institution. All dates denote close-confirmed market
sessions, not intraday execution. Outcomes never enter this module.
"""
from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd

# Only OHLC consistency accepts rounding at this relative scale (no tick-sized slack).
PRICE_VALIDATION_RTOL = 1e-12


def _price_ge(a, b):
    return a.ge(b) | (a-b).abs().le(PRICE_VALIDATION_RTOL*np.maximum(a.abs(), b.abs()))


STRATEGY_IDS = (
    'research_breakout20', 'smc_bull_break', 'smc_bos', 'smc_choch',
    'smc_sweep', 'fvg_form', 'fvg_retest', 'smc_orderblock_retest',
)


def compute_setups(f):
    """Return JSON-safe events and every FVG/order-block formation cohort.

    ``f`` is the aligned frame dictionary returned by scanner ``_prepare``.
    Setup status is an *as-of-last-session* observation, not a signal feature.
    Events are deduplicated by stock/date/strategy; ``setup_ids`` retains every
    corresponding cohort when several active zones retest on the same day.
    """
    required = ('c', 'h', 'l', 'open', 'close', 'volume', 'valid', 'eligible')
    if any(k not in f for k in required):
        raise ValueError('SMC requires prepared OHLC, volume, valid and eligible frames')
    c = f['c']
    if not isinstance(c, pd.DataFrame) or not isinstance(c.index, pd.DatetimeIndex):
        raise ValueError('SMC requires DataFrames indexed by market dates')
    days = c.index
    if (days.hasnans or days.tz is not None or days.has_duplicates
            or not days.is_monotonic_increasing or not days.equals(days.normalize())):
        raise ValueError('SMC requires sorted unique naive market dates')
    if c.columns.has_duplicates or any(not isinstance(x, str) or not re.fullmatch(r'\d{4}', x) for x in c.columns):
        raise ValueError('SMC requires unique four-digit stock identifiers')
    for k in required:
        if not isinstance(f[k], pd.DataFrame) or not f[k].index.equals(days) or not f[k].columns.equals(c.columns):
            raise ValueError('SMC frame coordinates differ: ' + k)
    h, l, raw, volume = (f[k] for k in ('h', 'l', 'close', 'volume'))
    adj_open = f['open'] * (c / raw)
    good = f['valid'].eq(True) & f['eligible'].eq(True) & volume.gt(0)
    for a in (c, h, l, adj_open, raw, volume):
        good &= np.isfinite(a) & a.gt(0)
    strict_ohlc = h.ge(c) & h.ge(adj_open) & l.le(c) & l.le(adj_open)
    consistent = _price_ge(h, c) & _price_ge(h, adj_open) & _price_ge(c, l) & _price_ge(adj_open, l)
    roundoff_count = int((good & ~strict_ohlc & consistent).to_numpy().sum())
    good &= consistent
    known = good.rolling(60, min_periods=60).sum().eq(60)
    turnover = (raw * volume).where(good).rolling(20, min_periods=20).mean()
    common = known & turnover.ge(50_000_000)
    masked_c = c.where(good)
    ma60 = masked_c.rolling(60, min_periods=60).mean()
    prior_c = masked_c.shift(1)
    tr = pd.DataFrame(np.maximum.reduce([
        (h-l).to_numpy(), (h-prior_c).abs().to_numpy(), (l-prior_c).abs().to_numpy(),
    ]), index=days, columns=c.columns).where(good)
    prior_atr = tr.rolling(14, min_periods=14).mean().shift(1)
    prior_high20 = h.where(good).rolling(20, min_periods=20).max().shift(1)
    prior_volume20 = volume.where(good).rolling(20, min_periods=20).mean().shift(1)
    # Candle colour is exactly invariant to adjustment; compare raw inputs so a
    # roundoff-sized difference cannot turn a doji into a red/black candle.
    red = raw.gt(f['open'])
    black = raw.lt(f['open'])
    baseline = common & red & c.gt(prior_high20) & volume.ge(prior_volume20*1.5)
    first_baseline = baseline & ~baseline.shift(1, fill_value=False) & known.shift(1, fill_value=False)
    date_strings = [str(x.date()) for x in days]
    events = {}
    setups = []
    counts = Counter(adjusted_ohlc_roundoff_tolerated=roundoff_count)
    matrices = {k: v.to_numpy() for k, v in dict(
        c=c, h=h, l=l, o=adj_open, good=good, common=common, red=red, black=black,
        ma60=ma60, atr=prior_atr, baseline=first_baseline,
    ).items()}

    def event(strategy_id, sid, i, setup_id, setup_date, **fields):
        key = (strategy_id, sid, i)
        if key in events:
            events[key]['setup_ids'].append(setup_id)
            counts['event_duplicates_merged'] += 1
            return
        events[key] = dict(strategy_id=strategy_id, stock_id=sid,
            signal_date=date_strings[i], setup_date=setup_date, setup_id=setup_id,
            setup_ids=[setup_id], **fields)

    for j, sid in enumerate(c.columns):
        if sid == '0050':
            continue  # Benchmark only; never an individual-stock setup.
        x = {k: v[:, j] for k, v in matrices.items()}
        high_level = low_level = None
        direction = 0
        active = []
        for i, date in enumerate(date_strings):
            if not x['good'][i]:
                for zone in active:
                    zone['status'] = 'data_missing'
                    zone['data_missing_date'] = date
                active = []
                high_level = low_level = None
                direction = 0
                counts['unavailable_stock_sessions'] += 1
                continue
            # Existing zones only: a formation candle can never retest itself.
            remaining = []
            for zone in active:
                age = i-zone['_formed_i']
                if x['c'][i] < zone['zone_lower']:
                    zone['status'] = 'invalidated'
                    zone['invalidation_date'] = date
                elif (age <= 10 and x['common'][i] and x['red'][i]
                      and x['l'][i] <= zone['zone_upper']
                      and x['h'][i] >= zone['zone_lower']
                      and x['c'][i] > zone['zone_upper']):
                    zone['status'] = 'retested'
                    zone['retest_date'] = date
                    zone['retest_sessions'] = age
                    strategy_id = 'fvg_retest' if zone['kind'] == 'fvg' else 'smc_orderblock_retest'
                    event(strategy_id, sid, i, zone['setup_id'], zone['setup_date'],
                        zone_lower=zone['zone_lower'], zone_upper=zone['zone_upper'],
                        structure_kind=zone.get('structure_kind'))
                elif age >= 10:
                    zone['status'] = 'expired'
                    zone['expiry_date'] = date
                else:
                    remaining.append(zone)
            active = remaining
            if x['baseline'][i]:
                event('research_breakout20', sid, i, f'breakout20:{sid}:{date}', date)

            def add_zone(kind, lower, upper, **extra):
                zone_id = f'{kind}:{sid}:{date}'
                zone = dict(kind=kind, stock_id=sid, setup_id=zone_id,
                    setup_date=date, status='pending', zone_lower=float(lower),
                    zone_upper=float(upper), retest_date=None, invalidation_date=None,
                    expiry_date=None, data_missing_date=None, retest_sessions=None,
                    _formed_i=i, **extra)
                setups.append(zone)
                active.append(zone)
                return zone

            # Ambiguous OHLC paths are explicitly excluded from structural signals.
            # Even a bullish close cannot reveal which side was taken first.
            ambiguous = (high_level is not None and low_level is not None
                and x['h'][i] > high_level['price'] and x['l'][i] < low_level['price'])
            if ambiguous:
                counts['ambiguous_structure_sessions'] += 1
                high_level = low_level = None
                direction = 0
            else:
                bull = (high_level is not None and not high_level['consumed'] and i > 0
                    and x['c'][i-1] <= high_level['price'] < x['c'][i])
                bear = (low_level is not None and not low_level['consumed'] and i > 0
                    and x['c'][i-1] >= low_level['price'] > x['c'][i])
                if bull:
                    structure_kind = {0:'initial', 1:'bos', -1:'choch'}[direction]
                    direction = 1
                    high_level['consumed'] = True
                    counts['bull_structure_' + structure_kind] += 1
                    if x['common'][i] and x['red'][i]:
                        setup_id = f'structure:{sid}:{high_level["pivot_date"]}:{date}'
                        fields = dict(structure_kind=structure_kind,
                            level=float(high_level['price']), pivot_date=high_level['pivot_date'],
                            confirmed_at=high_level['confirmed_at'])
                        event('smc_bull_break', sid, i, setup_id, date, **fields)
                        if structure_kind != 'initial':
                            event('smc_' + structure_kind, sid, i, setup_id, date, **fields)
                        for k in range(i-1, max(-1, i-11), -1):
                            if not x['good'][k]:
                                break
                            if x['black'][k]:
                                add_zone('orderblock', x['l'][k], x['h'][k],
                                    structure_kind=structure_kind, origin_date=date_strings[k],
                                    structural_setup_id=setup_id)
                                break
                elif bear:
                    direction = -1
                    low_level['consumed'] = True
                    counts['bear_structure_breaks'] += 1
                if (low_level is not None and not low_level['swept'] and not low_level['consumed']
                        and x['l'][i] < low_level['price'] < x['c'][i]
                        and x['common'][i] and x['red'][i]):
                    low_level['swept'] = True
                    event('smc_sweep', sid, i, f'sweep:{sid}:{low_level["pivot_date"]}', date,
                        structure_kind='low_sweep', level=float(low_level['price']),
                        pivot_date=low_level['pivot_date'], confirmed_at=low_level['confirmed_at'])

            if (i >= 2 and x['common'][i] and x['red'][i] and x['red'][i-1]
                    and x['l'][i] > x['h'][i-2]
                    and x['c'][i-1] > x['h'][i-2]
                    and x['l'][i]-x['h'][i-2] >= .1*x['atr'][i]
                    and x['c'][i] > x['ma60'][i]):
                zone = add_zone('fvg', x['h'][i-2], x['l'][i])
                event('fvg_form', sid, i, zone['setup_id'], date,
                    zone_lower=zone['zone_lower'], zone_upper=zone['zone_upper'])
            # Pivot at i-2 is learned at today's close and usable from i+1.
            if i >= 4 and x['good'][i-4:i+1].all():
                p = i-2
                neighbours = [i-4, i-3, i-1, i]
                if all(x['h'][p] > x['h'][n] for n in neighbours):
                    high_level = dict(price=x['h'][p], pivot_date=date_strings[p],
                        confirmed_at=date, consumed=False)
                if all(x['l'][p] < x['l'][n] for n in neighbours):
                    low_level = dict(price=x['l'][p], pivot_date=date_strings[p],
                        confirmed_at=date, consumed=False, swept=False)
    for zone in setups:
        zone.pop('_formed_i')
        counts[zone['kind'] + '_' + zone['status']] += 1
    result_events = sorted(events.values(), key=lambda r: (r['signal_date'], r['stock_id'], r['strategy_id']))
    by_strategy = Counter(e['strategy_id'] for e in result_events)
    return dict(events=result_events, setups=setups, definitions=dict(
        research_only=True, price_basis='adjusted daily OHLC; raw close times volume for turnover',
        adjusted_ohlc_validation_relative_tolerance=PRICE_VALIDATION_RTOL,
        candle_colour='raw close versus raw open; equal values remain doji',
        pivot='strict 2 left / 2 right; known at confirmation close; usable next market session',
        structural_break='close crosses latest unconsumed confirmed level; direction from prior break; initial is not BOS/CHoCH',
        ambiguity='daily high above prior swing high and daily low below prior swing low: skip structural signals and reset direction',
        common='60 consecutive valid eligible positive-volume sessions; current 20-session mean raw close times volume >= NTD 50 million',
        fvg='low[T] > high[T-2]; middle close > high[T-2], middle and T red; gap >= 0.1 * prior simple ATR14; T close > MA60',
        retest='sessions 1..10 after formation; overlap zone, red close above upper edge, common eligibility; invalidate close below lower edge first',
        orderblock='bull close-break: last black candle within previous 10 consecutive sessions, full low-high range; same retest rule',
        sweep='low below unconsumed confirmed swing low, red close above it; once per swing low',
        breakout20='red close above previous 20 highs and volume >= 1.5 prior 20-session average; previous known condition false',
        missing='invalid, ineligible, nonfinite or nontrading session resets pivots and direction; active zones become data_missing',
        expiry='retest allowed on session 10; otherwise expire at that session close; sample-end zones stay pending',
        cohort='all qualifying FVG and orderblock formations retained, even if no later retest; same-day entry duplicates merged without deleting cohorts',
        variants='fixed research assumptions, not canonical or exhaustive SMC; no intraday path or institutional ownership inferred',
        signal_timing='T close confirmed, entry no earlier than T+1; no outcome or future row used',
    ), counts=dict(counts, event_count=len(result_events), setup_count=len(setups),
        event_counts={k: by_strategy[k] for k in STRATEGY_IDS}))
