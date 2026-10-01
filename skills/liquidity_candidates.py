"""Fixed liquidity screens applied at the signal close, before cash replay."""
from copy import deepcopy

import numpy as np
import pandas as pd

from skills.liquidity_diagnostics import liquidity_features

ARMS = ('original', 'cap40', 'median50m', 'prior50m', 'persistent50m', 'benchmark')
FILTERS = ('median50m', 'prior50m', 'persistent50m')
NAMES = dict(original='原版', cap40='既有集中度控制', median50m='20 日中位數至少五千萬',
             prior50m='前 20 日均值至少五千萬', persistent50m='兩條件同時符合',
             benchmark='0050 股息再投入')


def filter_candidates(raw_close, volume, entries, cutoff):
    features = liquidity_features(raw_close, volume)
    days = raw_close.index
    result = {arm: [] for arm in ARMS if arm != 'benchmark'}
    decisions, seen = [], set()
    for event in entries:
        if event['signal_date'] > cutoff:
            continue
        if event['event_id'] in seen:
            raise ValueError('Duplicate candidate event')
        seen.add(event['event_id'])
        if len(event['members']) != 1:
            raise ValueError('Individual-stock candidate required')
        sid = event['members'][0]
        if len(sid) != 4 or not sid.isdigit() or sid.startswith('00'):
            raise ValueError('Only individual stocks are permitted')
        day = pd.Timestamp(event['signal_date'])
        i = days.get_loc(day)
        if i+1 >= len(days) or str(days[i+1].date()) != event['entry_date']:
            raise ValueError('Signal must precede entry by one market session')
        values = {key: float(frame.at[day, sid]) for key, frame in features.items()}
        known = {key: bool(np.isfinite(value)) for key, value in values.items()}
        if not known['mean20'] or values['mean20'] < 50_000_000:
            raise ValueError('Original mean liquidity threshold differs')
        passes = dict(median50m=known['median20'] and values['median20'] >= 50_000_000,
                      prior50m=known['prior_mean20'] and values['prior_mean20'] >= 50_000_000)
        passes['persistent50m'] = passes['median50m'] and passes['prior50m']
        decisions.append(dict(event_id=event['event_id'], stock_id=sid,
            signal_date=event['signal_date'], entry_date=event['entry_date'],
            values={k: v if known[k] else None for k, v in values.items()},
            unknown_features=[k for k in values if not known[k]], passes=passes))
        for arm in result:
            if arm in ('original', 'cap40') or passes[arm]:
                result[arm].append(deepcopy(event))
    return result, decisions
