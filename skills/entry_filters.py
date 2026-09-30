"""Three preregistered filters. Only signal-close observations are consumed."""
from copy import deepcopy
import numpy as np
import pandas as pd
from skills.candidate_quality import candidate_features

ARMS = ('control3', 'not_extended3', 'strong_close3', 'breadth3')


def filter_features(frames, companies, quotes):
    f = candidate_features(frames, companies)
    c = f['close']
    required = ['date', 'stock_id', 'open', 'high', 'low', 'close']
    if quotes.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate signal OHLC')
    q = quotes[required].copy()
    q['date'] = pd.to_datetime(q.date)
    ohlc = {k: q.pivot(index='date', columns='stock_id', values=k).reindex(index=c.index, columns=c.columns)
            for k in ('open', 'high', 'low', 'close')}
    o, h, l, raw = [ohlc[k] for k in ('open', 'high', 'low', 'close')]
    valid = (np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(raw)
             & o.gt(0) & l.gt(0) & h.gt(l) & raw.ge(l) & raw.le(h) & o.ge(l) & o.le(h)
             & (raw-frames['raw-close']).abs().le(.000001))
    location = ((raw-l)/(h-l)).where(valid)
    distance = c/f['ma20']-1
    r5 = c/c.shift(5)-1
    quality = f['quality'].copy()
    quality['0050'] = False
    count = quality.sum(axis=1)
    breadth = ((quality & c.gt(f['ma20'])).sum(axis=1)/count).where(count.ge(30))
    return dict(distance20=distance, return5=r5, close_location=location,
                green=(raw-o).where(valid), breadth=breadth, breadth5=breadth.shift(5),
                breadth_count=count)


def apply_filters(entries, features, cutoff=None):
    output = {k: [] for k in ARMS}
    rows = []
    for e in entries:
        if cutoff and e['signal_date'] > cutoff:
            continue
        sid, day = e['members'][0], pd.Timestamp(e['signal_date'])
        values = {k: float(features[k].at[day, sid]) for k in ('distance20', 'return5', 'close_location', 'green')}
        values.update({k: float(features[k].at[day]) for k in ('breadth', 'breadth5', 'breadth_count')})
        rules = dict(
            not_extended3=(('distance20', 'return5'), values['distance20'] <= .15 and values['return5'] <= .15),
            strong_close3=(('close_location', 'green'), values['close_location'] >= .70 and values['green'] >= 0),
            breadth3=(('breadth', 'breadth5'), values['breadth'] >= .50 and values['breadth'] >= values['breadth5']))
        decisions = dict(control3=True)
        reasons = dict(control3='control')
        for arm, (required, passed) in rules.items():
            known = all(np.isfinite(values[k]) for k in required)
            decisions[arm] = bool(known and passed)
            reasons[arm] = 'keep' if decisions[arm] else ('filter' if known else 'unknown')
        for arm, keep in decisions.items():
            if keep:
                output[arm].append(deepcopy(e))
        rows.append(dict(event_id=e['event_id'], stock_id=sid, signal_date=e['signal_date'],
                         **{k: v if np.isfinite(v) else None for k, v in values.items()},
                         decisions=decisions, reasons=reasons))
    return output, rows
