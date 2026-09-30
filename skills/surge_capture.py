"""Retrospective labels and causal gates. Labels never feed the selector."""
import numpy as np
import pandas as pd

from skills.candidate_quality import candidate_features

DEFINITIONS = ((20, .30), (60, .50), (120, 1.00))
STATUS = {0: 'ineligible_at_anchor', 1: 'immature', 2: 'missing_path_or_identity',
          3: 'price_review', 4: 'not_surge', 5: 'surge'}


def forward_sum(frame, horizon, include_anchor=True):
    """Inclusive [T,T+h], or strictly future [T+1,T+h]."""
    width = horizon + int(include_anchor)
    return frame.astype(float).rolling(width, min_periods=width).sum().shift(-horizon)


def labels(close, other, eligible, horizon, threshold):
    if horizon < 1 or not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError('Positive horizon and unique ordered calendar required')
    if any(not f.index.equals(close.index) or not f.columns.equals(close.columns)
           for f in (other, eligible)) or eligible.isna().any().any():
        raise ValueError('Label frames require identical axes and known eligibility')
    r = close.shift(-horizon)/close-1
    alt = other.shift(-horizon)/other-1
    excess = r.sub(r['0050'], axis=0)
    valid_price = close.gt(0) & other.gt(0) & np.isfinite(close) & np.isfinite(other)
    full_path = forward_sum(valid_price & eligible, horizon).eq(horizon+1)
    # Daily changes before the anchor do not affect a future label.
    a, b = close/close.shift(1)-1, other/other.shift(1)-1
    jumps = a.abs().gt(.20) | b.abs().gt(.20) | (a-b).abs().gt(.005)
    checked = forward_sum(jumps, horizon, False).eq(0) & (r-alt).abs().le(.02)
    benchmark_valid = (valid_price['0050'] & valid_price['0050'].shift(-horizon, fill_value=False)
                       & (r['0050']-alt['0050']).abs().le(.02))
    full_path = full_path.mul(benchmark_valid, axis=0).astype(bool)
    mature = np.arange(len(close))+horizon < len(close)
    status = np.full(close.shape, 2, dtype=np.uint8)
    status[full_path.to_numpy() & ~checked.to_numpy()] = 3
    known = full_path & checked
    positive = known & r.ge(threshold) & excess.ge(.20)
    status[known.to_numpy()] = 4
    status[positive.to_numpy()] = 5
    status[~mature] = 1
    status[~eligible.to_numpy(dtype=bool)] = 0
    return dict(status=pd.DataFrame(status, index=close.index, columns=close.columns),
                price_return=r, excess_return=excess)


def nonoverlap_starts(positive_indices, horizon):
    """Retain earliest window, skip every subsequent start <= its end."""
    out, last = [], -1
    for i in positive_indices:
        if i <= last:
            continue
        out.append(int(i))
        last = int(i)+horizon
    return out


def causal_gates(frames, companies):
    f = candidate_features(frames, companies)
    c = f['close']
    v = frames['raw-volume'].where(c.notna() & frames['raw-volume'].gt(0))
    previous_volume = v.shift(1).rolling(20, min_periods=20).mean()
    own = c/c.shift(20)-1
    gates = dict(quality=f['quality'],
                 breakout=c.gt(c.shift(1).rolling(60, min_periods=60).max()),
                 relative=own.gt(0) & f['relative20'].gt(0),
                 volume=v.ge(previous_volume*1.5))
    combined = gates['quality'] & gates['breakout'] & gates['relative'] & gates['volume']
    combined = combined.mul(f['trend'].eq('ON'), axis=0).astype(bool)
    combined['0050'] = False
    gates['signal'] = combined
    return dict(gates=gates, trend=f['trend'], relative20=f['relative20'],
                volume_ratio=v/previous_volume,
                adv20=(frames['raw-close']*frames['raw-volume']).rolling(20, min_periods=20).mean())


def window_gate_counts(gates, column, start, end):
    # End-day signals cannot execute inside the window: exclude the endpoint.
    stages, active = {}, np.ones(end-start, dtype=bool)
    for name in ('quality', 'breakout', 'relative', 'volume', 'signal'):
        active &= gates[name].iloc[start:end, column].to_numpy(dtype=bool)
        stages[name] = int(active.sum())
    reason = next((name for name, count in stages.items() if count == 0), None)
    return stages, reason


def label_statistics(status, returns):
    status, returns = np.asarray(status), np.asarray(returns, dtype=float)
    known = (status == 4) | (status == 5)
    n = int(known.sum())
    return dict(total=int(status.size), **{name: int((status == k).sum()) for k, name in STATUS.items()},
                known=n, surge_rate=float((status == 5).sum()/n) if n else None,
                negative_price_return=int((known & (returns < 0)).sum()),
                median_price_return=float(np.median(returns[known])) if n else None)
