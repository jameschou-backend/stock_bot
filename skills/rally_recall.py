"""Causal entry diagnostics; future labels are isolated from signal generation."""
import numpy as np
import pandas as pd

from skills.diffusion_signals import LOOKBACK, MIN_COMMON, MIN_TURNOVER
from skills.regime_state import build_trend

NAMED = ('2308', '2327', '2492', '3026', '2344', '2408', '1303', '2303', '3481', '3491', '6446')
HORIZON = 63


def signals(close, alternate, raw, volume, eligibility, companies):
    if (not close.index.is_unique or not close.index.is_monotonic_increasing
            or not close.columns.is_unique or '0050' not in close):
        raise ValueError('Unique ordered dates, symbols and benchmark required')
    for frame in (alternate, raw, volume, eligibility):
        if not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError('Research inputs must be aligned')
    if any(t != bool for t in eligibility.dtypes):
        raise ValueError('Historical eligibility must be explicit booleans')
    listed = companies.set_index('stock_id').listed_date
    if listed.index.has_duplicates or set(close.columns)-{'0050'}-set(listed.index):
        raise ValueError('Missing or duplicate company identity')
    valid = eligibility.copy()
    for sid in close.columns:
        if sid != '0050':
            if pd.isna(listed[sid]): raise ValueError('Missing listing date: '+sid)
            valid[sid] &= close.index >= pd.Timestamp(listed[sid])
    c, q, p, v = [f.where(valid & np.isfinite(f) & f.gt(0)) for f in (close, alternate, raw, volume)]
    ret, alt_ret = c/c.shift()-1, q/q.shift()-1
    common = ret['0050'].notna() & alt_ret['0050'].notna()
    enough = common.rolling(LOOKBACK).sum().ge(MIN_COMMON)
    incomplete = ((ret.isna() | alt_ret.isna()).mul(common, axis=0)).rolling(LOOKBACK).max().gt(0)
    anomalies = (ret.abs().gt(.20) | alt_ret.abs().gt(.20) | (ret-alt_ret).abs().gt(.005))
    anomaly_window = anomalies.rolling(LOOKBACK).max().gt(0)
    adv = (p*v).rolling(20).mean()
    seasoned = pd.DataFrame(False, index=c.index, columns=c.columns)
    for sid in c.columns:
        if sid == '0050': seasoned.iloc[LOOKBACK:, seasoned.columns.get_loc(sid)] = True
        else: seasoned.iloc[LOOKBACK:, seasoned.columns.get_loc(sid)] = c.index[:-LOOKBACK] >= pd.Timestamp(listed[sid])
    quality = (valid & seasoned & ~incomplete & ~anomaly_window & adv.ge(MIN_TURNOVER))
    quality = quality.mul(enough & ~anomaly_window['0050'], axis=0).astype(bool)
    quality['0050'] = False
    high60, ma20 = c.shift().rolling(60).max(), c.rolling(20).mean()
    r20, r1 = c/c.shift(20)-1, c/c.shift()-1
    excess20 = r20.sub(r20['0050'], axis=0)
    vr = v/v.shift().rolling(20).mean()
    trend = build_trend(c['0050'])
    on = trend['state'].eq('ON')
    base = quality.mul(on, axis=0).astype(bool)
    breakout = base & c.gt(high60) & r20.gt(0) & excess20.gt(0) & vr.ge(1.5)
    first = (base & r1.ge(.03) & vr.ge(1.5) & c.gt(ma20) & ma20.gt(ma20.shift(5))
             & excess20.gt(0) & (c.shift()/c.shift(11)-1).le(.10))
    return dict(eligible=base, quality=quality, breakout_unrestricted=breakout,
                first_expansion=first, volume_ratio=vr, relative20=excess20)


def sample(mask, start='2022-01-03', end='2026-09-08', cooldown=HORIZON):
    """Past-only refractory period; failed or unresolved labels do not reopen it."""
    rows = []
    for sid in mask.columns:
        next_index = 0
        for i in np.flatnonzero(mask[sid].to_numpy() & (mask.index >= start) & (mask.index <= end)):
            if i < next_index: continue
            rows.append((int(i), sid))
            next_index = i+cooldown+1
    return sorted(rows)


def labels(close, alternate, raw, raw_open, raw_quote_close, horizon=HORIZON):
    """T+1 opening PRICE PROXY, never a fill or portfolio return."""
    for f in (alternate, raw, raw_open, raw_quote_close):
        if not f.index.equals(close.index) or not f.columns.equals(close.columns):
            raise ValueError('Label inputs must be aligned')
    entry = (raw_open*close/raw).shift(-1)
    alt_entry = (raw_open*alternate/raw).shift(-1)
    result = close.shift(-(horizon+1))/entry-1
    other = alternate.shift(-(horizon+1))/alt_entry-1
    bad = (~np.isfinite(close) | ~np.isfinite(alternate) | close.le(0) | alternate.le(0)
           | (close/close.shift()-1).abs().gt(.20) | (alternate/alternate.shift()-1).abs().gt(.20))
    window_bad = bad.rolling(horizon+1).max().shift(-(horizon+1))
    opening_known = (np.isfinite(raw_open) & raw_open.gt(0) & raw.gt(0)
                     & np.isfinite(raw_quote_close) & (raw_quote_close-raw).abs().le(.011)).shift(-1, fill_value=False)
    known = window_bad.eq(0) & opening_known & (result-other).abs().le(.05) & np.isfinite(result)
    known = known.mul(known['0050'], axis=0).astype(bool)
    return result.where(known), known


def phase(day, end):
    return '2022_2024' if end <= '2024-12-31' else '2025_2026' if day >= '2025-01-01' else 'boundary'


def summarize(observations):
    rows = []
    for (arm, group, period), frame in observations.groupby(['arm', 'group', 'phase']):
        known = frame.dropna(subset=['forward_return', 'benchmark_return'])
        rows.append(dict(arm=arm, group=group, phase=period, signals=len(frame), known=len(known),
            unknown=len(frame)-len(known), stocks=frame.stock_id.nunique(),
            mean_return=known.forward_return.mean(), median_return=known.forward_return.median(),
            mean_excess=(known.forward_return-known.benchmark_return).mean(),
            mean_vs_same_date_eligible=(known.forward_return-known.eligible_mean).mean(),
            loss_fraction=known.forward_return.lt(0).mean(),
            surge_fraction=(known.forward_return.ge(.50) & (known.forward_return-known.benchmark_return).ge(.20)).mean()))
    return rows


def legacy_barrier(sid, day, monthly, events, entries, rejections, plans):
    """Explain original selector at a NEW signal date; never pretend it was an order."""
    group = next((g for g in monthly if g['month'] == day[:7]), None)
    if group is None: return 'month_unavailable'
    if sid not in group['selected_ids']:
        reasons = [k for k, ids in group['exclusions'].items() if sid in ids]
        return 'monthly_excluded:'+','.join(reasons) if reasons else 'outside_monthly_top300'
    cluster = next((c for c in group['clusters'] if sid in c['members']), None)
    if cluster is None: return 'cluster_size_or_variance'
    previous = [e for e in events if e['group_id']==cluster['group_id'] and e['leader_date']<day]
    if previous: return 'group_already_used:'+previous[0]['leader_id']
    rejection = next((r for r in group['leader_rejections'] if r['leader_id']==sid and r['date']==day), None)
    if rejection: return rejection['reason']
    e = next((e for e in entries if e['members']==[sid] and e['signal_date']==day), None)
    if e:
        p = next((p for p in plans if p['event_id']==e['event_id'] and p['side']=='buy'), None)
        return (p.get('rejection') or 'admitted_to_execution') if p else 'no_plan'
    if any(e['members']==[sid] and e['signal_date']==day for e in rejections): return 'market_trend_gate'
    competing = next((e for e in events if e['group_id']==cluster['group_id'] and e['leader_date']==day), None)
    return 'same_day_other_leader:'+competing['leader_id'] if competing else 'no_original_technical_signal'
