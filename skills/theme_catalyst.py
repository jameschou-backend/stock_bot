"""Dated issuer evidence and causal price rules; no outcome or network inputs."""
from copy import deepcopy
import re

import numpy as np
import pandas as pd

START, END = '2025-04-16', '2026-09-09'
COHORT = ('2313', '2383', '2392', '2454', '3105', '6282', '6285')
ARMS = ('watchlist_breakout', 'confirmed_catalyst')


def validate_ledger(events):
    seen = set()
    for e in events:
        sid = e['stock_id']
        if not re.fullmatch(r'[1-9][0-9]{3}', sid) or sid not in (*COHORT, '3491'):
            raise ValueError('Issuer outside the preregistered ordinary-stock cohort')
        if e['event_id'] in seen or e['theme'] != 'leo':
            raise ValueError('Duplicate identity or unsupported theme')
        seen.add(e['event_id'])
        for field in ('source_date', 'observed_at'):
            value = e[field]
            if not isinstance(value, str) or pd.Timestamp(value).strftime('%Y-%m-%d') != value:
                raise ValueError('Evidence requires explicit ISO dates')
        if e['source_date'] > e['observed_at']:
            raise ValueError('Source date is after observation date')
        if type(e.get('signal_eligible')) is not bool:
            raise ValueError('Evidence eligibility must be explicitly reviewed')
        for field in ('material_exposure', 'realized_growth', 'order_or_production', 'negative'):
            if e.get(field) is not None and type(e[field]) is not bool:
                raise ValueError('Evidence flags must be bool or unknown')
        if not e['source_url'].startswith('https://') or not e.get('facts'):
            raise ValueError('Evidence needs a source and supporting facts')
    return True


def evidence_state(events, days, *, mode='historical_assumption', delay=0):
    """Unknown new disclosures supersede good old news; ineligible revisions do not.

    Document date alone never establishes historical first-publication proof.
    In observed mode only actual collection dates authorize the evidence.
    """
    validate_ledger(events)
    days = pd.DatetimeIndex(days)
    if not days.is_unique or not days.is_monotonic_increasing or days.tz is not None:
        raise ValueError('Ordered unique market dates required')
    if mode not in ('historical_assumption', 'observed') or type(delay) is not int or delay not in (0, 5):
        raise ValueError('Unknown evidence clock or preregistered delay')
    columns = sorted((*COHORT, '3491'))
    state = pd.DataFrame(index=days, columns=columns, dtype=object)
    evidence = pd.DataFrame(index=days, columns=columns, dtype=object)
    state[:] = None
    evidence[:] = None
    scheduled = []
    for e in sorted(events, key=lambda x:(x['source_date'], x['event_id'])):
        if not e['signal_eligible']:
            continue
        if len(days) == 0 or days[0] > pd.Timestamp(e['source_date']):
            raise ValueError('Calendar must cover document dates; truncated history cannot reset evidence age')
        date = e['source_date'] if mode == 'historical_assumption' else e['observed_at']
        position = int(days.searchsorted(pd.Timestamp(date) + pd.Timedelta(days=1))) + delay
        expiry = int(days.searchsorted(pd.Timestamp(e['source_date']) + pd.Timedelta(days=1))) + 126
        scheduled.append((position, expiry, e))
    # Same issuer/date: combine explicit positive evidence, adverse dominates.
    groups = {}
    for pos, expiry, e in scheduled:
        groups.setdefault((pos, e['stock_id']), []).append((expiry,e))
    by_stock = {}
    for (pos, sid), rows in sorted(groups.items()):
        # Bulk collection can make many old reports observable simultaneously.
        # Use the newest disclosure, not an OR over every historical good fact.
        newest = max(e['source_date'] for _,e in rows)
        expiry = min(end for end,e in rows if e['source_date'] == newest)
        rows = [e for _,e in rows if e['source_date'] == newest]
        prior = by_stock.get(sid)
        if prior is not None:
            old_pos, old_status, old_ids, old_source, old_expiry = prior
            if newest <= old_source:
                continue  # Late backfill must not roll state backward or renew it.
            end = min(pos, old_expiry, len(days))
            state.iloc[old_pos:end, state.columns.get_loc(sid)] = old_status
            evidence.iloc[old_pos:end, state.columns.get_loc(sid)] = old_ids
        adverse = any(e['negative'] is True for e in rows)
        positive = any(e[k] is True for e in rows for k in ('material_exposure', 'realized_growth', 'order_or_production'))
        status = False if adverse else True if positive else None
        by_stock[sid] = (pos, status, '|'.join(sorted(e['event_id'] for e in rows)), newest, expiry)
    for sid, (pos, status, identities, _, expiry) in by_stock.items():
        end = min(expiry, len(days))
        state.iloc[pos:end, state.columns.get_loc(sid)] = status
        evidence.iloc[pos:end, state.columns.get_loc(sid)] = identities
    return state, evidence


def build_signals(close, raw, volume, events, *, augmented=False, delay=0,
                  mode='historical_assumption', start=START, end=END):
    days = close.index
    pool = list(COHORT) + (['3491'] if augmented else [])
    if not set(pool + ['0050']).issubset(close.columns):
        raise ValueError('Missing cohort or benchmark price column')
    if (not days.equals(raw.index) or not days.equals(volume.index)
            or not close.columns.equals(raw.columns) or not close.columns.equals(volume.columns)):
        raise ValueError('Aligned prices and volumes required')
    close = close.where(np.isfinite(close) & close.gt(0))
    raw = raw.where(np.isfinite(raw) & raw.gt(0))
    volume = volume.where(np.isfinite(volume) & volume.ge(0))
    eligible, evidence = evidence_state(events, days, mode=mode, delay=delay)
    relative = (close / close.shift(20) - 1).sub(close['0050'] / close['0050'].shift(20) - 1, axis=0)
    adv = (raw * volume).rolling(20, min_periods=20).mean()
    shares = volume.rolling(20, min_periods=20).mean()
    full = close.notna().rolling(60, min_periods=60).sum().eq(60)
    breakout = (full & close.gt(close.rolling(60, min_periods=60).mean())
                & close.gt(close.shift(1).rolling(20, min_periods=20).max())
                & relative.gt(0) & adv.ge(50_000_000) & shares.gt(0))
    entries = {arm:[] for arm in ARMS}
    decisions = []
    last = {}
    for i, day in enumerate(days):
        if not pd.Timestamp(start) <= day <= pd.Timestamp(end):
            continue
        for sid in pool:
            membership = '2025-08-12' if sid == '3491' else '2025-04-15'
            member_date = membership if mode == 'historical_assumption' else '2026-09-27'
            member_index = int(days.searchsorted(pd.Timestamp(member_date) + pd.Timedelta(days=1))) + delay
            member = i >= member_index
            value = eligible.at[day, sid]
            status = value if type(value) is bool else None
            price_ok = bool(breakout.at[day, sid])
            cooling = i - last.get(sid, -9999) < 20
            signal = member and price_ok and not cooling
            can_schedule = i+1 < len(days) and days[i+1] <= pd.Timestamp(end)
            decisions.append(dict(stock_id=sid, signal_date=str(day.date()), member=member,
                price_confirmed=price_ok, cooling=cooling, evidence_status=status,
                evidence_id=evidence.at[day,sid], signal=signal, scheduled=signal and can_schedule))
            if not signal:
                continue
            last[sid] = i
            if not can_schedule:
                continue
            date = str(day.date())
            liquidity = dict(as_of=date, complete_20_sessions=True, observations=20,
                adv20_shares=float(shares.at[day,sid]), mean_turnover20_twd=float(adv.at[day,sid]))
            event = dict(event_id=f'leo-{date}-{sid}', members=[sid], stock_id=sid,
                signal_date=date, entry_date=str(days[i+1].date()), priority=float(adv.at[day,sid]),
                feature_cutoff_date=date, group_cutoff_date=date,
                membership_point_in_time=False, membership_snapshot_date='2026-09-27',
                liquidity_at_signal=liquidity, liquidity_before_entry=deepcopy(liquidity),
                relative20=float(relative.at[day,sid]), theme='leo', evidence_id=evidence.at[day,sid],
                evidence_status=status, source_clock=mode, document_delay=delay)
            entries['watchlist_breakout'].append(event)
            if status is True:
                entries['confirmed_catalyst'].append(deepcopy(event))
    for rows in entries.values():
        rows.sort(key=lambda e:(e['signal_date'], -e['priority'], e['stock_id']))
    return dict(entries_by_arm=entries, decisions=decisions, mode=mode, augmented=augmented, delay=delay,
                historical_first_publication_verified=False, future_labels_used=False)
