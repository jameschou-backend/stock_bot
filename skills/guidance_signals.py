"""Point-in-time joins and causal price confirmation for the fixed official-source pilot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import re

RULES = {'beat': '實績超過指引', 'beat_confirm': '實績超標＋價量確認',
         'guidance_up': '全年展望上修（時間診斷）', 'combined': '兩者同時成立（時間診斷）'}


def quarterly_comparison(row, previous):
    """A forecast is only comparable to the quarter it originally targeted."""
    required = ('actual_revenue_usd_billion', 'actual_gross_margin_pct')
    forecast = ('next_revenue_low', 'next_revenue_high', 'next_gm_low', 'next_gm_high')
    if (previous is None or row.get('availability_status') != 'dated_official_release'
            or previous.get('availability_status') != 'dated_official_release'):
        return {'status': 'missing_source', 'beat': False}
    if row.get('stock_id') != '2330' or previous.get('stock_id') != row['stock_id']:
        raise ValueError('Quarterly pilot requires the same company 2330')
    if previous.get('next_guidance_quarter') != row['reported_quarter']:
        return {'status': 'target_mismatch', 'beat': False}
    if pd.Timestamp(previous['meeting_date']) >= pd.Timestamp(row['meeting_date']):
        raise ValueError('Prior guidance must precede the actual release')
    values = [row.get(k) for k in required] + [previous.get(k) for k in forecast]
    if any(v is None or not np.isfinite(float(v)) for v in values):
        return {'status': 'missing_values', 'beat': False}
    revenue, margin, low, high, gm_low, gm_high = map(float, values)
    if low <= 0 or high < low or not 0 <= gm_low <= gm_high <= 100:
        raise ValueError('Invalid guidance units or bounds')
    return {'status': 'matched', 'beat': revenue > high and margin >= gm_low,
            'prior_event_id': previous['event_id'], 'target_quarter': row['reported_quarter'],
            'actual_usd_billion': revenue, 'prior_usd_low': low, 'prior_usd_high': high,
            'revenue_vs_upper': revenue / high - 1, 'actual_gm_pct': margin,
            'prior_gm_low': gm_low, 'prior_gm_high': gm_high,
            'prior_source_url': previous['source_url']}


def version_signal_index(source, days):
    """Conservative diagnostic assumption; it does not certify first publication."""
    meeting = pd.Timestamp(source['meeting_date'])
    index = int(days.searchsorted(meeting, side='right')) + 1
    for key in ('known_version_date', 'availability_not_before'):
        if source.get(key):
            index = max(index, int(days.searchsorted(pd.Timestamp(source[key]), side='left')))
    match = re.search(r'/reports/(\d{4}-\d{2})/', source.get('source_url', ''))
    if match and match[1] > meeting.strftime('%Y-%m'):
        assumed = pd.Period(match[1], freq='M').end_time.normalize()
        index = max(index, int(days.searchsorted(assumed, side='left')))
    return index


def build_signals(close, volume, quarterly, annual):
    """No future returns enter event labels; entries are always after signal close."""
    days = pd.DatetimeIndex(close.index)
    if not days.is_monotonic_increasing or not days.is_unique:
        raise ValueError('Price dates must be unique and ordered')
    expected = [f'2330-{y}Q{q}' for y, qs in [(2022, [4]), (2023, range(1,5)),
                                            (2024, range(1,5)), (2025, range(1,5))] for q in qs]
    if [r['event_id'] for r in quarterly] != expected:
        raise ValueError('Retain all 13 consecutive quarters, including missing sources')
    annual_by_id = {r['event_id']: r for r in annual}
    if set(annual_by_id) != set(expected) or len(annual) != len(expected):
        raise ValueError('Annual ledger must retain the same 13 events')
    entries = {key: [] for key in RULES}
    events = []
    avg_volume = volume.rolling(20, min_periods=20).mean().shift(1)
    for n, row in enumerate(quarterly):
        event = {'event_id': row['event_id'], 'meeting_date': row['meeting_date'],
                 'reported_quarter': row['reported_quarter'], 'source_url': row.get('source_url'),
                 'comparison': quarterly_comparison(row, quarterly[n-1] if n else None)}
        outlook = annual_by_id[row['event_id']]
        if outlook['meeting_date'] != row['meeting_date'] or outlook['stock_id'] != row['stock_id']:
            raise ValueError('Annual and quarterly sources disagree on event identity or meeting date')
        event['annual_direction'] = outlook['direction']
        event['annual_source_url'] = outlook.get('source_url')
        event['annual_wording'] = outlook.get('wording')
        event['annual_strict_eligible'] = False  # Current edited PDFs do not establish first-version history.
        event['confirmation_date'] = None
        event['entries'] = {}
        if n == 0:
            event['status'] = 'warmup'
            events.append(event)
            continue
        if not row.get('meeting_date'):
            event['status'] = 'missing_meeting_date'
            events.append(event)
            continue
        meeting = pd.Timestamp(row['meeting_date'])
        before = int(days.searchsorted(meeting, side='left')) - 1
        after = int(days.searchsorted(meeting, side='right'))
        if before < 0 or after >= len(days):
            event['status'] = 'outside_prices'
            events.append(event)
            continue
        event['status'] = 'evaluated'
        confirmed = None
        if event['comparison']['beat']:
            p0 = close.iloc[before][['2330', '0050']]
            for i in range(after, min(after+5, len(days))):
                now = close.iloc[i][['2330', '0050']]
                avg, vol = avg_volume.iloc[i], volume.iloc[i]
                if (np.isfinite(p0).all() and p0.gt(0).all()
                        and np.isfinite(now).all() and now.gt(0).all()
                        and np.isfinite(avg) and avg > 0 and np.isfinite(vol)):
                    relative = (now['2330']/p0['2330'])/(now['0050']/p0['0050'])-1
                    if now['2330'] > p0['2330'] and relative > 0 and vol >= avg * 1.2:
                        confirmed = i
                        event['confirmation_date'] = str(days[i].date())
                        event['confirmation_relative_return'] = float(relative)
                        event['confirmation_volume_ratio'] = float(vol/avg)
                        break
        annual_signal = None
        if outlook['direction'] == 'up':
            # Explicit same-target comparison is mandatory even for timing diagnostics.
            previous = annual_by_id.get(outlook.get('previous_event_id'))
            if previous is None or previous.get('target_year') != outlook.get('target_year'):
                raise ValueError('Annual revision must match the prior forecast target year')
            target = outlook.get('period_key', {})
            year = outlook['target_year']
            required_key = {'stock_id':'2330', 'target_start':f'{year}-01-01', 'target_end':f'{year}-12-31',
                            'metric':'company_total_revenue_yoy', 'currency':'USD', 'unit':'percent',
                            'comparison_basis':'same_calendar_year_vs_prior_calendar_year'}
            if target != required_key or previous.get('period_key') != target or previous.get('stock_id') != '2330':
                raise ValueError('Annual comparison must match company, period, currency, unit and metric')
            if pd.Timestamp(previous['meeting_date']) >= meeting:
                raise ValueError('Annual comparison uses a later or same-date source')
            annual_signal = max(version_signal_index(outlook, days), version_signal_index(previous, days))
            event['annual_signal_date_assumed'] = str(days[annual_signal].date()) if annual_signal < len(days) else None
        candidates = {'beat': after if event['comparison']['beat'] else None,
                      'beat_confirm': confirmed+1 if confirmed is not None else None,
                      'guidance_up': annual_signal+1 if annual_signal is not None else None,
                      'combined': max(annual_signal, confirmed)+1 if annual_signal is not None and confirmed is not None else None}
        for rule, entry in candidates.items():
            if entry is not None and entry < len(days):
                date = str(days[entry].date())
                entries[rule].append({'event_id': row['event_id'], 'stock_id': '2330', 'entry_date': date})
                event['entries'][rule] = date
        events.append(event)
    return entries, events
