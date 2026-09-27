"""Frozen signal-time chip increments on the sealed sector-leader account."""
from copy import deepcopy

import numpy as np
import pandas as pd

from skills.residual_slot_replay import ResidualSlotReplay
from skills.theme_chips import align, changes

LAGS = {'main': (1, 8), 'delayed': (3, 15)}
ARMS = ('baseline', 'rank', 'filter', 'coverage')


def features(entries, days, flows, weekly, timing):
    """No returns or future prices are inputs; missing sessions cannot become zero."""
    flow_lag, chip_lag = LAGS[timing]
    days = pd.DatetimeIndex(days)
    if not days.is_unique or not days.is_monotonic_increasing:
        raise ValueError('Invalid market calendar')
    if flows.duplicated(['stock_id', 'date']).any():
        raise ValueError('Duplicate institutional stock/day')
    queries = pd.DataFrame([dict(event_id=e['event_id'], stock_id=e['members'][0],
        signal_date=e['signal_date']) for e in entries])
    if queries.empty or queries.event_id.duplicated().any():
        raise ValueError('Unique nonempty event cohort required')
    rows = align(queries, changes(weekly), chip_lag)
    by_stock = {sid: f.set_index('date') for sid, f in flows.groupby('stock_id')}
    result = {}
    for row in rows.to_dict('records'):
        sid, signal = row['stock_id'], pd.Timestamp(row['signal_date'])
        if signal not in days:
            raise ValueError('Signal absent from market calendar')
        last = days.get_loc(signal) - flow_lag
        window = days[last-4:last+1] if last >= 4 else days[:0]
        values = by_stock.get(sid, pd.DataFrame(columns=['foreign', 'trust'])).reindex(window)
        known_flow = len(window) == 5 and bool(np.isfinite(values[['foreign', 'trust']]).all().all())
        foreign = float(values.foreign.sum()) if known_flow else None
        trust = float(values.trust.sum()) if known_flow else None
        known_chip = bool(row['chip_known'])
        concentrated = bool(row['large_pct_delta4'] >= .005 and row['small_pct_delta4'] < 0
            and row['large_units_delta4'] > 0) if known_chip else None
        both = bool(foreign > 0 and trust > 0) if known_flow else None
        known = known_flow and known_chip
        result[row['event_id']] = dict(event_id=row['event_id'], stock_id=sid,
            signal_date=row['signal_date'], timing=timing, known=known,
            passed=bool(concentrated and both) if known else None,
            concentrated=concentrated, both_buy=both,
            foreign5=foreign, trust5=trust, flow_known=known_flow,
            flow_start=str(window[0].date()) if len(window) else None,
            flow_end=str(window[-1].date()) if len(window) else None,
            observed_date=row['observed_date'] if pd.notna(row['observed_date']) else None,
            available_date=row['available_date'] if pd.notna(row['available_date']) else None,
            large_pct_delta4=float(row['large_pct_delta4']) if known_chip else None,
            small_pct_delta4=float(row['small_pct_delta4']) if known_chip else None,
            large_units_delta4=float(row['large_units_delta4']) if known_chip else None,
            chip_known=known_chip, chip_reason=row['change_reason'],
            reason='ok' if known else ('unknown_flow' if not known_flow else 'unknown_holders'))
    return result


def choose(events, scores, arm):
    if arm not in ARMS:
        raise ValueError('Unregistered chip arm')
    if arm == 'baseline':
        return list(events)
    if arm == 'rank':
        return sorted(events, key=lambda e: scores[e['event_id']]['passed'] is not True)
    key = 'passed' if arm == 'filter' else 'known'
    return [e for e in events if scores[e['event_id']][key] is True]


class LeaderChipReplay(ResidualSlotReplay):
    def __init__(self, *args, chip_arm, chip_scores, **kwargs):
        if chip_arm not in ARMS:
            raise ValueError('Unregistered chip arm')
        super().__init__(*args, **kwargs)
        self.chip_arm = chip_arm
        self.chip_scores = deepcopy(chip_scores)
        self.chip_orders = []
        if chip_arm != 'baseline':
            for key, event in self.source_events.items():
                s = self.chip_scores[key]
                if (s['stock_id'] != event['members'][0] or s['signal_date'] != event['signal_date']
                        or (s['flow_end'] and s['flow_end'] >= s['signal_date'])
                        or (s['available_date'] and s['available_date'] > s['signal_date'])
                        or s['passed'] not in (True, False, None)
                        or (not s['known'] and s['passed'] is not None)):
                    raise ValueError('Chip score identity or availability mismatch')

    def corporate_day(self, day):
        income = super().corporate_day(day)
        original = self.events.get(day, [])
        selected = choose(original, self.chip_scores, self.chip_arm)
        if original:
            self.chip_orders.append(dict(date=str(day.date()),
                original=[e['event_id'] for e in original], selected=[e['event_id'] for e in selected]))
        self.events[day] = selected
        return income


def audit_order(journal, scores, arm):
    """Independently verify coverage and within-group order from the saved journal."""
    for row in journal:
        before, after = row['original'], row['selected']
        if len(set(before)) != len(before) or len(set(after)) != len(after):
            raise ValueError('Duplicate candidate order')
        if arm == 'baseline':
            expected = before
        elif arm == 'rank':
            expected = [e for e in before if scores[e]['passed'] is True]
            expected += [e for e in before if scores[e]['passed'] is not True]
        else:
            key = 'passed' if arm == 'filter' else 'known'
            expected = [e for e in before if scores[e][key] is True]
        if after != expected:
            raise ValueError('Chip ordering or coverage differs from frozen rule')
    return dict(passed=True, decision_days=len(journal))


def causality(entries, days, flows, weekly):
    """Future truncation AND mutation, including not-yet-available weekly rows."""
    checks = []
    for cutoff in ('2023-12-29', '2024-12-31', '2025-12-31'):
        subset = [e for e in entries if e['signal_date'] <= cutoff]
        if not subset:
            raise ValueError('Causality cutoff has no observations')
        for timing, (_, lag) in LAGS.items():
            expected = features(subset, days, flows, weekly, timing)
            future_week = pd.to_datetime(weekly.date) + pd.Timedelta(days=lag) > pd.Timestamp(cutoff)
            future_flow = pd.to_datetime(flows.date) >= pd.Timestamp(cutoff)
            for mode in ('truncate', 'mutate'):
                f, w = flows.copy(), weekly.copy()
                if mode == 'truncate':
                    f, w = f.loc[~future_flow], w.loc[~future_week]
                else:
                    f.loc[future_flow, ['foreign', 'trust']] = -1e12
                    w.loc[future_week, ['large_pct', 'small_pct', 'large_units']] = 1e12
                    w.loc[future_week, 'valid'] = False
                if features(subset, days, f, w, timing) != expected:
                    raise ValueError('Future source changed historical chip decisions')
                checks.append(dict(cutoff=cutoff, timing=timing, mode=mode, events=len(subset), passed=True))
    return dict(passed=True, checks=checks, count=len(checks))
