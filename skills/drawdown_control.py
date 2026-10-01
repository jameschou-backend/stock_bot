"""Fixed weak-trend and loss-cluster controls on the sealed cash account."""
from copy import deepcopy
import math
import pandas as pd

ARMS = ('control', 'weak20', 'half15', 'cooldown', 'benchmark')
NAMES = dict(control='中位數原版', weak20='個股轉弱全出', half15='高點回落先減半',
             cooldown='近期兩次停損後冷卻', benchmark='0050 含息再投入')


class CoolingState:
    def __init__(self):
        self.first_stops = {}
        self.until = -1

    def observe(self, today, fills):
        for eid, index in fills:
            if not isinstance(index, int) or index >= today or index < 0:
                raise ValueError('Cooling requires an earlier executed stop')
            if eid in self.first_stops:
                if index < self.first_stops[eid]:
                    raise ValueError('Stop journal is not chronological')
                continue
            if self.first_stops and index < max(self.first_stops.values()):
                raise ValueError('Stop journal is not chronological')
            self.first_stops[eid] = index
            if sum(index-19 <= day <= index for day in self.first_stops.values()) >= 2:
                self.until = max(self.until, index+10)
        return today <= self.until


def weak_exit(index, entry_index, sid, signals):
    previous = index-1
    if index-entry_index < 2:
        return False
    values = [signals.adjusted_close[sid].iloc[j] for j in (previous-1, previous)]
    averages = [signals.ma20[sid].iloc[j] for j in (previous-1, previous)]
    relative = signals.relative20[sid].iloc[previous]
    return bool(all(pd.notna(v) and math.isfinite(v) for v in [*values, *averages, relative])
                and all(v < m and not math.isclose(v, m, rel_tol=1e-12, abs_tol=1e-12)
                        for v, m in zip(values, averages)) and relative < -1e-12)


class DrawdownControl:
    def __init__(self, *args, drawdown_arm, **kwargs):
        if drawdown_arm not in ('control', 'weak20', 'cooldown'):
            raise ValueError('Unregistered drawdown control')
        self.drawdown_arm = drawdown_arm
        self.cooling = CoolingState()
        self.trade_cursor = 0
        self.control_log, self.cooling_log = [], []
        super().__init__(*args, **kwargs)

    def corporate_day(self, day):
        index = self.positions[day]
        signal = str(self.days[index-1].date())
        if self.drawdown_arm == 'cooldown':
            fills = [(t['event_id'], self.positions[pd.Timestamp(t['date'])])
                     for t in self.trades[self.trade_cursor:]
                     if t['side'] == 'sell' and t['reason'] == 'loss12']
            self.trade_cursor = len(self.trades)
            blocked = self.cooling.observe(index, fills)
            events = self.events.get(day, [])
            self.cooling_log.append(dict(date=str(day.date()), signal_date=signal,
                blocked=blocked, until_index=self.cooling.until,
                blocked_events=[e['event_id'] for e in events] if blocked else []))
            if blocked:
                self.events[day] = []
        income = super().corporate_day(day)
        if self.drawdown_arm != 'weak20':
            return income
        for sid, holding in self.holdings.items():
            eid = holding['event_id']
            state = self.exit_states.get(eid)
            if not holding['qty'] or not state or state['trigger_reason']:
                continue
            trigger = weak_exit(index, state['entry_index'], sid, self.exit_signals)
            self.control_log.append(dict(date=str(day.date()), signal_date=signal,
                event_id=eid, stock_id=sid, trigger=trigger))
            if trigger:
                state.update(trigger_reason='weak_ma20', signal_date=signal,
                             target_date=str(day.date()), target_index=index)
                holding['due_index'] = index
                self._plan(day, sid, 'sell', eid, signal,
                           holding['qty']//1000*1000, 0., self.opening_limit, None)
        return income

    def run(self):
        result = super().run()
        if self.drawdown_arm != 'control':
            result['settings']['drawdown_arm'] = self.drawdown_arm
            result['control_log'] = deepcopy(self.control_log)
            result['cooling_log'] = deepcopy(self.cooling_log)
        return result


def audit_drawdown(account, signals, entries):
    """Reconstruct controls from prior prices / first stop fills, not engine state."""
    arm = account['settings'].get('drawdown_arm', 'control')
    days = signals.days
    positions = {str(day.date()): i for i, day in enumerate(days)}
    cohorts = {c['event_id']: c for c in account['cohorts']}
    triggers, seen = {}, set()
    for row in account.get('control_log', []):
        eid, sid = row['event_id'], row['stock_id']
        i, entry = positions[row['date']], positions[cohorts[eid]['entry_date']]
        if ((eid, i) in seen or eid in triggers or row['signal_date'] != str(days[i-1].date())
                or cohorts[eid]['stock_id'] != sid):
            raise ValueError('Weak control identity, timing or latch differs')
        seen.add((eid, i))
        c = signals.adjusted_close[sid]
        windows = [c.iloc[j-19:j+1] for j in (i-2, i-1)]
        p0, p1 = c.iloc[i-21], c.iloc[i-1]
        b = signals.adjusted_close['0050']
        valid = all(len(w)==20 and w.notna().all() for w in windows)
        valid &= all(pd.notna(v) and math.isfinite(v) and v > 0
                     for v in (p0, p1, b.iloc[i-21], b.iloc[i-1]))
        expected = bool(i-entry >= 2 and valid
                        and all(w.iloc[-1] < w.mean()-max(1e-12, abs(w.mean())*1e-12) for w in windows)
                        and p1/p0-b.iloc[i-1]/b.iloc[i-21] < -1e-12)
        if expected != row['trigger']:
            raise ValueError(f'Weak signal differs from historical windows: {eid} {row["date"]} '
                             f'engine={row["trigger"]} rebuilt={expected}')
        if expected:
            triggers[eid] = row
    for trade in account['trades']:
        if trade['reason'] == 'weak_ma20':
            row = triggers.get(trade['event_id'])
            if not row or trade['signal_date'] != row['signal_date'] or trade['date'] < row['date']:
                raise ValueError('Weak exit has no earlier latched decision')
    if arm == 'cooldown':
        first = {}
        for trade in account['trades']:
            if trade['side']=='sell' and trade['reason']=='loss12':
                first.setdefault(trade['event_id'], positions[trade['date']])
        stops = sorted(first.values())
        activations = [d for k, d in enumerate(stops) if k and stops[k-1] >= d-19]
        candidates = {}
        for e in entries:
            candidates.setdefault(e['entry_date'], []).append(e['event_id'])
        logs = account['cooling_log']
        if [r['date'] for r in logs] != [r['date'] for r in account['daily']]:
            raise ValueError('Cooling log does not cover every account day')
        for row in logs:
            i = positions[row['date']]
            until = max([d+10 for d in activations if d < i], default=-1)
            expected = i <= until
            if (row['blocked'] != expected or row['until_index'] != until
                    or row['signal_date'] != str(days[i-1].date())
                    or row['blocked_events'] != (candidates.get(row['date'], []) if expected else [])):
                raise ValueError('Cooling gate differs from prior execution history')
            if expected and any(t['date']==row['date'] and t['side']=='buy' for t in account['trades']):
                raise ValueError('Purchase executed during cooldown')
    return dict(weak_decisions_rebuilt=len(seen), weak_exits=len(triggers),
                cooling_days_rebuilt=len(account.get('cooling_log', [])))
