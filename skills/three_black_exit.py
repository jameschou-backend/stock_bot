"""Three completed bearish candles, each with a lower adjusted close, then T+1 exit."""
from copy import deepcopy
import math

import numpy as np
import pandas as pd

ARMS = ('control', 'three_black', 'benchmark')
NAMES = dict(control='中位數原版', three_black='三黑K且收盤逐日降低', benchmark='0050 含息再投入')


class ThreeBlackSignals:
    def __init__(self, adjusted_close, quotes, days):
        self.days = pd.DatetimeIndex(days)
        if (not self.days.is_unique or not self.days.is_monotonic_increasing
                or not adjusted_close.index.equals(self.days)
                or not adjusted_close.columns.is_unique
                or quotes.duplicated(['date', 'stock_id']).any()):
            raise ValueError('Three-black inputs require aligned unique market observations')
        self.adjusted = adjusted_close.copy()
        self.raw = {name: quotes.pivot(index='date', columns='stock_id', values=name)
                    .reindex(index=self.days, columns=adjusted_close.columns)
                    for name in ('open', 'high', 'low', 'close', 'volume')}
        opened, closed, volume = [self.raw[name] for name in ('open', 'close', 'volume')]
        high, low = self.raw['high'], self.raw['low']
        complete = (opened.gt(0) & closed.gt(0) & high.gt(0) & low.gt(0) & volume.gt(0))
        for frame in (opened, closed, high, low, volume):
            complete &= np.isfinite(frame)
        self.invalid_ohlc = complete & (opened.lt(low) | opened.gt(high) | closed.lt(low) | closed.gt(high))
        previous = self.adjusted.shift(1)
        valid = (opened.gt(0) & closed.gt(0) & volume.gt(0)
                 & self.adjusted.gt(0) & previous.gt(0))
        for frame in (opened, closed, volume, self.adjusted, previous):
            valid &= np.isfinite(frame)
        tolerance = np.maximum(1e-12, np.maximum(abs(self.adjusted), abs(previous))*1e-12)
        single = valid & closed.lt(opened) & self.adjusted.lt(previous-tolerance)
        self.trigger = single & single.shift(1, fill_value=False) & single.shift(2, fill_value=False)

    def exits(self, execution_index, entry_index, sid):
        if sid not in self.adjusted:
            raise ValueError('Missing three-black stock: '+sid)
        if execution_index < 4 or execution_index-entry_index < 3:
            return False
        if self.invalid_ohlc[sid].iloc[execution_index-3:execution_index].any():
            raise ValueError('Three-black signal uses impossible OHLC: '+sid+' '+str(self.days[execution_index-1].date()))
        return bool(self.trigger.at[self.days[execution_index-1], sid])


class ThreeBlackControl:
    def __init__(self, *args, drawdown_arm, black_signals, **kwargs):
        if drawdown_arm not in ('control', 'three_black'):
            raise ValueError('Unregistered three-black arm')
        self.drawdown_arm, self.black_signals = drawdown_arm, black_signals
        self.black_log = []
        super().__init__(*args, **kwargs)
        if not black_signals.days.equals(self.days):
            raise ValueError('Three-black signal and account calendars differ')

    def corporate_day(self, day):
        income = super().corporate_day(day)  # Original loss12/time63 has priority.
        if self.drawdown_arm == 'control':
            return income
        index = self.positions[day]
        signal = str(self.days[index-1].date())
        for sid, holding in self.holdings.items():
            eid = holding['event_id']
            state = self.exit_states.get(eid)
            if not holding['qty'] or not state or state['trigger_reason']:
                continue
            trigger = self.black_signals.exits(index, state['entry_index'], sid)
            self.black_log.append(dict(date=str(day.date()), signal_date=signal,
                                       event_id=eid, stock_id=sid, trigger=trigger))
            if trigger:
                state.update(trigger_reason='three_black', signal_date=signal,
                             target_date=str(day.date()), target_index=index)
                holding['due_index'] = index
                self._plan(day, sid, 'sell', eid, signal,
                           holding['qty']//1000*1000, 0., self.opening_limit, None)
        return income

    def run(self):
        result = super().run()
        if self.drawdown_arm == 'three_black':
            result['settings']['drawdown_arm'] = 'three_black'
            result['black_log'] = deepcopy(self.black_log)
        return result


def audit_three_black(account, signals):
    """Rebuild logged decisions directly from four closes and three raw candles."""
    days = signals.days
    positions = {str(day.date()): i for i, day in enumerate(days)}
    cohorts = {row['event_id']: row for row in account['cohorts']}
    seen, triggers = set(), {}
    for row in account.get('black_log', []):
        eid, sid = row['event_id'], row['stock_id']
        i, entry = positions[row['date']], positions[cohorts[eid]['entry_date']]
        if ((eid, i) in seen or eid in triggers or sid != cohorts[eid]['stock_id']
                or row['signal_date'] != str(days[i-1].date())):
            raise ValueError('Three-black decision identity, timing or latch differs')
        seen.add((eid, i))
        expected, bars = i >= 4 and i-entry >= 3, []
        if expected:
            for j in range(i-3, i):
                day = days[j]
                opened, closed, volume = [float(signals.raw[k].at[day, sid])
                                          for k in ('open', 'close', 'volume')]
                previous, current = [float(signals.adjusted.at[days[k], sid]) for k in (j-1, j)]
                high, low = [float(signals.raw[k].at[day, sid]) for k in ('high', 'low')]
                if all(math.isfinite(v) and v > 0 for v in (opened, closed, volume, high, low)):
                    if not low <= opened <= high or not low <= closed <= high:
                        raise ValueError('Three-black audit found impossible OHLC')
                valid = all(math.isfinite(v) and v > 0 for v in
                            (opened, closed, volume, previous, current))
                expected &= (valid and closed < opened and current < previous
                             and not math.isclose(current, previous, rel_tol=1e-12, abs_tol=1e-12))
                bars.append(dict(date=str(day.date()), open=opened, close=closed, volume=volume,
                                 previous_adjusted_close=previous, adjusted_close=current))
        if bool(expected) != row['trigger']:
            raise ValueError('Three-black decision differs from historical candles')
        if expected:
            triggers[eid] = dict(**row, bars=bars)
    for trade in account['trades']:
        if trade['reason'] != 'three_black':
            continue
        row = triggers.get(trade['event_id'])
        if (not row or trade['side'] != 'sell' or trade['signal_date'] != row['signal_date']
                or trade['stock_id'] != row['stock_id'] or trade['date'] < row['date']):
            raise ValueError('Three-black sale lacks a prior completed-candle decision')
    return dict(decisions_rebuilt=len(seen), exits=len(triggers), triggers=list(triggers.values()))
