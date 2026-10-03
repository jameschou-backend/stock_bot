"""Fixed signal-red gate and early volume exits; decisions use completed days only."""
from collections import Counter
from copy import deepcopy
import math

import numpy as np
import pandas as pd

MODES = ('none', 'dry', 'dry_weak')
REASONS = {'dry': 'volume_dry', 'dry_weak': 'volume_dry_weak'}


class EntryCandleUnavailable(ValueError):
    def __init__(self, message, decisions):
        super().__init__(message)
        self.decisions = deepcopy(decisions)


def _day(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tz is not None or value != value.normalize():
        raise ValueError('A timezone-naive market date is required')
    return value


def _number(value):
    return float(value) if not isinstance(value, (bool, np.bool_)) and pd.notna(value) and math.isfinite(float(value)) else None


def _ohlc_known(bar):
    return (all(bar[k] is not None and bar[k] > 0 for k in ('open', 'high', 'low', 'close'))
            and bar['low'] <= min(bar['open'], bar['close'])
            and max(bar['open'], bar['close']) <= bar['high'])


class CandleVolumeSignals:
    """T red candles and two adjacent post-entry volumes below 50% of T volume.

    ``action_dates`` contains (stock_id, effective market date), including cash
    dividends. No adjusted-price factor is used to transform share volume.
    """
    def __init__(self, black_signals, entries, action_dates):
        self.days = pd.DatetimeIndex(black_signals.days)
        if (not self.days.is_unique or not self.days.is_monotonic_increasing
                or self.days.tz is not None or not self.days.equals(self.days.normalize())):
            raise ValueError('Unique ordered naive market dates required')
        self.positions = {day: i for i, day in enumerate(self.days)}
        self.adjusted = black_signals.adjusted.copy(deep=True)
        self.raw = {k: black_signals.raw[k].copy(deep=True)
                    for k in ('open', 'high', 'low', 'close', 'volume')}
        for frame in [self.adjusted, *self.raw.values()]:
            if (not frame.index.equals(self.days) or not frame.columns.equals(self.adjusted.columns)
                    or not frame.columns.is_unique):
                raise ValueError('Candle/volume matrices must share unique axes')
        self.entries = {}
        for event in entries:
            self._validate_event(event)
            if event['event_id'] in self.entries:
                raise ValueError('Duplicate candidate event')
            self.entries[event['event_id']] = deepcopy(event)
        self.action_dates = {(str(sid), _day(day)) for sid, day in action_dates}

    def _validate_event(self, event):
        if len(event.get('members', [])) != 1 or not isinstance(event.get('event_id'), str):
            raise ValueError('Individual-stock event identity required')
        sid = event['members'][0]
        if sid not in self.adjusted:
            raise ValueError('Candidate is missing candle data: '+str(sid))
        signal, entry = _day(event['signal_date']), _day(event['entry_date'])
        if signal not in self.positions:
            raise ValueError('Signal is outside the market calendar')
        i = self.positions[signal]
        if i+1 >= len(self.days) or self.days[i+1] != entry:
            raise ValueError('Candidate must enter one market session after signal T')

    def _bar(self, index, sid):
        day = self.days[index]
        return dict(date=str(day.date()), **{k: _number(v.at[day, sid]) for k, v in self.raw.items()},
                    adjusted_close=_number(self.adjusted.at[day, sid]))

    def filter_entries(self, entries):
        kept, decisions, seen = [], [], set()
        for event in entries:
            self._validate_event(event)
            eid, sid = event['event_id'], event['members'][0]
            if eid in seen or eid not in self.entries or event != self.entries[eid]:
                raise ValueError('Entry gate candidate identity changed or duplicated')
            seen.add(eid)
            bar = self._bar(self.positions[_day(event['signal_date'])], sid)
            if not _ohlc_known(bar):
                decisions.append(dict(event_id=eid,stock_id=sid,signal_date=event['signal_date'],
                    entry_date=event['entry_date'],passed=None,status='unknown_or_invalid_ohlc',
                    signal_open=bar['open'],signal_close=bar['close']))
                raise EntryCandleUnavailable('Unknown or impossible signal-day OHLC: '+eid,decisions)
            passed = bar['close'] > bar['open']
            decisions.append(dict(event_id=eid, stock_id=sid, signal_date=event['signal_date'],
                entry_date=event['entry_date'], passed=passed,
                status='red' if passed else 'black' if bar['close'] < bar['open'] else 'doji',
                signal_open=bar['open'], signal_close=bar['close']))
            if passed:
                kept.append(deepcopy(event))
        return kept, decisions

    def evaluate(self, execution_index, entry_index, sid, signal_date, mode):
        if mode not in MODES:
            raise ValueError('Unregistered volume exit mode')
        if (type(execution_index) is not int or type(entry_index) is not int
                or not 0 <= entry_index < execution_index < len(self.days) or sid not in self.adjusted):
            raise ValueError('Volume exit requires an existing earlier filled entry')
        baseline = _day(signal_date)
        if baseline not in self.positions or self.positions[baseline] >= entry_index:
            raise ValueError('Volume baseline must precede the filled entry')
        j = execution_index-1
        result = dict(trigger=False, status='disabled', mode=mode,
            execution_date=str(self.days[execution_index].date()), decision_date=str(self.days[j].date()),
            baseline_date=str(baseline.date()), entry_date=str(self.days[entry_index].date()),
            baseline_volume=None, threshold=None, observations=[], corporate_dates=[], weak=None)
        if mode == 'none':
            return result
        if j > entry_index+4:
            return dict(result, status='outside_early_window')
        if j-1 < entry_index:
            return dict(result, status='insufficient_post_entry_sessions')
        actions = sorted(str(day.date()) for stock, day in self.action_dates
                         if stock == sid and baseline <= day <= self.days[j])
        if actions:
            return dict(result, status='corporate_action_window', corporate_dates=actions)
        base = self._bar(self.positions[baseline], sid)
        bars = [self._bar(k, sid) for k in (j-1, j)]
        result.update(baseline_volume=base['volume'], observations=bars)
        if not all(_ohlc_known(b) for b in [base, *bars]):
            return dict(result, status='unknown_or_invalid_ohlc')
        volumes = [b['volume'] for b in [base, *bars]]
        if any(v is None or v < 0 for v in volumes):
            return dict(result, status='unknown_or_invalid_volume')
        if any(v == 0 for v in volumes):
            return dict(result, status='nontrading_volume')
        result['threshold'] = base['volume']*.5
        dry = all(b['volume'] < result['threshold'] for b in bars)
        if mode == 'dry_weak':
            prices = [base['adjusted_close'], *[b['adjusted_close'] for b in bars]]
            if any(v is None or v <= 0 for v in prices):
                return dict(result, status='unknown_or_invalid_adjusted_close')
            result['weak'] = prices[2] < prices[1] and prices[2] < prices[0]
        return dict(result, status='evaluated', trigger=bool(dry and (mode == 'dry' or result['weak'])))


class CandleVolumeExit:
    """Run after original loss/time/three-black decisions; latch pending exits."""
    def __init__(self, *args, candle_volume_signals, volume_exit_mode='none', **kwargs):
        if volume_exit_mode not in MODES or not isinstance(candle_volume_signals, CandleVolumeSignals):
            raise ValueError('Explicit registered candle-volume signals/mode required')
        self.candle_volume_signals = candle_volume_signals
        self.volume_exit_mode = volume_exit_mode
        self.volume_exit_log = []
        super().__init__(*args, **kwargs)
        if not self.days.equals(candle_volume_signals.days):
            raise ValueError('Volume signal and account calendars differ')

    def corporate_day(self, day):
        income = super().corporate_day(day)
        if self.volume_exit_mode == 'none':
            return income
        i = self.positions[day]
        for sid, holding in self.holdings.items():
            eid = holding['event_id']
            state = self.exit_states.get(eid)
            if not holding['qty'] or not state or state['trigger_reason']:
                continue
            event = self.candle_volume_signals.entries.get(eid)
            if event is None or event['members'] != [sid]:
                raise ValueError('Held volume-exit event is not a bound candidate')
            result = self.candle_volume_signals.evaluate(i, state['entry_index'], sid,
                                                       event['signal_date'], self.volume_exit_mode)
            self.volume_exit_log.append(dict(event_id=eid, stock_id=sid, date=str(day.date()),
                signal_date=result['decision_date'], **result))
            if result['trigger']:
                state.update(trigger_reason=REASONS[self.volume_exit_mode], signal_date=result['decision_date'],
                             target_date=str(day.date()), target_index=i)
                holding['due_index'] = i
                self._plan(day, sid, 'sell', eid, result['decision_date'],
                           holding['qty']//1000*1000, 0., self.opening_limit, None)
        return income

    def run(self):
        result = super().run()
        if self.volume_exit_mode != 'none':
            result['volume_exit_log'] = deepcopy(self.volume_exit_log)
            result['settings']['volume_exit_mode'] = self.volume_exit_mode
        return result


def _funded_entries(account, signals):
    cohorts = {c['event_id']: c for c in account['cohorts']}
    if len(cohorts) != len(account['cohorts']):
        raise ValueError('Duplicate funded event identity')
    buy_dates = {}
    for trade in account['trades']:
        if trade['side'] == 'buy':
            if trade['qty'] <= 0 or trade['event_id'] not in cohorts:
                raise ValueError('Funded cohort requires positive fills')
            buy_dates.setdefault(trade['event_id'], []).append(trade['date'])
    result = {}
    for eid, cohort in cohorts.items():
        if eid not in buy_dates or min(buy_dates[eid]) != cohort['entry_date']:
            raise ValueError('Entry clock differs from first positive fill')
        event = signals.entries.get(eid)
        if (event is None or event['members'] != [cohort['stock_id']]
                or event['signal_date'] != cohort['signal_date']):
            raise ValueError('Funded event/source signal differs')
        result[eid] = (cohort, signals.positions[_day(cohort['entry_date'])])
    return result


def audit_entry_gate(account, signals):
    """Independently check only signal T, not entry-day candles or outcomes."""
    funded = _funded_entries(account, signals)
    for eid, (cohort, _) in funded.items():
        day, sid = _day(cohort['signal_date']), cohort['stock_id']
        values = [float(signals.raw[k].at[day, sid]) for k in ('open', 'high', 'low', 'close')]
        opened, high, low, close = values
        if (not all(math.isfinite(v) and v > 0 for v in values)
                or not low <= opened <= high or not low <= close <= high or not close > opened):
            raise ValueError('Bought event did not have a valid red signal-T candle: '+eid)
    return dict(funded_events_checked=len(funded),signal_red_required=True,uses_entry_day_close=False)


def audit_candle_volume(account, signals, mode):
    """Scalar oracle: never calls evaluate/filter_entries; verify first latched exits."""
    if mode not in MODES:
        raise ValueError('Unregistered volume audit mode')
    if mode == 'none':
        if account.get('volume_exit_log') or any(t.get('reason') in REASONS.values() for t in account['trades']):
            raise ValueError('Disabled volume exit changed the account')
        return dict(decisions_rebuilt=0, exits=0, mode=mode)
    funded = _funded_entries(account, signals)
    positions = {str(d.date()): i for i, d in enumerate(signals.days)}
    seen, latched, statuses = set(), {}, Counter()
    logs = account.get('volume_exit_log', [])
    # ThreeBlackControl logs precisely the positive-quantity, not-yet-latched
    # holdings reached by this later mixin; it also supplies an omission check.
    if 'black_log' in account:
        expected = {(r['event_id'], r['date']) for r in account['black_log'] if not r['trigger']}
        actual = {(r['event_id'], r['date']) for r in logs}
        if expected != actual:
            raise ValueError('Volume audit missing or extra post-priority decisions')
    earlier = {(r['event_id'], r['date']) for r in account.get('exit_decisions', []) if r['exit']}
    earlier |= {(r['event_id'], r['date']) for r in account.get('black_log', []) if r['trigger']}
    for row in logs:
        eid, sid, day = row['event_id'], row['stock_id'], row['date']
        if eid not in funded:
            raise ValueError('Volume decision lacks positive funded entry')
        cohort, entry = funded[eid]
        i = positions[day];j = i-1;baseline = _day(cohort['signal_date'])
        key = (eid, day)
        if key in seen or eid in latched or key in earlier or sid != cohort['stock_id'] or not entry < i:
            raise ValueError('Volume exit priority, first latch, or event identity differs')
        if seen and day < max(d for _, d in seen):
            raise ValueError('Volume decisions must retain chronological order')
        seen.add(key)
        base_index = positions[str(baseline.date())]
        if base_index >= entry:
            raise ValueError('Volume audit baseline is not before entry')
        rebuilt = dict(trigger=False,status='disabled',mode=mode,execution_date=day,
            decision_date=str(signals.days[j].date()),baseline_date=str(baseline.date()),
            entry_date=cohort['entry_date'],baseline_volume=None,threshold=None,
            observations=[],corporate_dates=[],weak=None)
        actions = sorted(str(d.date()) for stock, d in signals.action_dates
                         if stock == sid and baseline <= d <= signals.days[j])
        if j > entry+4:
            rebuilt['status'] = 'outside_early_window'
        elif j-1 < entry:
            rebuilt['status'] = 'insufficient_post_entry_sessions'
        elif actions:
            rebuilt.update(status='corporate_action_window',corporate_dates=actions)
        else:
            bars = []
            for index in (base_index,j-1,j):
                stamp = signals.days[index]
                values = {}
                for k in ('open','high','low','close','volume'):
                    raw = signals.raw[k].at[stamp,sid]
                    value = float(raw) if pd.notna(raw) else float('nan')
                    values[k] = value if math.isfinite(value) else None
                raw = signals.adjusted.at[stamp,sid]
                value = float(raw) if pd.notna(raw) else float('nan')
                bars.append(dict(date=str(stamp.date()),**values,
                                 adjusted_close=value if math.isfinite(value) else None))
            base, left, right = bars
            rebuilt.update(baseline_volume=base['volume'],observations=[left,right])
            known = all(all(b[k] is not None and b[k] > 0 for k in ('open','high','low','close'))
                        and b['low'] <= b['open'] <= b['high'] and b['low'] <= b['close'] <= b['high'] for b in bars)
            volumes = [b['volume'] for b in bars]
            if not known:
                rebuilt['status'] = 'unknown_or_invalid_ohlc'
            elif any(v is None or v < 0 for v in volumes):
                rebuilt['status'] = 'unknown_or_invalid_volume'
            elif any(v == 0 for v in volumes):
                rebuilt['status'] = 'nontrading_volume'
            else:
                rebuilt['threshold'] = base['volume']*.5
                prices = [b['adjusted_close'] for b in bars]
                if mode == 'dry_weak' and any(v is None or v <= 0 for v in prices):
                    rebuilt['status'] = 'unknown_or_invalid_adjusted_close'
                else:
                    rebuilt['status'] = 'evaluated'
                    if mode == 'dry_weak':
                        rebuilt['weak'] = prices[2] < prices[1] and prices[2] < prices[0]
                    rebuilt['trigger'] = bool(left['volume'] < base['volume']/2
                        and right['volume'] < base['volume']/2 and (mode == 'dry' or rebuilt['weak']))
        for field, value in rebuilt.items():
            if row.get(field) != value:
                raise ValueError('Volume scalar audit differs: '+field)
        if row['signal_date'] != rebuilt['decision_date']:
            raise ValueError('Volume decision timestamp differs')
        statuses[rebuilt['status']] += 1
        if rebuilt['trigger']:
            latched[eid] = dict(date=day,signal_date=rebuilt['decision_date'],stock_id=sid)
    for trade in account['trades']:
        if trade.get('reason') not in REASONS.values():
            continue
        first = latched.get(trade['event_id'])
        if (not first or trade['side'] != 'sell' or trade['reason'] != REASONS[mode]
                or trade['stock_id'] != first['stock_id'] or trade['signal_date'] != first['signal_date']
                or trade['date'] < first['date']):
            raise ValueError('Volume sale lacks its first prior-close latched decision')
    return dict(decisions_rebuilt=len(seen),exits=len(latched),mode=mode,
                statuses=dict(statuses),triggers=[dict(event_id=eid,**r) for eid,r in sorted(latched.items())])
