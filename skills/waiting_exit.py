"""Entry-age patience experiments. The execution day's observations are unavailable."""
import math
import pandas as pd

ARMS = ('control3', 'stall53', 'stall103', 'stall5weak3')
MODES = ('control', 'stall5', 'stall10', 'stall5weak')


def waiting_context(close, ma20, sid, start, end):
    if not 0 <= start <= end < len(close):
        raise ValueError('Invalid entry/decision price window')
    p = close[sid].iloc[start:end+1]
    known = bool(p.notna().all() and p.gt(0).all() and p.map(math.isfinite).all())
    own = float(p.iloc[-1]/p.iloc[0]-1) if known else None
    peak = float(p.max()/p.iloc[0]-1) if known else None
    b0, b1, ma = float(close['0050'].iloc[start]), float(close['0050'].iloc[end]), float(ma20[sid].iloc[end])
    guard_known = known and all(math.isfinite(v) and v > 0 for v in (b0, b1, ma))
    return dict(observed_closes=end-start+1, known=known, entry_return=own,
                peak_return=peak, relative_entry=own-(b1/b0-1) if guard_known else None,
                below_ma20=bool(p.iloc[-1] < ma) if guard_known else None)


def waiting_choice(context, mode):
    if mode not in MODES:
        raise ValueError('Unregistered waiting policy')
    if mode == 'control':
        return False
    deadline = 10 if mode == 'stall10' else 5
    if context['observed_closes'] < deadline or not context['known']:
        return False
    stagnant = context['entry_return'] < .03-1e-12 and context['peak_return'] < .05-1e-12
    if mode == 'stall5weak':
        return bool(stagnant and context['below_ma20'] is True
                    and context['relative_entry'] is not None and context['relative_entry'] <= 0)
    return bool(stagnant)


class WaitingExit:
    def __init__(self, *args, waiting_mode, **kwargs):
        if waiting_mode not in MODES:
            raise ValueError('Unregistered waiting policy')
        self.waiting_mode = waiting_mode
        self.waiting_decisions = []
        super().__init__(*args, **kwargs)

    def corporate_day(self, day):
        # The original stop/deadline and any previously latched exit take priority.
        income = super().corporate_day(day)
        if self.waiting_mode == 'control':
            return income
        index = self.positions[day]
        for sid, h in self.holdings.items():
            state = self.exit_states.get(h['event_id'])
            if not state or not h['qty'] or state['trigger_reason']:
                continue
            context = waiting_context(self.exit_signals.adjusted_close, self.exit_signals.ma20,
                                      sid, state['entry_index'], index-1)
            trigger = waiting_choice(context, self.waiting_mode)
            signal = str(self.days[index-1].date())
            self.waiting_decisions.append(dict(date=str(day.date()), signal_date=signal,
                stock_id=sid, event_id=h['event_id'], trigger=trigger, **context))
            if trigger:
                state.update(trigger_reason=self.waiting_mode, signal_date=signal,
                             target_date=str(day.date()), target_index=index)
                h['due_index'] = index
                self._plan(day, sid, 'sell', h['event_id'], signal,
                           h['qty']//1000*1000, 0., self.opening_limit, None)
        return income

    def run(self):
        result = super().run()
        if self.waiting_mode != 'control':
            result['settings']['waiting_mode'] = self.waiting_mode
            result['waiting_decisions'] = self.waiting_decisions
        return result


def audit_waiting(account, signals):
    """Rebuild every recorded decision without the runtime decision helper."""
    mode = account['settings'].get('waiting_mode', 'control')
    cohorts = {r['event_id']: r for r in account['cohorts']}
    c, days = signals.adjusted_close, signals.days
    ma = c.rolling(20, min_periods=20).mean()
    seen, triggers = set(), {}
    for row in account.get('waiting_decisions', []):
        eid, sid = row['event_id'], row['stock_id']
        i = days.get_loc(pd.Timestamp(row['date']))
        j = days.get_loc(pd.Timestamp(cohorts[eid]['entry_date']))
        if (eid, i) in seen or eid in triggers or row['signal_date'] != str(days[i-1].date()):
            raise ValueError('Duplicate, unlatched or same-day waiting decision')
        seen.add((eid, i))
        prices = c[sid].iloc[j:i]
        known = bool(len(prices) and prices.notna().all() and prices.gt(0).all() and prices.map(math.isfinite).all())
        own = float(prices.iloc[-1]/prices.iloc[0]-1) if known else None
        peak = float(prices.max()/prices.iloc[0]-1) if known else None
        b0, b1, m = float(c['0050'].iloc[j]), float(c['0050'].iloc[i-1]), float(ma[sid].iloc[i-1])
        guard = known and all(math.isfinite(v) and v > 0 for v in (b0, b1, m))
        relative = own-(b1/b0-1) if guard else None
        below = bool(prices.iloc[-1] < m) if guard else None
        expected = dict(observed_closes=i-j, known=known, entry_return=own,
                        peak_return=peak, relative_entry=relative, below_ma20=below)
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError('Waiting context does not match historical prefix')
        deadline = 10 if mode == 'stall10' else 5
        trigger = bool(mode != 'control' and known and i-j >= deadline
                       and own < .03-1e-12 and peak < .05-1e-12)
        if mode == 'stall5weak':
            trigger = bool(trigger and below is True and relative is not None and relative <= 0)
        if trigger != row['trigger']:
            raise ValueError('Waiting decision differs from preregistration')
        if trigger:
            triggers[eid] = row
    for t in account['trades']:
        if t['reason'] not in MODES[1:]:
            continue
        row = triggers.get(t['event_id'])
        if not row or t['reason'] != mode or t['signal_date'] != row['signal_date'] or t['date'] < row['date']:
            raise ValueError('Waiting fill lacks an earlier decision')
    return dict(waiting_prefix_verified=True, waiting_context_count=len(seen), waiting_exits=len(triggers))
