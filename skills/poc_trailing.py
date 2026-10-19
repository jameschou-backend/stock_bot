"""Add one close-confirmed trailing exit after all original POC exit rules."""
from copy import deepcopy
import math

import pandas as pd

VARIANT = 'trail20_12'
REASON = 'poc_trailing12'


def trailing_decision(entry, current, peak):
    valid = all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                for x in (entry, current, peak))
    return bool(valid and peak / entry >= 1.20 - 1e-12
                and current / peak <= .88 + 1e-12)


class TrailingExit:
    def __init__(self, *args, **kwargs):
        self.trailing_records = []
        super().__init__(*args, **kwargs)

    def corporate_day(self, day):
        income = super().corporate_day(day)
        index = self.positions[day]
        signal = str(self.days[index - 1].date())
        # The original loop handles cash, resource locks and all prior exits.
        # In particular, no same-day sale proceeds or freed slots are reused.
        for sid, holding in self.holdings.items():
            eid = holding['event_id']
            state = self.exit_states.get(eid)
            if not holding['qty'] or state is None:
                continue
            before = deepcopy(state)
            signal_sid = state['stock_id']  # economic unit survives an exchange
            current = self.exit_signals.price(index - 1, signal_sid)
            triggered = (not state['trigger_reason'] and trailing_decision(
                state['entry_price'], current, state['peak_price']))
            if triggered:
                state.update(trigger_reason=REASON, signal_date=signal,
                             target_date=str(day.date()), target_index=index)
                holding['due_index'] = index
                self._plan(day, sid, 'sell', eid, signal, holding['qty']//1000*1000,
                           0., self.opening_limit, None)
            self.trailing_records.append(dict(date=str(day.date()), signal_date=signal,
                stock_id=sid, event_id=eid, signal_stock_id=signal_sid,
                current=current, before=before, after=deepcopy(state), triggered=bool(triggered)))
        return income

    def run(self):
        result = super().run()
        result['settings']['trailing_exit'] = dict(variant=VARIANT, arm_gain=.20,
            peak_drawdown=.12, close_only=True, execution_lag=1,
            original_exits_preserved=True, original_exits_first=True)
        result['trailing_records'] = self.trailing_records
        result['trailing_audit'] = audit_trailing(result, self.exit_signals.adjusted_close)
        return result


def audit_trailing(account, adjusted):
    """Rebuild every recorded peak from source closes, not the stored peak."""
    days = list(adjusted.index)
    positions = {str(d.date()): i for i, d in enumerate(days)}
    cohorts = {c['event_id']: c for c in account['cohorts']}
    seen, triggers = set(), {}
    for row in account['trailing_records']:
        eid, before, after = row['event_id'], row['before'], row['after']
        i = positions[row['date']]
        first = positions[cohorts[eid]['entry_date']]
        if ((eid, i) in seen or first >= i or before['entry_index'] != first
                or row['signal_date'] != str(days[i-1].date())):
            raise ValueError('Trailing decision identity or T+1 boundary differs')
        seen.add((eid, i))
        prices = adjusted[row['signal_stock_id']].iloc[first:i].tolist()
        valid = lambda x: isinstance(x, (int, float)) and math.isfinite(x) and x > 0
        entry = prices[0] if valid(prices[0]) else None
        current = prices[-1] if valid(prices[-1]) else None
        peak = max((x for x in prices if valid(x)), default=None)
        for value, recorded in ((entry, before['entry_price']), (peak, before['peak_price']),
                                (current, row['current'])):
            if ((value is None) != (recorded is None) or (value is not None
                    and not math.isclose(value, recorded, rel_tol=1e-11, abs_tol=1e-10))):
                raise ValueError('Trailing source close or lifetime peak differs')
        expected = bool(not before['trigger_reason'] and entry and current and peak
                        and peak >= entry * 1.2 - abs(entry)*1e-12
                        and current <= peak * .88 + abs(peak)*1e-12)
        wanted = deepcopy(before)
        if expected:
            wanted.update(trigger_reason=REASON, signal_date=row['signal_date'],
                          target_date=row['date'], target_index=i)
            triggers[eid] = row
        if row['triggered'] is not expected or after != wanted:
            raise ValueError('Trailing rule changed or overwrote an original exit')
        if before['trigger_reason'] == REASON:
            trigger = triggers.get(eid)
            if not trigger or before['target_date'] != trigger['date']:
                raise ValueError('Trailing pending exit was not kept latched')
    for trade in account['trades']:
        if trade['reason'] != REASON:
            continue
        trigger = triggers.get(trade['event_id'])
        if (not trigger or trade['side'] != 'sell' or trade['date'] < trigger['date']
                or trade['signal_date'] != trigger['signal_date']):
            raise ValueError('Trailing sale lacks an earlier close-confirmed trigger')
    return dict(passed=True, decisions_checked=len(seen), triggers=len(triggers),
                source_peaks_rebuilt=True, original_exits_preserved=True,
                first_trigger=min((r['date'] for r in triggers.values()), default=None))


def install(runner, variant, metadata):
    if variant != VARIANT:
        raise ValueError('Only preregistered trail20_12 may run')
    original = runner.engine_types
    def engines(*args, **kwargs):
        stock, benchmark = original(*args, **kwargs)
        return type('Trailing'+stock.__name__, (TrailingExit, stock), {}), benchmark
    runner.engine_types = engines
    writer = runner.write
    def write(path, value):
        if 'account' in value or str(path).endswith('/report.json'):
            value['risk_controls'] = metadata
        return writer(path, value)
    runner.write = write
