"""Frozen stagnation and stronger-candidate decisions, beginning in 2024 only."""
import math


def stagnant(age, return20, relative20):
    return bool(age >= 20 and all(math.isfinite(v) for v in (return20, relative20))
                and return20 < .03 and relative20 < 0)


def stronger(age, held_relative20, candidate_relative20):
    return bool(age >= 10 and all(math.isfinite(v) for v in (held_relative20, candidate_relative20))
                and held_relative20 <= 0 and candidate_relative20-held_relative20 >= .10)


class Rotation2024:
    def __init__(self, *args, rotation_arm, **kwargs):
        if rotation_arm not in ('original', 'relaxed', 'stagnant', 'stronger', 'combined'):
            raise ValueError('Unregistered rotation arm')
        self.rotation_arm = rotation_arm
        self.rotation_decisions = []
        super().__init__(*args, **kwargs)
        close = self.exit_signals.adjusted_close
        self.rotation_return20 = close/close.shift(20)-1

    def corporate_day(self, day):
        income = super().corporate_day(day)
        if day.year != 2024 or self.rotation_arm in ('original', 'relaxed'):
            return income
        i = self.positions[day]
        previous = self.days[i-1]
        candidates = [e for e in self.events.get(day, []) if e['members'][0] not in self.holdings]
        candidate = max(candidates, key=lambda e: (e['priority'], e['event_id'])) if candidates else None
        contexts = []
        for sid, holding in self.holdings.items():
            state = self.exit_states.get(holding['event_id'])
            if not state or not holding['qty'] or state['trigger_reason']:
                continue
            contexts.append(dict(stock_id=sid, event_id=holding['event_id'],
                age=i-state['entry_index'], return20=float(self.rotation_return20.at[previous, sid]),
                relative20=float(self.exit_signals.relative20.at[previous, sid])))
        exits = []
        if self.rotation_arm in ('stagnant', 'combined'):
            exits = [(r, 'stagnant20') for r in contexts if stagnant(r['age'], r['return20'], r['relative20'])]
        pending = any(self.exit_states.get(h['event_id'], {}).get('trigger_reason')
                      for sid, h in self.holdings.items() if sid in self.opening_members)
        if (not exits and not pending and self.rotation_arm in ('stronger', 'combined')
                and len(self.opening_members) >= self.slots and candidate):
            pool = [r for r in contexts if r['stock_id'] in self.opening_members
                    and stronger(r['age'], r['relative20'], candidate['priority'])]
            if pool:
                exits = [(min(pool, key=lambda r: (r['relative20'], r['stock_id'])), 'stronger_signal')]
        for context, reason in exits:
            sid, eid = context['stock_id'], context['event_id']
            state = self.exit_states[eid]
            state.update(trigger_reason=reason, signal_date=str(previous.date()),
                         target_date=str(day.date()), target_index=i)
            self.holdings[sid]['due_index'] = i
            self._plan(day, sid, 'sell', eid, str(previous.date()),
                       self.holdings[sid]['qty']//1000*1000, 0., self.opening_limit, None)
            self.rotation_decisions.append(dict(date=str(day.date()), signal_date=str(previous.date()),
                **context, reason=reason, candidate=candidate, opening_members=sorted(self.opening_members)))
        return income

    def run(self):
        account = super().run()
        account['rotation_decisions'] = self.rotation_decisions
        account['settings']['rotation_arm'] = self.rotation_arm
        return account
