"""Isolated, lagged slot-release policies; never renew or force a new buy."""
from copy import deepcopy
import math
import pandas as pd


def release_choices(contexts, candidate, opening_members, pending, slots, mode):
    if mode not in ('control', 'stagnant', 'stronger') or slots not in (3, 5):
        raise ValueError('Unregistered holding-release policy')
    valid = [r for r in contexts if all(math.isfinite(r[k]) for k in ('return20', 'relative20'))]
    if mode == 'stagnant':
        return [(r, 'stagnant20') for r in valid
                if r['age'] >= 20 and r['return20'] < .03 and r['relative20'] < 0]
    if mode != 'stronger' or pending or len(opening_members) < slots or candidate is None:
        return []
    if not math.isfinite(candidate['priority']):
        raise ValueError('Missing candidate strength')
    pool = [r for r in valid if r['stock_id'] in opening_members and r['age'] >= 10
            and r['relative20'] <= 0 and candidate['priority']-r['relative20'] >= .10]
    return [(min(pool, key=lambda r: (r['relative20'], r['stock_id'])), 'stronger_signal')] if pool else []


class HoldingRelease:
    def __init__(self, *args, release_mode, **kwargs):
        if release_mode not in ('control', 'stagnant', 'stronger'):
            raise ValueError('Unregistered holding-release mode')
        self.release_mode = release_mode
        self.release_decisions = []
        super().__init__(*args, **kwargs)
        c = self.exit_signals.adjusted_close
        self.release_return20 = c/c.shift(20)-1

    def corporate_day(self, day):
        income = super().corporate_day(day)
        if self.release_mode == 'control':
            return income
        index = self.positions[day]
        previous = self.days[index-1]
        # Preserve the frozen engine's actual ordering, including its tie break.
        candidates = [e for e in self.events.get(day, []) if e['members'][0] not in self.holdings]
        candidate = candidates[0] if candidates else None
        contexts = []
        for sid, holding in self.holdings.items():
            state = self.exit_states.get(holding['event_id'])
            if not state or not holding['qty'] or state['trigger_reason']:
                continue
            contexts.append(dict(stock_id=sid, event_id=holding['event_id'],
                age=index-state['entry_index'], return20=float(self.release_return20.at[previous, sid]),
                relative20=float(self.exit_signals.relative20.at[previous, sid])))
        pending = any(self.exit_states.get(h['event_id'], {}).get('trigger_reason')
                      for sid, h in self.holdings.items() if sid in self.opening_members)
        choices = release_choices(contexts, candidate, self.opening_members, pending, self.slots, self.release_mode)
        for context, reason in choices:
            sid, eid = context['stock_id'], context['event_id']
            state = self.exit_states[eid]
            state.update(trigger_reason=reason, signal_date=str(previous.date()),
                         target_date=str(day.date()), target_index=index)
            self.holdings[sid]['due_index'] = index
            self._plan(day, sid, 'sell', eid, str(previous.date()),
                       self.holdings[sid]['qty']//1000*1000, 0., self.opening_limit, None)
            self.release_decisions.append(dict(date=str(day.date()), signal_date=str(previous.date()),
                **context, reason=reason, candidate=deepcopy(candidate),
                opening_members=sorted(self.opening_members), pending=bool(pending),
                eligible_contexts=deepcopy([r for r in contexts if all(math.isfinite(r[k]) for k in ('return20', 'relative20'))])))
        return income

    def run(self):
        result = super().run()
        if self.release_mode != 'control':
            result['release_decisions'] = self.release_decisions
            result['settings']['release_mode'] = self.release_mode
        return result


def audit_release(account, signals, entries, snapshots):
    """Recompute recorded decision inputs independently of the policy helper."""
    mode = account['settings'].get('release_mode', 'control')
    days, c = signals.days, signals.adjusted_close
    returns = c/c.shift(20)-1
    cohorts = {r['event_id']: r for r in account['cohorts']}
    source = {r['event_id']: r for r in entries}
    openings = {r['date']: r['opening_active'] for r in snapshots}
    indexed = {}
    for row in account.get('release_decisions', []):
        if row['event_id'] in indexed:
            raise ValueError('Release instruction must remain latched')
        indexed[row['event_id']] = row
        day = pd.Timestamp(row['date']); i = days.get_loc(day); previous = days[i-1]
        if row['signal_date'] != str(previous.date()) or row['opening_members'] != openings[row['date']]:
            raise ValueError('Release date or opening membership differs')
        for ctx in [row, *row['eligible_contexts']]:
            sid = ctx['stock_id']; cohort = cohorts[ctx['event_id']]
            age = i-days.get_loc(pd.Timestamp(cohort['entry_date']))
            own = float(returns.at[previous, sid]); relative = own-float(returns.at[previous, '0050'])
            def equal(a, b): return a == b or (math.isnan(a) and math.isnan(b))
            if ctx['age'] != age or not equal(ctx['return20'], own) or not equal(ctx['relative20'], relative):
                raise ValueError('Release used different or future prices')
        if row['reason'] == 'stagnant20':
            valid = mode == 'stagnant' and row['age'] >= 20 and row['return20'] < .03 and row['relative20'] < 0
        elif row['reason'] == 'stronger_signal':
            candidate = row['candidate']; sid = candidate['members'][0]
            expected = source.get(candidate['event_id'])
            strength = float(returns.at[previous, sid]-returns.at[previous, '0050'])
            pool = [r for r in row['eligible_contexts'] if r['stock_id'] in row['opening_members']
                    and r['age'] >= 10 and r['relative20'] <= 0 and strength-r['relative20'] >= .10]
            valid = (mode == 'stronger' and candidate == expected and not row['pending']
                and candidate['signal_date'] == row['signal_date'] and candidate['entry_date'] == row['date']
                and abs(candidate['priority']-strength) < 1e-12
                and sid not in row['opening_members'] and len(row['opening_members']) >= account['settings']['slots']
                and pool and min(pool, key=lambda r: (r['relative20'], r['stock_id']))['event_id'] == row['event_id'])
        else:
            valid = False
        if not valid:
            raise ValueError('Release rule differs from preregistration')
    for trade in account['trades']:
        if trade['reason'] not in ('stagnant20', 'stronger_signal'):
            continue
        row = indexed.get(trade['event_id'])
        if not row or trade['reason'] != row['reason'] or trade['signal_date'] != row['signal_date'] or trade['date'] < row['date']:
            raise ValueError('Release fill lacks an earlier instruction')
    return dict(release_prior_inputs_rebuilt=True, release_instruction_count=len(indexed))
