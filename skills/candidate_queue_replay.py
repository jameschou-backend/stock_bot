"""Bounded next-session candidate retries above the sealed residual account."""
from collections import defaultdict
from copy import deepcopy
import math

import pandas as pd

from skills.residual_slot_replay import ResidualSlotReplay


class CandidateQueueReplay(ResidualSlotReplay):
    def __init__(self, *args, validity_sessions, **kwargs):
        if type(validity_sessions) is not int or validity_sessions not in (1, 2):
            raise ValueError('Only one or two candidate sessions are preregistered')
        super().__init__(*args, residual_policy='release', **kwargs)
        self.validity_sessions = validity_sessions
        self.queue_decisions = []
        if validity_sessions == 2:
            expanded = defaultdict(list)
            for day, events in self.events.items():
                for event in events:
                    for offset in range(2):
                        index = self.positions[day] + offset
                        if index >= len(self.days) or self.days[index] > self.end:
                            continue
                        attempt = deepcopy(event)
                        attempt.update(scheduled_entry_date=event['entry_date'],
                            entry_date=str(self.days[index].date()), attempt_number=offset+1)
                        expanded[self.days[index]].append(attempt)
            self.events = expanded

    def corporate_day(self, day):
        records, eligible = [], []
        for event in self.events.get(day, []):
            identity, sid = event['event_id'], event['members'][0]
            attempt = event.get('attempt_number', 1)
            action = attempt > 1 and self._action_between(sid, pd.Timestamp(event['signal_date']), day)
            if identity in self.finished:
                reason = 'already_filled'
            elif action:
                reason = 'corporate_action_since_signal'
                self.finished.add(identity)
            else:
                reason = 'eligible'
                eligible.append(event)
            amount = float(self.amount20.at[day, sid])
            records.append(dict(date=str(day.date()), event_id=identity, stock_id=sid,
                signal_date=event['signal_date'], attempt_number=attempt,
                scheduled_entry_date=event.get('scheduled_entry_date', event['entry_date']),
                reason=reason, reference_date=str(self.days[self.positions[day]-1].date()),
                prior_amount20=amount if math.isfinite(amount) else None, rank=None))
        self.events[day] = eligible
        income = super().corporate_day(day)
        # Parent applies the unchanged capacity ranking from prior quotes.
        ranks = {e['event_id']: i+1 for i, e in enumerate(self.events.get(day, []))}
        for record in records:
            record['rank'] = ranks.get(record['event_id'])
        self.queue_decisions.extend(sorted(records, key=lambda r: r['event_id']))
        return income

    def run(self):
        account = super().run()
        if self.validity_sessions == 2:
            account['settings']['candidate_validity_sessions'] = 2
        return account


def audit_queue(account, decisions, entries, calendar, quotes, action_dates, validity, mask):
    """Rebuild all scheduled decisions and their prior-price ranks from inputs."""
    dates = [r['date'] for r in account['daily']]
    days = pd.DatetimeIndex(calendar)
    positions = {str(d.date()): i for i, d in enumerate(days)}
    q = quotes.copy()
    q['date'] = pd.to_datetime(q['date'])
    close = q.pivot(index='date', columns='stock_id', values='close').reindex(days)
    volume = q.pivot(index='date', columns='stock_id', values='volume').reindex(days)
    adv = (close*volume).rolling(20, min_periods=20).mean().shift(1)
    fills = defaultdict(list)
    for trade in account['trades']:
        if trade['side'] == 'buy':
            fills[trade['event_id']].append(trade)
    scheduled, expected, outcomes = defaultdict(list), [], []
    actions = {(str(s), str(pd.Timestamp(d).date())) for s, d in action_dates}
    for event in entries:
        identity, sid = event['event_id'], event['members'][0]
        first = positions[event['entry_date']] + bool(mask & 2)
        window = [str(days[i].date()) for i in range(first, min(first+validity, len(days)))
                  if dates[0] <= str(days[i].date()) <= dates[-1]]
        fill_days = sorted({t['date'] for t in fills[identity]})
        if len(fill_days) > 1 or any(d not in window for d in fill_days):
            raise ValueError('Candidate filled repeatedly or outside its validity window')
        cancelled = False
        for attempt, day in enumerate(window, 1):
            if fill_days and fill_days[0] < day:
                reason = 'already_filled'
            elif attempt > 1 and any(s == sid and event['signal_date'] < d <= day for s, d in actions):
                reason = 'corporate_action_since_signal'
                cancelled = True
            else:
                reason = 'eligible'
            if reason != 'eligible' and day in fill_days:
                raise ValueError('Cancelled or completed candidate filled again')
            amount = float(adv.at[pd.Timestamp(day), sid])
            scheduled[day].append(dict(date=day, event_id=identity, stock_id=sid,
                signal_date=event['signal_date'], attempt_number=attempt,
                scheduled_entry_date=str(days[first].date()), reason=reason,
                reference_date=str(days[positions[day]-1].date()),
                prior_amount20=amount if math.isfinite(amount) else None, rank=None))
        outcomes.append(dict(event_id=identity, stock_id=sid, eligible_window=window,
            fill_date=fill_days[0] if fill_days else None,
            outcome=('filled' if fill_days else 'cancelled' if cancelled else
                     'sample_end_truncated' if len(window) < validity else 'expired_unfilled')))
    for day in dates:
        rows = scheduled[day]
        eligible = sorted((r for r in rows if r['reason']=='eligible'), key=lambda r:
            (-(r['prior_amount20'] if r['prior_amount20'] is not None else -math.inf), r['stock_id'], r['event_id']))
        for rank, row in enumerate(eligible, 1):
            row['rank'] = rank
        expected.extend(sorted(rows, key=lambda r:r['event_id']))
    if expected != decisions:
        raise ValueError('Candidate calendar, cancellation or causal ranking did not reconstruct')
    known = {e['event_id']:e for e in entries}
    for cohort in account['cohorts']:
        event = known[cohort['event_id']]
        if cohort['signal_date'] != event['signal_date'] or cohort['entry_date'] not in {t['date'] for t in fills[cohort['event_id']]}:
            raise ValueError('Cohort lost its original signal or actual entry date')
    return dict(queue_calendar_rebuilt=True, queue_prior_ranking_rebuilt=True,
                no_duplicate_candidate_fills=True, candidate_outcomes=outcomes)
