"""One extra session for unfilled candidates under fully locked resources."""
from collections import defaultdict
from copy import deepcopy
import pandas as pd

from skills.slot_reuse_replay import SlotReuseReplay


class DeferredEntryReplay(SlotReuseReplay):
    def __init__(self, *args, validity_sessions=1, **kwargs):
        if type(validity_sessions) is not int or validity_sessions not in (1, 2):
            raise ValueError('Only the preregistered one/two session policies are supported')
        super().__init__(*args, opening_cash_only=True, lock_unused=True,
                         lock_opening_slots=True, lock_failed_slots=True, **kwargs)
        self.validity_sessions = validity_sessions
        self.retry_decisions = []
        if validity_sessions == 2:
            expanded = defaultdict(list)
            for day, events in self.events.items():
                for event in events:
                    for offset in range(validity_sessions):
                        i = self.positions[day] + offset
                        if i >= len(self.days) or self.days[i] > self.end:
                            continue
                        attempt = deepcopy(event)
                        attempt.update(scheduled_entry_date=event['entry_date'],
                            entry_date=str(self.days[i].date()), attempt_number=offset+1)
                        expanded[self.days[i]].append(attempt)
            self.events = expanded

    def corporate_day(self, day):
        if self.validity_sessions == 2:
            keep = []
            for event in self.events.get(day, []):
                identity, sid = event['event_id'], event['members'][0]
                retry = event['attempt_number'] > 1
                action = retry and self._action_between(sid, pd.Timestamp(event['signal_date']), day)
                if identity in self.finished:
                    reason = 'already_filled'
                elif action:
                    reason = 'corporate_action_since_signal'
                    self.finished.add(identity)
                else:
                    reason = 'eligible'
                    keep.append(event)
                self.retry_decisions.append(dict(date=str(day.date()), event_id=identity,
                    signal_date=event['signal_date'], attempt_number=event['attempt_number'], reason=reason))
            self.events[day] = keep
        return super().corporate_day(day)
