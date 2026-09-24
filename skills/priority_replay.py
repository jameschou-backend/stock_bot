"""Research-only candidate priority; never selects parameters from future returns."""
import math

import numpy as np
import pandas as pd

from skills.five_axis_replay import FiveAxisReplay


def group_strength_scores(close, entries):
    """Use exactly 20 calendar rows ending at the original signal date, no fill."""
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError('Expected unique ordered market calendar')
    valid = close.where(np.isfinite(close) & close.gt(0))
    returns = valid / valid.shift(20) - 1
    scores = {}
    for event in entries:
        key = event['event_id']
        if key in scores:
            raise ValueError('Duplicate event identity')
        day = pd.Timestamp(event['signal_date'])
        sid = event['members'][0]
        group = sorted(set(event.get('group_members', event['members'])))
        values = returns.reindex(index=[day], columns=group).iloc[0].dropna()
        own = returns.at[day, sid] if day in returns.index and sid in returns else np.nan
        known = bool(group) and len(values) >= math.ceil(.8 * len(group)) and np.isfinite(own)
        scores[key] = dict(score=float(own-values.median()) if known else None,
                           signal_date=str(day.date()), group_size=len(group),
                           valid_members=len(values), stock_id=sid)
    return scores


class PriorityReplay(FiveAxisReplay):
    def __init__(self, *args, priority='control', scores=None, **kwargs):
        if priority not in ('control', 'capacity', 'group_strength'):
            raise ValueError('Unknown priority')
        super().__init__(*args, arm='capacity' if priority == 'capacity' else 'control', **kwargs)
        self.priority = priority
        self.priority_scores = scores or {}
        self.priority_decisions = []

    def corporate_day(self, day):
        income = super().corporate_day(day)
        if self.priority == 'group_strength':
            candidates = self.events.get(day, [])
            def key(event):
                evidence = self.priority_scores[event['event_id']]
                if pd.Timestamp(evidence['signal_date']) >= day:
                    raise ValueError('Priority score is not available before execution')
                score = evidence['score']
                return (-(score if score is not None else -math.inf), event['members'][0], event['event_id'])
            candidates.sort(key=key)
            for rank, event in enumerate(candidates, 1):
                self.priority_decisions.append(dict(date=str(day.date()), rank=rank,
                    event_id=event['event_id'], **self.priority_scores[event['event_id']]))
        return income
