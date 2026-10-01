"""Explicitly evidenced ex-date moves; unreviewed non-session rights fail closed."""
from copy import deepcopy
import math
import pandas as pd

from skills.million_replay import UnresolvedAction


class CorporateCalendar:
    def __init__(self, days, corrections):
        self.days = pd.DatetimeIndex(days)
        self.sessions = {str(d.date()) for d in self.days}
        self.corrections = {}
        for item in corrections:
            sid, old, new = item['stock_id'], item['scheduled_date'], item['effective_date']
            i = self.days.searchsorted(pd.Timestamp(old))
            if (old in self.sessions or i >= len(self.days) or str(self.days[i].date()) != new
                    or not old < new or not item['source']):
                raise ValueError('Ex-date move must be the evidenced next market session')
            key = (sid, old)
            if key in self.corrections:
                raise ValueError('Duplicate ex-date correction')
            values = [item[k] for k in ('lower', 'reference_price', 'upper')]
            if (not all(type(v) in (int, float) and math.isfinite(v) for v in values)
                    or not 0 < values[0] <= values[1] <= values[2]):
                raise ValueError('Invalid official ex-rights price band')
            self.corrections[key] = dict(item)

    def effective(self, sid, day):
        item = self.corrections.get((sid, day))
        return item['effective_date'] if item else day

    def actions(self, sid, rows):
        result = deepcopy(rows)
        for row in result:
            old = row['date']
            new = self.effective(sid, old)
            if new != old:
                row.update(date=new, scheduled_ex_date=old)
        return sorted(result, key=lambda r: (r['date'], r['action_id']))

    def guard(self, sid, day, rows):
        i = self.days.get_loc(pd.Timestamp(day))
        if not i:
            return
        previous = str(self.days[i-1].date())
        for row in rows:
            if previous < row['date'] < day and row['date'] not in self.sessions:
                raise UnresolvedAction(f'Unreviewed non-session corporate action: {sid} {row["date"]} -> {day}')

    def opening_reference(self, sid, day):
        rows = [r for (stock, _), r in self.corrections.items()
                if stock == sid and r['effective_date'] == day]
        if len(rows) > 1:
            raise ValueError('Multiple moved corporate references')
        return rows[0]['reference_price'] if rows else None

    def limits(self, sid, limits):
        result = deepcopy(limits)
        for (stock, _), item in self.corrections.items():
            if stock != sid:
                continue
            day = item['effective_date']
            if ('expected_finmind_limits' in item
                    and result.get(day) != item['expected_finmind_limits']):
                raise ValueError('Limit correction no longer matches its reviewed source')
            result[day] = dict(lower=item['lower'], upper=item['upper'])
        return result
