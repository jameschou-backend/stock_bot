"""Causal reclaim entries after a fully executed close-confirmed peak stop."""
from collections import Counter
from copy import deepcopy
import math
import pandas as pd

from skills.close_confirmed_exit import CloseConfirmedExit
from skills.replay_market_feeds import ReplayDataUnavailable

ARMS = ('control','reclaim','confirm2')


def recovered(close, ma10, excess5, floor):
    values = (close,ma10,excess5,floor)
    return all(pd.notna(v) and math.isfinite(float(v)) for v in values) and close > floor > 0 and close > ma10 > 0 and excess5 > 0


class CloseReentry(CloseConfirmedExit):
    def __init__(self,*args,reentry_arm,**kwargs):
        if reentry_arm not in ARMS:
            raise ValueError('Unknown close reentry arm')
        super().__init__(*args,**kwargs)
        self.reentry_arm = reentry_arm
        close = self.exit_signals.adjusted_close
        self.reentry_ma10 = close.rolling(10,min_periods=10).mean()
        ret5 = close/close.shift(5)-1
        self.reentry_excess5 = ret5.sub(ret5['0050'],axis=0)
        self.reentry_attempted, self.reentry_counts = set(),Counter()
        self.reentry_log, self.reentry_screens, self.reentry_queue = [],[],[]

    def corporate_day(self,day):
        if self.reentry_arm == 'control':
            return super().corporate_day(day)
        i = self.positions[day]; j = i-1
        prior = self.days[j]; signal = str(prior.date())
        candidates = list(self.events.get(day,[]))
        original = [e['event_id'] for e in candidates]
        original_stocks = {e['members'][0] for e in candidates}
        latest = {c['stock_id']:c for c in self.cohorts}
        added = []
        for sid,c in sorted(latest.items()):
            parent = c['event_id']
            root = self.source_events[parent].get('reentry_root',parent)
            state = self.exit_states.get(parent)
            if (parent in self.reentry_attempted or self.reentry_counts[root]>=2
                    or not c.get('exit_date') or not state or state['trigger_reason']!='close_confirmed_peak15'):
                continue
            exited = self.positions[pd.Timestamp(c['exit_date'])]
            if not 1 <= j-exited <= 20:
                continue
            if sid in self.holdings or any(r['stock_id']==sid and r.get('qty',0)>0 for r in self.receivables):
                continue
            trigger = next(r for r in self.close_evidence if r['event_id']==parent and r['triggered'])
            signal_index = self.positions[pd.Timestamp(trigger['signal_date'])]
            adjusted = self.exit_signals.price(signal_index,sid)
            if adjusted is None:
                raise ReplayDataUnavailable('Missing adjusted stop basis for reentry')
            floor = trigger['threshold'] * adjusted / trigger['close']
            indices = [j] if self.reentry_arm=='reclaim' else [j-1,j]
            observations = [dict(signal_date=str(self.days[k].date()),
                close=self.exit_signals.price(k,sid),ma10=float(self.reentry_ma10[sid].iloc[k]),
                excess5=float(self.reentry_excess5[sid].iloc[k])) for k in indices]
            ok = min(indices)>exited and all(recovered(r['close'],r['ma10'],r['excess5'],floor) for r in observations)
            market = self.exit_signals.trend.state.at[prior]=='ON'
            why = ('same_stock_original_signal' if sid in original_stocks else
                   'market_off' if not market else 'not_recovered' if not ok else 'candidate')
            row = dict(date=str(day.date()),signal_date=signal,stock_id=sid,parent=parent,root=root,
                exit_date=c['exit_date'],adjusted_stop_floor=floor,observations=observations,
                market_on=bool(market),outcome=why)
            # JSON reports preserve unknown observations explicitly, not NaN.
            for r in row['observations']:
                for k in ('ma10','excess5'):
                    if not math.isfinite(r[k]):r[k]=None
            self.reentry_screens.append(row)
            if why!='candidate':
                continue
            event = deepcopy(self.source_events[parent])
            event.update(event_id=root+'-reclaim-'+signal,signal_date=signal,entry_date=str(day.date()),
                reentry_root=root,reentry_parent=parent,reentry_attempt=self.reentry_counts[root]+1,
                selection_reason='Close-stop recovery: reclaim stop floor, above MA10, excess5 positive')
            added.append(event)
            self.source_events[event['event_id']] = deepcopy(event)
            self.reentry_attempted.add(parent); self.reentry_counts[root]+=1
            self.reentry_log.append(dict(row,event_id=event['event_id'],attempt=self.reentry_counts[root]))
        added.sort(key=lambda e:(-self.reentry_excess5.at[prior,e['members'][0]],e['members'][0],e['event_id']))
        self.events[day] = candidates+added
        self.reentry_queue.append(dict(date=str(day.date()),original=original,
            reentries=[e['event_id'] for e in added],ordered=[e['event_id'] for e in candidates+added]))
        return super().corporate_day(day)

    def run(self):
        result = super().run()
        if self.reentry_arm!='control':
            result.update(reentry_log=self.reentry_log,reentry_screens=self.reentry_screens,reentry_queue=self.reentry_queue)
            result['settings'].update(reentry_arm=self.reentry_arm,reentry_window=20,
                max_reentry_attempts_per_root=2,reentry_priority='original_candidates_then_excess5')
        return result
