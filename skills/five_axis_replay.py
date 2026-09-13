"""Causal entry, candidate ordering, and allocation contrasts on cash accounting."""
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np
import pandas as pd
from skills.cash_risk_replay import CashRiskReplay

ARMS = ('control','limit3','retest3','capacity','capacity_vol','group_one','slots5','staged','joint')


class FiveAxisReplay(CashRiskReplay):
    def __init__(self,*args,arm='control',action_dates=(),**kwargs):
        if arm not in ARMS:
            raise ValueError('Unknown five-axis arm')
        if kwargs.get('risk_mode','none')!='none':
            raise ValueError('Account risk controls are fixed off in this experiment')
        super().__init__(*args,slots=5 if arm=='slots5' else 3,**kwargs)
        self.arm=arm
        self.policy_decisions=[]
        self.finished=set()
        self.targets={}
        self.add_attempted=set()
        self.source_events={e['event_id']:deepcopy(e) for group in self.events.values() for e in group}
        self.action_dates={(str(s),pd.Timestamp(d)) for s,d in action_dates}
        close=self.exit_signals.adjusted_close
        self.vol20=(close/close.shift(1)-1).rolling(20,min_periods=20).std().shift(1)
        self.support=close.rolling(60,min_periods=60).max().shift(1)
        if arm in ('limit3','retest3','joint'):
            expanded=defaultdict(list)
            for day,events in self.events.items():
                for event in events:
                    for offset in range(3):
                        i=self.positions[day]+offset
                        if i>=len(self.days) or self.days[i]>self.end:
                            continue
                        attempt=deepcopy(event)
                        attempt['scheduled_entry_date']=event['entry_date']
                        attempt['entry_date']=str(self.days[i].date())
                        attempt['attempt_number']=offset+1
                        expanded[self.days[i]].append(attempt)
            self.events=expanded

    def _record(self,day,sid,event_id,reason,**details):
        self.policy_decisions.append(dict(date=str(day.date()),stock_id=sid,event_id=event_id,
                                          reason=reason,**details))

    def _reject(self,day,sid,qty,event_id,signal_date,reason):
        self._record(day,sid,event_id,reason,requested_qty=qty)
        self.orders.append(dict(date=str(day.date()),stock_id=sid,name=self.names.get(sid,sid),
            event_id=event_id,signal_date=signal_date,side='buy',channel='event',
            requested_qty=qty,filled_qty=0,reason='leader_entry',failure=reason))
        return 0

    def _action_between(self,sid,first,last):
        return any(stock==sid and first<day<=last for stock,day in self.action_dates)

    def corporate_day(self,day):
        income=super().corporate_day(day)
        candidates=[e for e in self.events.get(day,[]) if e['event_id'] not in self.finished]
        if self.arm in ('capacity','capacity_vol','joint'):
            def score(event):
                sid=event['members'][0]
                value=float(self.amount20.at[day,sid])
                if self.arm=='capacity_vol':
                    vol=float(self.vol20.at[day,sid])
                    value=value/vol if math.isfinite(vol) and vol>0 else float('nan')
                return (-(value if math.isfinite(value) else -math.inf),sid,event['event_id'])
            candidates.sort(key=score)
        self.events[day]=candidates
        return income

    def order(self,day,sid,side,qty,reason,event_id,signal_date=None):
        if side!='buy' or sid=='0050' or reason!='leader_entry':
            return super().order(day,sid,side,qty,reason,event_id,signal_date)
        event=self.source_events[event_id]
        if self.arm in ('group_one','joint'):
            group=set(event.get('group_members',event['members']))
            for stock,holding in self.holdings.items():
                if stock==sid or not holding['qty']:
                    continue
                other=self.source_events[holding['event_id']]
                if group.intersection(other.get('group_members',other['members'])):
                    return self._reject(day,sid,qty,event_id,signal_date,'overlapping_signal_group')
        if self.arm in ('limit3','retest3','joint'):
            signal=pd.Timestamp(signal_date)
            previous=self.days[self.positions[day]-1]
            raw=self.prior(day,sid)
            prior_adj=self.exit_signals.adjusted_close.at[previous,sid]
            support=self.support.at[signal,sid]
            signal_adj=self.exit_signals.adjusted_close.at[signal,sid]
            if self._action_between(sid,signal,day):
                self.finished.add(event_id)
                return self._reject(day,sid,qty,event_id,signal_date,'cancel_pending_corporate_action')
            if not raw or not all(math.isfinite(float(x)) and x>0 for x in (prior_adj,support,signal_adj)):
                return self._reject(day,sid,qty,event_id,signal_date,'missing_entry_reference')
            if prior_adj<support:
                self.finished.add(event_id)
                return self._reject(day,sid,qty,event_id,signal_date,'cancel_broken_support')
            if self.arm=='retest3':
                low=self.raw(previous,sid,'low')
                if previous<=signal or not low or low*prior_adj/raw>support:
                    return self._reject(day,sid,qty,event_id,signal_date,'waiting_prior_day_retest')
            limit=signal_adj*1.02*raw/prior_adj
            # This is fill validation for a pre-existing resting order, not a
            # same-day entry signal. Requiring the whole board-price range below
            # the limit deliberately avoids assuming a touch implies a fill.
            high=self.raw(day,sid,'high')
            if not high or high>limit:
                return self._reject(day,sid,qty,event_id,signal_date,'waiting_limit_all_day')
            if qty%1000:
                odd=self.feeds.get_odd(str(day.date()),sid,self.markets[sid])
                ref=odd.get('odd_ask' if self.stress_quote else 'odd_last') if odd else None
                if not ref or ref>limit:
                    return self._reject(day,sid,qty,event_id,signal_date,'waiting_odd_limit')
            self._record(day,sid,event_id,'resting_limit_validated',limit=limit,
                         reference_date=str(previous.date()))
        target=qty
        if self.arm=='staged':
            qty//=2
        filled=super().order(day,sid,side,qty,reason,event_id,signal_date)
        if filled:
            self.finished.add(event_id)
            if self.arm=='staged':
                self.targets[event_id]=dict(qty=target,entry_date=str(day.date()))
        return filled

    def buy_etf(self,day,reason):
        super().buy_etf(day,reason)
        if self.arm!='staged' or reason!='idle_cash':
            return
        i=self.positions[day]
        for sid,holding in list(self.holdings.items()):
            event_id=holding['event_id']
            if event_id not in self.targets or event_id in self.add_attempted or not holding['qty']:
                continue
            target=self.targets[event_id]
            entry=pd.Timestamp(target['entry_date'])
            if i<self.positions[entry]+5:
                continue
            self.add_attempted.add(event_id)
            state=self.exit_states.get(event_id)
            previous=self.days[i-1]
            a,b=(self.exit_signals.adjusted_close.at[d,sid] for d in (previous,entry))
            if (not state or state['trigger_reason'] or not np.isfinite([a,b]).all() or a<=b
                    or self._action_between(sid,entry,day)):
                self._record(day,sid,event_id,'cancel_staged_add')
                continue
            qty=max(0,target['qty']-holding['qty'])
            filled=super().order(day,sid,'buy',qty,'staged_add',event_id,str(previous.date()))
            cohort=next(c for c in self.cohorts if c['event_id']==event_id)
            cohort['bought_qty']+=filled
            self._record(day,sid,event_id,'staged_add_attempt',requested_qty=qty,filled_qty=filled)
