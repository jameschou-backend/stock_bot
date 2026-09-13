"""Point-in-time planning and cross-session accounting for research only.

Providers must supply dated views, complete corporate-action coverage, and
validated tapes. Missing evidence aborts the entire current session atomically.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from decimal import Decimal,ROUND_HALF_UP,ROUND_FLOOR

from skills.contingent_execution import Plan,integer
from skills.contingent_replay import cumulative_bill
from skills.contingent_session import replay_session
from skills.replay_market_feeds import ReplayDataUnavailable


def cents(value):
    return int(Decimal(str(value)).quantize(Decimal(1),rounding=ROUND_HALF_UP))


def channel_orders(day,sid,side,qty,price,signal,slip):
    result=[]
    for channel,amount in [('board',qty//1000*1000),('odd',qty%1000)]:
        if amount:
            gross=amount*price
            budget=gross+cumulative_bill(gross,'buy',sid,slip)['total'] if side=='buy' else 0
            result.append(asdict(Plan(f'{day}:{side}:{sid}:{channel}',sid,side,channel,amount,price,budget,signal)))
    return result


def sized_orders(day,sid,budget,price,signal,slip):
    low,high=0,max(0,budget//price)
    # Cost is monotonic inside each 1000-share block, but a new board/odd
    # split adds a minimum fee. Check the boundary and remainder separately.
    board=high//1000
    while board and sum(p['budget_cents'] for p in channel_orders(day,sid,'buy',board*1000,price,signal,slip))>budget:
        board-=1
    base=board*1000
    low,high=0,min(999,max(0,(budget-base*price)//price))
    while low<high:
        mid=(low+high+1)//2
        if sum(p['budget_cents'] for p in channel_orders(day,sid,'buy',base+mid,price,signal,slip))<=budget:low=mid
        else:high=mid-1
    return channel_orders(day,sid,'buy',base+low,price,signal,slip)


class PortfolioReplay:
    def __init__(self,calendar,entries,view,actions,tapes,*,start,end,initial_cents=100_000_000,
                 slots=3,credit_delay_us=1_000_000,settlement_lag=3,benchmark=False):
        self.calendar=list(calendar)
        if self.calendar!=sorted(set(self.calendar)):raise ValueError('Unique ordered calendar required')
        for d in self.calendar:date.fromisoformat(d)
        if start not in calendar or end not in calendar or start>end or calendar.index(start)==0:
            raise ValueError('Calendar must cover endpoints and prior decision session')
        self.index={d:i for i,d in enumerate(calendar)}
        self.start,self.end=start,end
        integer(initial_cents,1);integer(slots,1);integer(settlement_lag,1)
        self.initial=initial_cents;self.slots=1 if benchmark else slots
        self.view,self.actions,self.tapes=view,actions,tapes
        self.delay,self.lag,self.benchmark=credit_delay_us,settlement_lag,benchmark
        self.entries=defaultdict(list)
        for e in entries:
            day=e['entry_date'];signal=e['signal_date']
            if day not in self.index or self.index[day]==0 or calendar[self.index[day]-1]!=signal:
                raise ValueError('Candidate must use the preceding session signal')
            self.entries[day].append(deepcopy(e))
        self.state=dict(available_cents=initial_cents,holdings={},cohorts={},pending=[],nav_cents=initial_cents)
        self.applied=set();self.daily=[];self.sessions=[];self.plans=[];self.action_log=[]

    def quotes(self,day,symbols):
        data=self.view(day,set(symbols))
        for sid in symbols:
            q=data.get(sid)
            if not q or q.get('date')!=day:
                raise ReplayDataUnavailable(f'Missing dated quote: {sid} {day}')
            for field in ('raw_cents','adv20_shares','amount20_cents'):integer(q[field],1)
            if not Decimal(str(q['adjusted_close'])).is_finite() or Decimal(str(q['adjusted_close']))<=0:
                raise ReplayDataUnavailable(f'Missing adjusted close: {sid} {day}')
        return deepcopy(data)

    def opening(self,day,decision,relevant,quotes):
        state=self.state
        # Entitlements use yesterday's actual shares, before today's trading.
        opening_holdings=dict(state['holdings'])
        events=self.actions(day,set(relevant))
        for a in events:
            sid=a['stock_id'];identity=a['action_id']
            if (a['date']!=day or a['known_date']>decision or not a['source_id']
                    or a.get('verified') is not True):
                raise ReplayDataUnavailable(f'Unverified corporate terms: {sid} {day}')
            if identity in self.applied:raise ValueError('Duplicate corporate action')
            if a['kind'] not in ('cash_dividend','stock_dividend','split','capital_reduction'):
                raise ReplayDataUnavailable(f'Unsupported corporate action: {sid} {day} {a["kind"]}')
            if a['available_date'] not in self.index or a['available_date']<day:
                raise ReplayDataUnavailable('Corporate available session missing')
            integer(a['reference_cents'],1)
            if sid in quotes:quotes[sid]['raw_cents']=a['reference_cents']
            qty=opening_holdings.get(sid,0)
            cash=shares=0
            if a['kind']=='cash_dividend':
                rate=Decimal(str(a['cash_per_share_cents']))
                if not rate.is_finite() or rate<0:raise ValueError('Invalid dividend rate')
                cash=cents(qty*rate)
            else:
                exchange=a['kind'] in ('split','capital_reduction')
                if exchange and any(p['shares'] and p['stock_id']==sid for p in state['pending']):
                    raise ReplayDataUnavailable('Exchange with outstanding stock rights requires verified conversion terms')
                rate=Decimal(str(a['exchange_ratio'] if exchange else a['new_shares_per_share']))
                if not rate.is_finite() or rate<0 or exchange and rate==0:raise ValueError('Invalid stock conversion ratio')
                right=qty*rate;shares=int(right.to_integral_value(rounding=ROUND_FLOOR));fraction=right-shares
                if exchange:state['holdings'].pop(sid,None)
                if a['kind']=='capital_reduction':
                    return_rate=Decimal(str(a['cash_return_per_old_share_cents']))
                    if not return_rate.is_finite() or return_rate<0:raise ValueError('Invalid capital return')
                    cash=cents(qty*return_rate)
                if fraction:
                    if a.get('fractional_cash_per_share_cents') is None:
                        raise ReplayDataUnavailable('Fractional right settlement terms missing')
                    fractional_rate=Decimal(str(a['fractional_cash_per_share_cents']))
                    if not fractional_rate.is_finite() or fractional_rate<0:raise ValueError('Invalid fractional cash rate')
                    cash+=cents(fraction*fractional_rate)
            if cash or shares:
                state['pending'].append(dict(kind='corporate',stock_id=sid,cash_cents=cash,shares=shares,
                    available_date=a['available_date'],source_id=identity))
            self.applied.add(identity)
            self.action_log.append(dict(a,entitled_shares=qty,receivable_cash_cents=cash,receivable_shares=shares))
        pending=[]
        for item in state['pending']:
            if item['available_date']<=day:
                state['available_cents']+=item['cash_cents']
                if item['shares']:state['holdings'][item['stock_id']]=state['holdings'].get(item['stock_id'],0)+item['shares']
                self.action_log.append(dict(item,kind='delivery',date=day))
            else:pending.append(item)
        state['pending']=pending

    def plan(self,day,decision,candidates,q):
        state=self.state;orders=[];reasons={}
        reserved={p['stock_id'] for p in state['pending'] if p['shares']}
        forecast=state['available_cents']
        exiting=set()
        for sid,qty in sorted(state['holdings'].items()):
            cohort=state['cohorts'][sid]
            if not self.benchmark and not cohort.get('exit_reason'):
                current=Decimal(str(q[sid]['adjusted_close']))
                if current<=Decimal(cohort['entry_adjusted_close'])*Decimal('.88'):
                    cohort.update(exit_reason='loss12',exit_signal_date=decision)
                elif self.index[day]-self.index[cohort['entry_date']]>=63:
                    cohort.update(exit_reason='time63',exit_signal_date=decision)
            if cohort.get('exit_reason'):
                planned=channel_orders(day,sid,'sell',qty,q[sid]['raw_cents'],cohort['exit_signal_date'],45)
                orders+=planned;exiting.add(sid);reasons[sid]=cohort['exit_reason']
                forecast+=sum(p['qty']*p['limit_cents']-cumulative_bill(p['qty']*p['limit_cents'],'sell',sid,45)['total'] for p in planned)
        occupied=(set(state['holdings'])-exiting)|reserved
        candidates=sorted(candidates,key=lambda e:(-q[e['members'][0]]['amount20_cents'],e['members'][0],e['event_id']))
        decisions=[]
        for event in candidates:
            sid=event['members'][0]
            if not self.benchmark and (sid in state['holdings'] or sid in occupied or len(occupied)>=self.slots):continue
            if not self.benchmark and q[sid]['amount20_cents']<5_000_000_000:continue
            budget=max(0,min(state['nav_cents']//self.slots,forecast))
            if self.benchmark and budget<500_000:continue
            planned=sized_orders(day,sid,budget,q[sid]['raw_cents'],decision,45)
            if not planned:continue
            orders+=planned;occupied.add(sid);forecast-=sum(p['budget_cents'] for p in planned)
            decisions.append(dict(stock_id=sid,event_id=event['event_id'],qty=sum(p['qty'] for p in planned),
                prior_amount20_cents=q[sid]['amount20_cents']))
        return dict(date=day,decision_date=decision,prior_data_date=decision,calendar=self.calendar,
            holdings=dict(state['holdings']),available_cents=state['available_cents'],slots=self.slots,
            prior_avg_volume_shares={s:v['adv20_shares'] for s,v in q.items()},plans=orders,
            credit_delay_us=self.delay,slippage_bps=45,reserved_members=sorted(reserved)),decisions,reasons

    def run(self):
        blocked=None;peak=self.initial;synthetic=False
        for day in self.calendar[self.index[self.start]:self.index[self.end]+1]:
            previous=self.calendar[self.index[day]-1]
            saved=deepcopy(self.state);applied=set(self.applied);action_count=len(self.action_log)
            spec=None;stage='prior_data'
            try:
                candidates=[] if self.benchmark else self.entries[day]
                if self.benchmark:
                    candidates=[dict(event_id='benchmark',members=['0050'])]
                symbols=set(self.state['holdings'])|{p['stock_id'] for p in self.state['pending'] if p['shares']}|{e['members'][0] for e in candidates}
                quotes=self.quotes(previous,symbols)
                stage='corporate_actions';self.opening(day,previous,symbols,quotes)
                stage='planning';spec,decisions,reasons=self.plan(day,previous,candidates,quotes)
                self.plans.append(dict(date=day,spec=deepcopy(spec),candidates=decisions,exit_reasons=reasons))
                stage='tapes'
                needed={(p['stock_id'],p['channel']) for p in spec['plans']}
                result=replay_session(spec,self.tapes(day,needed))
                if not result['completed']:
                    blocked=dict(date=day,stage=stage,missing=result['missing']);raise ReplayDataUnavailable('Required market tapes missing')
                synthetic|=result['synthetic']
                state=self.state;state['available_cents']=result['available_cents'];state['holdings']=result['holdings']
                credits={e['fill_id'] for e in result['events'] if e['kind']=='available'}
                unpaid=[f for f in result['fills'] if f['side']=='sell' and f['cash_cents']>0 and f['fill_id'] not in credits]
                for f in unpaid:
                    if self.index[day]+self.lag>=len(self.calendar):raise ReplayDataUnavailable('Settlement calendar coverage missing')
                    state['pending'].append(dict(kind='sale',stock_id=f['stock_id'],cash_cents=f['cash_cents'],shares=0,
                        source_id=f['fill_id'],available_date=self.calendar[self.index[day]+self.lag]))
                if sum(f['cash_cents'] for f in unpaid)!=result['receivable_cents']:raise ValueError('Uncredited sale reconciliation failed')
                stage='valuation'
                marked=set(state['holdings'])|{p['stock_id'] for p in state['pending'] if p['shares']}
                endquotes=self.quotes(day,marked)
                for f in result['fills']:
                    if f['side']=='buy' and f['stock_id'] not in state['cohorts']:
                        sid=f['stock_id'];state['cohorts'][sid]=dict(entry_date=day,entry_adjusted_close=str(endquotes[sid]['adjusted_close']))
                reserved={p['stock_id'] for p in state['pending'] if p['shares']}
                for sid in list(state['cohorts']):
                    if sid not in state['holdings'] and sid not in reserved:del state['cohorts'][sid]
                if len(set(state['holdings'])|reserved)>self.slots:raise ValueError('Cross-day slot conservation failed')
                shares_value=sum(qty*endquotes[sid]['raw_cents'] for sid,qty in state['holdings'].items())
                pending_value=sum(p['cash_cents']+p['shares']*(endquotes[p['stock_id']]['raw_cents'] if p['shares'] else 0) for p in state['pending'])
                nav=state['available_cents']+shares_value+pending_value;state['nav_cents']=nav;peak=max(peak,nav)
                self.daily.append(dict(date=day,nav_cents=nav,available_cents=state['available_cents'],
                    market_value_cents=shares_value,receivable_value_cents=pending_value,holdings=dict(state['holdings']),
                    drawdown=nav/peak-1,cost_cents=sum(f['costs_cents']['total'] for f in result['fills'])))
                self.sessions.append(result)
            except ReplayDataUnavailable as exc:
                self.state=saved;self.applied=applied;del self.action_log[action_count:]
                blocked=blocked or dict(date=day,stage=stage,reason=str(exc))
                break
        return dict(completed=blocked is None,blocked=blocked,daily=self.daily,sessions=self.sessions,plans=self.plans,
            state=self.state,corporate_actions=self.action_log,live_qualified=False,synthetic=synthetic,
            total_return=self.state['nav_cents']/self.initial-1 if blocked is None else None,
            initial_cents=self.initial,start=self.start,end=self.end,
            assumptions=dict(credit_delay_us=self.delay,uncredited_sale_available_lag=self.lag,
                limit='prior_close_or_verified_corporate_reference',slots=self.slots,benchmark=self.benchmark))
