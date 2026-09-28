"""Frozen pre-open limit orders and explicitly inferred opening-batch fills."""
from copy import deepcopy
import math

import pandas as pd

from skills.execution_stress import StressOrder
from skills.million_replay import money
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.residual_tick_replay import TickExecution, audit_tick_plans


def opening_match(tape, opening, limit, qty, adv, participation):
    if (type(qty) is not int or qty<0 or qty%1000 or participation not in (.01,.005)
            or not math.isfinite(limit) or limit<=0):
        raise ValueError('Invalid frozen opening assumptions')
    if (not math.isfinite(opening) or opening <= 0 or tape.empty
            or not math.isfinite(adv) or adv <= 0):
        raise ReplayDataUnavailable('Missing opening price, tape or prior ADV')
    if opening>limit+1e-8:
        raise ReplayDataUnavailable('Opening price exceeds legal upper limit')
    first = tape.time.min()
    if not pd.Timedelta('09:00:00') <= first < pd.Timedelta('13:25:00'):
        raise ReplayDataUnavailable('First tick is outside supported opening window')
    batch = tape.loc[tape.time.eq(first)]
    if not batch.price.eq(opening).all():
        raise ReplayDataUnavailable('First tick batch conflicts with raw daily open')
    shares = int(batch.shares.sum())
    if shares <= 0:
        raise ReplayDataUnavailable('Opening batch has no positive trade volume')
    capacity = int(min(shares, adv)*participation)//1000*1000
    # A limit at the auction price gives no evidence about time priority.
    fill = min(qty, capacity) if opening < limit-1e-8 else 0
    return dict(opening_time=str(first), opening_shares=shares, capacity_qty=capacity,
                reference_price=opening, filled_qty=fill,
                failure=None if fill==qty else 'opening_at_upper_limit' if opening>=limit-1e-8
                else 'partial_opening_capacity' if fill else 'opening_capacity_below_one_lot')


class OpeningPlanning:
    def __init__(self, *args, **kwargs):
        self.base_tick_plans=[]
        super().__init__(*args, **kwargs)

    def _plan(self, day, sid, side, eid, signal, qty, budget, opening_cash, failure):
        super()._plan(day,sid,side,eid,signal,qty,budget,opening_cash,failure)
        plan=self.day_plans[(eid,side)]
        self.base_tick_plans.append(deepcopy(plan))
        if side=='buy':
            plan.update(order_time='08:59:00',expires_at='opening_batch_only',
                        sizing_budget=budget if self.benchmark else min(budget,self.residual_budget))
            if plan['planned_qty']:
                limits=self.feeds.get_limits(sid).get(str(day.date()))
                if not limits or not 0<limits['lower']<=limits['upper']:
                    raise ReplayDataUnavailable(f'Missing bounded opening limit: {sid} {day.date()}')
                plan['limit_price']=limits['upper']
                plan['planned_qty']=self._affordable(plan['planned_qty'],1000,plan['limit_price'],plan['sizing_budget'],sid)
                if not plan['planned_qty']:plan['rejection']='opening_reserve_below_one_lot'
        self.tick_plans[-1]=deepcopy(plan)

    def run(self):
        result=super().run()
        result['base_tick_plans']=self.base_tick_plans
        result['settings'].update(execution='next_session_opening_entry_v1',
            opening_auction_inferred=True,cancellation_latency_verified=False,live_qualified=False)
        return result


class OpeningExecution(TickExecution):
    def _execute_order(self,day,sid,side,qty,reason,event_id,signal_date=None):
        if side!='buy':return super()._execute_order(day,sid,side,qty,reason,event_id,signal_date)
        plan=self.day_plans[(event_id,side)]
        if sid in self.tick_attempts or plan['stock_id']!=sid or plan['signal_date']!=signal_date:
            raise ValueError('Opening order duplicated or differs from frozen identity')
        self.tick_attempts.add(sid)
        if qty<plan['planned_qty']:raise ValueError('Opening resources shrank a frozen order')
        adv,amount=(float(frame.at[day,sid]) for frame in (self.volume20,self.amount20))
        row=dict(date=str(day.date()),stock_id=sid,name=self.names.get(sid,sid),side=side,event_id=event_id,
            signal_date=signal_date,channel='board',requested_qty=plan['planned_qty'],filled_qty=0,
            reason=reason,failure=plan['rejection'],prior_avg_volume20=adv,prior_avg_amount20=amount,
            limit_price=plan['limit_price'],order_time=plan['order_time'],expires_at=plan['expires_at'])
        if not plan['planned_qty']:
            self.orders.append(row);return 0
        tape,digest=self.ticks.get(sid,row['date'],self.markets[sid])
        high,low,volume=(self.raw(day,sid,k) for k in ('high','low','volume'))
        if (not high or not low or not volume or tape.price.max()>high+1e-6
                or tape.price.min()<low-1e-6 or int(tape.shares.sum())>volume*1.01):
            raise ReplayDataUnavailable(f'Opening tick/daily price or unit conflict: {sid} {day.date()}')
        opening=float(self.raw(day,sid,'open') or 0)
        limits=self.feeds.get_limits(sid).get(row['date'])
        if not limits or opening<limits['lower']-1e-8:
            raise ReplayDataUnavailable('Opening price below dated legal lower limit')
        row.update(ticks_sha256=digest,day_volume=volume,tick_daily_volume_ratio=float(tape.shares.sum()/volume),
            **opening_match(tape,opening,plan['limit_price'],plan['planned_qty'],adv,self.tick_participation))
        filled,price=row['filled_qty'],row['reference_price']
        if filled:
            paid=self._costs(price,filled,side,sid)
            if -paid['cash_change']>plan['sizing_budget']+.005 or money(self.cash+paid['cash_change'])<0:
                raise ValueError('Opening fill exceeded frozen cash')
            self.cash_move(day,side,paid['cash_change'],stock_id=sid,event_id=event_id,channel='board')
            self.holdings[sid]['qty']+=filled
            mark=self.raw(day,sid)
            self.marks[sid]=dict(price=mark,date=row['date'])
            self.used[(sid,'board')]+=filled
            self.day_cost+=paid['total_cost'];self.day_basis+=filled*(mark-price)
            trade=dict(row,**paid,qty=filled,cash_after=self.cash,remaining_shares=self.holdings[sid]['qty'],
                day_participation=filled/volume,sequence=len(self.trades)+1)
            trade.pop('filled_qty');trade.pop('failure');self.trades.append(trade)
        self.orders.append(row)
        return filled


def factory(cls):
    class OpeningReplay(OpeningPlanning,cls,OpeningExecution):
        pass
    return OpeningReplay


def audit_opening(account,ticks,markets,quotes,calendar,corporate,*,feeds):
    # Original references, signal timing and unchanged sell matching are audited
    # against the sealed engine. Buy fills are reconstructed separately below.
    base=dict(account,tick_plans=account['base_tick_plans'],
              orders=[r for r in account['orders'] if r['side']=='sell'],
              trades=[r for r in account['trades'] if r['side']=='sell'])
    audit_tick_plans(base,ticks,markets,quotes,calendar,corporate)
    before,after=account['base_tick_plans'],account['tick_plans']
    if len(before)!=len(after):raise ValueError('Opening plan count changed')
    cost=StressOrder();cost.stress_slippage=account['settings']['slippage']
    plans={}
    for original,plan in zip(before,after):
        expected=deepcopy(original)
        if original['side']=='buy':
            budget=plan['sizing_budget']
            if not math.isfinite(budget) or not 0<=budget<=original['reserved_cash']:
                raise ValueError('Opening budget exceeds frozen cash')
            expected.update(order_time='08:59:00',expires_at='opening_batch_only',sizing_budget=budget)
            if original['planned_qty']:
                limits=feeds.get_limits(plan['stock_id']).get(plan['date'])
                if not limits or not 0<limits['lower']<=limits['upper']:
                    raise ReplayDataUnavailable('Opening audit lacks dated limits')
                expected['limit_price']=limits['upper']
                expected['planned_qty']=cost._affordable(original['planned_qty'],1000,limits['upper'],budget,plan['stock_id'])
                if not expected['planned_qty']:expected['rejection']='opening_reserve_below_one_lot'
        if expected!=plan:raise ValueError('Opening plan differs from pre-open rule')
        plans[(plan['date'],plan['event_id'],plan['side'])]=plan
    source=quotes.copy();source['date']=pd.to_datetime(source['date'])
    price=source.pivot(index='date',columns='stock_id',values='open').reindex(calendar)
    close=source.pivot(index='date',columns='stock_id',values='close').reindex(calendar)
    volume=source.pivot(index='date',columns='stock_id',values='volume').reindex(calendar)
    adv=volume.rolling(20,min_periods=20).mean().shift(1)
    amount=(close*volume).rolling(20,min_periods=20).mean().shift(1)
    matched={};seen=set()
    for row in account['orders']:
        if row['side']!='buy' or 'ticks_sha256' not in row:continue
        key=(row['date'],row['event_id'],row['side']);plan=plans[key]
        sid,day=row['stock_id'],pd.Timestamp(row['date'])
        stockday=(sid,row['date'])
        if stockday in seen or row['requested_qty']!=plan['planned_qty'] or row['limit_price']!=plan['limit_price']:
            raise ValueError('Opening execution changed plan or reused volume')
        seen.add(stockday)
        if row['prior_avg_volume20']!=adv.at[day,sid] or row['prior_avg_amount20']!=amount.at[day,sid]:
            raise ValueError('Opening liquidity differs from prior window')
        tape,digest=ticks.get(sid,row['date'],markets.get(stockday,markets.get(sid)))
        limits=feeds.get_limits(sid).get(row['date'])
        if not limits or price.at[day,sid]<limits['lower']-1e-8:
            raise ReplayDataUnavailable('Opening audit price below legal lower limit')
        rebuilt=opening_match(tape,price.at[day,sid],plan['limit_price'],plan['planned_qty'],adv.at[day,sid],account['settings']['participation'])
        if digest!=row['ticks_sha256'] or any(row[k]!=v for k,v in rebuilt.items()):
            raise ValueError('Opening tick fill did not independently reproduce')
        matched[key]=row
    fills={}
    for trade in account['trades']:
        if trade['side']!='buy':continue
        key=(trade['date'],trade['event_id'],trade['side']);row=matched.get(key)
        if not row or trade['reference_price']!=row['reference_price']:
            raise ValueError('Opening trade lacks rebuilt price')
        paid=cost._costs(row['reference_price'],trade['qty'],'buy',trade['stock_id'])
        if any(trade[k]!=v for k,v in paid.items()) or -trade['cash_change']>plans[key]['sizing_budget']+.005:
            raise ValueError('Opening trade costs or budget disagree')
        fills[key]=fills.get(key,0)+trade['qty']
    if any(fills.get(k,0)!=r['filled_qty'] for k,r in matched.items()):raise ValueError('Opening fill ledger mismatch')
    return dict(precommitted_limits=True,opening_cash_reserved=True,tick_fills_rebuilt=True,
                opening_buy_rule_rebuilt=True,unchanged_sell_rule_rebuilt=True)
