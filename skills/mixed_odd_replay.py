"""Separate board ticks and explicitly estimated whole-session odd-lot fills."""
from copy import deepcopy
import math
import pandas as pd
from skills.board_only_verified_replay import VerifiedBoardOnlyOrders
from skills.execution_stress import StressOrder
from skills.intraday_limit_replay import match_ticks
from skills.opening_entry_replay import opening_match
from skills.million_replay import money
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.residual_tick_replay import audit_tick_plans


def sized_quantity(maximum,limit,budget,cost,sid):
    qty=min(maximum,max(0,int(budget/limit)))
    while qty:
        board,odd=qty//1000*1000,qty%1000
        needed=sum(-cost._costs(limit,n,'buy',sid)['cash_change'] for n in (board,odd) if n)
        if needed<=budget+1e-7:return qty
        qty-=1
    return 0


def odd_match(row,side,limit,qty,participation):
    if side not in ('buy','sell') or type(qty) is not int or not 0<qty<1000 or participation not in (.01,.005):
        raise ValueError('Invalid odd-lot child order')
    if row and row.get('odd_shares')==0:
        return dict(odd_volume=0,odd_high=row.get('odd_high'),odd_low=row.get('odd_low'),capacity_qty=0,
            reference_price=None,filled_qty=0,participation_limit=participation,
            execution_evidence='daily_envelope_estimate',failure='official_odd_zero_volume')
    if not row or any(row.get(k) is None for k in ('odd_shares','odd_high','odd_low')):
        raise ReplayDataUnavailable('Missing independent odd-lot daily evidence')
    volume,high,low=(row[k] for k in ('odd_shares','odd_high','odd_low'))
    if not all(math.isfinite(v) for v in (volume,high,low)) or volume<=0 or not 0<low<=high:
        raise ReplayDataUnavailable('Invalid odd-lot daily price or volume')
    eligible=high<limit-1e-8 if side=='buy' else low>limit+1e-8
    capacity=int(volume*participation) if eligible else 0
    filled=min(qty,capacity)
    return dict(odd_volume=int(volume),odd_high=high,odd_low=low,capacity_qty=capacity,
        reference_price=high if side=='buy' else low,filled_qty=filled,
        participation_limit=participation,execution_evidence='daily_envelope_estimate',
        failure=None if filled==qty else 'odd_daily_boundary_uncertain' if not eligible
        else 'partial_odd_daily_capacity' if filled else 'odd_daily_capacity_zero')


class MixedOrders:
    def __init__(self,*args,odd_feeds,**kwargs):
        self.odd_feeds=odd_feeds;self.base_tick_plans=[]
        super().__init__(*args,**kwargs)

    def _plan(self,day,sid,side,eid,signal,qty,budget,opening_cash,failure):
        super()._plan(day,sid,side,eid,signal,qty,budget,opening_cash,failure)
        p=self.day_plans[(eid,side)];self.base_tick_plans.append(deepcopy(p))
        p.update(odd_limit=None,sizing_budget=budget if self.benchmark else min(budget,self.residual_budget),
                 order_time='08:59:00',odd_order_time='09:00:00',odd_expires_at='13:30:00')
        eligible=(side=='sell' or (budget>0 and p['prior_reference'] and p['rejection'] in (None,'cash_below_one_lot_or_missing_prior')))
        if eligible:
            limits=self.feeds.get_limits(sid).get(str(day.date()))
            if not limits or not 0<limits['lower']<=limits['upper']:
                raise ReplayDataUnavailable(f'Missing mixed dated legal limits: {sid} {day.date()}')
            p['odd_limit']=limits['upper'] if side=='buy' else limits['lower']
            if side=='buy':
                p['limit_price']=limits['upper']
                maximum=max(0,math.floor((p['sizing_budget']-40)/(p['prior_reference']*(1+.001425+.0045))))
                p['planned_qty']=sized_quantity(maximum,limits['upper'],p['sizing_budget'],self,sid)
                p['rejection']=None if p['planned_qty'] else 'mixed_budget_below_one_share'
            else:p['planned_qty']=self.holdings[sid]['qty']
        p['board_qty']=p['planned_qty']//1000*1000;p['odd_qty']=p['planned_qty']%1000
        self.tick_plans[-1]=deepcopy(p)

    def order(self,*args,**kwargs):
        # Retain the resource/slot/residual layers; skip only the board-only
        # wrapper's final-fill assertion because this derivative permits odds.
        return super(VerifiedBoardOnlyOrders,self).order(*args,**kwargs)

    def _execute_order(self,day,sid,side,qty,reason,event_id,signal_date=None):
        if side=='sell' and reason=='scheduled_exit' and hasattr(self,'exit_states'):
            state=self.exit_states.get(event_id)
            if not state or not state['trigger_reason']:raise ValueError('Unlatched mixed exit')
            reason,signal_date=state['trigger_reason'],state['signal_date']
        p=self.day_plans[(event_id,side)]
        if p['stock_id']!=sid or p['signal_date']!=signal_date or sid in self.tick_attempts:
            raise ValueError('Mixed order identity changed or repeated')
        self.tick_attempts.add(sid)
        if hasattr(self,'identity'):
            identity=self.identity(day,sid)
            if identity['status'] not in ('identified','official_trading_suspension','not_general_board'):
                raise ReplayDataUnavailable(f'Mixed order identity unavailable: {sid} {day.date()}')
            if identity['status']=='not_general_board' or (identity['status']=='identified' and identity['category']!=('ETF' if sid=='0050' else '股票')):
                raise ReplayDataUnavailable('Mixed order is not an eligible dated security')
            self.markets[sid]=identity['market'].upper()
        if self.official_halt(day,sid):
            self.orders.append(dict(date=str(day.date()),stock_id=sid,event_id=event_id,signal_date=signal_date,
                side=side,channel='event',requested_qty=p['planned_qty'],filled_qty=0,reason=reason,failure='official_full_session_halt'))
            return 0
        if qty<p['planned_qty']:raise ValueError('Mixed resource sizing shrank frozen order')
        if p['planned_qty']:self.require_prior_inputs(day,sid)
        total=0
        for channel,n in (('board',p['board_qty']),('odd',p['odd_qty'])):
            if not n:continue
            limit=p['limit_price'] if channel=='board' else p['odd_limit']
            row=dict(date=str(day.date()),stock_id=sid,name=self.names.get(sid,sid),side=side,event_id=event_id,
                signal_date=signal_date,reason=reason,channel=channel,requested_qty=n,limit_price=limit,
                prior_avg_amount20=float(self.amount20.at[day,sid]),prior_avg_volume20=float(self.volume20.at[day,sid]),
                order_time=('08:59:00' if side=='buy' else '09:01:00') if channel=='board' else '09:00:00',
                expires_at=('opening_batch_only' if side=='buy' else '13:25:00') if channel=='board' else '13:30:00')
            if channel=='board':
                tape,digest=self.ticks.get(sid,row['date'],self.markets[sid])
                high,low,volume=(self.raw(day,sid,k) for k in ('high','low','volume'))
                if (not high or not low or not volume or tape.price.max()>high+1e-6 or tape.price.min()<low-1e-6
                        or int(tape.shares.sum())>volume*1.01):raise ReplayDataUnavailable('Mixed tick/daily conflict')
                adv=float(self.volume20.at[day,sid]);row.update(ticks_sha256=digest,prior_avg_volume20=adv,day_volume=volume)
                limits=self.feeds.get_limits(sid).get(row['date'])
                if side=='buy':
                    opening=float(self.raw(day,sid,'open') or 0)
                    if opening<limits['lower']-1e-8:raise ReplayDataUnavailable('Opening below legal lower limit')
                    row.update(opening_match(tape,opening,limit,n,adv,self.tick_participation))
                elif not limits['lower']<=limit<=limits['upper']:
                    row.update(filled_qty=0,failure='precommitted_limit_outside_legal_range',reference_price=limit)
                else:
                    row.update(match_ticks(tape,side,limit,n,adv,self.tick_participation),reference_price=limit)
                    row['failure']=None if row['filled_qty']==n else 'partial_trade_through_capacity' if row['filled_qty'] else 'no_trade_through_capacity'
            else:
                odd=self.odd_feeds.get_odd(row['date'],sid,self.markets[sid])
                row.update(odd_match(odd,side,limit,n,self.tick_participation))
                limits=self.feeds.get_limits(sid).get(row['date'])
                if odd['odd_shares'] and (odd['odd_low']<limits['lower']-1e-8 or odd['odd_high']>limits['upper']+1e-8):
                    raise ReplayDataUnavailable('Odd daily price conflicts with legal range')
                volume=odd['odd_shares']
            filled=row['filled_qty'];price=row['reference_price']
            if filled:
                paid=self._costs(price,filled,side,sid)
                if money(self.cash+paid['cash_change'])<0:
                    row.update(filled_qty=0,failure='proceeds_below_costs_insufficient_cash');filled=0
                else:
                    self.cash_move(day,side,paid['cash_change'],stock_id=sid,event_id=event_id,channel=channel)
                    self.holdings[sid]['qty']+=filled if side=='buy' else -filled
                    mark=self.raw(day,sid);self.marks[sid]=dict(price=mark,date=row['date'])
                    self.used[(sid,channel)]+=filled;self.day_cost+=paid['total_cost'];self.day_basis+=filled*(mark-price)*(1 if side=='buy' else -1)
                    trade=dict(row,**paid,qty=filled,cash_after=self.cash,remaining_shares=self.holdings[sid]['qty'],
                        day_participation=filled/volume,sequence=len(self.trades)+1)
                    trade.pop('filled_qty');trade.pop('failure',None);self.trades.append(trade)
            total+=filled;self.orders.append(row)
        return total

    def run(self):
        result=super().run();result['base_tick_plans']=self.base_tick_plans
        result['settings'].update(execution='mixed_board_open_odd_daily_v1',execution_policy='board_and_odd',
            odd_execution_evidence='daily_envelope_estimate',odd_tick_verified=False,live_qualified=False,
            opening_auction_inferred=True,cancellation_latency_verified=False)
        return result


def factory(cls):
    class MixedReplay(MixedOrders,cls):pass
    return MixedReplay
