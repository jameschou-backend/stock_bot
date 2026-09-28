"""Explicit daily HL2 price proxy, never an auction or VWAP fill claim."""
import math
from skills.million_replay import money
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.high_return_replay import HighReturnReplay
from skills.strict_tick_inputs import StrictResidualBenchmark
from skills.mixed_odd_replay import factory


def midpoint_match(high, low, volume, adv, qty, side, limit, lower, upper, channel):
    if channel not in ('board', 'odd') or side not in ('buy', 'sell'):
        raise ValueError('Unknown midpoint channel or side')
    if type(qty) is not int or qty <= 0 or (channel=='board' and qty%1000) or (channel=='odd' and qty>=1000):
        raise ValueError('Invalid midpoint child shares')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (volume,adv,limit,lower,upper)):
        raise ReplayDataUnavailable('Missing midpoint volume or limits')
    if volume<0 or adv<=0 or not 0<lower<=upper or limit<=0:
        raise ReplayDataUnavailable('Invalid midpoint volume or limits')
    if volume==0:
        return dict(reference_price=None,capacity_qty=0,filled_qty=0,failure='official_zero_volume')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (high,low)) or not lower-1e-8<=low<=high<=upper+1e-8:
        raise ReplayDataUnavailable('Midpoint range conflicts with legal limits')
    price=(high+low)/2
    eligible=(price<min(limit,upper)-1e-8 if side=='buy' else price>max(limit,lower)+1e-8)
    capacity=int((min(volume,adv) if channel=='board' else volume)*.01)
    if channel=='board':capacity=capacity//1000*1000
    if not eligible:capacity=0
    filled=min(qty,capacity)
    return dict(reference_price=price,capacity_qty=capacity,filled_qty=filled,
        failure=None if filled==qty else 'midpoint_limit_not_crossed' if not eligible
        else 'partial_midpoint_daily_capacity' if filled else 'midpoint_daily_capacity_zero')


class MidpointOrders:
    def _plan(self,*args,**kwargs):
        super()._plan(*args,**kwargs)
        p=self.tick_plans[-1]
        p['expires_at']='13:30:00'
        self.day_plans[(p['event_id'],p['side'])]['expires_at']='13:30:00'

    def _execute_order(self,day,sid,side,qty,reason,event_id,signal_date=None):
        if side=='sell' and reason=='scheduled_exit' and hasattr(self,'exit_states'):
            state=self.exit_states.get(event_id)
            if not state or not state['trigger_reason']:raise ValueError('Unlatched midpoint exit')
            reason,signal_date=state['trigger_reason'],state['signal_date']
        p=self.day_plans[(event_id,side)]
        if p['stock_id']!=sid or p['signal_date']!=signal_date or sid in self.tick_attempts:
            raise ValueError('Midpoint identity changed or stock/day reused')
        self.tick_attempts.add(sid)
        if hasattr(self,'identity'):
            identity=self.identity(day,sid)
            if identity['status'] not in ('identified','official_trading_suspension'):
                raise ReplayDataUnavailable('Midpoint dated identity unavailable')
            if identity['status']=='identified' and identity['category']!=('ETF' if sid=='0050' else '股票'):
                raise ReplayDataUnavailable('Midpoint security is not eligible')
            self.markets[sid]=identity['market'].upper()
        if self.official_halt(day,sid):
            self.orders.append(dict(date=str(day.date()),stock_id=sid,event_id=event_id,signal_date=signal_date,
                side=side,channel='event',requested_qty=p['planned_qty'],filled_qty=0,reason=reason,failure='official_full_session_halt'))
            return 0
        if qty<p['planned_qty']:raise ValueError('Post-plan midpoint sizing shrank')
        if p['planned_qty']:self.require_prior_inputs(day,sid)
        total=0
        for channel,n in (('board',p['board_qty']),('odd',p['odd_qty'])):
            if not n:continue
            adv=float(self.volume20.at[day,sid]);date=str(day.date())
            limit=p['limit_price'] if channel=='board' else p['odd_limit']
            limits=self.feeds.get_limits(sid).get(date)
            if not limits:raise ReplayDataUnavailable('Missing midpoint legal limits')
            if channel=='board':
                high,low=(float(self.fields[k].at[day,sid]) for k in ('high','low'))
                volume=float(self.fields['volume'].at[day,sid])
            else:
                odd=self.odd_feeds.get_odd(date,sid,self.markets[sid])
                if odd is None:raise ReplayDataUnavailable('Missing midpoint odd row')
                high,low,volume=(odd.get(k) for k in ('odd_high','odd_low','odd_shares'))
            row=dict(date=date,stock_id=sid,name=self.names.get(sid,sid),side=side,event_id=event_id,
                signal_date=signal_date,reason=reason,channel=channel,requested_qty=n,limit_price=limit,
                prior_avg_amount20=float(self.amount20.at[day,sid]),prior_avg_volume20=adv,
                order_time='08:59:00' if channel=='board' else '09:00:00',expires_at='13:30:00',
                source_high=high,source_low=low,source_volume=volume,participation_limit=.01,
                execution_evidence='daily_high_low_midpoint_proxy')
            row.update(midpoint_match(high,low,volume,adv,n,side,limit,limits['lower'],limits['upper'],channel))
            filled=row['filled_qty'];price=row['reference_price']
            if filled:
                paid=self._costs(price,filled,side,sid)
                if money(self.cash+paid['cash_change'])<0:
                    row.update(filled_qty=0,failure='proceeds_below_costs_insufficient_cash');filled=0
                else:
                    self.cash_move(day,side,paid['cash_change'],stock_id=sid,event_id=event_id,channel=channel)
                    self.holdings[sid]['qty']+=filled if side=='buy' else -filled
                    mark=self.raw(day,sid);self.marks[sid]=dict(price=mark,date=date)
                    self.used[(sid,channel)]+=filled;self.day_cost+=paid['total_cost']
                    self.day_basis+=filled*(mark-price)*(1 if side=='buy' else -1)
                    trade=dict(row,**paid,qty=filled,cash_after=self.cash,remaining_shares=self.holdings[sid]['qty'],
                        day_participation=filled/volume,sequence=len(self.trades)+1)
                    trade.pop('filled_qty');trade.pop('failure',None);self.trades.append(trade)
            total+=filled;self.orders.append(row)
        return total

    def run(self):
        result=super().run()
        result['settings'].update(execution='daily_high_low_midpoint_proxy_v1',
            price_formula='(high+low)/2',board_capacity='min(day_volume,prior_adv20)*0.01',
            odd_execution_evidence='daily_high_low_midpoint_proxy',opening_auction_inferred=False,
            actual_fill_verified=False,live_qualified=False)
        return result


class MidpointStock(MidpointOrders,HighReturnReplay):
    pass


class MidpointBenchmark(MidpointOrders,factory(StrictResidualBenchmark)):
    pass
