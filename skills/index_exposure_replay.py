"""Single-ETF cash account with prior-session plans and immutable delayed orders."""
from bisect import bisect_left
from copy import deepcopy
from decimal import Decimal,ROUND_HALF_UP,ROUND_FLOOR,ROUND_CEILING
import math

SID='00631L'
SPLIT_DAY='2026-03-31'
SPLIT_KNOWN='2026-03-30'

def costs(price,qty,side,slippage):
    if side not in ('buy','sell') or type(qty) is not int or qty<=0 or qty%1000:
        raise ValueError('ETF fills require a positive whole lot and side')
    p=Decimal(str(price));s=Decimal(str(slippage))
    if not p.is_finite() or p<=0 or not s.is_finite() or s<0:raise ValueError('Invalid costs')
    gross=p*qty
    fee=max(Decimal(20),(gross*Decimal('.001425')).quantize(Decimal(1),rounding=ROUND_HALF_UP))
    tax=(gross*Decimal('.001')).quantize(Decimal(1),rounding=ROUND_FLOOR) if side=='sell' else Decimal(0)
    slip=(gross*s).quantize(Decimal(1),rounding=ROUND_CEILING)
    total=fee+tax+slip
    return dict(gross=float(gross),commission=float(fee),tax=float(tax),slippage=float(slip),
        total_cost=float(total),cash_change=float((-gross if side=='buy' else gross)-total))

def affordable(maximum,price,budget,slip):
    lo,hi=0,maximum//1000
    while lo<hi:
        mid=(lo+hi+1)//2
        if -costs(price,mid*1000,'buy',slip)['cash_change']<=budget+1e-8:lo=mid
        else:hi=mid-1
    return lo*1000

class IndexExposureReplay:
    def __init__(self,inputs,window,mask):
        if window not in (0,180,200,220) or type(mask) is not int or mask not in range(8):
            raise ValueError('Use preregistered ETF windows and stress masks')
        self.data=inputs;self.window=window;self.mask=mask
        self.slip=.009 if mask&1 else .0045
        self.days=inputs['days'];self.calendar=inputs['calendar']
        self.quotes=inputs['quotes'];self.signals=inputs['signals']
        self.signal_dates=sorted(self.signals)
        self.cash=1_000_000.;self.qty=0;self.mark=0.;self.mark_date=None
        self.pending=None;self.exit_latched=False;self.retry=False;self.previous_trend=False
        self.decisions=[];self.plans=[];self.actions=[];self.orders=[];self.trades=[]
        self.daily=[];self.holdings=[]
        self.cash_ledger=[dict(date=self.days[0],kind='initial_deposit',cash_change=1_000_000.,cash_after=1_000_000.)]
        self.peak=1_000_000.

    def context(self,day):
        i=bisect_left(self.calendar,day)
        if i==0:raise ValueError('No previous market day')
        signal=self.calendar[i-1];n=bisect_left(self.signal_dates,day)
        dates=self.signal_dates[max(0,n-self.window):n] if self.window else self.signal_dates[max(0,n-1):n]
        if not dates or (self.window and len(dates)!=self.window):raise ValueError('Signal warmup missing')
        close=self.signals[dates[-1]];mean=math.fsum(self.signals[d] for d in dates)/len(dates)
        prior=next((d for d in reversed(self.calendar[:i]) if d in self.quotes),None)
        if prior is None:raise ValueError('No prior ETF quote')
        ref=self.quotes[prior]['close']/(22 if prior<SPLIT_DAY<=day else 1)
        history=self.calendar[max(0,i-20):i]
        volumes=[self.quotes.get(d,{}).get('volume',0)*(22 if d<SPLIT_DAY<=day else 1) for d in history]
        return dict(signal_date=signal,signal_price_date=dates[-1],signal_close=close,sma=mean,
            signal_first_observation=dates[0],observations=len(dates),trend=not self.window or close>mean,
            reference_date=prior,reference_price=ref,prior_avg_volume20=math.fsum(volumes)/20 if len(history)==20 else None,
            month_first=day[:7]!=signal[:7])

    def split(self,day):
        if day!=SPLIT_DAY:return
        action=dict(date=day,known_date=SPLIT_KNOWN,stock_id=SID,kind='split',multiplier=22,
            old_qty=self.qty,new_qty=self.qty*22,old_mark=self.mark,new_mark=self.mark/22,
            pending_before=deepcopy(self.pending))
        self.qty*=22;self.mark/=22
        if self.pending:
            self.pending['qty']*=22
            self.pending['reference_price']/=22
            self.pending['split_adjustments'].append(dict(date=day,multiplier=22))
        action['pending_after']=deepcopy(self.pending);self.actions.append(action)

    def plan(self,day,index,ctx,full_exit):
        nav=self.cash+self.qty*ctx['reference_price']
        if full_exit:
            side,qty,reason='sell',self.qty,'trend_off'
        else:
            budget=.75*nav-self.qty*ctx['reference_price']
            if budget>=0:
                side='buy';maximum=int(budget/ctx['reference_price'])//1000*1000
                qty=affordable(maximum,ctx['reference_price'],min(budget,self.cash),self.slip)
            else:
                side='sell';desired=int(.75*nav/ctx['reference_price'])//1000*1000
                qty=self.qty-desired
            reason='initial_or_monthly_or_reentry'
        order=dict(order_id=len(self.plans)+1,date=day,signal_date=ctx['signal_date'],side=side,qty=qty,
            original_qty=qty,reference_price=ctx['reference_price'],original_reference_price=ctx['reference_price'],
            prior_nav=nav,available_cash=self.cash,prior_qty=self.qty,reason=reason,
            due_index=index+(bool(self.mask&2) if side=='buy' else bool(self.mask&4)),
            split_adjustments=[],context=deepcopy(ctx))
        self.plans.append(deepcopy(order));return order if qty else None

    def execute(self,day,ctx):
        p=self.pending;quote=self.quotes.get(day)
        row=dict(date=day,stock_id=SID,name='元大台灣50正2',event_id='index_exposure',
            order_id=p['order_id'],signal_date=p['signal_date'],side=p['side'],reason=p['reason'],
            requested_qty=p['qty'],filled_qty=0,channel='board',failure=None,
            prior_avg_volume20=ctx['prior_avg_volume20'],price_limit_source='derived_domestic_2x_rule',
            capacity_qty=0,day_volume=quote['volume'] if quote else 0,
            reference_price=quote['close'] if quote else None)
        if not quote or not quote['volume']:row['failure']='missing_or_zero_quote_volume'
        elif quote['high']==quote['low']:row['failure']='single_price_session'
        elif not quote.get('lower') or not quote.get('upper'):row['failure']='missing_price_limits'
        elif p['side']=='buy' and quote['close']>=quote['upper']-1e-8:row['failure']='at_upper_limit'
        elif p['side']=='sell' and quote['close']<=quote['lower']+1e-8:row['failure']='at_lower_limit'
        elif not ctx['prior_avg_volume20']:row['failure']='missing_adv20'
        else:
            cap=int(min(quote['volume'],ctx['prior_avg_volume20'])*.01)//1000*1000
            row['capacity_qty']=cap
            fill=min(p['qty'],cap)
            if p['side']=='buy':fill=affordable(fill,quote['close'],self.cash,self.slip)
            else:fill=min(fill,self.qty)
            if fill:
                paid=costs(quote['close'],fill,p['side'],self.slip)
                self.cash=round(self.cash+paid['cash_change'],2)
                self.qty+=fill if p['side']=='buy' else -fill
                trade=dict(row,**paid,qty=fill,sequence=len(self.trades)+1,cash_after=self.cash,
                    remaining_shares=self.qty,participation_limit=.01,day_participation=fill/quote['volume'])
                trade.pop('filled_qty');trade.pop('failure');self.trades.append(trade)
                self.cash_ledger.append(dict(date=day,kind=p['side'],cash_change=paid['cash_change'],
                    cash_after=self.cash,stock_id=SID,order_id=p['order_id']))
                row['filled_qty']=fill
            if fill<p['qty']:row['failure']='partial_capacity_or_cash' if fill else 'capacity_or_cash_zero'
        self.orders.append(row)
        # A submitted instruction ends today. A remainder requires tomorrow's fresh plan.
        self.retry=row['filled_qty']<p['qty'];self.pending=None
        if self.exit_latched and self.qty==0:self.exit_latched=False;self.retry=False
        return row

    def run(self):
        for i,day in enumerate(self.days):
            opening_nav=self.daily[-1]['nav'] if self.daily else 1_000_000.
            self.split(day);ctx=self.context(day)
            decision=dict(date=day,context=deepcopy(ctx),opening_cash=self.cash,opening_qty=self.qty,
                pending_before=deepcopy(self.pending),exit_latched_before=self.exit_latched,status='hold')
            if self.pending and self.pending['side']=='buy' and not ctx['trend']:
                decision['cancelled_order']=deepcopy(self.pending);self.pending=None;self.retry=False
            if not ctx['trend'] and self.qty and not self.exit_latched:
                self.exit_latched=True
                if self.pending:
                    decision['cancelled_order']=deepcopy(self.pending);self.pending=None
            if self.pending is None:
                rebalance=ctx['trend'] and (i==0 or self.qty==0 or ctx['month_first'] or not self.previous_trend or self.retry)
                if self.exit_latched or rebalance:
                    self.pending=self.plan(day,i,ctx,self.exit_latched)
                    decision['created_plan']=deepcopy(self.plans[-1]);self.retry=False
            if self.pending:
                decision['executing_instruction']=deepcopy(self.pending)
                if i<self.pending['due_index']:decision['status']='waiting_extra_delay'
                else:
                    row=self.execute(day,ctx)
                    decision['status']='filled' if row['filled_qty'] else 'unfilled'
                    decision['execution']=deepcopy(row)
            if day in self.quotes:self.mark=self.quotes[day]['close'];self.mark_date=day
            value=self.qty*self.mark;nav=round(self.cash+value,2);self.peak=max(self.peak,nav)
            if self.cash<0 or self.qty<0:raise ValueError('Unfunded ETF account')
            if self.qty:
                self.holdings.append(dict(date=day,stock_id=SID,name='元大台灣50正2',event_id='index_exposure',
                    qty=self.qty,price=self.mark,market_value=value,mark_date=self.mark_date,stale=self.mark_date!=day))
            self.daily.append(dict(date=day,opening_nav=opening_nav,nav=nav,cash=self.cash,market_value=value,
                receivable=0.,total_return=nav/1_000_000.-1,daily_return=nav/opening_nav-1,
                drawdown=nav/self.peak-1,stale_holdings=int(self.qty>0 and self.mark_date!=day),holdings=int(self.qty>0)))
            decision.update(closing_cash=self.cash,closing_qty=self.qty,pending_after=deepcopy(self.pending),
                            exit_latched_after=self.exit_latched)
            self.decisions.append(decision);self.previous_trend=ctx['trend']
        return dict(settings=dict(initial_cash=1_000_000,commission=.001425,minimum_fee=20.,slippage=self.slip,
            participation=.01,odd_participation=.05,execution_policy='board_only',slots=1,benchmark=False,
            instrument_type='domestic_leveraged_ETF',stock_id=SID,etf_sell_tax=.001,window=self.window,
            target_weight=.75,idle_capital='cash',derived_price_limits=True),
            daily=self.daily,trades=self.trades,orders=self.orders,cash_ledger=self.cash_ledger,
            holdings=self.holdings,corporate_actions=self.actions,receivables=[],cohorts=[])
