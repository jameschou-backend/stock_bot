"""Independent source, plan, price, capacity and cost checks for HL2 proxies."""
from copy import deepcopy
import math
import pandas as pd
from skills.mixed_odd_audit import audit_mixed_execution
from skills.execution_stress import StressOrder


def audit_midpoint(account,ticks,odds,markets,quotes,calendar,corporate,feeds):
    settings=account['settings']
    if (settings.get('price_formula')!='(high+low)/2' or settings.get('actual_fill_verified') is not False
            or settings.get('opening_auction_inferred') is not False or settings.get('live_qualified') is not False
            or settings.get('odd_execution_evidence')!='daily_high_low_midpoint_proxy'
            or settings['participation']!=.01 or settings['slippage']!=.0045):
        raise ValueError('Midpoint assumptions or qualification changed')
    # Reuse the sealed prior-only sizing audit without passing any fills to its
    # opening execution checker. Only the window annotation is normalized.
    plans=deepcopy(account['tick_plans'])
    for p in plans:
        if p['expires_at']!='13:30:00':raise ValueError('Midpoint window differs')
        p['expires_at']='13:25:00'
    skeleton=dict(account,tick_plans=plans,orders=[],trades=[],settings=dict(settings,
        odd_execution_evidence='daily_envelope_estimate'))
    audit_mixed_execution(skeleton,ticks,odds,markets,quotes,calendar,corporate,feeds)
    indexed=quotes.set_index(['date','stock_id'])
    volumes=quotes.pivot(index='date',columns='stock_id',values='volume').reindex(calendar)
    closes=quotes.pivot(index='date',columns='stock_id',values='close').reindex(calendar)
    adv=volumes.rolling(20,min_periods=20).mean().shift(1)
    amount=(volumes*closes).rolling(20,min_periods=20).mean().shift(1)
    frozen={(p['date'],p['event_id'],p['side']):p for p in account['tick_plans']}
    orders={};fills={};stockdays=set();cash_by_day={}
    for i,daily in enumerate(account['daily']):
        cash=account['daily'][i-1]['cash'] if i else settings['initial_cash']
        cash+=sum(r['cash_change'] for r in account['cash_ledger'] if r['date']==daily['date']
                  and r['kind'] not in ('initial_deposit','buy','sell'))
        cash_by_day[daily['date']]=cash
    cost=StressOrder();cost.stress_slippage=.0045
    for row in account['orders']:
        if row['channel'] not in ('board','odd') or not row['requested_qty']:continue
        key=(row['date'],row['event_id'],row['side']);child=(*key,row['channel']);p=frozen[key]
        sid=row['stock_id'];day=pd.Timestamp(row['date']);channel=row['channel']
        stockday=(row['date'],sid,channel)
        if child in orders or stockday in stockdays:raise ValueError('Midpoint daily capacity reused')
        stockdays.add(stockday)
        limit=p['limit_price'] if channel=='board' else p['odd_limit']
        if (row['requested_qty']!=p[channel+'_qty'] or row['limit_price']!=limit
                or row['signal_date']!=p['signal_date'] or row['signal_date']>=row['date']
                or row['stock_id']!=p['stock_id'] or row['expires_at']!='13:30:00'
                or row['execution_evidence']!='daily_high_low_midpoint_proxy'
                or row['prior_avg_volume20']!=adv.at[day,sid]
                or row['prior_avg_amount20']!=amount.at[day,sid]):
            raise ValueError('Midpoint order differs from frozen plan or prior inputs')
        limits=feeds.get_limits(sid)[row['date']]
        if channel=='board':
            source=indexed.loc[(day,sid)];high,low,volume=(float(source[k]) for k in ('high','low','volume'))
        else:
            source=odds.get_odd(row['date'],sid,markets.get((sid,row['date']),markets.get(sid)))
            high,low,volume=(source[k] for k in ('odd_high','odd_low','odd_shares'))
        if (row['source_high'],row['source_low'],row['source_volume'])!=(high,low,volume):
            raise ValueError('Midpoint price/volume evidence differs')
        price=(high+low)/2 if volume else None
        if volume and not limits['lower']-1e-8<=low<=high<=limits['upper']+1e-8:
            raise ValueError('Midpoint source outside legal price range')
        eligible=volume>0 and (price<min(limit,limits['upper'])-1e-8 if row['side']=='buy'
                               else price>max(limit,limits['lower'])+1e-8)
        capacity=math.floor((min(volume,adv.at[day,sid]) if channel=='board' else volume)*.01) if eligible else 0
        if channel=='board':capacity=capacity//1000*1000
        fill=min(row['requested_qty'],capacity)
        if fill:
            change=cost._costs(price,fill,row['side'],sid)['cash_change']
            if round(cash_by_day[row['date']]+change,2)<0:
                if row.get('failure')!='proceeds_below_costs_insufficient_cash':
                    raise ValueError('Midpoint order overspent cash')
                fill=0
            else:cash_by_day[row['date']]+=change
        if row['reference_price']!=price or row['capacity_qty']!=capacity or row['filled_qty']!=fill:
            raise ValueError('Midpoint price or volume arithmetic differs')
        orders[child]=row
    for trade in account['trades']:
        key=(trade['date'],trade['event_id'],trade['side'],trade['channel']);row=orders.get(key)
        if not row or trade['reference_price']!=row['reference_price']:
            raise ValueError('Midpoint trade lacks source-priced order')
        expected=cost._costs(row['reference_price'],trade['qty'],trade['side'],trade['stock_id'])
        if any(trade[k]!=v for k,v in expected.items()):raise ValueError('Midpoint costs differ')
        fills[key]=fills.get(key,0)+trade['qty']
    if any(fills.get(k,0)!=r['filled_qty'] for k,r in orders.items()):
        raise ValueError('Midpoint trade quantities differ')
    if any(abs(cash_by_day[r['date']]-r['cash'])>.02 for r in account['daily']):
        raise ValueError('Midpoint order cash differs from account')
    return dict(prior_only_plans_rebuilt=True,midpoint_prices_rebuilt=True,daily_capacity_rebuilt=True,
                separate_channel_costs_rebuilt=True,price_proxy_only=True,actual_fill_verified=False)
