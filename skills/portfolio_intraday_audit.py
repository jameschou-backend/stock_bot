"""Rebuild stop signals/capacities from source tapes, not engine fill records."""
from collections import defaultdict
import math
import pandas as pd


def audit_sources(account,quotes,calendar,actions,ticks,odds,feeds,markets):
    q=quotes.copy();q['date']=pd.to_datetime(q.date)
    indexed=q.set_index(['date','stock_id'])
    volumes=q.pivot(index='date',columns='stock_id',values='volume').reindex(calendar)
    adv=volumes.rolling(20,min_periods=20).mean().shift(1)
    peaks,pending={},{}
    checked=0
    for row in account['intraday_evidence']:
        sid,eid,date=row['stock_id'],row['event_id'],row['date'];day=pd.Timestamp(date)
        if eid not in peaks:
            buys=[t for t in account['trades'] if t['event_id']==eid and t['side']=='buy']
            entry=pd.Timestamp(buys[0]['date'])
            if entry>=day:raise ValueError('Stop armed on entry session')
            peaks[eid]=max(float(indexed.loc[(entry,sid),'close']),*[t['reference_price'] for t in buys])
        changes=actions.loc[actions.stock_id.eq(sid)&pd.to_datetime(actions.event_date).eq(day)]
        for a in changes.itertuples():peaks[eid]*=float(a.ratio)
        peak=peaks[eid]
        if not math.isclose(peak,row['prior_peak'],rel_tol=0,abs_tol=1e-8):
            raise ValueError('Prior peak differs from held source history')
        if row['status']=='official_full_session_halt':continue
        source=indexed.loc[(day,sid)]
        if (row['high'],row['low'])!=(source.high,source.low):
            raise ValueError('Stop certificate differs from raw price')
        peaks[eid]=max(peak,float(source.high))
        if row['status']=='no_crossing_under_any_bar_order':
            if not source.low>max(peak,source.high)*.85+1e-10:
                raise ValueError('False no-crossing certificate')
            continue
        tape,digest=ticks.get(sid,date,markets[sid])
        if digest!=row['tape_sha256']:raise ValueError('Stop tape digest differs')
        limits=feeds.get_limits(sid)[date]
        tape=tape[tape.time.ge(pd.Timedelta('09:00:00'))&tape.time.lt(pd.Timedelta('13:25:00'))&tape.shares.gt(0)]
        trigger=pending.get(eid)
        quantity=row['board_position_qty'];remaining=quantity;used=volume=0
        expected=[]
        for time,group in tape.groupby('time',sort=True):
            at=(pd.Timestamp(date,tz='Asia/Taipei')+time).isoformat()
            if trigger is None:
                low,high=float(group.price.min()),float(group.price.max())
                if low<=peak*.85+1e-10:
                    trigger=dict(at=at,peak=peak,threshold=peak*.85,observed_low=low,fill_price=None)
                    pending[eid]=trigger
                    if quantity==0:break
                    continue
                if low<=max(peak,high)*.85+1e-10:
                    raise ValueError('Ambiguous source timestamp group')
                peak=max(peak,high)
                continue
            eligible=group[group.price.gt(limits['lower'])]
            volume+=int(eligible.shares.sum())
            capacity=int(min(volume,adv.at[day,sid])*.01)//1000*1000
            fill=min(remaining,max(0,capacity-used))
            if fill and len(eligible):
                expected.append((at,fill,float(eligible.price.min())))
                used+=fill;remaining-=fill
        if trigger!=row['trace']['trigger']:
            raise ValueError('First stop signal differs from independent tick scan')
        actual=[(t['fill_time'],t['qty'],t['reference_price']) for t in account['trades']
            if t['event_id']==eid and t['date']==date and t['channel']=='board' and t['reason']=='intraday_peak_stop15']
        if actual!=expected:raise ValueError('Post-trigger board fills differ from source capacity')
        checked+=1
    # Independently verify every HL2 buy/ordinary sell and delayed odd stop.
    totals=defaultdict(int)
    for trade in account['trades']:
        if trade['channel']=='board' and trade['reason']=='intraday_peak_stop15':continue
        sid,date=trade['stock_id'],trade['date'];day=pd.Timestamp(date)
        if trade['channel']=='board':
            source=indexed.loc[(day,sid)];high,low,vol=map(float,(source.high,source.low,source.volume))
            capacity=int(min(vol,adv.at[day,sid])*.01)//1000*1000
        else:
            source=odds.get_odd(date,sid,markets[sid]);high,low,vol=(source[k] for k in ('odd_high','odd_low','odd_shares'))
            capacity=int(vol*.01)
        if trade['reference_price']!=(high+low)/2:
            raise ValueError('HL2 trade differs from its own market source')
        totals[(sid,date,trade['channel'])]+=trade['qty']
        if totals[(sid,date,trade['channel'])]>capacity:
            raise ValueError('HL2 trade exceeded independent market capacity')
        limits=feeds.get_limits(sid)[date];price=trade['reference_price']
        if not limits['lower']<=low<=high<=limits['upper']:
            raise ValueError('HL2 source exceeds dated legal limits')
        if (trade['side']=='buy' and price>=min(trade['limit_price'],limits['upper'])-1e-8
                or trade['side']=='sell' and price<=max(trade['limit_price'],limits['lower'])+1e-8):
            raise ValueError('HL2 trade was not eligible under precommitted limit')
    return dict(source_stop_tapes_rebuilt=checked,prior_peaks_rebuilt=True,
        all_trade_prices_and_source_capacity_rebuilt=True)
