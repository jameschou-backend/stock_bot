"""Causal first-volume-bar events and separately computed paired price outcomes."""
import numpy as np
import pandas as pd
from skills.surge_anatomy import features as base_features
from skills.theme_chips import align
from skills.stock_launch import TARGETS

EXCLUDED = {*TARGETS, '3491'}
START, END = '2022-01-03', '2026-09-09'


def features(close, raw, volume, ohlc, companies):
    base = base_features(close, raw, volume, companies)
    ids = base['eligible'].columns
    c = close[ids].where(np.isfinite(close[ids]) & close[ids].gt(0))
    v = volume[ids].where(np.isfinite(volume[ids]) & volume[ids].gt(0))
    q = {}
    for key in ('open','high','low','close','volume'):
        q[key] = ohlc[key].reindex(index=c.index, columns=ids)
    valid = pd.DataFrame(True,index=c.index,columns=ids)
    for frame in q.values():
        valid &= np.isfinite(frame) & frame.gt(0)
    valid &= (q['high'].ge(q['open']) & q['high'].ge(q['close']) &
              q['low'].le(q['open']) & q['low'].le(q['close']) & q['high'].gt(q['low']) &
              q['close'].sub(raw[ids]).abs().le(1e-8) & q['volume'].eq(volume[ids]))
    location = ((q['close']-q['low'])/(q['high']-q['low'])).where(valid)
    ratio = v/v.shift(1).rolling(20,min_periods=20).mean()
    ret = c/c.shift(1)-1
    burst_known = valid & ratio.notna() & ret.notna()
    burst = (ratio.ge(2) & ret.ge(.02) & q['close'].gt(q['open']) & location.ge(.65)) & burst_known
    prior_burst = burst.shift(1).rolling(10,min_periods=10).sum()
    prior_known = burst_known.shift(1).rolling(10,min_periods=10).sum().eq(10)
    high = c.shift(1).rolling(20,min_periods=20).max()
    low = c.shift(1).rolling(20,min_periods=20).min()
    first = base['eligible'] & burst & prior_known & prior_burst.eq(0) & (high/low-1).le(.15)
    strength = base['numeric']['relative20']
    breakout = base['eligible'] & c.gt(high) & c.gt(c.rolling(60,min_periods=60).mean()) & strength.gt(0)
    known = (c.notna().rolling(120,min_periods=120).sum().eq(120) &
             base['adv20'].notna() & strength.notna())
    return dict(first=first, breakout=breakout, breakout_known=known, burst_known=burst_known,
        adv=base['adv20'], shares=v.rolling(20,min_periods=20).mean(),
        volume_ratio=ratio, return1=ret, close_location=location, range20=high/low-1,
        relative20=strength, eligible=base['eligible'])


def episodes(computed, *, start=START, end=END):
    first=computed['first'];days=first.index;rows=[];last={}
    for pos,col in np.argwhere(first.to_numpy()):
        day=days[pos];sid=first.columns[col]
        if not pd.Timestamp(start)<=day<=pd.Timestamp(end) or pos-last.get(sid,-9999)<21:
            continue
        last[sid]=pos
        row=dict(event_id=f'first-{day.date()}-{sid}',stock_id=sid,signal_date=str(day.date()),
            named_case=sid in EXCLUDED)
        for k in ('adv','shares','volume_ratio','return1','close_location','range20','relative20'):
            row[k]=float(computed[k].iat[pos,col])
        row['already_breakout']=bool(computed['breakout'].iat[pos,col])
        rows.append(row)
    return pd.DataFrame(rows,columns=['event_id','stock_id','signal_date','named_case','adv','shares',
        'volume_ratio','return1','close_location','range20','relative20','already_breakout'])


def wait_for_breakout(events, computed, *, end=END):
    days=computed['breakout'].index;positions={str(d.date()):i for i,d in enumerate(days)}
    rows=[]
    for e in events.to_dict('records'):
        pos=positions[e['signal_date']];sid=e['stock_id']
        state='not_triggered';signal=None;lag=None
        for i in range(pos,pos+21):
            if i>=len(days) or days[i]>pd.Timestamp(end):
                state='unmatured';break
            if not computed['breakout_known'].at[days[i],sid]:
                state='missing_price_before_breakout';break
            if computed['breakout'].at[days[i],sid]:
                state='triggered';signal=str(days[i].date());lag=i-pos;break
        rows.append(dict(event_id=e['event_id'],wait_state=state,wait_signal_date=signal,wait_sessions=lag))
    return pd.DataFrame(rows,columns=['event_id','wait_state','wait_signal_date','wait_sessions'])


def with_chips(events, weekly, lag):
    rows=align(events,weekly,lag)
    rows['concentration']=(rows.large_pct_delta4.ge(.005)&rows.small_pct_delta4.lt(0)&
        rows.large_units_delta4.gt(0)).astype('boolean').where(rows.chip_known,pd.NA)
    return rows


def entries(events, waits, computed, *, arm, end=END):
    if arm not in ('first','wait'):raise ValueError('Unknown registered entry arm')
    rows=events.merge(waits,on='event_id',validate='one_to_one')
    days=computed['breakout'].index;out=[]
    for e in rows.to_dict('records'):
        if e['named_case'] or pd.isna(e['concentration']) or not e['concentration']:
            continue
        date=e['signal_date'] if arm=='first' else e['wait_signal_date']
        if date is None or pd.isna(date):continue
        pos=days.get_loc(pd.Timestamp(date))
        if pos+1>=len(days) or days[pos+1]>pd.Timestamp(end):continue
        sid=e['stock_id'];day=days[pos]
        liq=dict(as_of=date,complete_20_sessions=True,observations=20,
            adv20_shares=float(computed['shares'].at[day,sid]),
            mean_turnover20_twd=float(computed['adv'].at[day,sid]))
        out.append(dict(event_id=e['event_id'],members=[sid],stock_id=sid,
            signal_date=date,entry_date=str(days[pos+1].date()),
            priority=liq['mean_turnover20_twd'],feature_cutoff_date=date,group_cutoff_date=date,
            membership_point_in_time=False,membership_snapshot_date='2026-09-27',
            liquidity_at_signal=liq,liquidity_before_entry=dict(liq),
            launch_date=e['signal_date'],chip_available_date=e['available_date'],
            chip_observed_date=e['observed_date']))
    return sorted(out,key=lambda r:(r['signal_date'],-r['priority'],r['stock_id']))


def _path(close, quality, sid, begin, end):
    if begin<0 or end>=len(close):return None
    result=[]
    for matrix in (close,quality):
        a=matrix.iloc[begin:end+1][[sid,'0050']].to_numpy(float)
        if (not np.isfinite(a).all() or (a<=0).any() or
                (np.abs(a[1:]/a[:-1]-1)>.15).any()):return None
        result.append(a)
    if (np.abs((result[0][-1]/result[0][0]-1)-(result[1][-1]/result[1][0]-1))>.05).any():
        return None
    return result


def outcomes(events, waits, close, quality):
    """Future outcomes never enter event detection, chip selection or orders."""
    rows=[];days=close.index;positions={str(d.date()):i for i,d in enumerate(days)}
    for e in events.merge(waits,on='event_id',validate='one_to_one').to_dict('records'):
        pos=positions[e['signal_date']];sid=e['stock_id']
        for horizon in (20,60):
            end=pos+horizon+1;exit_date=str(days[end].date()) if end<len(days) else None
            r=dict(e,horizon=horizon,exit_date=exit_date,
                phase='discovery' if exit_date and exit_date<='2024-12-31' else
                'replication' if e['signal_date']>='2025-01-01' else 'boundary',
                first_return=None,wait_return=None,benchmark_return=None,first_excess=None,wait_excess=None,
                first_surge=None,wait_surge=None,paired_difference=None,mfe=None,mae=None,
                false_start5=None,wait_bought=False,outcome_reason='missing_or_unmatured_price')
            path=_path(close,quality,sid,pos+1,end)
            false_path=_path(close,quality,sid,pos-1,pos+5)
            if false_path is not None:
                r['false_start5']=bool((false_path[0][2:,0]<false_path[0][0,0]).any())
            if path is None:rows.append(r);continue
            p=path[0];ret=float(p[-1,0]/p[0,0]-1);benchmark=float(p[-1,1]/p[0,1]-1)
            a,b=(.30,.20) if horizon==20 else (.50,.30)
            r.update(first_return=ret,benchmark_return=benchmark,first_excess=ret-benchmark,
                first_surge=ret>=a and ret-benchmark>=b,mfe=float((p[:,0]/p[0,0]-1).max()),
                mae=float((p[:,0]/p[0,0]-1).min()),outcome_reason='waiting_unresolved')
            if e['wait_state']=='not_triggered':wait_return=0.
            elif e['wait_state']=='triggered':
                begin=positions[e['wait_signal_date']]+1
                if begin>=end:wait_return=0.
                else:
                    waited=_path(close,quality,sid,begin,end)
                    if waited is None:rows.append(r);continue
                    wait_return=float(waited[0][-1,0]/waited[0][0,0]-1);r['wait_bought']=True
            else:rows.append(r);continue
            r.update(wait_return=wait_return,wait_excess=wait_return-benchmark,
                wait_surge=wait_return>=a and wait_return-benchmark>=b,
                paired_difference=ret-wait_return,outcome_reason='known')
            rows.append(r)
    return pd.DataFrame(rows)


def summarize(frame):
    rows=[]
    for (lag,horizon,phase),part in frame[~frame.named_case].groupby(['lag','horizon','phase'],sort=True):
        for group,mask in [('all_events',pd.Series(True,index=part.index)),('chip_known',part.chip_known),
                ('concentrated',part.concentration.eq(True).fillna(False)),
                ('not_concentrated',part.concentration.eq(False).fillna(False))]:
            f=part[mask];paired=f.dropna(subset=['paired_difference'])
            row=dict(lag=int(lag),horizon=int(horizon),phase=phase,group=group,events=len(f),
                stocks=f.stock_id.nunique(),dates=f.signal_date.nunique(),
                first_known=int(f.first_return.notna().sum()),paired_known=len(paired),
                unknown=int(f.paired_difference.isna().sum()),first_surge=int(f.first_surge.eq(True).sum()),
                wait_surge=int(f.wait_surge.eq(True).sum()),first_mean=f.first_return.mean(),
                first_median=f.first_return.median(),wait_mean=f.wait_return.mean(),
                paired_first_mean=paired.first_return.mean(),paired_wait_mean=paired.wait_return.mean(),
                paired_difference_mean=paired.paired_difference.mean(),paired_difference_median=paired.paired_difference.median(),
                paired_win_rate=paired.paired_difference.gt(0).mean(),benchmark_mean=paired.benchmark_return.mean(),
                first_excess=paired.first_excess.mean(),wait_excess=paired.wait_excess.mean(),
                first_beat_benchmark=paired.first_excess.gt(0).mean(),wait_beat_benchmark=paired.wait_excess.gt(0).mean(),
                same_day=int(f.wait_sessions.eq(0).sum()),later=int(f.wait_sessions.gt(0).sum()),
                never=int(f.wait_state.eq('not_triggered').sum()),wait_unknown=int((~f.wait_state.isin(['triggered','not_triggered'])).sum()),
                later_median=f.loc[f.wait_sessions.gt(0),'wait_sessions'].median(),
                false_start_known=int(f.false_start5.notna().sum()),false_start5=f.false_start5.dropna().astype(float).mean(),
                mean_mae=f.mae.mean(),mean_mfe=f.mfe.mean())
            # Same-day entries must produce identical paired returns.
            if not paired.loc[paired.wait_sessions.eq(0),'paired_difference'].eq(0).all():
                raise ValueError('Same-day arms have different price outcomes')
            rows.append(row)
    return rows
