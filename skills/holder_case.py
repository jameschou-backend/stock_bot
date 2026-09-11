"""Observed holder tiers and explicitly lagged single-stock diagnostics."""
import re
import numpy as np
import pandas as pd
from skills.chip_research import holder_rows


def aggregate(raw):
    checked=holder_rows(raw).set_index(['date','stock_id'])
    records=[]
    for (day,sid),group in raw.groupby(['date','stock_id']):
        valid=bool(checked.loc[(pd.Timestamp(day),sid),'valid'])
        levels={}
        total=group[group.HoldingSharesLevel.str.lower().eq('total')]
        for row in group.to_dict('records'):
            label=str(row['HoldingSharesLevel']).replace(',','').strip().lower()
            match=re.fullmatch(r'(\d+)-(\d+)',label) or re.fullmatch(r'(?:more than|over)\s*(\d+)',label)
            if match:
                levels[int(match.group(1))]=row
        quantities=[float(r[k]) for r in levels.values() for k in ('people','unit')]
        valid=valid and len(total)==1 and all(np.isfinite(x) and x>=0 and x==int(x) for x in quantities)
        record=dict(date=pd.Timestamp(day),stock_id=sid,valid=valid)
        if valid:
            denominator=float(total.iloc[0].unit)
            people=float(total.iloc[0].people)
            valid=denominator>0 and abs(sum(r['unit'] for r in levels.values())/denominator-1)<=.005
            valid=valid and sum(r['people'] for r in levels.values())==people
            valid=valid and all(abs(r['percent']-100*r['unit']/denominator)<=.011 for r in levels.values())
            record['valid']=valid
            if valid:
                record.update(total_shares=denominator,total_people=people)
                for name,rows in [('small',[r for low,r in levels.items() if low<=50001]),
                                  ('large',[r for low,r in levels.items() if low>=1000001])]:
                    shares=sum(r['unit'] for r in rows)
                    record.update({name+'_pct':100*shares/denominator,name+'_shares':shares,
                                   name+'_people':sum(r['people'] for r in rows),
                                   name+'_reported_pct':sum(r['percent'] for r in rows)})
        records.append(record)
    frame=pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    for key in ('small_pct','large_pct','small_people','large_people','small_shares','large_shares','total_shares','total_people'):
        if key not in frame:
            frame[key]=np.nan
    return frame


def features(weekly,adjusted,benchmark,lag=7):
    """Weekly observation date is never treated as its first publication time."""
    if lag not in (7,14):
        raise ValueError('Only preregistered publication lags')
    result=weekly.copy()
    days=benchmark.index
    stock=adjusted.reindex(days)
    good=result.valid.rolling(5,min_periods=5).sum().eq(5)
    good &= (result.date-result.date.shift(4)).dt.days.between(21,35)
    for who in ('small','large'):
        result[who+'_change4']=result[who+'_pct'].diff(4).where(good)
    result['joint']=((result.large_change4>0)&(result.small_change4<0)).astype(float).where(good)
    current=stock.reindex(pd.DatetimeIndex(result.date)).to_numpy()
    result['same_period_return4']=pd.Series(current,index=result.index)/pd.Series(current,index=result.index).shift(4)-1
    result['same_period_return4']=result.same_period_return4.where(good)
    indices=days.searchsorted(pd.DatetimeIndex(result.date)+pd.Timedelta(days=lag),side='right')
    result['available_entry_date']=[days[i] if i<len(days) else pd.NaT for i in indices]
    for horizon in (20,40,60):
        own,excess=[],[]
        for i in indices:
            if i+horizon>=len(days):
                own.append(np.nan);excess.append(np.nan);continue
            a,b=stock.iloc[i],stock.iloc[i+horizon]
            c,d=benchmark.iloc[i],benchmark.iloc[i+horizon]
            if not all(np.isfinite(v) and v>0 for v in (a,b,c,d)):
                own.append(np.nan);excess.append(np.nan);continue
            ret=b/a-1
            own.append(ret);excess.append(ret-(d/c-1))
        result['forward'+str(horizon)]=own
        result['excess'+str(horizon)]=excess
    return result


def diagnostics(frame):
    result={}
    for target in ('same_period_return4','forward20','forward40','forward60','excess20','excess40','excess60'):
        row={}
        for feature in ('large_change4','small_change4'):
            pair=frame[[feature,target]].dropna()
            rho=pair[feature].rank().corr(pair[target].rank()) if len(pair)>=3 and pair[feature].nunique()>1 and pair[target].nunique()>1 else None
            row[feature]=dict(n=len(pair),spearman=None if rho is None or not np.isfinite(rho) else float(rho))
        known=frame[frame.joint.notna() & frame[target].notna()]
        for flag,name in ((1,'joint_pass'),(0,'joint_fail')):
            values=known.loc[known.joint==flag,target]
            row[name]=dict(n=len(values),mean=None if values.empty else float(values.mean()),
                positive_rate=None if values.empty else float(values.gt(0).mean()))
        result[target]=row
    return result
