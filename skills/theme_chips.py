"""Point-in-time holder proxies and descriptive chip/price comparisons.

Concentration is an account-size distribution, never proof of buying intent.
News coverage must be established separately; missing evidence stays unknown.
"""
import numpy as np
import pandas as pd
from skills.holding_validation import TIERS

KEYS = ['stock_id', 'date']
RULES = ['all_eligible', 'concentration', 'relative_strength', 'both',
         'concentration_only', 'strength_only', 'neither']


def aggregate_week(raw, day, cohort):
    """Vectorized strict reconciliation, including explicit missing-stock rows."""
    ids = sorted(cohort)
    if any(not isinstance(s,str) or len(s)!=4 or not s.isdecimal() for s in ids):
        raise ValueError('Only four-digit ordinary-stock cohort identifiers')
    out = pd.DataFrame(index=pd.Index(ids,name='stock_id'))
    out['date'] = pd.Timestamp(day)
    out['valid'] = False; out['reason'] = 'missing_stock_week'
    for field in ('total_units','large_units','small_units','large_pct','small_pct','total_people'):
        out[field] = np.nan
    if raw.empty:return out.reset_index()
    required = {'stock_id','date','HoldingSharesLevel','unit','people','percent'}
    if not required.issubset(raw):raise ValueError('Missing holder schema')
    frame = raw.copy();frame['stock_id'] = frame.stock_id.astype(str)
    frame = frame[frame.stock_id.isin(ids)].copy()
    if frame.empty:return out.reset_index()
    if not pd.to_datetime(frame.date).eq(pd.Timestamp(day)).all():
        raise ValueError('Holder data outside the requested observation date')
    frame['level'] = frame.HoldingSharesLevel.astype(str).str.replace(',','',regex=False).str.strip().str.lower()
    frame['level'] = frame.level.replace({'more than 1000001':'1000001+','over 1000001':'1000001+'})
    frame.loc[frame.level.str.startswith('差異數調整'),'level'] = 'adjustment'
    known = [*TIERS,'total','adjustment']
    bad_levels = frame.loc[~frame.level.isin(known),'stock_id'].unique()
    duplicates = frame.loc[frame.duplicated(['stock_id','level'],keep=False),'stock_id'].unique()
    values = frame[frame.level.isin(known)].drop_duplicates(['stock_id','level']).copy()
    for field in ('unit','people','percent'):values[field] = pd.to_numeric(values[field],errors='coerce')
    def pivot(field):
        return values.pivot(index='stock_id',columns='level',values=field).reindex(index=ids,columns=known)
    unit,people,percent = (pivot(field) for field in ('unit','people','percent'))
    levels = [*TIERS,'total']
    present = frame.groupby('stock_id').size().reindex(ids,fill_value=0).gt(0)
    reason = pd.Series('',index=out.index)
    def reject(mask, why):reason.loc[mask & reason.eq('')] = why
    reject(~present,'missing_stock_week')
    reject(out.index.isin(bad_levels),'unknown_tier')
    reject(out.index.isin(duplicates),'duplicate_tier')
    reject(unit[levels].isna().any(axis=1)|people[levels].isna().any(axis=1)|percent[levels].isna().any(axis=1),'incomplete_or_nonnumeric')
    good_numbers = (np.isfinite(unit[levels]).all(axis=1)&np.isfinite(people[levels]).all(axis=1)
        &np.isfinite(percent[levels]).all(axis=1)&unit[levels].ge(0).all(axis=1)
        &people[levels].ge(0).all(axis=1)&percent[levels].ge(0).all(axis=1)
        &percent[levels].le(100).all(axis=1)&unit[levels].mod(1).eq(0).all(axis=1)
        &people[levels].mod(1).eq(0).all(axis=1)&unit['total'].gt(0))
    reject(~good_numbers,'invalid_value')
    adjustment_present = values[values.level.eq('adjustment')].stock_id
    adjustment = unit['adjustment'].copy()
    adjustment.loc[~adjustment.index.isin(adjustment_present)] = 0
    reject(~np.isfinite(adjustment)|adjustment.mod(1).ne(0),'invalid_adjustment')
    total = unit['total']
    reject((unit[list(TIERS)].sum(axis=1)+adjustment).ne(total)
           |people[list(TIERS)].sum(axis=1).ne(people['total']),'totals_mismatch')
    reject((percent[list(TIERS)]-unit[list(TIERS)].div(total,axis=0)*100).abs().gt(.011).any(axis=1)
           |percent['total'].sub(100).abs().gt(.011),'percent_mismatch')
    reject(adjustment.abs().div(total).gt(.005),'excessive_adjustment')
    valid = reason.eq('')
    out['valid']=valid;out['reason']=reason.where(~valid,'ok')
    out['total_units']=total.where(valid)
    out['total_people']=people['total'].where(valid)
    out['large_units']=unit['1000001+'].where(valid)
    out['small_units']=unit[list(TIERS[:9])].sum(axis=1).where(valid)
    out['large_pct']=out.large_units/out.total_units
    out['small_pct']=out.small_units/out.total_units
    return out.reset_index()


def changes(weekly):
    """Keep invalid/missing weeks in place so rolling windows cannot skip them."""
    rows=weekly.sort_values(['stock_id','date']).copy()
    rows['date']=pd.to_datetime(rows.date)
    if rows.duplicated(KEYS).any():raise ValueError('Duplicate holder stock/week')
    grouped=rows.groupby('stock_id',sort=False)
    for field in ('large_pct','small_pct','large_units'):
        rows[field+'_delta4']=rows[field]-grouped[field].shift(4)
    span=(rows.date-grouped.date.shift(4)).dt.days
    gaps=grouped.date.diff().dt.days
    denominator=(rows.total_units/grouped.total_units.shift(1)-1).abs()
    def rolling(series,window,operation):
        return series.groupby(rows.stock_id).transform(lambda s:getattr(s.rolling(window,min_periods=window),operation)())
    complete=rolling(rows.valid.astype(int),5,'sum').eq(5)
    regular=span.between(21,35)&rolling(gaps,4,'max').le(10)
    stable=rolling(denominator,4,'max').le(.01)
    rows['change_known']=complete&regular&stable
    rows['change_reason']=np.select([~complete,~regular,~stable],
        ['incomplete_five_weeks','irregular_week_span','inventory_denominator_change'],default='ok')
    for key in ('large_pct_delta4','small_pct_delta4','large_units_delta4'):
        rows[key]=rows[key].where(rows.change_known)
    return rows


def align(queries, weekly_changes, lag):
    if lag not in (8,15):raise ValueError('Use preregistered 8/15 calendar-day availability')
    left=queries.copy();left['_ordinal']=range(len(left))
    left['signal_date']=pd.to_datetime(left.signal_date)
    right=weekly_changes.copy();right['observed_date']=pd.to_datetime(right.pop('date'))
    right['available_date']=right.observed_date+pd.Timedelta(days=lag)
    result=pd.merge_asof(left.sort_values('signal_date'),right.sort_values('available_date'),
        left_on='signal_date',right_on='available_date',by='stock_id',direction='backward')
    fresh=(result.signal_date-result.available_date).dt.days.le(14)
    result['chip_known']=result.change_known.fillna(False)&fresh
    result.loc[~fresh,'change_reason']='stale_or_unavailable'
    for key in ('large_pct_delta4','small_pct_delta4','large_units_delta4'):
        result[key]=result[key].where(result.chip_known)
    result=result.sort_values('_ordinal').drop(columns='_ordinal').reset_index(drop=True)
    for key in ('signal_date','observed_date','available_date'):
        result[key]=result[key].dt.strftime('%Y-%m-%d')
    return result


def conditions(frame, threshold=.005):
    if threshold not in (0,.005,.01):raise ValueError('Unregistered concentration threshold')
    result=frame.copy()
    large=result.large_pct_delta4
    rising=large.gt(0) if threshold==0 else large.ge(threshold)
    valid=result.chip_known & result.relative_strength.notna()
    c=(rising & result.small_pct_delta4.lt(0)&result.large_units_delta4.gt(0)).astype('boolean').where(valid)
    r=result.relative_strength.astype('boolean').where(valid)
    result['all_eligible']=pd.Series(True,index=result.index,dtype='boolean').where(valid)
    result['concentration']=c;result['relative_strength']=r
    result['both']=c&r;result['concentration_only']=c&~r
    result['strength_only']=~c&r;result['neither']=~c&~r
    return result


def increment(frame):
    """Paired date-block descriptive interval for C+R versus R without C."""
    frame=frame[frame.event.notna() & frame.concentration.notna() & frame.relative_strength.eq(True)].copy()
    frame['hit']=frame.event.astype(int)
    frame['positive']=frame.concentration.astype(int)
    frame['positive_hit']=frame.hit*frame.positive
    frame['negative']=1-frame.positive;frame['negative_hit']=frame.hit*frame.negative
    cols=['positive_hit','positive','negative_hit','negative']
    block=frame.groupby('signal_date')[cols].sum().to_numpy(dtype=float)
    totals=block.sum(axis=0)
    def ratio(a,b):return float(a/b) if b else None
    cp=ratio(totals[0],totals[1]);cn=ratio(totals[2],totals[3])
    result=dict(concentrated_rows=int(totals[1]),other_rows=int(totals[3]),
        concentrated_rate=cp,other_rate=cn,difference=cp-cn if cp is not None and cn is not None else None,
        ci_low=None,ci_high=None,dates=len(block),valid_bootstrap=0,bootstrap_replicates=1000)
    if len(block)>=5:
        weights=np.random.default_rng(20260927).multinomial(len(block),np.full(len(block),1/len(block)),size=1000)
        samples=weights@block;valid=(samples[:,1]>0)&(samples[:,3]>0)
        diff=samples[valid,0]/samples[valid,1]-samples[valid,2]/samples[valid,3]
        result['valid_bootstrap']=len(diff)
        if len(diff)>=950:result['ci_low'],result['ci_high']=map(float,np.quantile(diff,[.025,.975]))
    return result


def news_at(events, sid, day):
    """Positive existence evidence only; absence never means no theme."""
    rows=[r for r in events if r['stock_id']==sid and r['date_verified']
          and pd.Timestamp(r['source_date'])+pd.Timedelta(days=1)<=pd.Timestamp(day)]
    return dict(theme_observed=True if rows else None,
                evidence_ids=';'.join(r['id'] for r in rows),
                theme_absence_verified=False)
