"""Broad revenue-surprise diagnostics; assumed lags are never publication facts."""
import numpy as np
import pandas as pd


def revenue_months(rows):
    frame=rows[['stock_id','trading_date','revenue_current_month']].copy()
    frame['trading_date']=pd.to_datetime(frame.trading_date)
    if (frame.empty or frame.duplicated(['stock_id','trading_date']).any() or
        frame.trading_date.isna().any() or not frame.trading_date.dt.is_month_start.all()):
        raise ValueError('Unique FinMind following-month dates required')
    values=frame.pivot(index='trading_date',columns='stock_id',values='revenue_current_month').astype(float)
    calendar=pd.date_range(values.index.min(),values.index.max()+pd.offsets.MonthBegin(1),freq='MS')
    return values.reindex(calendar).where(lambda x:np.isfinite(x)&x.ge(0))


def expected_revenue(monthly):
    past3=monthly.shift(1).rolling(3,min_periods=3).sum()
    previous_year=past3.shift(12)
    forecast=monthly.shift(12)*past3/previous_year.where(previous_year>0)
    return forecast.where(np.isfinite(forecast)&forecast.gt(0))


def revenue_daily(rows,days,ids,lag):
    if lag not in (15,30):
        raise ValueError('Revenue lag must be 15 or 30 assumption days')
    monthly=revenue_months(rows).reindex(columns=ids)
    forecast=expected_revenue(monthly)
    surprise=monthly/forecast-1
    surprise.index=monthly.index-pd.Timedelta(days=1)+pd.Timedelta(days=lag)
    # Reindex entire rows; a missing latest month must not resurrect an older signal.
    return surprise.reindex(days,method='ffill')


def quality_daily(rows,days,ids,lag):
    if lag not in (120,150):
        raise ValueError('Financial lag must be 120 or 150 assumption days')
    required={'stock_id','date','type','value'}
    if not required.issubset(rows):
        raise ValueError('Missing financial source fields')
    selected=rows[rows.type.isin(['Revenue','GrossProfit','IncomeAfterTaxes'])].copy()
    selected['date']=pd.to_datetime(selected.date)
    if selected.duplicated(['stock_id','date','type']).any():
        raise ValueError('Financial duplicates need explicit reconciliation')
    if not selected.date.dt.is_quarter_end.all():
        raise ValueError('Quarter end labels required; they are not announcement dates')
    result=pd.DataFrame(np.nan,index=days,columns=ids)
    for sid,group in selected.groupby('stock_id'):
        if sid not in result:
            continue
        quarterly=group.pivot(index='date',columns='type',values='value').reindex(
            columns=['Revenue','GrossProfit','IncomeAfterTaxes']).astype(float)
        quarters=pd.period_range(quarterly.index.min(),days.max(),freq='Q').end_time.normalize()
        quarterly=quarterly.reindex(quarters)
        margin=quarterly.GrossProfit/quarterly.Revenue.where(quarterly.Revenue>0)
        prior=margin.shift(4)
        valid=np.isfinite(margin)&np.isfinite(prior)&np.isfinite(quarterly.IncomeAfterTaxes)
        good=((margin>prior)&quarterly.IncomeAfterTaxes.gt(0)).astype(float).where(valid)
        good.index=quarters+pd.Timedelta(days=lag)
        result[sid]=good.reindex(days,method='ffill')
    return result


def filter_entries(entries,surprise,quality,mode):
    if mode not in ('revenue_covered','surprise','quality_covered','surprise_quality'):
        raise ValueError('Unknown earnings experiment')
    accepted, decisions=[],[]
    for event in entries:
        day=pd.Timestamp(event['signal_date']);sid=event['members'][0]
        value=surprise.at[day,sid]
        q=quality.at[day,sid]
        covered=pd.notna(value) and (pd.notna(q) if mode in ('quality_covered','surprise_quality') else True)
        passed=covered and (mode in ('revenue_covered','quality_covered') or value>=.10)
        if mode=='surprise_quality':
            passed=passed and q==1
        decisions.append(dict(event_id=event['event_id'],stock_id=sid,signal_date=str(day.date()),
            covered=bool(covered),accepted=bool(passed),
            revenue_surprise=float(value) if pd.notna(value) else None,
            margin_and_profit_pass=bool(q) if pd.notna(q) else None,
            publication_timing='assumed_not_verified'))
        if passed: accepted.append(event)
    return accepted,decisions
