#!/usr/bin/env python3
"""Freeze future revenue forecasts; never manufacture historical forecast dates."""
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.research_exit_scenarios import read,write,sha
from scripts.prepare_five_axis import OUTPUT
from skills.earnings_expectations import revenue_months,expected_revenue


def freeze(output=OUTPUT):
    target=output/'revenue-forecast.json'
    if target.exists():raise ValueError('Forecast is immutable; use a new dated output directory')
    now=datetime.now(timezone.utc)
    month=pd.Timestamp(now.astimezone(ZoneInfo('Asia/Taipei')).date()).to_period('M')
    raw_source=ROOT/'.cache/revenue-research/revenue.parquet'
    ledger_source=ROOT/'artifacts/revenue_announcements/announcements.parquet'
    inputs=output/'forecast-inputs';inputs.mkdir(parents=True,exist_ok=True)
    raw_path,ledger_path=inputs/'revenue.parquet',inputs/'announcements.parquet'
    for src,dest in ((raw_source,raw_path),(ledger_source,ledger_path)):
        if dest.exists():raise ValueError('Forecast input path already exists')
        dest.write_bytes(src.read_bytes())
    monthly=revenue_months(pd.read_parquet(raw_path))
    ledger=pd.read_parquet(ledger_path)
    ledger['announcement_date']=pd.to_datetime(ledger.announcement_date)
    ledger=ledger[ledger.announcement_date<=pd.Timestamp(now.astimezone(ZoneInfo('Asia/Taipei')).date())]
    conflicts=[];overrides=0
    for (sid,year,number),group in ledger.groupby(['stock_id','revenue_year','revenue_month']):
        stamp=pd.Timestamp(year=int(year),month=int(number),day=1)+pd.offsets.MonthBegin(1)
        latest=group[group.announcement_date.eq(group.announcement_date.max())]
        if latest.revenue.nunique()!=1:
            conflicts.append(dict(stock_id=str(sid),revenue_month=f'{int(year)}-{int(number):02d}'))
            if sid in monthly and stamp in monthly.index:monthly.at[stamp,sid]=float('nan')
            continue
        if sid in monthly and stamp in monthly.index:
            # The append-only MOPS crawler stores thousands of TWD; FinMind
            # revenue history is TWD. Only this explicitly sourced unit changes.
            monthly.at[stamp,sid]=float(latest.revenue.iloc[0])*1000
            overrides+=1
    target_index=(month+1).start_time
    calendar=pd.date_range(monthly.index.min(),max(monthly.index.max(),target_index),freq='MS')
    monthly=monthly.reindex(calendar)
    # A forecast for the current revenue month cannot consume a preexisting
    # actual for that same month, even if a malformed source supplied one.
    monthly.loc[monthly.index>=target_index]=float('nan')
    forecast=expected_revenue(monthly)
    entries=read(output/'rebuild/signals.json')['entries']
    ids=sorted({e['members'][0] for e in entries})
    rows=[]
    for sid in ids:
        expected=forecast.at[target_index,sid] if sid in forecast else float('nan')
        rows.append(dict(stock_id=sid,revenue_month=str(month),expected_revenue_twd=
            float(expected) if pd.notna(expected) else None,
            status='await_future_actual' if pd.notna(expected) else 'insufficient_history'))
    result=dict(frozen_at=now.isoformat(),revenue_month=str(month),rows=rows,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (raw_path,ledger_path)},
        code_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                     'skills/earnings_expectations.py':sha(ROOT/'skills/earnings_expectations.py')},
        official_monthly_overrides=overrides,ambiguous_observations=conflicts,
        actual_announcement_timestamps_verified=False,historical_forecast=False,
        model='prior-year same-month revenue times preceding-three-month YoY ratio',
        note='First recorded date is not exact announcement time. Initial batches are not historical events; predictions exist only from this freeze onward.',
        network_calls=0,live_qualified=False)
    write(target,result)
    return result


if __name__=='__main__':
    result=freeze()
    print('Future forecasts',sum(r['expected_revenue_twd'] is not None for r in result['rows']),
          'of',len(result['rows']),'month',result['revenue_month'])
