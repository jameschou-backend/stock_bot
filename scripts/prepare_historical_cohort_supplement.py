#!/usr/bin/env python3
"""Freeze existing local prices omitted from the old current-company cohort.

Only SELECTs are used. This is a raw-data supplement, not a revised signal set
or proof of eligibility. Old companies, matrices and accounts are unchanged.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from skills.backtest_data_evidence import digest
from scripts.audit_historical_universe_followup import verify_report as verify_identity

IDENTITY=ROOT/'.cache/historical-universe-followup-20260925/report-final.json'
COHORT=ROOT/'.cache/million-replay-signals/companies.parquet'
ORIGINAL=ROOT/'.cache/million-replay-inputs'
DIRECTORY=ROOT/'.cache/historical-cohort-supplement-20260925/verified-snapshot'
START,END='2021-01-01','2026-09-09'
FIELDS=['stock_id','date','open','high','low','close','volume']


def encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'


def plan():
    identity=verify_identity(IDENTITY)
    inputs={str(IDENTITY.relative_to(ROOT)):digest(IDENTITY)}
    for directory,names in ((COHORT.parent,['companies.parquet']),
                            (ORIGINAL,['quotes.parquet'])):
        manifest=directory/'manifest.json'
        values=json.loads(manifest.read_text())['files_sha256']
        inputs[str(manifest.relative_to(ROOT))]=digest(manifest)
        for name in names:
            path=directory/name
            if digest(path)!=values[name]:raise ValueError('Old sealed input changed: '+name)
            inputs[str(path.relative_to(ROOT))]=values[name]
    companies=pd.read_parquet(COHORT)
    episodes=[e for e in identity['episodes'] if e['category'] in ('股票','unconfirmed')
        and e['start'] is not None and e['start']<=END
        and (e['end'] is None or e['end']>'2022-01-03')]
    ids=sorted({e['stock_id'] for e in episodes}-set(companies.stock_id))
    if len(ids)!=49:raise ValueError('The reviewed missing-cohort inventory changed')
    old=pd.read_parquet(ORIGINAL/'quotes.parquet',columns=['stock_id'],filters=[('stock_id','in',ids)])
    if len(old):raise ValueError('These stocks are already present in the sealed quote source')
    return dict(stock_ids=ids,start=START,end=END,original_cohort_stocks=len(companies),
        original_quote_rows_for_supplement=0,identity_inventory_includes_unconfirmed_categories=True,
        input_sha256=inputs)


def summarize(frame,ids,*,allow_quarantine=False):
    if list(frame.columns)!=FIELDS or frame.empty:
        raise ValueError('Historical raw-price schema or data is missing')
    if set(frame.stock_id)!=set(ids) or frame.duplicated(['stock_id','date']).any():
        raise ValueError('Missing, unexpected or duplicate raw-price identity')
    dates=pd.to_datetime(frame.date,errors='raise')
    if (dates.dt.tz is not None or not dates.equals(dates.dt.normalize())
            or not dates.between(START,END).all()):
        raise ValueError('Raw-price dates exceed the fixed inventory scope')
    values=frame[['open','high','low','close','volume']].astype(float)
    if not np.isfinite(values.to_numpy()).all() or (values<0).any().any():
        raise ValueError('Invalid raw price or share volume')
    if not frame.volume.eq(np.floor(frame.volume)).all():
        raise ValueError('Raw volume must contain integer shares')
    positive=values[['open','high','low','close']].gt(0).all(axis=1)
    missing_trade_price=values.volume.gt(0) & ~positive
    bad_range=(values['low']>values['high']) | (positive & (
        (values['low']>values[['open','close']].min(axis=1))
        | (values['high']<values[['open','close']].max(axis=1))))
    bad=missing_trade_price | bad_range
    if bad.any() and not allow_quarantine:
        raise ValueError('Raw OHLC range is inconsistent')
    quarantined=[dict(stock_id=str(row.stock_id),date=str(pd.Timestamp(row.date).date()),
        open=float(row.open),high=float(row.high),low=float(row.low),close=float(row.close),
        reasons=([ 'positive_volume_without_complete_ohlc'] if missing_trade_price.loc[index] else [])
                + (['ohlc_range_inconsistent'] if bad_range.loc[index] else []),
        accepted_for_features=False)
        for index,row in frame.loc[bad].iterrows()]
    result=[]
    for sid,rows in frame.assign(date=dates).groupby('stock_id',sort=True):
        result.append(dict(stock_id=sid,rows=len(rows),first_date=str(rows.date.min().date()),
            last_date=str(rows.date.max().date()),positive_volume_rows=int(rows.volume.gt(0).sum()),
            zero_volume_rows=int(rows.volume.eq(0).sum()),
            warmup_rows=int(rows.date.lt('2022-01-03').sum()),
            research_period_rows=int(rows.date.ge('2022-01-03').sum())))
    return dict(stock_count=len(result),quote_rows=len(frame),stocks=result,
        quarantined_rows=len(quarantined),quarantine=quarantined,
        warmup_rows=int(dates.lt('2022-01-03').sum()),research_period_rows=int(dates.ge('2022-01-03').sum()),
        missing_market_sessions_not_inferred=True)


def verify(directory=DIRECTORY):
    directory=Path(directory).resolve()
    manifest_path=directory/'manifest.json'
    if digest(manifest_path)!=manifest_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Historical supplement manifest hash differs')
    manifest=json.loads(manifest_path.read_text())
    if (manifest['schema']!='historical_cohort_local_prices_v1'
            or manifest['plan']!=plan() or manifest['code_sha256']!=digest(__file__)):
        raise ValueError('Historical supplement source or recipe changed')
    prices=directory/'quotes.parquet'
    if digest(prices)!=manifest['quotes_sha256']:
        raise ValueError('Frozen supplemental prices changed')
    if summarize(pd.read_parquet(prices),manifest['plan']['stock_ids'],allow_quarantine=True)!=manifest['summary']:
        raise ValueError('Supplemental price inventory cannot be reproduced')
    if any(manifest[k] is not False for k in ('performance_recomputed','strict_data_ready','live_qualified')):
        raise ValueError('Raw prices cannot promote backtest or trading qualification')
    return manifest


def prepare(directory=DIRECTORY):
    from sqlalchemy import select,text
    from app.db import get_session
    from app.models import RawPrice
    code_before=digest(__file__)
    fixed=plan()
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ROOT/'.cache'):
        raise ValueError('Use a new project cache directory')
    directory.mkdir(parents=True,exist_ok=False)
    observed=datetime.now(timezone.utc).isoformat()
    with get_session() as session:
        if session.execute(text('SELECT DATABASE()')).scalar_one()!='stock_bot':
            raise ValueError('Unexpected database; no price query executed')
        query=select(RawPrice.stock_id,RawPrice.trading_date,RawPrice.open,RawPrice.high,
            RawPrice.low,RawPrice.close,RawPrice.volume).where(
            RawPrice.stock_id.in_(fixed['stock_ids']),RawPrice.trading_date>=START,
            RawPrice.trading_date<=END).order_by(RawPrice.stock_id,RawPrice.trading_date)
        rows=session.execute(query).all()
    frame=pd.DataFrame(rows,columns=FIELDS)
    frame['date']=pd.to_datetime(frame['date'])
    for column in ('open','high','low','close'):frame[column]=frame[column].astype(float)
    summary=summarize(frame,fixed['stock_ids'],allow_quarantine=True)
    if digest(__file__)!=code_before or plan()!=fixed:
        raise ValueError('Supplement preparation code or sealed sources changed during export')
    prices=directory/'quotes.parquet';frame.to_parquet(prices,index=False)
    value=dict(schema='historical_cohort_local_prices_v1',observed_at=observed,
        source='local stock_bot.raw_prices SELECT snapshot',plan=fixed,summary=summary,
        quotes_sha256=digest(prices),code_sha256=digest(__file__),
        database_mutations=0,external_market_data_requests=0,finmind_requests=0,
        old_sealed_inputs_changed=False,performance_recomputed=False,strict_data_ready=False,live_qualified=False,
        limitations=['Only raw prices are supplemented; no historical eligibility is inferred from their presence.',
            'Missing dates, delisting settlements and corporate actions still need independent evidence.',
            'A new historical-universe signal/account version must be rebuilt before using these stocks in reported returns.',
            'This is a current database observation, not a contemporaneously archived price version.'])
    path=directory/'manifest.json';path.write_text(encoded(value))
    path.with_suffix('.sha256').write_text(digest(path)+'\n')
    return verify(directory)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--directory',type=Path,default=DIRECTORY)
    args=parser.parse_args()
    result=prepare(args.directory) if args.prepare else verify(args.directory)
    print(encoded({k:v for k,v in result['summary'].items() if k!='stocks'}))
