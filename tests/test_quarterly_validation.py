from copy import deepcopy
from datetime import datetime, timedelta
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.finmind import FinMindError
from app.models import Base, QuarterlyFundamentalSnapshot
from skills.quarterly_validation import calculate, align_prices, pivot, BS_MAP
from skills.ingest_quarterly_fundamental import persist_snapshots

OBS = '2025-03-15T13:00:00+00:00'


def inputs():
    bs, income, cf = [], [], []
    for i,period in enumerate(pd.period_range('2023Q4','2024Q4',freq='Q')):
        d=str(period.end_time.date());year_quarter=period.quarter
        for bucket,values in [(bs,dict(TotalAssets=500,Liabilities=300,Equity=100+i*25)),
                (income,dict(Revenue=100,OperatingIncome=20,IncomeAfterTaxes=10)),
                (cf,dict(CashFlowsFromOperatingActivities=100*year_quarter,PropertyAndPlantAndEquipment=-20*year_quarter))]:
            bucket.extend(dict(date=d,stock_id='2330',type=k,value=v) for k,v in values.items())
    return tuple(pd.DataFrame(x) for x in (bs,income,cf))


def test_true_ttm_ytd_conversion_and_denominator():
    result=calculate(*inputs(),observed_at=OBS)
    assert result[-1]['roe_ttm']==pytest.approx(40/150*100)
    assert result[-1]['roa_ttm']==8
    assert result[-1]['debt_ratio']==60
    assert result[-1]['fcf_ttm']==320
    assert result[-1]['fcf_per_share'] is None
    assert result[0]['roe_ttm'] is None
    shares=[dict(stock_id='2330',report_date='2024-12-31',shares=10,source='verified outstanding shares fixture')]
    assert calculate(*inputs(),observed_at=OBS,shares=shares)[-1]['fcf_per_share']==32


def test_gap_does_not_become_four_consecutive_quarters():
    raw=tuple(x[x.date.ne('2024-06-30')] for x in inputs())
    result=calculate(*raw,observed_at=OBS)[-1]
    assert result['roe_ttm'] is None and result['fcf_ttm'] is None


def test_capital_cannot_masquerade_as_share_count():
    bs,inc,cf=inputs()
    bs=pd.concat([bs,pd.DataFrame([dict(stock_id='2330',date='2024-12-31',type='OrdinaryShare',value=100)])])
    assert calculate(bs,inc,cf,observed_at=OBS)[-1]['fcf_per_share'] is None


@pytest.mark.parametrize('kind',['conflict','nan','date','cash_sign'])
def test_invalid_source_fails_explicitly(kind):
    bs,inc,cf=inputs()
    if kind=='conflict':bs=pd.concat([bs,pd.DataFrame([dict(bs.iloc[0],value=1000)])])
    if kind=='nan':bs.loc[0,'value']=float('nan')
    if kind=='date':bs.loc[0,'date']='2024-05-15'
    if kind=='cash_sign':cf.loc[cf.type.eq('PropertyAndPlantAndEquipment'),'value']=20
    with pytest.raises(FinMindError):calculate(bs,inc,cf,observed_at=OBS)


def test_later_period_cannot_change_earlier_metrics_or_fingerprint():
    before=calculate(*inputs(),observed_at=OBS)
    data=list(inputs());data[1].loc[data[1].date.eq('2024-12-31'),'value']*=2
    after=calculate(*data,observed_at=OBS)
    assert before[:-1]==after[:-1]
    assert before[-1]['source_sha256']!=after[-1]['source_sha256']


def test_availability_is_observation_not_report_plus_delay():
    rows=calculate(*inputs(),observed_at=OBS)
    snapshots=pd.DataFrame(rows)
    prices=pd.DataFrame(dict(stock_id='2330',trading_date=pd.to_datetime(['2024-12-31','2025-03-15','2025-03-16'])))
    result=align_prices(prices,snapshots)
    assert result.roe_raw.iloc[:2].isna().all()
    assert result.roe_raw.iloc[2]==pytest.approx(40/150*100)
    snapshots['available_date']=pd.Timestamp('2024-12-31')
    with pytest.raises(FinMindError):align_prices(prices,snapshots)


def test_late_old_report_revision_does_not_replace_newer_period():
    rows=calculate(*inputs(),observed_at=OBS)
    old=dict(rows[0],observed_at=datetime(2025,3,17),available_date=datetime(2025,3,18).date(),roe_ttm=999)
    prices=pd.DataFrame(dict(stock_id='2330',trading_date=pd.to_datetime(['2025-03-18'])))
    assert align_prices(prices,pd.DataFrame(rows+[old])).roe_raw.iloc[0]==pytest.approx(40/150*100)


def test_versions_are_append_only_idempotent_and_reversion_is_new_evidence():
    engine=create_engine('sqlite://');Base.metadata.create_all(engine,tables=[QuarterlyFundamentalSnapshot.__table__])
    a=calculate(*inputs(),observed_at=OBS)[-1];a['source_manifest']='fixture.json'
    with Session(engine) as session:
        assert persist_snapshots(session,[a])==1
        assert persist_snapshots(session,[dict(a,observed_at=a['observed_at']+timedelta(days=1))])==0
        b=dict(a,observed_at=a['observed_at']+timedelta(days=1),source_sha256='b'*64,roe_ttm=20)
        assert persist_snapshots(session,[b])==1
        reverted=dict(a,observed_at=a['observed_at']+timedelta(days=2))
        assert persist_snapshots(session,[reverted])==1
        assert len(session.execute(select(QuarterlyFundamentalSnapshot)).scalars().all())==3


def test_ingest_budget_rotation_and_unchanged_observation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from app.models import Stock,Job,QuarterlyIngestState
    from skills import ingest_quarterly_fundamental as ingest
    engine=create_engine('sqlite://')
    Base.metadata.create_all(engine,tables=[m.__table__ for m in (Stock,Job,QuarterlyIngestState,QuarterlyFundamentalSnapshot)])
    monkeypatch.setattr(ingest,'ROOT',tmp_path)
    calls=[]
    def fetch(dataset,start,end,**kw):
        calls.append((dataset,kw['data_id']))
        assert kw['max_retries']==0
        frame=inputs()[ingest.DATASETS.index(dataset)].copy();frame['stock_id']=kw['data_id']
        return frame
    monkeypatch.setattr(ingest,'fetch_dataset',fetch)
    config=SimpleNamespace(tz='Asia/Taipei',finmind_token=None,finmind_requests_per_hour=6000)
    with Session(engine) as session:
        session.add_all([Stock(stock_id=sid,name=sid,is_listed=True,security_type='stock',market='TWSE') for sid in ('1101','2330')]);session.commit()
        first=ingest.run(config,session,max_requests=3);session.commit()
        assert first['stocks']==1 and first['deferred_stocks']==1 and first['rows']==5
        second=ingest.run(config,session,max_requests=3);session.commit()
        assert second['rows']==5 and calls[3][1]=='2330'
        original=list(session.execute(select(QuarterlyFundamentalSnapshot.observed_at).order_by(QuarterlyFundamentalSnapshot.stock_id,QuarterlyFundamentalSnapshot.report_date)).scalars())
        third=ingest.run(config,session,max_requests=3);session.commit()
        assert third['rows']==0 and calls[6][1]=='1101'
        assert list(session.execute(select(QuarterlyFundamentalSnapshot.observed_at).order_by(QuarterlyFundamentalSnapshot.stock_id,QuarterlyFundamentalSnapshot.report_date)).scalars())==original
        assert len(list(session.execute(select(QuarterlyIngestState)).scalars()))==2
        with pytest.raises(ValueError,match='budget'):ingest.run(config,session,stock_ids=['1101','2330'],max_requests=3)
        assert len(calls)==9
        with pytest.raises(ValueError,match='nonordinary'):ingest.run(config,session,stock_ids=['0050'],max_requests=3)
        assert len(calls)==9


def test_unrecognized_statement_does_not_count_as_success():
    bs,inc,cf=inputs();cf['type']='provider_schema_changed'
    with pytest.raises(FinMindError,match='No supported'):calculate(bs,inc,cf,observed_at=OBS)
