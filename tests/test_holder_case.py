import numpy as np
import pandas as pd
import pytest
from skills.holder_case import aggregate,features,diagnostics
from test_chip_research import LEVELS


def raw():
    rows=[dict(date='2026-04-02',stock_id='2492',HoldingSharesLevel=level,
        unit=0,people=0,percent=0.) for level in LEVELS]
    rows[1].update(unit=1_000_000,people=1000,percent=25.)
    rows[-1].update(unit=3_000_000,people=1,percent=75.)
    rows.append(dict(date='2026-04-02',stock_id='2492',HoldingSharesLevel='total',
        unit=4_000_000,people=1001,percent=100.))
    return pd.DataFrame(rows)


def test_small_and_thousand_lot_tiers_use_units_without_total_double_count():
    f=aggregate(raw()).iloc[0]
    assert f.valid and f.small_pct==25 and f.large_pct==75
    assert f.small_people==1000 and f.large_people==1


@pytest.mark.parametrize('column,value',[('people',-1),('unit',np.nan),('percent',30)])
def test_inconsistent_people_or_share_pct_is_unknown(column,value):
    data=raw();data.loc[1,column]=value
    f=aggregate(data).iloc[0]
    assert not f.valid and np.isnan(f.small_pct)


def test_publication_plus_seven_requires_next_market_day_and_no_partial_horizon():
    days=pd.bdate_range('2026-03-02','2026-05-08')
    weeks=pd.date_range('2026-03-06',periods=8,freq='W-FRI')
    frame=pd.DataFrame(dict(date=weeks,valid=True,small_pct=np.arange(8)[::-1]+20.,large_pct=np.arange(8)+40.))
    stock=pd.Series(np.arange(len(days))+100.,index=days)
    benchmark=pd.Series(100.,index=days)
    f=features(frame,stock,benchmark,7)
    assert f.loc[4,'available_entry_date']==pd.Timestamp('2026-04-13')
    assert f.loc[4,'joint']==1 and f.loc[4,'large_change4']==4
    assert f.forward60.isna().all()
    assert diagnostics(f)['forward60']['large_change4']['n']==0
    assert features(frame,stock,benchmark,14).loc[4,'available_entry_date']==pd.Timestamp('2026-04-20')


def test_invalid_intermediate_week_breaks_four_week_signal():
    days=pd.bdate_range('2026-01-01','2026-04-30')
    f=pd.DataFrame(dict(date=pd.date_range('2026-01-02',periods=5,freq='W-FRI'),
        valid=[True,True,False,True,True],small_pct=[5,4,3,2,1],large_pct=[1,2,3,4,5]))
    p=pd.Series(100.,index=days)
    assert np.isnan(features(f,p,p).iloc[-1].joint)
