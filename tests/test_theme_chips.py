import numpy as np
import pandas as pd
import pytest
from skills.holding_validation import TIERS,aggregate as strict_reference
from skills.theme_chips import aggregate_week,changes,align,conditions,news_at,increment


def raw_week(day='2026-01-02',shift=0):
    # Consistent inventory, transferring shares from the small to large tier.
    amounts=[1_000_000]*15;amounts[0]-=shift;amounts[-1]+=shift
    rows=[dict(stock_id='2492',date=day,HoldingSharesLevel=t,people=1,unit=u,
        percent=round(u/sum(amounts)*100,2)) for t,u in zip(TIERS,amounts)]
    return pd.DataFrame(rows+[dict(stock_id='2492',date=day,HoldingSharesLevel='total',
        people=15,unit=sum(amounts),percent=100)])


def history(n=8):
    return pd.concat([aggregate_week(raw_week(str(d.date()),i*50_000),d,{'2492'})
        for i,d in enumerate(pd.date_range('2026-01-02',periods=n,freq='7D'))],ignore_index=True)


def test_exact_tiers_not_all_retail_and_missing_stock():
    raw=raw_week();row=aggregate_week(raw,'2026-01-02',{'2492','2308'}).set_index('stock_id')
    assert row.loc['2492','small_pct']==.6
    assert row.loc['2492','large_pct']==1/15
    assert row.loc['2308','reason']=='missing_stock_week'
    assert np.isnan(row.loc['2308','large_pct'])
    assert round(row.loc['2492','large_pct'],4)==strict_reference(raw).iloc[0].large_holder_pct


@pytest.mark.parametrize('kind',['duplicate','missing','unknown','percent','people','nan','fraction','adjustment'])
def test_broken_week_is_unknown(kind):
    raw=raw_week()
    if kind=='duplicate':raw=pd.concat([raw,raw.iloc[[0]]])
    elif kind=='missing':raw=raw.iloc[1:]
    elif kind=='unknown':raw.loc[0,'HoldingSharesLevel']='unknown'
    elif kind=='percent':raw.loc[0,'percent']=20
    elif kind=='people':raw.loc[0,'people']=2
    elif kind=='nan':raw.loc[0,'unit']=np.nan
    elif kind=='fraction':raw.loc[0,'unit']=3.5
    else:raw.loc[len(raw)]={**raw.iloc[-1].to_dict(),'HoldingSharesLevel':'差異數調整','unit':1_000_000}
    row=aggregate_week(raw,'2026-01-02',{'2492'}).iloc[0]
    assert not row.valid and pd.isna(row.large_pct)


def test_availability_staleness_and_no_future_features():
    weekly=history();queries=pd.DataFrame(dict(stock_id=['2492']*3,
        signal_date=['2026-02-06','2026-02-07','2026-05-01'],relative_strength=[True]*3))
    full=conditions(align(queries,changes(weekly),8))
    assert pd.isna(full.iloc[0].concentration)  # Fifth observation not yet available.
    assert bool(full.iloc[1].concentration)
    assert pd.isna(full.iloc[2].concentration)
    cutoff=pd.Timestamp('2026-02-07')
    prefix=weekly[weekly.date<=cutoff].copy()
    mutated=weekly.copy();mutated.loc[mutated.date>cutoff,'large_pct']=.9
    for source in (prefix,mutated):
        result=conditions(align(queries.iloc[:2],changes(source),8))
        pd.testing.assert_frame_equal(full.iloc[:2].reset_index(drop=True),result)
    delayed=conditions(align(queries.iloc[[1]],changes(weekly),15))
    assert pd.isna(delayed.iloc[0].concentration)


def test_bad_middle_week_and_inventory_jump_do_not_skip():
    weekly=history(5);weekly.loc[2,'valid']=False
    assert not changes(weekly).iloc[-1].change_known
    weekly=history(5);weekly.loc[2:,'total_units']*=2
    row=changes(weekly).iloc[-1]
    assert not row.change_known and row.change_reason=='inventory_denominator_change'
    weekly=history(5);weekly.loc[2:,'date']+=pd.Timedelta(days=7)
    assert not changes(weekly).iloc[-1].change_known


def test_four_cells_and_unknown_do_not_become_negative():
    f=pd.DataFrame(dict(large_pct_delta4=[.01,.01,-.01,-.01,np.nan],small_pct_delta4=[-.01]*5,
        large_units_delta4=[100]*5,chip_known=[True]*4+[False],relative_strength=[True,False,True,False,False]))
    result=conditions(f)
    assert result.both.iloc[0] and result.concentration_only.iloc[1]
    assert result.strength_only.iloc[2] and result.neither.iloc[3]
    assert all(pd.isna(result.iloc[4][key]) for key in ['all_eligible','concentration','both','neither'])


def test_news_date_only_and_missing_not_no_theme():
    events=[dict(stock_id='3026',id='a',source_date='2026-03-24',date_verified=True),
            dict(stock_id='3026',id='b',source_date='2026-03-01',date_verified=False)]
    assert news_at(events,'3026','2026-03-24')['theme_observed'] is None
    assert news_at(events,'3026','2026-03-25')['evidence_ids']=='a'
    assert news_at(events,'2492','2026-04-02')['theme_observed'] is None


def test_increment_unknown_excluded_and_empty_arm_not_zero():
    rows=pd.DataFrame(dict(event=[True,False,pd.NA],concentration=[True,True,False],
        relative_strength=[True]*3,signal_date=['2026-01-02']*3))
    result=increment(rows)
    assert result['concentrated_rate']==.5
    assert result['other_rate'] is None and result['difference'] is None
