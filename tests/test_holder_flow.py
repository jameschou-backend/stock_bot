import numpy as np
import pandas as pd
import pytest
from skills.holder_flow import FLOWS,ARMS,combine,orders,summaries,matched_pairs,comparisons
from skills.theme_chips import changes


def fixture():
    ids=[str(2000+i) for i in range(8)]
    f=[100.,-100.,100.,-100.,0.,np.nan,100.,-100.]
    t=[100.,100.,-100.,-100.,100.,100.,100.,100.]
    base=pd.DataFrame([dict(event_id='sample-'+sid,stock_id=sid,name=sid,signal_date='2022-01-14',
        feature_date='2022-01-13',flow_lag=1,foreign5=f[i],trust5=t[i],shares5=100000.,
        relative20=.15,momentum20=.2,adv20=100000000.,adv20_shares=1000000.,
        industry='01',named_case=False,feature_known=True) for i,sid in enumerate(ids)])
    rows=[]
    for sid in ids:
        for j,date in enumerate(pd.date_range('2021-11-05',periods=10,freq='W-FRI')):
            step=-1 if sid=='2006' else 1
            large=4000000+step*j*20000;small=2000000-step*j*20000
            rows.append(dict(stock_id=sid,date=date,valid=True,reason='ok',total_units=10000000.,
                large_units=float(large),small_units=float(small),large_pct=large/10000000,
                small_pct=small/10000000,total_people=1000.))
    weekly=changes(pd.DataFrame(rows))
    return base,weekly


def test_actor_directions_do_not_cancel_each_other_and_zero_is_separate():
    b,w=fixture();f=combine(b,w,8).set_index('stock_id')
    assert f.loc['2001','actor_state']=='foreign_sell_trust_buy'
    assert f.loc['2001','foreign_ratio']==pytest.approx(-.001)
    assert f.loc['2001','trust_ratio']==pytest.approx(.001)
    assert f.loc['2002','actor_state']=='foreign_buy_trust_sell'
    assert f.loc['2004','actor_state']=='zero_actor' and f.loc['2004','joint_known']
    assert f.loc['2004','concentrated_strength']==1
    assert f.loc['2004',list(ARMS[4:])].eq(0).all()


def test_missing_actor_is_unknown_in_every_arm():
    b,w=fixture();f=combine(b,w,8).set_index('stock_id')
    assert not f.loc['2005','joint_known']
    assert f.loc['2005',list(ARMS)].isna().all()


def test_concentration_distribution_and_neutral_are_distinct():
    b,w=fixture();f=combine(b,w,8).set_index('stock_id')
    assert f.loc['2000','holder_state']=='concentrated'
    assert f.loc['2006','holder_state']=='distributed'
    w.loc[w.stock_id.eq('2000'),'large_pct_delta4']=.001
    assert combine(b,w,8).set_index('stock_id').loc['2000','holder_state']=='neutral'


def test_publication_lag_is_not_observation_date_and_stale_is_unknown():
    b,w=fixture();f=combine(b,w,8)
    assert f.observed_date.eq('2021-12-31').all()
    assert f.available_date.eq('2022-01-08').all()
    assert combine(b,w,15).observed_date.eq('2021-12-24').all()
    b.signal_date='2022-03-01';f=combine(b,w,8)
    assert f.all_known.isna().all() and f.change_reason.eq('stale_or_unavailable').all()


def test_unpublished_week_cannot_affect_current_candidates():
    b,w=fixture();baseline=combine(b,w,8)
    future=w.date+pd.Timedelta(days=8)>pd.Timestamp('2022-01-14')
    w.loc[future,'valid']=False;w.loc[future,'large_units']*=9
    pd.testing.assert_frame_equal(baseline,combine(b,changes(w),8))


def test_bad_week_is_not_skipped_in_rolling_change():
    b,w=fixture();w.loc[w.date.eq(pd.Timestamp('2021-12-24')),'valid']=False
    f=combine(b,changes(w),8)
    assert not f.chip_known.any() and f.concentration.isna().all()


def test_strength_uses_its_own_features_not_old_selling_resilience():
    b,w=fixture();b.feature_known=False
    f=combine(b,w,8);assert f.iloc[0].concentrated_strength==1
    b.loc[0,'relative20']=.09;assert combine(b,w,8).iloc[0].concentrated_strength==0
    b.loc[0,'relative20']=np.nan;assert pd.isna(combine(b,w,8).iloc[0].concentration)


def test_matching_uses_same_actor_for_concentration_and_both_buy_for_divergence():
    b,w=fixture();table=combine(b,w,8);pairs=matched_pairs(table)
    row=pairs[(pairs.kind=='concentration')&(pairs.event_id=='sample-2000')].iloc[0]
    assert row.control_id=='sample-2006'
    row=pairs[(pairs.kind=='concentration')&(pairs.event_id=='sample-2001')].iloc[0]
    assert pd.isna(row.control_id)
    row=pairs[(pairs.kind=='divergence_vs_both_buy')&(pairs.event_id=='sample-2001')].iloc[0]
    assert row.control_id=='sample-2000'
    table['fake_future_return']=100000
    pd.testing.assert_frame_equal(pairs,matched_pairs(table))


def test_matching_never_crosses_industry_and_keeps_missing_control():
    b,w=fixture();b.loc[b.stock_id.eq('2006'),'industry']='02'
    pairs=matched_pairs(combine(b,w,8))
    row=pairs[(pairs.kind=='concentration')&(pairs.event_id=='sample-2000')].iloc[0]
    assert pd.isna(row.control_id)


def labelled(table):
    result=table.copy()
    for name,value in [('horizon',20),('phase','replication'),('year',2025),('forward_return',.4),
        ('fee_return',.38),('fee_excess',.3),('stress_fee_return',.37),('stress_fee_excess',.29),('surge',True)]:
        result[name]=value
    return result


def test_all_cells_partition_known_rows_and_unknown_results_are_retained():
    b,w=fixture();f=labelled(combine(b,w,8));f.loc[0,'forward_return']=np.nan;f.loc[0,'surge']=None
    cells=summaries(f,cells=True);stats=summaries(f)
    assert len(cells)==30 and cells.signals.sum()==int(f.joint_known.sum())
    assert cells.known.sum()==int(f.joint_known.sum())-1
    assert stats[stats.arm.eq('all_known')].iloc[0].outcome_unknown==1


def test_comparison_missing_pairs_and_small_date_counts_are_not_fake_intervals():
    b,w=fixture();table=combine(b,w,8);f=labelled(table)
    attached,report=comparisons(f,matched_pairs(table))
    assert attached.control_id.isna().any()
    assert report.unknown.gt(0).any() and report.date_bootstrap_low.isna().all()


def test_orders_use_next_session_main_lags_and_exclude_named_cases():
    b,w=fixture();b.loc[0,'named_case']=True
    table=pd.concat([combine(b,w,8),combine(b,w,15)],ignore_index=True)
    days=pd.bdate_range('2022-01-03','2022-01-31')
    result=orders(table,days)
    for rows in result.values():
        assert all(r['entry_date']=='2022-01-17' and r['stock_id']!='2000' for r in rows)
        assert all(r['holder_available_date']<=r['signal_date']<r['entry_date'] for r in rows)
    assert len(result['concentrated_strength'])==5


def test_duplicate_candidate_is_rejected():
    b,w=fixture()
    with pytest.raises(ValueError,match='Duplicate'):
        combine(pd.concat([b,b.iloc[:1]]),w,8)


def test_every_arm_time_and_horizon_variant_has_its_own_trial():
    from scripts.research_holder_flow import trial_definitions
    variants=trial_definitions()
    assert len(variants)==64
    assert len({tuple(sorted(v.items())) for v in variants})==64
