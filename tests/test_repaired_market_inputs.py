from copy import deepcopy
import pandas as pd
import pytest

from skills.repaired_market_inputs import merge_quote_repairs, candidate_diff, apply_entry_policy, observed_roster_check


def fixture():
    d = pd.Timestamp('2020-01-02')
    q = pd.DataFrame(columns=['date','stock_id','open','high','low','close','volume'])
    frames = {k:pd.DataFrame({'1101':[float('nan')]},index=[d]) for k in ['raw-close','raw-volume','close-quality']}
    a = pd.DataFrame([dict(date=d,stock_id='1101',open=10.,high=12.,low=9.,close=11.,total_daily_volume=1200.,quality_adjusted_close=8.)])
    return q,frames,a


def test_repairs_add_explicit_totals_and_independent_adjusted_price_without_mutation():
    q,f,a=fixture(); old=deepcopy(f)
    quotes,frames=merge_quote_repairs(q,f,a)
    assert quotes.iloc[0].volume==1200 and frames['close-quality'].iloc[0,0]==8
    for k in f:pd.testing.assert_frame_equal(f[k],old[k])
    assert q.empty


@pytest.mark.parametrize('field,value',[('total_daily_volume',float('nan')),('total_daily_volume',1.5),
    ('quality_adjusted_close',None),('open',999),('stock_id','2222')])
def test_missing_scope_adjusted_price_or_conflicting_identity_cannot_be_inserted(field,value):
    q,f,a=fixture();a.loc[0,field]=value
    with pytest.raises(ValueError):merge_quote_repairs(q,f,a)


def test_existing_record_or_conflicting_raw_cell_cannot_be_overwritten():
    q,f,a=fixture();f['raw-close'].iloc[0,0]=12
    with pytest.raises(ValueError,match='conflicts'):merge_quote_repairs(q,f,a)
    q,f,a=fixture();q=a.rename(columns={'total_daily_volume':'volume'}).drop(columns='quality_adjusted_close')
    with pytest.raises(ValueError,match='recorded quote'):merge_quote_repairs(q,f,a)


def test_account_restrictions_are_logged_without_erasing_signal():
    rows=[dict(event_id='a',members=['1101'],signal_date='2020-01-01',entry_date='2020-01-02')]
    copied=deepcopy(rows)
    result=lambda *args,**kwargs:dict(allowed=False,reason='retail_restriction')
    accepted,blocked=apply_entry_policy(rows,{},result,research_risk_notice_assumed=False)
    assert accepted==[] and blocked[0]['decision']['reason']=='retail_restriction' and rows==copied
    with pytest.raises(ValueError,match='explicitly'):
        apply_entry_policy(rows,{},lambda *a,**k:{'allowed':None},research_risk_notice_assumed=False)


def test_candidate_diff_records_ranking_change_and_duplicate_ids_fail():
    assert not candidate_diff([{'event_id':'a'},{'event_id':'b'}],[{'event_id':'b'},{'event_id':'a'}])['ordering_unchanged']
    with pytest.raises(ValueError):candidate_diff([{'event_id':'a'}]*2,[])


def test_universe_cannot_certify_missing_dates_unknown_identities_or_missing_columns():
    f=pd.DataFrame([dict(market='TWSE',date='2020-01-02',stock_id='1101'),dict(market='TPEX',date='2020-01-02',stock_id='2222')])
    resolve=lambda identity,sid,day:dict(market='TWSE' if sid=='1101' else 'TPEX',category='股票')
    ok=observed_roster_check(f,{},resolve,['2020-01-02'],['1101','2222'])
    assert ok['complete_observed_daily_rosters'] and not ok['intraday_restrictions_certified']
    assert not observed_roster_check(f,{},resolve,['2020-01-02'],['1101'])['complete_observed_daily_rosters']
    assert not observed_roster_check(f.iloc[:1],{},resolve,['2020-01-02'],['1101','2222'])['complete_observed_daily_rosters']
    assert not observed_roster_check(f,{},lambda *a:None,['2020-01-02'],['1101','2222'])['complete_observed_daily_rosters']
