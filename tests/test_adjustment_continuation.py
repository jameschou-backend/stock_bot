from datetime import date
from dataclasses import replace
import pandas as pd
import pytest
from skills.official_adj_factors import AdjEvent, EVENT_COLUMNS
from scripts.research_adjustment_continuation import (
    checked_events, add_benchmark_split, parse_tpex_delistings, parse_twse_listings, parse_recent_tpex_listings, compare_factors, audit_membership)


def test_alias_resolution_requires_exact_source_and_equal_economics():
    e = AdjEvent('5906',date(2016,7,14),'TWSE','capital_reduction','減資',1.77,5.68,5.68,reason='彌補虧損',payload={'name':'old'})
    alias = replace(e,payload={'name':'new'})
    resolution = dict(key=['5906','2016-07-14','TWSE','capital_reduction'],source_sha256='approved',expected_rows=2)
    args = ('twse_capital_reduction', lambda _: [e,alias], {'data':[1,2]},date(2016,1,1),date(2016,12,31))
    events, records, count = checked_events(*args,'approved',resolution)
    assert len(events)==1 and count==2 and len(records[0]['raw_payloads'])==2
    with pytest.raises(ValueError,match='Unresolved duplicate'):
        checked_events(*args,'changed',resolution)
    with pytest.raises(ValueError,match='Unresolved duplicate'):
        checked_events(args[0],lambda _:[e,replace(alias,ref_price=5.7)],*args[2:],'approved',resolution)
    with pytest.raises(ValueError,match='omitted'):
        checked_events(args[0],lambda _:[e],*args[2:],'approved',resolution)


def test_split_uses_exact_ratio_and_preserves_window_and_event_day():
    evidence = dict(stock_id='0050',verified_terms=dict(units_after_per_unit_before='4',
        exact_split_price_factor='0.25',new_units_listing_date='2025-06-18',resumption_reference_twd='47.16'))
    ev = add_benchmark_split(pd.DataFrame(columns=EVENT_COLUMNS),evidence)
    quotes = pd.DataFrame(dict(stock_id=['0050']*3,trading_date=pd.to_datetime(['2025-06-10','2025-06-18','2025-06-19']),
        close=[188.65,47.16,48],adj_factor=[1.,1.,1.]))
    factors,_,stats = compare_factors(ev,quotes)
    assert factors.adj_factor.tolist()==[.25,1,1]
    assert stats['changed_rows']==1
    with pytest.raises(ValueError,match='already has an action'):
        add_benchmark_split(ev,evidence)
    quotes.loc[0,'trading_date']=pd.Timestamp('2015-12-31')
    with pytest.raises(ValueError,match='extrapolate'):
        compare_factors(ev,quotes)


def test_missing_old_factor_and_no_event_stock_are_not_certified():
    ev = pd.DataFrame([dict(stock_id='1111',event_date=date(2022,1,4),ratio=.05)])
    quotes = pd.DataFrame(dict(stock_id=['1111','2222'],trading_date=pd.to_datetime(['2022-01-03']*2),
        close=[10.,20.],adj_factor=[None,1.]))
    factors,_,stats = compare_factors(ev,quotes)
    assert factors.adj_factor.tolist()==[.05]  # Do not clip away a real corporate action.
    assert stats['common_valid_rows']==0 and stats['no_official_event_stocks']==['2222']
    assert stats['unclipped_rows_below_0_1']==1


def test_tpex_requires_requested_year_complete_pagination_and_valid_dates():
    payload = dict(stat='ok',date='2022',tables=[dict(fields=['股票代號','公司名稱','終止上櫃日期','終止上櫃原因','公司資料網址'],
        data=[['1752 ','南光','111-01-19','轉上市','https://example.test']],totalCount=1)])
    rows = parse_tpex_delistings(payload,2022)
    assert rows[0]['stock_id']=='1752' and rows[0]['end']=='2022-01-19'
    with pytest.raises(ValueError,match='year'):
        parse_tpex_delistings(payload,2023)
    payload['tables'][0]['totalCount']=2
    with pytest.raises(ValueError,match='pagination'):
        parse_tpex_delistings(payload,2022)


def test_post_delisting_quotes_are_flagged_without_deleting_transfer_prices():
    rows=[dict(stock_id='1752',end='2022-01-19',market='TPEx',name='南光',reason='轉上市')]
    quotes=pd.DataFrame(dict(stock_id=['1752']*2,trading_date=pd.to_datetime(['2022-01-18','2022-01-20'])))
    stocks=pd.DataFrame(dict(stock_id=['1752'],listed_date=[None]))
    result=audit_membership(rows,quotes,stocks)
    assert result[0]['prices_before']==1 and result[0]['prices_on_or_after']==1
    assert result[0]['db_listed_date'] is None and not result[0]['eligible_for_automatic_historical_cohort']
    assert len(quotes)==2
    listings=[dict(stock_id='1752',start='2022-01-19',market='TWSE',note='櫃轉市')]
    result=audit_membership(rows,quotes,stocks,listings)
    assert result[0]['post_end_classification']=='confirmed_transfer_to_twse'
    listings[0]['start']='2022-02-01'
    assert audit_membership(rows,quotes,stocks,listings)[0]['post_end_classification']=='unresolved'


def test_listing_uses_actual_trading_date_not_application_date():
    payload=dict(stat='OK',total=2,fields=['公司代號','公司簡稱','申請日期','股票上市買賣日期','備註'],
        data=[['1752','南光','110.04.28','111.01.19','櫃轉市'],['9999','未掛牌','111.01.01','','']])
    result=parse_twse_listings(payload)
    assert len(result)==1 and result[0]['start']=='2022-01-19'
    payload['total']=3
    with pytest.raises(ValueError,match='Incomplete'):
        parse_twse_listings(payload)


def test_reverse_transfer_is_tied_to_matching_market_dates():
    payload=dict(stat='ok',tables=[dict(fields=['索引','股票代號','公司名稱','上櫃日期'],
        data=[[21,'6423','億而得','115/01/22']])])
    listings=parse_recent_tpex_listings(payload)
    rows=[dict(stock_id='6423',end='2026-01-22',market='TWSE',name='億而得-創',reason='')]
    quotes=pd.DataFrame(dict(stock_id=['6423'],trading_date=pd.to_datetime(['2026-01-23'])))
    stocks=pd.DataFrame(dict(stock_id=['6423'],listed_date=[None]))
    assert audit_membership(rows,quotes,stocks,listings)[0]['post_end_classification']=='corroborated_transition_to_tpex'
