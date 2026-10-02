from copy import deepcopy

import pytest

from skills.market_input_validation import (MarketEvidenceError, compare_quote, excluded,
    execution_scope, ordinary_capacity, parse_market_day, require_complete, resolve_episode)


def payload(market='TPEX'):
    if market == 'TPEX':
        table = dict(title='上櫃股票每日收盤行情(不含定價)',date='115/09/09',
            category='所有證券(不含權證、牛熊證)',totalCount=1,
            fields=['代號','名稱','開盤 ','最高','最低','收盤','成交股數'],
            data=[['2330','測試','100','110','90','105','100,000']])
        return dict(stat='ok',date='20260909',tables=[table])
    table = dict(fields=['證券代號','證券名稱','開盤價','最高價','最低價','收盤價','成交股數'],
        notes=['本統計資訊含一般、零股、盤後定價、鉅額交易'],
        data=[['2330','測試','100','110','90','105','100,005']])
    return dict(stat='OK',date='20260909',type='ALLBUT0999',tables=[table])


def test_primary_parsers_keep_different_volume_scopes():
    tp = parse_market_day(payload(),'TPEX','2026-09-09')['2330']
    tw = parse_market_day(payload('TWSE'),'TWSE','2026-09-09')['2330']
    assert tp['volume_scope'] == 'ordinary_session'
    assert tw['volume_scope'] == 'all_daily_sessions'
    assert tp['high'] == tw['high'] == 110
    local = dict(open=100,high=110,low=90,close=105,volume=100005)
    a,b = compare_quote(local,tp),compare_quote(local,tw)
    assert a['high'] == b['high'] == 'matched'
    assert a['total_volume'] == 'different_scope'
    assert b['ordinary_volume'] == 'unverified'
    assert b['total_volume'] == 'matched'


@pytest.mark.parametrize('mutation', ['date','category','count','duplicate','width','impossible','negative','noninteger'])
def test_malformed_source_cannot_raise_verification_coverage(mutation):
    p = payload()
    table = p['tables'][0]
    if mutation == 'date': p['date'] = '20260908'
    elif mutation == 'category': table['category'] = '半導體'
    elif mutation == 'count': table['totalCount'] = 2
    elif mutation == 'duplicate':
        table['totalCount'] = 2
        table['data'].append(deepcopy(table['data'][0]))
    elif mutation == 'width': table['data'][0].pop()
    elif mutation == 'impossible': table['data'][0][3] = '99'
    elif mutation == 'negative': table['data'][0][-1] = '-1'
    else: table['data'][0][-1] = '1.5'
    with pytest.raises(MarketEvidenceError):
        parse_market_day(p,'TPEX','2026-09-09')


def test_missing_ohlc_is_unavailable_not_a_match():
    p = payload()
    p['tables'][0]['data'][0][2:6] = ['--']*4
    row = parse_market_day(p,'TPEX','2026-09-09')['2330']
    result = compare_quote(dict(open=100,high=110,low=90,close=105,volume=0),row)
    assert {result[k] for k in ('open','high','low','close')} == {'unavailable'}


def test_transfer_and_termination_dates_are_exclusive():
    episodes = [dict(stock_id='2330',start='2019-01-01',end='2020-01-01',market='TPEx'),
                dict(stock_id='2330',start='2020-01-01',end='2025-01-01',market='TWSE')]
    assert resolve_episode(episodes,'2330','2019-12-31')['market'] == 'TPEx'
    assert resolve_episode(episodes,'2330','2020-01-01')['market'] == 'TWSE'
    assert resolve_episode(episodes,'2330','2025-01-01') is None
    episodes[0]['end'] = '2021-01-01'
    with pytest.raises(MarketEvidenceError,match='Overlapping'):
        resolve_episode(episodes,'2330','2020-01-02')


def test_intraday_halt_does_not_exclude_a_whole_day():
    row = dict(stock_id='2330',market='TWSE',kind='information_halt',start='2024-01-01',end='2024-01-03',
               source_row=[0,'2330','test','date','10:00','date','11:00'])
    assert excluded([row],'2330','2024-01-02','TWSE') is None
    row['source_row'][4] = row['source_row'][6] = '8:00'
    assert excluded([row],'2330','2024-01-02','TWSE') == 'information_halt'
    assert excluded([row],'2330','2024-01-03','TWSE') is None
    assert excluded([row],'2330','2024-01-02','TPEX') is None


def test_capacity_requires_twenty_prior_ordinary_rows():
    current = dict(date='2026-09-21',stock_id='2330',market='TPEX',volume_scope='ordinary_session',volume=200000)
    prior = [dict(current,date=f'2026-09-{i:02d}',volume=100000) for i in range(1,21)]
    assert ordinary_capacity(current,prior) == 1000
    for rows in (prior[:-1], [None]+prior[1:], prior[:-1]+[current], prior[:-1]+[prior[0]]):
        with pytest.raises(MarketEvidenceError): ordinary_capacity(current,rows)
    with pytest.raises(MarketEvidenceError,match='not ordinary'):
        ordinary_capacity(dict(current,volume_scope='all_daily_sessions'),prior)


def test_known_day_bound_does_not_claim_prior_volume_verified():
    current = parse_market_day(payload(),'TPEX','2026-09-09')['2330']
    trade = dict(date='2026-09-09',stock_id='2330',channel='board',sequence=1,qty=1000,
        capacity_qty=2000,source_high=110,source_low=90,source_volume=200000,prior_avg_volume20=200000)
    episodes = [dict(stock_id='2330',market='TPEx',category='股票',start='2000-01-01',end=None)]
    result = execution_scope(dict(trades=[trade]),{('TPEX','2026-09-09','2330'):current},
                             ['2026-09-08','2026-09-09'],episodes,[])
    r = result['rows'][0]
    assert r['same_scope_day_upper_bound'] == 1000
    assert r['status'] == 'prior_ordinary_volume_incomplete'
    assert not result['all_capacity_verified']
    trade['qty'] = 2000
    assert execution_scope(dict(trades=[trade]),{('TPEX','2026-09-09','2330'):current},
        ['2026-09-08','2026-09-09'],episodes,[])['rows'][0]['status'] == 'day_bound_conflict'


def test_verified_mode_does_not_silently_drop_unknown_data():
    names = ('all_execution_prices_verified','all_holding_marks_verified','all_ordinary_capacity_verified',
             'all_signal_histories_verified','all_historical_market_days_observed','known_identity_checks_passed',
             'all_observed_positive_quotes_present','complete_historical_universe',
             'no_observed_price_conflicts')
    report = dict(checks=dict.fromkeys(names,True),live_qualified=False)
    require_complete(report)
    for field in names:
        bad = deepcopy(report)
        bad['checks'][field] = False
        with pytest.raises(MarketEvidenceError,match=field): require_complete(bad)


def test_multiple_orders_share_the_same_daily_capacity():
    current = parse_market_day(payload(),'TPEX','2026-09-09')['2330']
    trade = dict(date='2026-09-09',stock_id='2330',channel='board',sequence=1,qty=1000,
        capacity_qty=2000,source_high=110,source_low=90,source_volume=200000,prior_avg_volume20=200000)
    episodes = [dict(stock_id='2330',market='TPEx',category='股票',start='2000-01-01',end=None)]
    result = execution_scope(dict(trades=[trade,dict(trade,sequence=2)]),
        {('TPEX','2026-09-09','2330'):current},['2026-09-08','2026-09-09'],episodes,[])
    assert result['rows'][1]['cumulative_day_qty'] == 2000
    assert result['rows'][1]['status'] == 'day_bound_conflict'


def test_repair_requires_dated_primary_prices_without_substituting_volume():
    from scripts.repair_market_input_gaps import normalized_repair
    primary = parse_market_day(payload(),'TPEX','2026-09-09')['2330']
    raw = dict(query=dict(dataset='TaiwanStockPrice',data_id='2330',start_date='2026-09-09',end_date='2026-09-09'),
        data=[dict(stock_id='2330',date='2026-09-09',open=100,max=110,min=90,close=105,Trading_Volume=100500)])
    result = normalized_repair(raw,'2330','2026-09-09',primary)
    assert result['quote']['volume'] == 100500
    assert result['official']['volume'] == 100000
    assert result['field_checks']['total_volume'] == 'different_scope'
    for key,value in [('date','2026-09-08'),('stock_id','2337'),('close',106),('Trading_Volume',99999)]:
        bad = deepcopy(raw)
        bad['data'][0][key] = value
        with pytest.raises(MarketEvidenceError): normalized_repair(bad,'2330','2026-09-09',primary)


def test_unfilled_order_or_pending_entry_prevents_no_impact_claim():
    from scripts.repair_market_input_gaps import repair_exposure
    a = dict(holdings=[],trades=[],orders=[dict(stock_id='2330',date='2026-05-22')])
    entries = [dict(members=['2330'],signal_date='2026-05-21',entry_date='2026-05-22')]
    counts = repair_exposure(a,entries,{'2330'},'2026-05-22')
    assert counts['affected_holdings_after_repair'] == 0
    assert counts['affected_orders_after_repair'] == 1
    assert counts['affected_candidates_after_repair'] == 1


def test_quality_repair_uses_independent_series_with_both_side_anchors():
    import pandas as pd
    from scripts.repair_market_input_gaps import aligned_quality
    days = pd.to_datetime(['2026-05-21','2026-05-22','2026-05-25'])
    independent = pd.Series([100.,105.,110.],index=days)
    existing = pd.Series([50.,float('nan'),55.],index=days)
    assert aligned_quality(independent,existing,'2026-05-22')['value'] == 52.5
    with pytest.raises(MarketEvidenceError,match='both sides'):
        aligned_quality(independent,existing.iloc[:2],'2026-05-22')
    existing.iloc[-1] = 56.
    with pytest.raises(MarketEvidenceError,match='basis differs'):
        aligned_quality(independent,existing,'2026-05-22')
