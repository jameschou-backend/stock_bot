from copy import deepcopy

import pandas as pd
import pytest

from skills.market_input_validation import MarketEvidenceError
from skills.official_quote_repair_enrichment import (align_independent_close, provider_rows,
    ready_columns, with_provider_total)


def official():
    return dict(stock_id='4415',date='2019-06-14',open=6.,high=6.,low=6.,close=6.,
        total_daily_volume=None,ordinary_session_volume=2000,total_daily_volume_verified=False)


def provider():
    return dict(stock_id='4415',date='2019-06-14',open=6.,max=6.,min=6.,close=6.,Trading_Volume=2100)


def test_provider_resolves_total_but_does_not_certify_official_all_session_sum():
    row = with_provider_total(official(),provider(),source_path='receipt.json')
    assert row['total_daily_volume'] == 2100
    assert row['ordinary_session_volume'] == 2000
    assert row['total_daily_volume_verified'] is False
    assert row['ready_for_raw_quote_insert'] is True
    ready = ready_columns(row)
    assert ready['total_volume'] == 2100 and ready['quality_adjusted_close'] is None
    assert ready['ready_for_signal_rebuild'] is False


@pytest.mark.parametrize('change',[{'stock_id':'2330'},{'date':'2019-06-13'},
    {'close':6.1},{'Trading_Volume':1000},{'Trading_Volume':2.5}])
def test_provider_identity_price_and_volume_conflicts_rejected(change):
    row = provider(); row.update(change)
    with pytest.raises(MarketEvidenceError):
        with_provider_total(official(),row,source_path='receipt.json')


def test_provider_scope_duplicate_and_missing_are_not_silently_accepted():
    query = dict(dataset='TaiwanStockPrice',data_id='4415',start_date='2019-06-14',end_date='2019-07-09')
    receipt = dict(query=query,data=[provider()])
    assert list(provider_rows(receipt,query)) == ['2019-06-14']
    assert provider_rows(dict(query=query,data=[]),query) == {}
    for mutation in ('duplicate','identity','outside','query'):
        bad = deepcopy(receipt)
        if mutation == 'duplicate': bad['data'].append(provider())
        elif mutation == 'identity': bad['data'][0]['stock_id'] = '2330'
        elif mutation == 'outside': bad['data'][0]['date'] = '2019-06-13'
        else: bad['query']['dataset'] = 'TaiwanStockPriceAdj'
        with pytest.raises(MarketEvidenceError):
            provider_rows(bad,query)


def test_two_sided_adjusted_alignment_uses_independent_values_not_raw_price():
    days = pd.date_range('2020-01-01',periods=3)
    series = pd.Series([8.,9.,10.],index=days)
    existing = pd.Series([4.,float('nan'),5.],index=days)
    detail = align_independent_close(series,existing,'2020-01-02')
    assert detail['value'] == 4.5 and detail['scale'] == .5
    assert detail['method'] == 'two_sided_nearest_scale_alignment'
    existing.iloc[-1] = 6
    with pytest.raises(MarketEvidenceError,match='basis conflicts'):
        align_independent_close(series,existing,'2020-01-02')


def test_independent_prefix_requires_broad_constant_scale_overlap():
    days = pd.date_range('2020-01-01',periods=25)
    series = pd.Series(range(10,35),index=days,dtype=float)
    existing = series * .9; existing.iloc[:5] = float('nan')
    detail = align_independent_close(series,existing,'2020-01-01')
    assert detail['value'] == pytest.approx(9.)
    assert detail['overlap_count'] == 20
    assert detail['method'] == 'independent_prefix_all_overlap_scale_alignment'
    existing.iloc[-1] += 1
    with pytest.raises(MarketEvidenceError,match='basis conflicts'):
        align_independent_close(series,existing,'2020-01-01')
    existing.iloc[-1] = float('nan')
    with pytest.raises(MarketEvidenceError,match='prefix overlap'):
        align_independent_close(series,existing,'2020-01-01')


def test_empty_adjusted_basis_requires_explicit_full_series_initialization():
    days = pd.date_range('2020-01-01',periods=3)
    series = pd.Series([8.,9.,10.],index=days)
    empty = pd.Series(float('nan'),index=days)
    with pytest.raises(MarketEvidenceError,match='full-series'):
        align_independent_close(series,empty,'2020-01-02')
    detail = align_independent_close(series,empty,'2020-01-02',allow_empty_existing=True)
    assert detail['value'] == 9 and detail['overlap_count'] == 0
    assert detail['method'] == 'independent_full_series_initialization'
