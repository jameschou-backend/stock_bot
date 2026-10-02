from copy import deepcopy
import json

import pandas as pd
import pytest

from skills.market_input_validation import MarketEvidenceError
from skills.official_market_supplement import sha
from skills.official_quote_repair import daily_request_plan, normalize_repair, prepare


def target():
    return dict(market='TWSE',date='2026-09-09',stock_id='2330',positive_official_price=True,source_volume=100000)


def observation(scope='all_daily_sessions', volume=100000):
    return dict(market='TWSE',date='2026-09-09',stock_id='2330',open=100.,high=110.,low=90.,close=105.,
                volume=volume,volume_scope=scope,source_id='verified-source')


@pytest.mark.parametrize('scope',['ordinary_session','unclassified_daily'])
def test_non_total_volume_and_raw_close_never_fill_other_scopes(scope):
    value = normalize_repair(target(),[observation(scope)])
    assert value['volume'] is value['total_daily_volume'] is None
    assert value['adjusted_close'] is None
    assert value['total_daily_volume_verified'] is value['adjusted_price_verified'] is False
    assert value['ready_for_raw_quote_insert'] is False
    assert value['raw_prices_verified'] is True


def test_distinct_volume_scopes_survive_same_stock_day():
    rows = [observation(),dict(observation('ordinary_session',90000),source_id='ordinary-source')]
    value = normalize_repair(target(),rows)
    assert value['volume'] == 100000
    assert value['ordinary_session_volume'] == 90000
    assert value['source_ids'] == ['ordinary-source','verified-source']
    assert value['adjusted_price_verified'] is False


@pytest.mark.parametrize('changes',[{'market':'TPEX'},{'date':'2026-09-08'},{'stock_id':'2317'},
    {'close':None},{'open':0},{'high':99},{'low':float('nan')},{'volume':1.5},
    {'volume':-1},{'volume_scope':'unknown'},{'source_id':None}])
def test_invalid_quote_evidence_fails_closed(changes):
    row = observation(); row.update(changes)
    with pytest.raises((MarketEvidenceError, ValueError)):
        normalize_repair(target(),[row])


def test_conflicting_source_and_mismatched_recorded_gap_fail_closed():
    with pytest.raises(MarketEvidenceError,match='prices'):
        normalize_repair(target(),[observation(),dict(observation(),close=104)])
    with pytest.raises(MarketEvidenceError,match='same-scope'):
        normalize_repair(target(),[observation(),dict(observation(),volume=100001)])
    with pytest.raises(MarketEvidenceError,match='gap volume'):
        normalize_repair(dict(target(),source_volume=123),[observation()])


def test_missing_total_plan_groups_stock_ranges_retaining_exact_days():
    a = normalize_repair(target(),[observation('ordinary_session')])
    b = dict(a,date='2026-09-07')
    verified = normalize_repair(target(),[observation()])
    plan = daily_request_plan([a,b,verified])
    assert len(plan) == 1
    assert plan[0]['required_dates'] == ['2026-09-07','2026-09-09']
    assert plan[0]['start_date'] == '2026-09-07' and plan[0]['end_date'] == '2026-09-09'


def save(path, value):
    path.write_text(json.dumps(value))
    path.with_suffix('.sha256').write_text(sha(path))


def fixture_inputs(root):
    raw = root/'raw.json'
    raw.write_text(json.dumps(dict(date='20260909',stat='OK',type='ALLBUT0999',tables=[dict(
        fields=['證券代號','證券名稱','開盤價','最高價','最低價','收盤價','成交股數'],
        notes=['含一般、零股、盤後定價、鉅額交易'],data=[['2330','測試','100','110','90','105','100000']])])) )
    receipt = root/'receipt.json'
    receipt.write_text(json.dumps(dict(url='https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX',
        params=dict(date='20260909',type='ALLBUT0999',response='json'),http_status=200,
        raw_path='raw.json',raw_sha256=sha(raw))))
    audit = dict(schema='market_input_validation_v2',live_qualified=False,
        source_sha256={'raw.json':sha(raw),'receipt.json':sha(receipt)},
        sources=[dict(market='TWSE',date='2026-09-09',rows=1,path='raw.json',receipt='receipt.json',
            sha256=sha(raw),volume_scope='all_daily_sessions')],missing_positive_quotes_in_scope=[target()])
    save(root/'audit.json',audit)
    pd.DataFrame(dict(date=pd.to_datetime(['2026-09-08']),stock_id=['2330'])).to_parquet(root/'quotes-unmasked.parquet')
    repair = dict(schema='market_input_repairs_v1',live_qualified=False,source_sha256={},
                  output_sha256={'quotes-unmasked.parquet':sha(root/'quotes-unmasked.parquet')})
    save(root/'repair.json',repair)
    return audit, repair


def test_full_source_and_supplement_outputs_are_bound_and_create_only(tmp_path):
    fixture_inputs(tmp_path)
    result = prepare(tmp_path,tmp_path/'audit.json',tmp_path/'repair.json',tmp_path/'out')
    assert result['source_count'] == result['normalized_rows'] == result['raw_ohlc_repaired'] == 1
    assert result['total_daily_volume_verified'] == 1 and result['adjusted_price_verified'] == 0
    normalized = pd.read_parquet(tmp_path/'out/official-normalized.parquet')
    assert normalized.iloc[0].source_id in result['sources']
    assert normalized.iloc[0].volume_scope == 'all_daily_sessions'
    assert result['live_qualified'] is result['frozen_inputs_changed'] is False
    for name, digest in result['output_sha256'].items():
        assert sha(tmp_path/name) == digest
    with pytest.raises(MarketEvidenceError,match='new repository'):
        prepare(tmp_path,tmp_path/'audit.json',tmp_path/'repair.json',tmp_path/'out')


@pytest.mark.parametrize('mutation',['raw','receipt','quotes','descriptor','existing','duplicate_target'])
def test_changed_sources_or_overwrite_fail_before_publishing(tmp_path,mutation):
    audit, repair = fixture_inputs(tmp_path)
    if mutation in ('raw','receipt'):
        (tmp_path/(mutation+'.json')).write_text('{}')
    elif mutation == 'quotes':
        (tmp_path/'quotes-unmasked.parquet').write_text('modified')
    elif mutation == 'descriptor':
        audit['sources'][0]['rows'] = 2; save(tmp_path/'audit.json',audit)
    elif mutation == 'existing':
        pd.DataFrame(dict(date=pd.to_datetime(['2026-09-09']),stock_id=['2330'])).to_parquet(tmp_path/'quotes-unmasked.parquet')
        repair['output_sha256']['quotes-unmasked.parquet'] = sha(tmp_path/'quotes-unmasked.parquet')
        save(tmp_path/'repair.json',repair)
    else:
        audit['missing_positive_quotes_in_scope'].append(deepcopy(target())); save(tmp_path/'audit.json',audit)
    with pytest.raises(MarketEvidenceError):
        prepare(tmp_path,tmp_path/'audit.json',tmp_path/'repair.json',tmp_path/'out')
    assert not (tmp_path/'out/official-sources.json').exists()
