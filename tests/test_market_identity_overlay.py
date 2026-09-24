from copy import deepcopy
import pytest

from skills.market_identity_overlay import apply_starts, audit_rows


def episode(sid='1234', venue='TPEx', start=None, end='2023-06-01', category='unconfirmed'):
    return dict(stock_id=sid, market=venue, start=start, end=end, category=category)


def proof(**kwargs):
    return dict(stock_id='1234', market='TPEx', start='2010-02-03', category='股票', **kwargs)


def test_fill_is_copy_and_transfer_day_uses_new_market():
    original = [episode(), episode(venue='TWSE', start='2023-06-01', end=None, category='股票')]
    before = deepcopy(original)
    filled = apply_starts(original, [proof()])
    assert original == before
    assert audit_rows(filled, [{'stock_id':'1234','date':'2023-05-31'}], {'1234':'TPEX'})['passed']
    result = audit_rows(filled, [{'stock_id':'1234','date':'2023-06-01'}], {'1234':'TPEX'})
    assert result['issues'][0]['reasons'] == ['execution_market']
    assert result['issues'][0]['identity']['market'] == 'TWSE'


@pytest.mark.parametrize('rows,proofs', [
    ([episode()], [proof(), proof()]),
    ([episode(start='2010-02-03')], [proof()]),
    ([episode(end='2000-01-01')], [proof()]),
    ([episode(), episode(venue='TWSE',start='2022-01-01',end=None)], [proof()]),
    ([episode(category='臺灣存託憑證(TDR)')], [proof()]),
    ([episode(), episode()], [proof()]),
])
def test_conflicts_are_rejected(rows, proofs):
    with pytest.raises(ValueError): apply_starts(rows, proofs)


def test_unknown_stays_unknown_and_class_not_inferred():
    result = audit_rows([episode()], [{'stock_id':'1234','date':'2022-01-03'}], {'1234':'TPEX'})
    assert result['issues'][0]['reasons'] == ['unknown']
    assert not result['continuous_eligibility_proven']
    filled = apply_starts([episode()], [{**proof(), 'category':'unconfirmed'}])
    assert audit_rows(filled,[{'stock_id':'1234','date':'2022-01-03'}],{'1234':'TPEX'})['issues'][0]['reasons'] == ['security_category']


def test_etf_is_only_allowed_for_explicit_benchmark():
    eps=[episode(sid='0050',venue='TWSE',start='2003-06-30',end=None,category='ETF')]
    rows=[{'stock_id':'0050','date':'2022-01-03'}]
    assert not audit_rows(eps,rows,{'0050':'TWSE'})['passed']
    assert audit_rows(eps,rows,{'0050':'TWSE'},benchmark=True)['passed']


def test_outside_interval_or_missing_engine_market_cannot_pass():
    eps=[episode(start='2022-01-05',category='股票')]
    rows=[{'stock_id':'1234','date':'2022-01-04'},{'stock_id':'1234','date':'2022-01-05'},
          {'stock_id':'1234','date':'2023-06-01'}]
    result=audit_rows(eps,rows,{})
    assert [r['reasons'][0] for r in result['issues']] == ['outside_verified_intervals','execution_market','outside_verified_intervals']


@pytest.mark.parametrize('start', ['20220203','2022-02-30',None])
def test_invalid_dates_fail(start):
    with pytest.raises((ValueError,TypeError)):
        apply_starts([episode()],[{**proof(),'start':start}])


def test_reviewed_isin_update_date_can_be_corrected_without_losing_observation():
    original=[{**episode(start='2026-06-01',end=None,category='股票'),
               'start_evidence':'current_official_ISIN','snapshot_date':'2026-09-14'}]
    evidence={**proof(),'replaces_snapshot_start':'2026-06-01'}
    filled=apply_starts(original,[evidence])
    assert original[0]['start']=='2026-06-01'
    assert filled[0]['listing_evidence']['replaces_snapshot_start']=='2026-06-01'
    assert audit_rows(filled,[{'stock_id':'1234','date':'2022-01-03'}],{'1234':'TPEX'})['passed']
    for change in ({'replaces_snapshot_start':'2026-07-01'},{'start':'2026-07-01'}):
        with pytest.raises(ValueError): apply_starts(original,[{**evidence,**change}])
    original[0]['start_evidence']='official_listing_record'
    with pytest.raises(ValueError): apply_starts(original,[evidence])
