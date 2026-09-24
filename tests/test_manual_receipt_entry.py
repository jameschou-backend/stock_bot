from copy import deepcopy
import hashlib
import pytest
from app.manual_receipt_entry import cents, shares, new_document, append_event
from app.manual_execution_audit import audit


def draft():
    return new_document('2022-01-04','1000000',3,[],[],[{
        '委託代號':'B1','股票代號':'0050','買賣':'買進','盤別':'零股','股數':1,
        '限價（元）':'145.50','預算（元）':'165.50','訊號日':'2022-01-03'}])


@pytest.mark.parametrize('value',['NaN','Infinity','-1','1.005',True,''])
def test_reject_invalid_money_instead_of_rounding(value):
    with pytest.raises(ValueError):cents(value)


@pytest.mark.parametrize('value',['1.5','-1',True,'1,000'])
def test_share_units_cannot_be_rounded(value):
    with pytest.raises(ValueError):shares(value,1)


def test_yuan_to_cents_and_0050_identity_preserved():
    d=draft()
    assert d['opening_cash_cents']==100000000
    assert d['plans'][0]['limit_cents']==14550
    assert d['plans'][0]['stock_id']=='0050'
    assert cents('0.29')==29


def test_actual_bytes_hashed_and_consistent_prefix_stays_unqualified():
    d=draft();before=deepcopy(d);evidence=b'fixture only'
    d=append_event(d,'funds_snapshot','E1','09:10:00','09:10:01',evidence,
        dict(available_cents=100000000,covers_through=None))
    assert before['events']==[]
    assert d['events'][0]['source_sha256']==hashlib.sha256(evidence).hexdigest()
    d=append_event(d,'submit','E2','09:10:02','09:10:03',evidence,dict(order_id='B1'))
    d=append_event(d,'fill','E3','09:10:04','09:10:05',evidence,
        dict(order_id='B1',qty=1,price_cents=14550,fee_cents=2000,tax_cents=0))
    r=audit(d)
    assert r['decisions'][-1]['cash_cents']==99983450
    assert r['live_qualified'] is False and r['broker_execution_verified'] is False


def test_invalid_addition_does_not_mutate_draft():
    d=draft();before=deepcopy(d)
    with pytest.raises(ValueError,match='額度'):
        append_event(d,'submit','E1','09:10:00','09:10:01',b'proof',dict(order_id='B1'))
    assert d==before


def test_no_synthetic_hash_without_source_bytes():
    with pytest.raises(ValueError,match='證據檔'):
        append_event(draft(),'funds_snapshot','E1','09:10:00','09:10:01',b'',
            dict(available_cents=0,covers_through=None))


def test_cannot_override_timestamp_or_digest():
    with pytest.raises(ValueError,match='覆蓋'):
        append_event(draft(),'funds_snapshot','E1','09:10:00','09:10:01',b'proof',
            dict(available_cents=0,covers_through=None,source_sha256='a'*64))


def test_same_day_plan_is_rejected_before_event_entry():
    rows=[{'委託代號':'B1','股票代號':'2330','買賣':'買進','盤別':'零股','股數':1,
        '限價（元）':'10','預算（元）':'30','訊號日':'2022-01-04'}]
    with pytest.raises(ValueError,match='signal date'):
        new_document('2022-01-04','1000000',3,[],[],rows)
