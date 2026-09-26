"""00631L uses the same funding and late-receipt rules as existing stock IDs."""
from copy import deepcopy
import pytest
from app.manual_receipt_entry import new_document,append_event,holdings
from app.manual_execution_audit import audit
from skills.contingent_execution import Plan,ConfirmationGate,OPEN
from skills.manual_security_ids import valid_manual_security_id

@pytest.mark.parametrize('sid',['00631l','00632R','00631','00631LL','2330.TW','２３３０','١٢٣٤',' 00631L',None,631])
def test_exception_does_not_broaden_identity_acceptance(sid):
    assert not valid_manual_security_id(sid)
    with pytest.raises(ValueError):Plan('x',sid,'buy','board',1000,1000,1002000,'2022-01-03').validate('2022-01-04')

@pytest.mark.parametrize('sid',['00631L','0050','2330'])
def test_holdings_keep_exact_identifier_and_reject_duplicates(sid):
    row={'股票代號':sid,'股數':1000}
    assert holdings([row])=={sid:1000}
    with pytest.raises(ValueError):holdings([row,row])

def draft(side='buy'):
    return new_document('2022-01-04','1000000' if side=='buy' else '0',1,
        [] if side=='buy' else [{'股票代號':'00631L','股數':1000}],[],[{
            '委託代號':'A','股票代號':'00631L','買賣':'買進' if side=='buy' else '賣出','盤別':'整股',
            '股數':1000,'限價（元）':'20.00','預算（元）':'20050' if side=='buy' else '0','訊號日':'2022-01-03'}])

def test_manual_etf_buy_requires_funds_and_keeps_unverified_classification():
    d=draft();before=deepcopy(d);proof=b'synthetic unit-test receipt, not broker evidence'
    with pytest.raises(ValueError,match='額度'):
        append_event(d,'submit','S','09:10:00','09:10:01',proof,dict(order_id='A'))
    assert d==before
    d=append_event(d,'funds_snapshot','F','09:10:00','09:10:01',proof,dict(available_cents=100000000,covers_through=None))
    d=append_event(d,'submit','S','09:10:02','09:10:03',proof,dict(order_id='A'))
    d=append_event(d,'fill','T','09:10:04','09:10:05',proof,dict(order_id='A',qty=1000,price_cents=2000,fee_cents=2850,tax_cents=0))
    r=audit(d)
    assert r['decisions'][-1]['holdings']=={'00631L':1000}
    assert r['decisions'][-1]['cash_cents']==97997150
    assert not r['live_qualified'] and not r['broker_execution_verified']

def test_manual_etf_sale_records_actual_tax_without_crediting_availability():
    d=draft('sell');proof=b'synthetic unit-test receipt'
    d=append_event(d,'submit','S','09:10:00','09:10:01',proof,dict(order_id='A'))
    d=append_event(d,'fill','T','09:10:02','09:10:03',proof,dict(order_id='A',qty=1000,price_cents=2000,fee_cents=2850,tax_cents=2000))
    r=audit(d)['decisions'][-1]
    assert r['cash_cents']==1995150 and r['available_cents']==0 and r['holdings']=={}

def test_etf_confirmation_respects_reserved_cash_and_opening_shares():
    p=Plan('x','00631L','buy','board',1000,2000,2005000,'2022-01-03')
    g=ConfirmationGate('2022-01-04',[p],{},2005000,1)
    assert g.submit_ready(OPEN)==['x'] and g.available==0
    g.confirm_fill('f','x',OPEN+1,1000,2000,2002850)
    assert g.holdings=={'00631L':1000}
    bad=Plan('s','00631L','sell','board',2000,2000,0,'2022-01-03')
    with pytest.raises(ValueError):ConfirmationGate('2022-01-04',[bad],{'00631L':1000},0,1)
