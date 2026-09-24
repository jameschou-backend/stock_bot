from copy import deepcopy
import pytest
from app.manual_execution_audit import audit


def plan(oid,sid,side,qty=1000,channel='board'):
    return dict(order_id=oid,stock_id=sid,side=side,qty=qty,channel=channel,
        limit_cents=1000,budget_cents=qty*1000+2000 if side=='buy' else 0,signal_date='2026-09-23')


def event(n,kind,**kw):
    return dict(id=str(n),kind=kind,occurred_at=f'2026-09-24T09:10:{n*2:02}+08:00',
        received_at=f'2026-09-24T09:10:{n*2+1:02}+08:00',source_sha256='a'*64,**kw)


def doc(plans,holdings=None,cash=2000000,events=None):
    return dict(schema='dahu_manual_receipts_v1',session='2026-09-24',slots=1,
        opening_holdings=holdings or {},undelivered_stock_rights={},opening_cash_cents=cash,plans=plans,events=events or [])


def cancellation():
    return doc([plan('a','2330','buy'),plan('b','2317','buy')],events=[
        event(1,'funds_snapshot',available_cents=2000000,covers_through=None),
        event(2,'submit',order_id='a'),event(3,'cancel_request',order_id='a'),
        event(4,'cancel_ack',order_id='a',request_id='3',cumulative_filled_qty=0,cancelled_qty=1000),
        event(5,'funds_snapshot',available_cents=2000000,covers_through='4'),
        event(6,'submit',order_id='b')])


def test_cancel_ack_and_fresh_available_snapshot_release_slot_and_budget():
    r=audit(cancellation())
    assert r['sequence_consistent'] and r['open_orders']==['b']
    assert r['decisions'][-1]['available_cents']==998000
    assert r['live_qualified'] is False and r['broker_execution_verified'] is False


@pytest.mark.parametrize('omit,match',[(4,'名額'),(5,'額度')])
def test_cancel_request_alone_or_ack_without_funds_cannot_buy(omit,match):
    d=cancellation();d['events']=[x for x in d['events'] if int(x['id'])!=omit]
    if omit==4:d['events']=[x for x in d['events'] if x['kind']!='funds_snapshot' or x['id']=='1']
    with pytest.raises(ValueError,match=match):audit(d)


def test_cancel_rejected_keeps_slot():
    d=cancellation();d['events'][3]=event(4,'cancel_rejected',order_id='a',request_id='3')
    d['events'][4]['available_cents']=998000
    with pytest.raises(ValueError,match='名額'):audit(d)


def test_missing_late_fill_prevents_cancel_release():
    d=cancellation();d['events'][3]['cumulative_filled_qty']=1000
    with pytest.raises(ValueError,match='尚未收到'):audit(d)


def test_partial_cancel_keeps_actual_holding_and_blocks_slot():
    d=cancellation();d['plans'][0]=plan('a','2330','buy',100,'odd')
    d['events']=[event(1,'funds_snapshot',available_cents=2000000,covers_through=None),
        event(2,'submit',order_id='a'),event(3,'cancel_request',order_id='a'),
        event(4,'fill',order_id='a',qty=1,price_cents=1000,fee_cents=1,tax_cents=0),
        event(5,'cancel_ack',order_id='a',request_id='3',cumulative_filled_qty=1,cancelled_qty=99),
        event(6,'funds_snapshot',available_cents=1998999,covers_through='5'),event(7,'submit',order_id='b')]
    with pytest.raises(ValueError,match='名額'):audit(d)


@pytest.mark.parametrize('snapshot',[False,True])
def test_sale_confirmation_requires_separate_available_funds(snapshot):
    d=doc([plan('s','2330','sell'),plan('b','2317','buy',1,'odd')],{'2330':1000},0,[
        event(1,'submit',order_id='s'),event(2,'fill',order_id='s',qty=1000,price_cents=1000,fee_cents=2000,tax_cents=3000)])
    if snapshot:d['events'].append(event(3,'funds_snapshot',available_cents=995000,covers_through='2'))
    d['events'].append(event(4,'submit',order_id='b'))
    if snapshot:assert audit(d)['decisions'][-1]['available_cents']==992000
    else:
        with pytest.raises(ValueError,match='額度'):audit(d)


@pytest.mark.parametrize('rights',[False,True])
def test_one_share_or_stock_right_still_blocks_replacement(rights):
    d=doc([plan('b','2317','buy')],{'2330':1} if not rights else {},events=[
        event(1,'funds_snapshot',available_cents=2000000,covers_through=None),event(2,'submit',order_id='b')])
    if rights:d['undelivered_stock_rights']={'2330':1}
    with pytest.raises(ValueError,match='名額'):audit(d)


@pytest.mark.parametrize('key,value,match',[
    ('covers_through','3','涵蓋'),('available_cents',2000001,'超過'),
    ('occurred_at','2026-09-24T09:10:05+08:00','早於'),('source_sha256','wrong','SHA256')])
def test_snapshot_validation(key,value,match):
    d=cancellation();d['events'][4][key]=value
    with pytest.raises(ValueError,match=match):audit(d)


def test_duplicate_and_out_of_order_receipts_rejected_without_mutating_input():
    d=cancellation();d['events'][3]['id']='3';before=deepcopy(d)
    with pytest.raises(ValueError,match='重複'):audit(d)
    assert d==before
    d=cancellation();d['events'][3]['received_at']=d['events'][2]['received_at']
    with pytest.raises(ValueError,match='時間'):audit(d)


def test_fill_after_cancel_requires_reconciliation():
    d=cancellation();d['events']=d['events'][:4]+[
        event(5,'fill',order_id='a',qty=1000,price_cents=1000,fee_cents=2000,tax_cents=0)]
    with pytest.raises(ValueError,match='終結'):audit(d)


def test_blank_example_or_unmapped_fields_are_not_evidence():
    d=cancellation();d['example_only']=True
    with pytest.raises(ValueError,match='範本'):audit(d)
    d=cancellation();d['events']=[]
    with pytest.raises(ValueError,match='空白'):audit(d)
    d=cancellation();d['events'][0]['bank_cash']=2000000
    with pytest.raises(ValueError,match='欄位'):audit(d)


def test_backdated_submit_cannot_use_a_later_received_funds_snapshot():
    d=cancellation();d['events'][1]['occurred_at']='2026-09-24T09:10:01+08:00'
    with pytest.raises(ValueError,match='快照'):audit(d)
