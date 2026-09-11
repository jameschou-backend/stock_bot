from copy import deepcopy
import json
import pytest
from app import forward_odd_lot as q, forward_journal as j, forward_portfolio as p
from tests.test_forward_portfolio import setup,buy,fill,clock


def payload():
    return dict(rtcode='0000',msgArray=[dict(c='2492',ex='tse',ch='2492.tw',d='20260914',tlong=str(int(clock('2026-09-14',10)().timestamp()*1000)),a='99_100_',f='50_100_',b='98_97_',g='10_20_',v='500')])

def quote():return q.parse(payload(),'tse','2492',clock('2026-09-14',10)().isoformat())
def order():return dict(stock_id='2492',channel='odd',session='2026-09-14',side='buy',limit_price='100',qty=200,filled=0)


def test_share_units_limit_depth_and_stale_quotes_never_guarantee_fill():
    result=q.assess(quote(),order(),'2026-09-14T10:00:01+08:00')
    assert result['visible_shares_at_limit']==150 and not result['suitable_snapshot'] and not result['fill_guaranteed']
    assert not q.assess(quote(),dict(order(),qty=50),'2026-09-14T10:01:00+08:00')['suitable_snapshot']
    assert q.assess(quote(),dict(order(),qty=50),'2026-09-14T10:00:01+08:00')['suitable_snapshot']
    for change in [dict(channel='board'),dict(stock_id='0050')]:
        with pytest.raises(ValueError):q.assess(quote(),dict(order(),**change),'2026-09-14T10:00:01+08:00')


def test_invalid_ladders_market_order_missing_stock_future_time_rejected():
    for update in [dict(f='50_'),dict(f='0.5_1_'),dict(a='0_100_'),dict(a='100_99_'),dict(b='100_97_'),dict(c='0050'),dict(d='20260915')]:
        x=payload();x['msgArray'][0].update(update)
        with pytest.raises((ValueError,KeyError)):q.parse(x,'tse','2492','2026-09-14T10:00:01+08:00')


def test_shared_cache_error_cache_and_no_finmind_requests(tmp_path):
    class Response:
        text=json.dumps(payload())
        def raise_for_status(self):pass
    class Session:
        calls=0
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,*args,**kwargs):
            Session.calls+=1;return Response()
    path=tmp_path/'quotes'
    a=q.refresh('tse','2492',path,clock('2026-09-14',10),Session)
    assert a['body']['status']=='ok'
    assert q.refresh('tse','2492',path,clock('2026-09-14',10),Session)==a and Session.calls==1
    Response.text='not json'
    b=q.refresh('tse','2492',path,clock('2026-09-14',11),Session)
    assert b['body']['status']=='error'
    assert q.refresh('tse','2492',path,clock('2026-09-14',11),Session)==b and Session.calls==2


def test_link_partial_reports_keeps_accounting_unchanged_and_detects_double_depth(tmp_path):
    book=tmp_path/'book';evidence_path=tmp_path/'quotes'
    with j.connection(evidence_path) as con:
        obs=j.append(con,'q','odd_quote',dict(status='ok',quote=quote()),clock('2026-09-14',10))
    with j.connection(book) as con:
        setup(con);o=buy(con,n=200);a=fill(con,o,100,'99');b=fill(con,o,100,'99',key='second')
    proof=dict(reference='report',reviewer='tester',reason='test report',text='Synthetic quote and execution evidence for isolated tests only.')
    before=p.summary(book,clock('2026-09-14'))
    link=q.attach(book,a['hash'],obs['hash'],proof,evidence_path,clock('2026-09-14',11))
    assert q.attach(book,a['hash'],obs['hash'],proof,evidence_path,clock('2026-09-14',11))==link
    second=q.attach(book,b['hash'],obs['hash'],proof,evidence_path,clock('2026-09-14',11))
    assert any('累計' in r for r in second['body']['assessment']['reasons'])
    after=p.summary(book,clock('2026-09-14'))
    assert (before['cash'],before['holdings'],before['fill_count'])==(after['cash'],after['holdings'],after['fill_count'])
    assert len(q.review(book)['fills'])==2

    # Restated fills keep old attachments as history, without inheriting their attestation.
    from app import forward_restatement as repair
    preview=repair.preview(book,[dict(op='replace',target=a['hash'],body=dict(a['body'],fee='30'))],proof,clock('2026-09-15'))
    derived=repair.materialize(book,preview['command'],tmp_path/'versions',clock('2026-09-15'))
    assert all(x['quote_status']=='缺少零股行情／成交憑證連結' for x in q.review(derived)['fills'])
    assert len([x for x in repair.read(derived) if x['kind']=='historical_execution_evidence'])==2
