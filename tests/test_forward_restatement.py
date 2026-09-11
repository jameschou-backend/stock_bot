from copy import deepcopy
import pytest
from app import forward_restatement as r, forward_journal as j, forward_portfolio as p
from tests.test_forward_portfolio import setup,buy,fill,cancel,close,clock

PROOF=dict(reference='receipt-1',reviewer='fixture reviewer',reason='手續費／權益漏登核對',text='Synthetic test receipt only. Correct fee is 30 TWD; not a real trade.')

def account(tmp_path):
    path=tmp_path/'original.sqlite3'
    with j.connection(path) as con:
        setup(con);o=buy(con,n=999);f=fill(con,o,400,'99');cancel(con,o)
        close(con,'2026-09-14',{'2492':'100','0050':'100'})
    return path,f

def test_fee_correction_rebuilds_cash_cost_curve_in_separate_book(tmp_path):
    path,f=account(tmp_path);original=r.read(path)
    body=dict(f['body'],fee='30')
    report=r.preview(path,[dict(op='replace',target=f['hash'],body=body)],PROOF,clock('2026-09-15'))
    assert report['after']['cash']=='960370'
    assert report['after']['nav']=='1000370'
    assert report['after']['holdings']['2492']['cost']=='39630'
    target=r.materialize(path,report['command'],tmp_path/'versions',clock('2026-09-15'))
    assert r.read(path)==original
    assert p.summary(target,clock('2026-09-15'))['nav']=='1000370'
    assert r.materialize(path,report['command'],tmp_path/'versions',clock('2026-09-15'))==target
    rows=r.read(target)
    assert all(j.timestamp(x['recorded_at'])==clock('2026-09-15')() for x in rows)
    assert len(r.versions(path,tmp_path/'versions'))==1
    # Actual later trading remains possible; no original timestamps are forged.
    with j.connection(target) as con:
        close(con,'2026-09-15',{'2492':'100','0050':'100'})
        o=buy(con,sid='0050',n=100,day='2026-09-16',signalday='2026-09-15',key='later')
        fill(con,o,100,'99',day='2026-09-16',key='later-fill')
    assert sum(x['qty'] for x in p.summary(target,clock('2026-09-16'))['holdings'])==500


def test_missing_dividend_insert_recomputes_historical_nav_without_spending_receivable(tmp_path):
    path,f=account(tmp_path)
    with j.connection(path) as con:mark=close(con,'2026-09-15',{'2492':'90','0050':'100'})
    body=dict(action_id='div',action_type='cash',stock_id='2492',ex_date='2026-09-15',delivery_date='2026-09-17',eligible_qty=400,cash_per_share='10',amount='4000',evidence='fixture official')
    ops=[dict(op='insert',before=mark['hash'],kind='entitlement',body=body)]
    report=r.preview(path,ops,PROOF,clock('2026-09-16'))
    assert report['after']['nav']=='1000380' and report['after']['cash']=='960380'
    target=r.materialize(path,report['command'],tmp_path/'versions',clock('2026-09-16'))
    with j.connection(target) as con:
        p.submit(con,dict(kind='delivery',id='pay',action_id='div',date='2026-09-17',evidence='receipt'),clock('2026-09-17',8))
    assert p.summary(target,clock('2026-09-17'))['cash']=='964380'


def test_stale_preview_future_fill_overfill_and_tampering_rejected(tmp_path):
    path,f=account(tmp_path)
    for change in [dict(qty=1000),dict(executed_at='2026-09-18T10:00:00+08:00')]:
        with pytest.raises(ValueError):r.preview(path,[dict(op='replace',target=f['hash'],body=dict(f['body'],**change))],PROOF,clock('2026-09-15'))
    good=r.preview(path,[dict(op='replace',target=f['hash'],body=dict(f['body'],fee='30'))],PROOF,clock('2026-09-15'))
    with pytest.raises(ValueError):r.materialize(path,dict(good['command'],id='../bad'),tmp_path/'v',clock('2026-09-15'))
    with pytest.raises(ValueError):r.materialize(path,dict(good['command'],projection_sha256='bad'),tmp_path/'v',clock('2026-09-15'))
    with j.connection(path) as con:close(con,'2026-09-15',{'2492':'100','0050':'100'})
    with pytest.raises(ValueError,match='已更新'):r.materialize(path,good['command'],tmp_path/'v',clock('2026-09-15'))


def test_cannot_void_fill_that_subsequent_split_depends_on(tmp_path):
    path=tmp_path/'original'
    with j.connection(path) as con:
        setup(con);o=buy(con);f=fill(con,o,1000,'100');close(con,'2026-09-14',{'2492':'100','0050':'100'})
        p.submit(con,dict(kind='entitlement',id='split',action_id='split',action_type='split',stock_id='2492',ex_date='2026-09-15',delivery_date='2026-09-16',eligible_qty=1000,ratio='2',result_qty=2000,evidence='fixture'),clock('2026-09-15',8))
    with pytest.raises(ValueError):r.preview(path,[dict(op='void',target=f['hash'])],PROOF,clock('2026-09-16'))
