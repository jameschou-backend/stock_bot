from copy import deepcopy
import pytest
from app import forward_corporate_resolution as r, forward_corporate_audit as a, forward_portfolio as p, forward_journal as j
from test_forward_corporate_audit import account, sources, pair, codes
from test_forward_portfolio import clock


def evidence():
    return dict(url='https://www.twse.com.tw/fixture',title='TEST ONLY',published_at='2026-09-14T18:00:00+08:00',
                text='Synthetic test terms only; no real company announcement or customer receipt.',reviewer='test-reviewer')


def terms(kind='cash'):
    if kind=='cash':
        return dict(stock_id='2492',ex_date='2026-09-15',delivery_date='2026-09-17',action_type='cash',cash_per_share='10')
    return dict(stock_id='2492',ex_date='2026-09-16',delivery_date='2026-09-16',action_type='split',ratio='4',halt_start='2026-09-15')


def prepare(tmp_path, kind='cash'):
    path=account(tmp_path);src=tmp_path/'sources'
    at=clock('2026-09-15' if kind=='cash' else '2026-09-16',8)
    data=pair('9') if kind=='cash' else {a.SPLIT:[dict(stock_id='2492',date='2026-09-16',before_price=100,after_price=25)]}
    sources(path,src,data,at)
    return path,src,at,data


def test_cash_review_never_posts_money_missing_entitlement_stays_blocked(tmp_path):
    path,src,at,_=prepare(tmp_path)
    before=p.summary(path,at)
    preview=r.preview(path,terms(),evidence(),src,at)
    assert 'amount_conflict' in codes(preview['before'])
    assert 'amount_conflict' not in codes(preview['after'])
    assert 'reviewed_missing_entitlement' in codes(preview['after'])
    event=r.save(path,preview['command'],src,at)
    assert r.save(path,preview['command'],src,at)==event
    after=p.summary(path,at)
    for field in ('cash','nav','holdings','fill_count'): assert after[field]==before[field]
    assert len(after['rows'])==len(before['rows'])+1
    draft=r.entitlement_draft(path,event['hash'],src,at)
    assert draft['amount']=='10000' and draft['eligible_qty']==1000
    with j.connection(path) as con: p.submit(con,draft,at)
    report=a.inspect(path,src,at)
    assert not report['blocked'] and 'reviewed_delivery_pending' in codes(report)
    assert p.summary(path,at)['cash']==before['cash']


def test_split_requires_integer_shares_and_actual_delivery(tmp_path):
    path,src,at,_=prepare(tmp_path,'split')
    preview=r.preview(path,terms('split'),evidence(),src,at)
    assert 'reviewed_missing_entitlement' in codes(preview['after'])
    event=r.save(path,preview['command'],src,at)
    draft=r.entitlement_draft(path,event['hash'],src,at)
    assert draft['result_qty']==4000
    with j.connection(path) as con: p.submit(con,draft,at)
    assert a.inspect(path,src,at)['blocked']
    assert p.summary(path,at)['holdings'][0]['qty']==1000
    with j.connection(path) as con:
        p.submit(con,dict(kind='delivery',id='fixture-delivery',action_id=draft['action_id'],date='2026-09-16',evidence='test receipt'),at)
    assert not a.inspect(path,src,at)['blocked']
    assert p.summary(path,at)['holdings'][0]['qty']==4000


def test_fractional_reverse_split_remains_blocked(tmp_path):
    path,src,at,_=prepare(tmp_path,'split');c=terms('split');c['ratio']='0.33333333'
    preview=r.preview(path,c,evidence(),src,at)
    assert 'fractional_right' in codes(preview['after'])
    event=r.save(path,preview['command'],src,at)
    with pytest.raises(ValueError):r.entitlement_draft(path,event['hash'],src,at)


def test_source_changes_stale_preview_and_saved_resolution_then_supersede(tmp_path):
    path,src,at,data=prepare(tmp_path)
    preview=r.preview(path,terms(),evidence(),src,at)
    event=r.save(path,preview['command'],src,at)
    data[a.RESULT][0]['stock_and_cache_dividend']='8'
    sources(path,src,data,clock('2026-09-15',9))
    assert 'resolution_stale' in codes(a.inspect(path,src,clock('2026-09-15',9)))
    with pytest.raises(ValueError):r.entitlement_draft(path,event['hash'],src,clock('2026-09-15',9))
    new=r.preview(path,terms(),evidence(),src,clock('2026-09-15',9))
    assert new['command']['supersedes']==event['hash']
    saved=r.save(path,new['command'],src,clock('2026-09-15',9))
    assert saved['hash']!=event['hash']
    assert 'resolution_stale' not in codes(a.inspect(path,src,clock('2026-09-15',9)))
    data[a.RESULT][0]['stock_and_cache_dividend']='7'
    pending=r.preview(path,terms(),evidence(),src,clock('2026-09-15',9))
    sources(path,src,data,clock('2026-09-15',10))
    with pytest.raises(ValueError,match='變動'): r.save(path,pending['command'],src,clock('2026-09-15',10))


def test_deletion_invalidates_but_refresh_with_identical_content_does_not(tmp_path):
    path,src,at,data=prepare(tmp_path)
    event=r.save(path,r.preview(path,terms(),evidence(),src,at)['command'],src,at)
    sources(path,src,data,clock('2026-09-15',9))
    assert 'resolution_stale' not in codes(a.inspect(path,src,clock('2026-09-15',9)))
    sources(path,src,{},clock('2026-09-15',10))
    assert 'resolution_stale' in codes(a.inspect(path,src,clock('2026-09-15',10)))


def test_new_account_event_and_tampered_preview_are_rejected(tmp_path):
    path,src,at,_=prepare(tmp_path)
    command=r.preview(path,terms(),evidence(),src,at)['command']
    bad=deepcopy(command);bad['terms']['cash_per_share']='20'
    with pytest.raises(ValueError):r.save(path,bad,src,at)
    with j.connection(path) as con:j.append(con,'fixture','observation',{'note':'new'},at)
    with pytest.raises(ValueError,match='帳本已有'):r.save(path,command,src,at)


def test_existing_wrong_entitlement_is_not_rewritten(tmp_path):
    path,src,at,_=prepare(tmp_path)
    with j.connection(path) as con:
        p.submit(con,dict(kind='entitlement',id='wrong',action_id='wrong',action_type='cash',stock_id='2492',ex_date='2026-09-15',
            delivery_date='2026-09-17',eligible_qty=1000,cash_per_share='9',amount='9000',evidence='fixture'),at)
    before=p.summary(path,at)
    result=r.preview(path,terms(),evidence(),src,at)
    assert 'reviewed_ledger_conflict' in codes(result['after'])
    r.save(path,result['command'],src,at)
    assert p.summary(path,at)['cash']==before['cash']
    assert a.inspect(path,src,at)['blocked']


def test_missing_or_stale_sources_cannot_be_waived(tmp_path):
    path=account(tmp_path);src=tmp_path/'empty'
    with pytest.raises(ValueError,match='來源'):r.preview(path,terms(),evidence(),src,clock('2026-09-15'))
    sources(path,src,pair(),clock('2026-09-15'))
    with pytest.raises(ValueError,match='來源'):r.preview(path,terms(),evidence(),src,clock('2026-09-15',22))


def test_halt_fills_and_mixed_actions_remain_blocked(tmp_path):
    path,src,at,data=prepare(tmp_path,'split')
    c=terms('split');c['halt_start']='2026-09-14'
    assert 'halt_fill' in codes(r.preview(path,c,evidence(),src,at)['after'])
    cashdata=pair();cashdata[a.POLICY][0]['CashExDividendTradingDate']='2026-09-16';cashdata[a.RESULT][0]['date']='2026-09-16'
    data.update(cashdata);sources(path,src,data,at)
    assert 'mixed_actions' in codes(r.preview(path,terms('split'),evidence(),src,at)['after'])


@pytest.mark.parametrize('field,value',[('url','file:///secret'),('published_at','2026-09-20T00:00:00+08:00'),('text','too short')])
def test_invalid_evidence_rejected(field,value,tmp_path):
    path,src,at,_=prepare(tmp_path);ev=evidence();ev[field]=value
    with pytest.raises(ValueError):r.preview(path,terms(),ev,src,at)


def test_no_late_entitlement_draft(tmp_path):
    path,src,at,_=prepare(tmp_path)
    event=r.save(path,r.preview(path,terms(),evidence(),src,at)['command'],src,at)
    sources(path,src,pair(),clock('2026-09-16'))
    # Source revision may itself block, but old ex-date also cannot be backfilled.
    with pytest.raises(ValueError): r.entitlement_draft(path,event['hash'],src,clock('2026-09-16'))


def test_resolved_cash_reaches_guarded_close_with_neutral_nav(tmp_path,monkeypatch):
    from contextlib import contextmanager
    path,src,at,data=prepare(tmp_path)
    event=r.save(path,r.preview(path,terms(),evidence(),src,at)['command'],src,at)
    draft=r.entitlement_draft(path,event['hash'],src,at)
    with j.connection(path) as con:p.submit(con,draft,at)
    end=clock('2026-09-15',20);sources(path,src,data,end)
    monkeypatch.setattr('app.workbench_service.data_status',lambda:dict(data_ready=True,price_date='2026-09-15'))
    class DB:
        def scalars(self,q):return ['2026-09-11','2026-09-14','2026-09-15','2026-09-16','2026-09-17','2026-09-18']
        def execute(self,q):return [('2492',90),('0050',100)]
    @contextmanager
    def session():yield DB()
    monkeypatch.setattr('app.db.get_session',session)
    result=a.capture_close(path,True,end,src)
    assert result['body']['nav']=='999980'  # Ex-price adjustment offsets the receivable.
    assert p.summary(path,end)['cash']=='899980'
    rows=a.account_rows(path,end)
    assert rows[-2]['body']['report']['resolutions'][0]['hash']==event['hash']


def test_ui_preview_does_not_save_and_explicit_save_preserves_cash(tmp_path,monkeypatch):
    from datetime import date
    from streamlit.testing.v1 import AppTest
    path,src,at,_=prepare(tmp_path)
    original_preview,original_save=r.preview,r.save
    report=a.inspect(path,src,at)
    monkeypatch.setattr(r,'preview',lambda path,terms,evidence:original_preview(path,terms,evidence,src,at))
    monkeypatch.setattr(r,'save',lambda path,command:original_save(path,command,src,at))
    # Supply the test artifact through source literals; there is no network or production account access.
    app=AppTest.from_string(f'from app.forward_corporate_resolution_ui import render\nrender({str(path)!r},"fixture",{report!r})').run()
    app.date_input[0].set_value(date(2026,9,15));app.date_input[1].set_value(date(2026,9,17))
    for widget,value in zip(app.text_input,['10','Test title','https://www.twse.com.tw/fixture','2026-09-14T18:00:00+08:00','test-reviewer']):
        widget.set_value(value)
    app.text_area[0].set_value(evidence()['text'])
    before=p.summary(path,at)
    app.button[0].click().run()
    assert not app.exception and p.summary(path,at)==before
    next(b for b in app.button if b.label=='保存這份核對紀錄（不入帳）').click().run()
    assert not app.exception
    after=p.summary(path,at)
    assert len(after['rows'])==len(before['rows'])+1 and after['cash']==before['cash']
