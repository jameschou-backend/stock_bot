from decimal import Decimal as D
import pytest
from app import forward_halts as h, forward_journal as j, forward_portfolio as p
from tests.test_forward_portfolio import setup,buy,fill,close,clock

EVIDENCE=dict(url='https://www.twse.com.tw/fixture',title='TEST ONLY halt',published_at='2026-09-14T17:00:00+08:00',text='Synthetic halt announcement fixture, not an actual announcement.',reviewer='tester')

def account(tmp_path):
    path=tmp_path/'book'
    with j.connection(path) as con:
        setup(con);o=buy(con);fill(con,o,1000,'100');close(con,'2026-09-14',{'2492':'100','0050':'100'})
    return path

def notice(path):
    return h.save_notice(path,dict(stock_id='2492',halt_start='2026-09-15',resume_date='2026-09-17',withdrawn=False),EVIDENCE,clock('2026-09-14'))

def rows(path):
    with j.connection(path) as con:return j.read_events(con)

def test_halt_carry_does_not_hide_ordinary_missing_price_or_extend_past_resume(tmp_path):
    path=account(tmp_path)
    with pytest.raises(ValueError,match='無有效'):h.valuation_prices(rows(path),'2026-09-15',{'0050':'100'})
    event=notice(path)
    assert notice(path)==event
    prices,details=h.valuation_prices(rows(path),'2026-09-15',{'0050':'100'})
    assert prices['2492']=='100' and details['2492']['price_date']=='2026-09-14'
    assert not details['2492']['tradable']
    with pytest.raises(ValueError,match='無有效'):h.valuation_prices(rows(path),'2026-09-17',{'0050':'100'})
    with pytest.raises(ValueError,match='衝突'):h.valuation_prices(rows(path),'2026-09-15',{'2492':'99','0050':'100'})
    assert h.valuation_prices(rows(path),'2026-09-17',{'2492':'101','0050':'100'})[1]=={}


def test_carry_adjusts_cash_dividend_exactly_once_and_nav_is_neutral(tmp_path):
    path=account(tmp_path);notice(path)
    with j.connection(path) as con:
        p.submit(con,dict(kind='entitlement',id='div',action_id='div',action_type='cash',stock_id='2492',ex_date='2026-09-15',delivery_date='2026-09-17',eligible_qty=1000,cash_per_share='10',amount='10000',evidence='fixture'),clock('2026-09-15',8))
        data=j.read_events(con);prices,details=h.valuation_prices(data,'2026-09-15',{'0050':'100'})
        body=dict(date='2026-09-15',prices=prices,estimated_prices=details,actions_reviewed=True,source='fixture')
        nav=p.valuation({**p.state(data),'mark':{'body':body}})
        assert nav==D('999980') and prices['2492']=='90'
        p.submit(con,dict(kind='close',id='15',**body,nav=str(nav)),clock('2026-09-15'))
    assert h.valuation_prices(rows(path),'2026-09-16',{'0050':'100'})[0]['2492']=='90'
    assert h.save_plans(path,clock=clock('2026-09-15'))['exits']==[]
    assert h.compare(path,path)['ready'] is False


def test_future_halt_blocks_plan_and_fill_without_affecting_other_stock_exit(tmp_path):
    path=tmp_path/'book'
    with j.connection(path) as con:
        setup(con);o=buy(con);other=buy(con,sid='2491',key='other');fill(con,o,1000,'100');fill(con,other,1000,'100',key='otherfill')
        close(con,'2026-09-14',{'2492':'100','2491':'80','0050':'100'})
    notice(path)
    plans=h.save_plans(path,clock=clock('2026-09-14'))
    assert len(plans['exits'])==1 and plans['exits'][0]['body']['stock_id']=='2491'
    assert plans['entry_block']
    bad=dict(kind='fill',id='bad',order_id='buy',qty=1,price='100',fee='20',tax='0',executed_at='2026-09-15T10:00:00+08:00',evidence=dict(source='paper_execution_report',report_id='bad'))
    with pytest.raises(ValueError,match='停牌'):h.record(path,bad,clock=clock('2026-09-15',10))


def test_notice_revision_preserves_original_and_rejects_malformed_dates(tmp_path):
    path=account(tmp_path);a=notice(path)
    terms=dict(stock_id='2492',halt_start='2026-09-15',resume_date='2026-09-17',withdrawn=True)
    b=h.save_notice(path,terms,EVIDENCE,clock('2026-09-15'))
    assert b['body']['supersedes']==a['hash'] and not h.active(rows(path),'2026-09-15')
    with pytest.raises(ValueError):h.save_notice(path,dict(terms,resume_date='2026-09-14'),EVIDENCE,clock('2026-09-15'))
