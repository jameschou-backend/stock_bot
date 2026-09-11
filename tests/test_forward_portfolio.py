from datetime import datetime, timezone
from decimal import Decimal as D
from copy import deepcopy
import pytest
from app import forward_portfolio as p, forward_journal as j


def clock(day='2026-09-11', hour=20):
    return lambda:datetime.fromisoformat(f'{day}T{hour:02d}:00:00+08:00')


def setup(con):
    p.submit(con,dict(kind='calendar',id='cal',source='fixture',sessions=[
        '2026-09-11','2026-09-14','2026-09-15','2026-09-16','2026-09-17','2026-09-18']),clock())
    close(con,'2026-09-11',{'2492':'100','0050':'100'})


def close(con,day,prices,reviewed=True):
    s=p.state(j.read_events(con))
    body=dict(date=day,prices=prices,actions_reviewed=reviewed,source='dated_fixture')
    nav=p.valuation({**s,'mark':{'body':body}})
    return p.submit(con,dict(kind='close',id=day,**body,nav=str(nav) if nav is not None else None),clock(day))


def buy(con,sid='2492',n=1000,price='100',day='2026-09-14',signalday='2026-09-11',key='buy'):
    signal=dict(priority=.5,signal_date=signalday,entry_date=day,members=[sid],planning_reference_close=float(price),
                liquidity_before_entry=dict(complete_20_sessions=True,mean_turnover20_twd=100000000,adv20_shares=1000000))
    proof=dict(seq=1,event_key='signal:'+signalday,kind='signal',recorded_at=clock(signalday)().isoformat(),
        body=dict(eligible=True,candidates=[signal]),previous_hash='')
    proof['hash']=j.digest(proof)
    c=dict(kind='order',id=key,order_id=key,stock_id=sid,side='buy',channel='board' if n%1000==0 else 'odd',
        qty=n,limit_price=price,session=day,reason='idle_cash' if sid=='0050' else 'candidate',
        signal=signal,signal_proof=proof,source_hash=proof['hash'])
    return p.submit(con,c,clock(signalday))


def fill(con,order,n,price,day='2026-09-14',key='fill',fee='20',tax='0',hour=10):
    at=clock(day,hour)().isoformat()
    c=dict(kind='fill',id=key,order_id=order['body']['order_id'],qty=n,price=price,fee=fee,tax=tax,
           executed_at=at,evidence=dict(source='paper_execution_report',report_id=key))
    return p.submit(con,c,clock(day,hour))


def cancel(con,order,day='2026-09-14'):
    return p.submit(con,dict(kind='cancel',id=order['hash'],order_id=order['body']['order_id'],reason='核對未成交'),clock(day,14))


def test_partial_fills_cash_average_cost_reserve_and_idempotence(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con,n=999)
        a=fill(con,o,400,'99')
        assert fill(con,o,400,'99')==a
        s=p.state(j.read_events(con))
        assert s['cash']==D('960380') and s['holdings']['2492']['cost']==D('39620')
        reserved,_=p.reserves(s,'2026-09-14')
        assert reserved==D('59985')
        cancel(con,o)
        s=p.state(j.read_events(con))
        assert p.reserves(s,'2026-09-14')[0]==0 and s['holdings']['2492']['qty']==400
        with pytest.raises(ValueError,match='取消'):
            fill(con,o,1,'99',key='late')


def test_overfill_oversell_double_report_and_limit_rejected(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con)
        with pytest.raises(ValueError,match='剩餘'): fill(con,o,2000,'100')
        with pytest.raises(ValueError,match='限價'): fill(con,o,1000,'101')
        fill(con,o,1000,'100')
        with pytest.raises(ValueError,match='剩餘'): fill(con,o,1000,'100',key='again')
        with pytest.raises(ValueError,match='持股'):
            p.submit(con,dict(kind='order',id='sell',order_id='sell',stock_id='2492',side='sell',channel='board',
                qty=2000,limit_price='80',session='2026-09-15',reason='bad'),clock('2026-09-14'))


def test_partial_exit_realized_cost_and_retry_after_recovery(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con,n=999);fill(con,o,999,'100',fee='142')
        close(con,'2026-09-14',{'2492':'87','0050':'100'})
        decision=p.submit(con,dict(kind='decision',id='stop',stock_id='2492',date='2026-09-14',reason='stop12_close'),clock('2026-09-14'))
        c=dict(kind='order',id='exit',order_id='exit',stock_id='2492',side='sell',channel='odd',qty=999,
            limit_price='87',session='2026-09-15',reason=decision['hash'])
        sell=p.submit(con,c,clock('2026-09-14'))
        fill(con,sell,400,'90',day='2026-09-15',key='sellfill',tax='108')
        s=p.state(j.read_events(con));assert s['holdings']['2492']['qty']==599
        expected=D(36000)-20-108-(D(99900)+142)*400/999
        assert s['realized_pnl']==expected
        cancel(con,sell,'2026-09-15')
        close(con,'2026-09-15',{'2492':'110','0050':'100'})
        retry=p.submit(con,{**c,'id':'retry','order_id':'retry','qty':599,'session':'2026-09-16','limit_price':'110'},clock('2026-09-15'))
        assert retry['body']['reason']==decision['hash']


def test_compounding_uses_two_million_nav_not_initial_one_million(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con,sid='0050',n=9000)
        fill(con,o,9000,'100')
        mark=close(con,'2026-09-14',{'0050':'220','2492':'100'})
        assert mark['body']['nav']=='2079980'
        sell=p.submit(con,dict(kind='order',id='etfsell',order_id='etfsell',stock_id='0050',side='sell',channel='board',
            qty=9000,limit_price='220',session='2026-09-15',reason='fund_stock_entries'),clock('2026-09-14'))
        with pytest.raises(ValueError,match='現金不足'):
            buy(con,n=6000,day='2026-09-15',signalday='2026-09-14',key='unfunded')
        fill(con,sell,9000,'220',day='2026-09-15',key='etfexit',tax='1980')
        close(con,'2026-09-15',{'0050':'220','2492':'100'})
        assert buy(con,n=6000,day='2026-09-16',signalday='2026-09-15',key='compound')['body']['qty']==6000
        with pytest.raises(ValueError,match='三分之一'):
            buy(con,n=1000,day='2026-09-16',signalday='2026-09-15',key='too_big')


def test_missing_quotes_unknown_no_future_marks_or_rewriting(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con);fill(con,o,1000,'100')
        mark=close(con,'2026-09-14',{'0050':'100'})
        assert mark['body']['nav'] is None and p.valuation(p.state(j.read_events(con))) is None
        with pytest.raises(ValueError,match='已封存'):
            c=dict(kind='close',id='rewrite',**mark['body']);c['prices']['2492']='100'
            p.submit(con,c,clock('2026-09-14'))
        with pytest.raises(ValueError,match='當日'):
            p.submit(con,dict(kind='close',id='future',date='2026-09-16',prices={},actions_reviewed=True,source='x',nav='0'),clock('2026-09-15'))


def test_dividend_receivable_not_spendable_double_count_protected(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con);fill(con,o,1000,'100')
        close(con,'2026-09-14',{'2492':'100','0050':'100'})
        action=dict(kind='entitlement',id='div',action_id='div',action_type='cash',stock_id='2492',
            ex_date='2026-09-15',delivery_date='2026-09-17',eligible_qty=1000,cash_per_share='10',amount='10000',evidence='official fixture')
        p.submit(con,action,clock('2026-09-15',8))
        s=p.state(j.read_events(con));assert s['cash']==D('899980') and s['holdings']['2492']['stop_basis']==90
        close(con,'2026-09-15',{'2492':'90','0050':'100'})
        assert p.valuation(p.state(j.read_events(con)))==D('999980')
        with pytest.raises(ValueError,match='尚未達到'):
            p.submit(con,dict(kind='decision',id='false_stop',stock_id='2492',date='2026-09-15',reason='stop12_close'),clock('2026-09-15'))
        payment=dict(kind='delivery',id='paid',action_id='div',date='2026-09-17',evidence='payment report')
        p.submit(con,payment,clock('2026-09-17',8))
        assert p.submit(con,payment,clock('2026-09-17',8))['body']=={k:v for k,v in payment.items() if k not in ('kind','id')}
        assert p.state(j.read_events(con))['cash']==D('909980')
        with pytest.raises(ValueError,match='重複'):
            p.submit(con,{**payment,'id':'duplicate'},clock('2026-09-17',8))


def test_split_blocks_trading_until_shares_delivered(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con);fill(con,o,1000,'100');close(con,'2026-09-14',{'2492':'100','0050':'100'})
        p.submit(con,dict(kind='entitlement',id='split',action_id='split',action_type='split',stock_id='2492',
            ex_date='2026-09-15',delivery_date='2026-09-16',eligible_qty=1000,ratio='2',result_qty=2000,evidence='official fixture'),clock('2026-09-15',8))
        assert p.valuation(p.state(j.read_events(con))) is None
        with pytest.raises(ValueError,match='尚未交付'):
            buy(con,day='2026-09-16',signalday='2026-09-15',key='blocked')
        p.submit(con,dict(kind='delivery',id='delivery',action_id='split',date='2026-09-16',evidence='share report'),clock('2026-09-16',8))
        s=p.state(j.read_events(con));assert s['holdings']['2492']['qty']==2000 and s['holdings']['2492']['stop_basis']==50
        close(con,'2026-09-16',{'2492':'50','0050':'100'})
        assert p.valuation(p.state(j.read_events(con)))==D('999980')


def test_fees_cannot_consume_other_reserved_cash_and_expiry_needs_reconciliation(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con,sid='0050',n=9000)
        with pytest.raises(ValueError,match='預留'):
            fill(con,o,1000,'100',fee='200000')
        with pytest.raises(ValueError,match='核對'):
            buy(con,day='2026-09-16',signalday='2026-09-15',key='later')


def test_rules_changes_do_not_overwrite_history(tmp_path,monkeypatch):
    path=tmp_path/'j'
    with j.connection(path) as con: setup(con)
    monkeypatch.setitem(p.RULES,'initial_cash','2000000')
    with j.connection(path) as con:
        with pytest.raises(ValueError,match='版本'): p.initialize(con)


def test_funding_intent_cannot_buy_until_etf_execution(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);etf=buy(con,sid='0050',n=9000);fill(con,etf,9000,'100')
        close(con,'2026-09-14',{'0050':'100','2492':'100'})
        candidate=dict(signal_date='2026-09-14',entry_date='2026-09-15',members=['2492'],planning_reference_close=100,
            liquidity_before_entry=dict(complete_20_sessions=True,mean_turnover20_twd=100000000,adv20_shares=1000000))
        proof=dict(seq=1,event_key='sig',kind='signal',recorded_at=clock('2026-09-14')().isoformat(),
                   body=dict(eligible=True,candidates=[candidate]),previous_hash='')
        proof['hash']=j.digest(proof)
        intent=dict(kind='funding_intent',id='intent',order_id='intent',stock_id='2492',side='buy',channel='board',qty=3000,
                    limit_price='100',session='2026-09-15',reason='candidate',signal=candidate,signal_proof=proof,source_hash=proof['hash'])
        pending=p.submit(con,intent,clock('2026-09-14'))
        order={**intent,'kind':'order','funding_intent':pending['hash']}
        with pytest.raises(ValueError,match='現金不足'): p.submit(con,order,clock('2026-09-15',10))
        sell=p.submit(con,dict(kind='order',id='fundsell',order_id='fundsell',stock_id='0050',side='sell',channel='board',qty=3000,
            limit_price='100',session='2026-09-15',reason='fund_stock_entries'),clock('2026-09-14'))
        fill(con,sell,3000,'100',day='2026-09-15',key='fundfill',tax='300')
        result=p.submit(con,order,clock('2026-09-15',11))
        assert result['kind']=='order'
        with pytest.raises(ValueError,match='不符'):
            p.submit(con,{**order,'id':'extra','order_id':'extra','qty':1,'channel':'odd'},clock('2026-09-15',11))


def test_day_close_requires_unfilled_order_reconciliation(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con);o=buy(con,n=999);fill(con,o,100,'100')
        with pytest.raises(ValueError,match='未完成委託'): close(con,'2026-09-14',{'2492':'100'})
        cancel(con,o)
        close(con,'2026-09-14',{'2492':'100'})
        assert p.valuation(p.state(j.read_events(con)))==D('999980')


def test_three_slots_board_plus_odd_share_one_budget(tmp_path):
    with j.connection(tmp_path/'j') as con:
        setup(con)
        buy(con,n=3000)
        with pytest.raises(ValueError,match='三分之一'): buy(con,n=500,key='odd')
        buy(con,sid='2330',n=1000,key='second')
        buy(con,sid='1560',n=1000,key='third')
        with pytest.raises(ValueError,match='三檔'): buy(con,sid='2491',n=1000,key='fourth')


def test_service_proposals_idempotent_no_automatic_fills(tmp_path,monkeypatch):
    from app import forward_portfolio_service as service
    path=tmp_path/'j'
    with j.connection(path) as con:
        setup(con)
        # Obtain a correctly hashed proof from the test order, then use a fresh account.
        o=buy(con)
        proof=o['body']['signal_proof']
    path=tmp_path/'fresh'
    with j.connection(path) as con: setup(con)
    monkeypatch.setattr(service,'source_signal',lambda *args:proof)
    result=service.save_proposals(path,clock=clock())
    assert {r['body']['stock_id'] for r in result}=={'2492','0050'}
    data=p.summary(path,clock())
    assert data['fill_count']==0 and data['cash']=='1000000' and D(data['available_cash'])>=0
    service.save_proposals(path,clock=clock())
    with j.connection(path) as con:
        assert len([r for r in j.read_events(con) if r['kind']=='order'])==len(result)


def test_portfolio_ui_shows_unknown_nav_and_partial_order(monkeypatch):
    from app import forward_portfolio_ui as ui
    from streamlit.testing.v1 import AppTest
    data=dict(nav=None,available_cash='900000',reserved_cash='40000',price_date=None,fill_count=1,holdings=[],rows=[],
        orders=[dict(order_id='test',stock_id='2492',side='buy',session='2026-09-14',channel='odd',limit_price='100',
            qty=999,filled=400,remaining=599,status='部分成交')])
    monkeypatch.setattr(ui.p,'summary',lambda:data)
    app=AppTest.from_string('from app.forward_portfolio_ui import render\nrender()').run()
    assert not app.exception
    assert app.metric[0].value=='未知，待完整估值'
    assert app.dataframe[0].value.iloc[0]['剩餘']==599
