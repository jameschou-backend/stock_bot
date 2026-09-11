from decimal import Decimal as D
from pathlib import Path
import pytest
from app import forward_comparison as c, forward_portfolio as p, forward_journal as j
from test_forward_portfolio import setup, buy, fill, close, clock


def pair(tmp_path):
    strategy=tmp_path/'strategy';benchmark=tmp_path/'benchmark'
    with j.connection(strategy) as con:
        setup(con);order=buy(con)
    c.seed_benchmark(strategy,benchmark,clock('2026-09-12'))
    return strategy,benchmark,order


def fill_both(strategy,benchmark,order):
    with j.connection(strategy) as con: fill(con,order,1000,'100')
    with j.connection(benchmark) as con:
        orders=[r for r in j.read_events(con) if r['kind']=='order']
        for i,o in enumerate(orders):
            fill(con,o,o['body']['qty'],'100',key='b'+str(i),fee=str(p.fee(D(100)*o['body']['qty'])),hour=10+i)
    return orders


def test_seed_is_prospective_identical_start_no_fake_trades(tmp_path):
    strategy,benchmark,_=pair(tmp_path)
    with j.connection(benchmark) as con:
        rows=j.read_events(con);a=next(r for r in rows if r['kind']=='comparison_anchor')
        inherited=next(r for r in rows if r['kind']=='close')
        assert inherited['recorded_at'].startswith('2026-09-12')
        assert inherited['body']['date']=='2026-09-11'
        assert p.state(rows)['cash']==1000000
        assert not any(r['kind']=='fill' for r in rows)
    assert c.seed_benchmark(strategy,benchmark,clock('2026-09-15'))==a
    assert not c.comparison(strategy,benchmark)['ready']


def test_late_or_same_path_seed_rejected(tmp_path):
    strategy=tmp_path/'s'
    with j.connection(strategy) as con: setup(con);buy(con)
    with pytest.raises(ValueError,match='開盤前'):
        c.seed_benchmark(strategy,tmp_path/'late',clock('2026-09-14',10))
    with pytest.raises(ValueError,match='分開'): c.seed_benchmark(strategy,strategy,clock())


def test_execution_before_benchmark_seed_is_not_shared_start(tmp_path):
    strategy=tmp_path/'s'
    with j.connection(strategy) as con:
        setup(con);o=buy(con);fill(con,o,1000,'100')
    with pytest.raises(ValueError): c.seed_benchmark(strategy,tmp_path/'b',clock('2026-09-14',11))


def test_cost_inclusive_same_date_comparison_and_numeric_price_equivalence(tmp_path):
    strategy,benchmark,o=pair(tmp_path);orders=fill_both(strategy,benchmark,o)
    with j.connection(strategy) as con: close(con,'2026-09-14',{'2492':'110','0050':'102'})
    with j.connection(benchmark) as con: close(con,'2026-09-14',{'0050':'102.000000'})
    result=c.comparison(strategy,benchmark)
    assert result['ready'] and not result['live_qualified']
    assert result['strategy_return']==pytest.approx((1000000-100020+110000)/1000000-1)
    shares=sum(r['body']['qty'] for r in orders)
    cost=sum(p.fee(D(100)*r['body']['qty']) for r in orders)
    nav=D(1000000)+2*shares-cost
    assert result['benchmark_return']==pytest.approx(float(nav/1000000-1))
    assert result['excess_percentage_points']==pytest.approx((result['strategy_return']-result['benchmark_return'])*100)


def test_missing_day_and_mismatched_quotes_prevent_scorecard(tmp_path):
    strategy,benchmark,o=pair(tmp_path);fill_both(strategy,benchmark,o)
    with j.connection(strategy) as con: close(con,'2026-09-14',{'2492':'110','0050':'102'})
    assert not c.comparison(strategy,benchmark)['ready']
    with j.connection(benchmark) as con: close(con,'2026-09-14',{'0050':'103'})
    assert any('估價不一致' in r for r in c.comparison(strategy,benchmark)['reasons'])
    with j.connection(strategy) as con: close(con,'2026-09-16',{'2492':'110','0050':'102'})
    with j.connection(benchmark) as con: close(con,'2026-09-16',{'0050':'102'})
    assert any('每日結算' in r for r in c.comparison(strategy,benchmark)['reasons'])


def test_exit_persists_when_signal_source_fails(tmp_path,monkeypatch):
    path=tmp_path/'s'
    with j.connection(path) as con:
        setup(con);o=buy(con);fill(con,o,1000,'100')
        close(con,'2026-09-14',{'2492':'87','0050':'100'})
    monkeypatch.setattr(c.legacy,'source_signal',lambda *args:None)
    result=c.save_strategy_plans(path,clock=clock('2026-09-14'))
    assert len(result['exits'])==1 and result['entry_block']
    with j.connection(path) as con:
        rows=j.read_events(con)
        assert len([r for r in rows if r['kind']=='decision'])==1
        assert len([r for r in rows if r['kind']=='entry_gap'])==1
    c.save_strategy_plans(path,clock=clock('2026-09-14'))
    with j.connection(path) as con:
        assert len([r for r in j.read_events(con) if r['kind']=='order' and r['body']['side']=='sell'])==1


def test_failed_entry_validation_does_not_roll_back_exit(tmp_path,monkeypatch):
    path=tmp_path/'s'
    with j.connection(path) as con:
        setup(con);o=buy(con);fill(con,o,1000,'100');close(con,'2026-09-14',{'2492':'87','0050':'100'})
    def broken(*args): raise ValueError('bad candidate inputs')
    monkeypatch.setattr(c.legacy,'save_proposals',broken)
    result=c.save_strategy_plans(path,clock=clock('2026-09-14'))
    assert result['exits'] and 'bad candidate' in result['entry_block']
    with j.connection(path) as con: assert any(r['kind']=='decision' for r in j.read_events(con))


def test_dividends_only_reinvest_after_delivery_and_never_double_spend(tmp_path):
    strategy,benchmark,o=pair(tmp_path);fill_both(strategy,benchmark,o)
    with j.connection(benchmark) as con:
        close(con,'2026-09-14',{'0050':'100'})
        n=p.state(j.read_events(con))['holdings']['0050']['qty']
    action=dict(kind='entitlement',id='div',action_id='div',action_type='cash',stock_id='0050',
        ex_date='2026-09-15',delivery_date='2026-09-16',eligible_qty=n,cash_per_share='1',amount=str(n),evidence='fixture')
    c.record_benchmark(action,benchmark,clock('2026-09-15',8))
    with j.connection(benchmark) as con: close(con,'2026-09-15',{'0050':'99'})
    assert c.reinvest_dividends(benchmark,clock('2026-09-15'))==[]
    c.record_benchmark(dict(kind='delivery',id='pay',action_id='div',date='2026-09-16',evidence='fixture'),benchmark,clock('2026-09-16',8))
    with j.connection(benchmark) as con: close(con,'2026-09-16',{'0050':'99'})
    orders=c.reinvest_dividends(benchmark,clock('2026-09-16'))
    assert orders and sum(D(o['body']['limit_price'])*o['body']['qty']+p.fee(D(o['body']['limit_price'])*o['body']['qty']) for o in orders)<=n
    assert c.reinvest_dividends(benchmark,clock('2026-09-16'))==[]


def test_benchmark_rejects_discretionary_trading_and_other_stocks(tmp_path):
    _,benchmark,_=pair(tmp_path)
    with pytest.raises(ValueError,match='任意選股'):
        c.record_benchmark(dict(kind='order'),benchmark,clock())
    with pytest.raises(ValueError,match='0050'):
        c.record_benchmark(dict(kind='entitlement',stock_id='2492'),benchmark,clock())


def test_no_complete_evidence_means_no_return_widgets(monkeypatch):
    from app import forward_comparison_ui as ui
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(ui.c,'comparison',lambda:dict(ready=False,reasons=['strategy 尚無成交回報']))
    app=AppTest.from_string('from app.forward_comparison_ui import render\nrender()').run()
    assert not app.exception and not app.metric
    assert '尚無成交回報' in app.info[0].value
