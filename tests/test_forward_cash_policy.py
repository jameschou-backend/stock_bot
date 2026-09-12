from copy import deepcopy
import pytest
from streamlit.testing.v1 import AppTest
from app import forward_cash_policy as cash, forward_cash_automation as auto
from app import forward_journal as j, forward_portfolio as p, forward_simulation as sim
from tests.test_forward_simulation import initialized, observation
from tests.test_forward_portfolio import buy, close, clock


def create(tmp_path):
    old, source, benchmark = initialized(tmp_path)
    with j.connection(source) as con:
        buy(con, sid='0050', n=1000, key='idle-etf')
    before=(sim.read(source),sim.read(benchmark),sim.read(old/'strategy.sqlite3'))
    root=tmp_path/'cash'
    cash.initialize(root,source,benchmark,old/'signals.sqlite3',clock())
    assert before==(sim.read(source),sim.read(benchmark),sim.read(old/'strategy.sqlite3'))
    return root,source,benchmark


def test_initialization_cancels_only_new_strategy_etf_and_preserves_benchmark(tmp_path):
    root,source,benchmark=create(tmp_path)
    rows=cash.verify(root/'strategy.sqlite3','strategy'); state=p.state(rows)
    assert state['orders']['idle-etf']['closed']
    assert not state['orders']['buy']['closed']
    assert state['cash']==1000000
    b=p.state(cash.verify(root/'benchmark.sqlite3','benchmark'))
    assert any(o['stock_id']=='0050' and not o['closed'] for o in b['orders'].values())
    assert cash.initialize(root,source,benchmark,clock=clock())==root
    with pytest.raises(ValueError,match='開盤'):
        cash.initialize(tmp_path/'late',source,benchmark,root/'signals.sqlite3',clock('2026-09-14',10))


def test_cash_guard_rejects_new_etf_orders_and_rule_changes(tmp_path,monkeypatch):
    root,_,_=create(tmp_path);path=root/'strategy.sqlite3'
    monkeypatch.setitem(cash.RULES,'idle_asset','0050')
    with pytest.raises(ValueError,match='規則'):cash.verify(path)
    monkeypatch.setitem(cash.RULES,'idle_asset','cash')
    with j.connection(path) as con:buy(con,sid='0050',n=1000,key='forbidden')
    with pytest.raises(ValueError,match='0050'):cash.verify(path)


def test_new_day_stock_plans_preserved_and_no_idle_etf_recreated(tmp_path,monkeypatch):
    root,_,_=create(tmp_path);path=root/'strategy.sqlite3'
    auto._cancel_day(path,clock('2026-09-14',14))
    with j.connection(path) as con:close(con,'2026-09-14',{'2492':'100','0050':'100'})
    # Obtain a valid next-session proof without writing to the cash account.
    reference=tmp_path/'reference'
    with j.connection(reference) as con:
        from tests.test_forward_portfolio import setup
        setup(con);close(con,'2026-09-14',{'2492':'100','0050':'100'})
        template=buy(con,n=1000,day='2026-09-15',signalday='2026-09-14')
    proof=template['body']['signal_proof']
    proof['body']['signal_date']='2026-09-14'
    proof['hash']=j.digest({k:v for k,v in proof.items() if k!='hash'})
    monkeypatch.setattr(cash.service,'source_signal',lambda *a:proof)
    result=cash.save_plans(path,root/'signals.sqlite3',clock=clock('2026-09-14'))
    assert result['entry_block'] is None
    active=[o for o in p.state(cash.verify(path))['orders'].values() if not o['closed']]
    assert active and all(o['stock_id']=='2492' for o in active)
    assert sum(o['qty'] for o in active)==3328  # Original fee-inclusive one-third sizing.
    empty=deepcopy(proof);empty['body']['candidates']=[]
    monkeypatch.setattr(cash.service,'source_signal',lambda *a:empty)
    cash.save_plans(path,root/'signals.sqlite3',clock=clock('2026-09-14'))
    assert all(o['stock_id']!='0050' or o['closed'] for o in p.state(cash.verify(path))['orders'].values())


def test_cash_run_and_export_are_explicit_and_closed_day_skips(tmp_path,monkeypatch):
    root,_,_=create(tmp_path)
    monkeypatch.setattr(auto,'calendar_day',lambda day:False)
    assert auto.run(root,clock=clock('2026-09-12'))['status']=='closed_market'
    report=auto.export(root,clock=clock('2026-09-12'))
    assert report['allocation_policy']['idle_asset']=='cash'
    assert report['books']['strategy']['fill_count']==0 and not report['live_qualified']


def test_evening_dispatch_uses_cash_plans_and_preserves_entry_gap(tmp_path,monkeypatch):
    root,_,_=create(tmp_path);calls=[]
    monkeypatch.setattr(auto,'calendar_day',lambda day:True)
    monkeypatch.setattr(auto,'command',lambda *a:None)
    monkeypatch.setattr(auto,'_signals',lambda *a:{'hash':'fixture'})
    monkeypatch.setattr(auto.corporate,'refresh',lambda *a,**kw:None)
    monkeypatch.setattr(auto,'close',lambda *a,**kw:{'hash':'fixture'})
    def plan(path,signals,benchmark=False,clock=None):
        calls.append((path,benchmark))
        return {'entry_block':None if benchmark else 'fixture unavailable signal'}
    monkeypatch.setattr(cash,'save_plans',plan)
    result=auto.run(root,clock=clock('2026-09-14',18))
    assert result['status']=='needs_attention'
    assert calls==[(root/'strategy.sqlite3',False),(root/'benchmark.sqlite3',True)]


def test_cash_runner_refuses_swapped_roles_before_any_observation(tmp_path,monkeypatch):
    root,_,_=create(tmp_path)
    a=root/'strategy.sqlite3';b=root/'benchmark.sqlite3';temp=root/'swap'
    a.rename(temp);b.rename(a);temp.rename(b)
    monkeypatch.setattr(auto,'_intraday',lambda *a:pytest.fail('must not observe invalid books'))
    with pytest.raises(ValueError,match='規則'):
        auto.run(root,clock=clock('2026-09-14',10))


def test_cash_ui_explains_separate_benchmark_without_writing(tmp_path,monkeypatch):
    root,_,_=create(tmp_path)
    from app import forward_corporate_ui,forward_evidence_ui
    monkeypatch.setattr(forward_corporate_ui,'render',lambda *a:None)
    monkeypatch.setattr(forward_evidence_ui,'render_halts',lambda *a:None)
    before=sim.read(root/'strategy.sqlite3')
    app=AppTest.from_string('from pathlib import Path\nfrom app.forward_simulation_ui import render\nrender(Path('+repr(str(root))+'), cash_mode=True)').run()
    assert not app.exception
    assert any('0050僅作獨立比較' in x.value for x in app.info)
    assert sim.read(root/'strategy.sqlite3')==before
