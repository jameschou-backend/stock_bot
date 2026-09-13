from copy import deepcopy
from datetime import timedelta
from decimal import Decimal as D
import pytest
from app import capacity_forward as cap, capacity_forward_runner as runner
from app import forward_journal as j, forward_portfolio as p, forward_simulation as sim
from tests.test_forward_simulation import initialized,observation
from tests.test_forward_portfolio import clock,close,fill


def prepare(tmp_path):
    old,source,benchmark=initialized(tmp_path)
    original=sim.read(source)
    base=next(r['body']['signal'] for r in original if r['kind']=='order')
    candidates=[]
    for sid,amount,priority in [('2492',100000000,.9),('1101',500000000,.2),('1102',300000000,.4),('1103',200000000,.6)]:
        c=deepcopy(base);c.update(members=[sid],priority=priority,event_id='event-'+sid)
        c['liquidity_before_entry']['mean_turnover20_twd']=amount
        candidates.append(c)
    with j.connection(old/'signals.sqlite3') as con:
        j.append(con,'signal:2026-09-11','signal',dict(signal_date='2026-09-11',eligible=True,candidates=candidates),clock())
    root=tmp_path/'capacity'
    cap.initialize(root,source,benchmark,old/'signals.sqlite3',clock())
    assert original==sim.read(source)
    return root,source,benchmark


def test_three_books_rank_all_candidates_before_reserving_three_slots(tmp_path):
    root,source,benchmark=prepare(tmp_path)
    for role,expected in [('strategy',['1101','1102','1103']),('control',['2492','1103','1102'])]:
        rows=cap.verify(root/(role+'.sqlite3'),role)
        plan=next(r['body'] for r in rows if r['kind']=='capacity_plan')
        assert [r['stock_id'] for r in plan['decisions'] if r['selected']]==expected
        assert len(plan['decisions'])==4 and not plan['decisions'][-1]['selected']
        state=p.state(rows)
        assert '0050' not in state['holdings'] and state['cash']==1000000
        assert p.reserves(state,'2026-09-11')[0]<=1000000
    assert cap.initialize(root,source,benchmark,clock=clock())==root
    assert any(o['stock_id']=='0050' and not o['closed'] for o in p.state(cap.verify(root/'benchmark.sqlite3'))['orders'].values())


def test_late_initialization_refuses_backdated_start(tmp_path):
    root,source,benchmark=prepare(tmp_path)
    with pytest.raises(ValueError,match='開盤'):
        cap.initialize(tmp_path/'late',source,benchmark,root/'signals.sqlite3',clock('2026-09-14',10))


def test_plan_rerun_no_duplicate_and_code_or_role_changes_fail(tmp_path,monkeypatch):
    root,_,_=prepare(tmp_path);path=root/'strategy.sqlite3';before=cap.verify(path)
    cap.save_plans(path,root/'signals.sqlite3',clock())
    assert before==cap.verify(path)
    with pytest.raises(ValueError,match='角色'):cap.verify(path,'benchmark')
    monkeypatch.setitem(cap.RULES,'max_drawdown','0.30')
    with pytest.raises(ValueError,match='版本'):cap.verify(path)


def test_drawdown_latches_and_pause_cancels_only_new_buys(tmp_path):
    root,_,_=prepare(tmp_path);path=root/'strategy.sqlite3'
    event=cap.pause(path,True,'測試人工停止新增買入，等待重新核對',clock())
    assert cap.risk(cap.verify(path))['blocked']
    assert all(o['closed'] for o in p.state(cap.verify(path))['orders'].values())
    cap.pause(path,False,'測試核對完成可恢復後續新的每日計畫',clock())
    assert not cap.risk(cap.verify(path))['blocked']
    rows=cap.verify(path)
    synthetic=deepcopy(rows);synthetic.append(dict(kind='close',body=dict(nav='799999')))
    assert cap.risk(synthetic)['blocked'] and D(cap.risk(synthetic)['drawdown'])>D('.20')
    synthetic[-1]['body']['nav']=None
    assert cap.risk(synthetic)['blocked']


def test_missing_signal_cannot_suppress_confirmed_stop_exit(tmp_path):
    root,_,_=prepare(tmp_path);path=root/'strategy.sqlite3'
    with j.connection(path) as con:
        rows=sim.verify(con);orders=[r for r in rows if r['kind']=='order' and r['body']['order_id'].startswith('capacity:')]
        order=next(r for r in orders if r['body']['stock_id']=='1101' and r['body']['channel']=='board')
        fill(con,order,1000,'99',fee='141')
    runner.base._cancel_day(path,clock('2026-09-14',14))
    with j.connection(path) as con:close(con,'2026-09-14',{'1101':'80','0050':'100'})
    with pytest.raises(ValueError,match='候選來源'):
        cap.save_plans(path,tmp_path/'missing',clock('2026-09-14'))
    rows=cap.verify(path)
    assert any(r['kind']=='decision' and r['body']['reason']=='stop12_close' for r in rows)
    assert any(o['side']=='sell' and not o['closed'] for o in p.state(rows)['orders'].values())


def test_closed_market_no_network_and_status_no_fills(tmp_path,monkeypatch):
    root,_,_=prepare(tmp_path)
    monkeypatch.setattr(runner.base,'calendar_day',lambda day:False)
    def never(*a,**k):pytest.fail('closed market must not request data')
    assert runner.run(root,clock=clock('2026-09-12'),fetcher=never,sleeper=never)['status']=='closed_market'
    report=runner.status(root,clock())
    assert report['completed_days']==0 and not report['broker_execution_verified']
    assert all(b['fill_count']==0 for b in report['books'].values())


def test_shared_corporate_budget_never_exceeds_twelve(tmp_path,monkeypatch):
    calls=[]
    def refresh(path,clock,request_budget):
        n=min(7,request_budget);calls.append(n);return dict(calls=n,reused=0)
    monkeypatch.setattr(runner.corporate,'refresh',refresh)
    result=runner.refresh(dict(strategy='a',control='b',benchmark='c'),clock())
    assert calls==[7,5] and result['calls']==12


def test_failed_signal_stage_still_attempts_exits_and_nested_block_is_failure(tmp_path,monkeypatch):
    root,_,_=prepare(tmp_path);attempts=[]
    monkeypatch.setattr(runner.base,'calendar_day',lambda day:True)
    monkeypatch.setattr(runner.base,'command',lambda *a:None)
    monkeypatch.setattr(runner,'_signals',lambda *a:(_ for _ in ()).throw(ValueError('missing signal')))
    monkeypatch.setattr(runner,'refresh',lambda *a:None)
    monkeypatch.setattr(runner.base,'close',lambda path,clock:attempts.append(path.name))
    monkeypatch.setattr(cap,'save_plans',lambda *a:dict(entry_block='資料缺漏'))
    result=runner.run(root,clock=clock('2026-09-14',18))
    assert result['status']=='needs_attention' and len(attempts)==3
    stages={r['stage']:r['status'] for r in result['stages']}
    assert stages['signals']=='blocked' and stages['plan_strategy']=='blocked'


def test_real_observation_pair_only_creates_labelled_simulation_and_no_duplicates(tmp_path):
    root,source,_=prepare(tmp_path);path=root/'strategy.sqlite3';before=sim.read(source)
    def observed(seconds,volume):
        o=observation(seconds,volume);o['body']['quote']['stock_id']='1101'
        o['body']['quote']['channel']='board';o['body']['quote']['asks'][0]['shares']=50000
        o['hash']=j.digest(o['body']);return o
    assert not sim.match(path,observed(0,1000),clock('2026-09-14',10))['fills']
    now=lambda:clock('2026-09-14',10)()+timedelta(seconds=20)
    result=sim.match(path,observed(20,21000),now)
    assert result['fills'] and sim.match(path,observed(20,21000),now)==result
    rows=cap.verify(path)
    fills=[r for r in rows if r['kind']=='fill']
    assert fills[0]['body']['evidence']['classification']=='counterfactual_simulation_not_execution_evidence'
    assert p.state(rows)['holdings']['1101']['qty']==2000 and sim.read(source)==before


def test_future_signal_cannot_enter_before_it_was_observed(tmp_path):
    root,_,_=prepare(tmp_path)
    with pytest.raises(ValueError,match='尚未取得'):
        cap._proof(root/'signals.sqlite3','2026-09-11',clock('2026-09-11',19))


def test_pause_keeps_active_exit_orders(tmp_path):
    root,_,_=prepare(tmp_path);path=root/'strategy.sqlite3'
    with j.connection(path) as con:
        order=next(r for r in sim.verify(con) if r['kind']=='order' and r['body']['order_id'].startswith('capacity:') and r['body']['stock_id']=='1101' and r['body']['channel']=='board')
        fill(con,order,1000,'99',fee='141')
    runner.base._cancel_day(path,clock('2026-09-14',14))
    with j.connection(path) as con:close(con,'2026-09-14',{'1101':'80','0050':'100'})
    with pytest.raises(ValueError):cap.save_plans(path,tmp_path/'missing',clock('2026-09-14'))
    cap.pause(path,True,'人工重新核對系統只停止新的買進',clock('2026-09-14'))
    assert any(o['side']=='sell' and not o['closed'] for o in p.state(cap.verify(path))['orders'].values())
