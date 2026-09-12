from copy import deepcopy
from datetime import timedelta
from decimal import Decimal as D
import pytest
from streamlit.testing.v1 import AppTest
from app import forward_readiness as r, forward_cash_automation as a, forward_automation as base
from app import forward_cash_policy as cash, forward_portfolio as p, forward_journal as j, forward_simulation as sim
from tests.test_forward_cash_policy import create
from tests.test_forward_simulation import observation
from tests.test_forward_corporate_audit import sources
from tests.test_forward_portfolio import clock, close, setup, buy


def test_refresh_is_bounded_and_does_not_approve_or_change_books(tmp_path, monkeypatch):
    root, _, _ = create(tmp_path); evidence=tmp_path/'evidence'; before={x:sim.read(root/(x+'.sqlite3')) for x in ('strategy','benchmark')}
    calls=[]
    def refresh(path, evidence_path, at, request_budget):
        calls.append(request_budget)
        sources(path,evidence_path,at=at)
        return {'calls':request_budget,'reused':0}
    monkeypatch.setattr(r.corporate,'refresh',refresh)
    result=r.refresh(root,evidence,clock(),budget=3)
    assert calls==[3] and result['calls']==3
    assert result['report']['books']['benchmark']['source_blocked']
    assert not result['report']['books']['strategy']['review_current']
    assert before=={x:sim.read(root/(x+'.sqlite3')) for x in before}
    with pytest.raises(ValueError):r.refresh(root,evidence,clock(),budget=13)


def test_readiness_expiry_changes_next_action_without_ledger_write(tmp_path):
    root, _, _ = create(tmp_path); evidence=tmp_path/'evidence'
    for role in ('strategy','benchmark'):sources(root/(role+'.sqlite3'),evidence,at=clock())
    report=r.inspect(root,evidence,clock())
    assert report['completed_days']==0 and not report['broker_execution_verified']
    assert all(not b['source_blocked'] for b in report['books'].values())
    assert all(x['average_price'] is None for b in report['books'].values() for x in b['execution'])
    later=r.inspect(root,evidence,clock('2026-09-11',22))
    assert later['next_action']=='更新公司行動來源'


def test_workflow_requires_real_observation_details_and_all_successful_stages():
    rows=[{'body':dict(date='2026-09-14',stage=name,status='ok',detail={})} for name in r.STAGES]
    assert not r.workflow(rows)[0]['completed']
    rows[0]['body']['detail']=[{'role':role,'status':'collect'} for role in ('strategy','benchmark')]
    assert r.workflow(rows)[0]['completed']
    rows[-1]['body']['detail']={'entry_block':'missing signal'}
    assert r.workflow(rows)[0]['missing']==['plan_benchmark']
    rows[-1]['body']['detail']={}
    rows[0]['body']['detail'].append({'status':'blocked'})
    assert not r.workflow(rows)[0]['completed']


def test_cash_day_rehearsal_partial_fill_cancel_review_resume_and_no_etf(tmp_path, monkeypatch):
    """Real sealed orchestration/accounting; fixture market, database prices and signal input."""
    root, original, benchmark=create(tmp_path); before=(sim.read(original),sim.read(benchmark))
    monkeypatch.setattr(a,'calendar_day',lambda day:True)
    monkeypatch.setattr(base,'markets',lambda ids:{sid:'tse' for sid in ids})
    monkeypatch.setattr(a.corporate,'refresh',lambda *args,**kw:{})
    monkeypatch.setattr(a.corporate,'inspect',lambda *args,**kw:{'blocked':False})
    monkeypatch.setattr(base,'approval_signature',lambda *args:'fixture-same-day-sources')
    current=[clock('2026-09-14',10)()]
    def fetch(market,sid,channel):
        seconds=int((current[0]-clock('2026-09-14',10)()).total_seconds())
        obs=observation(seconds,1000+seconds*100)
        q=obs['body']['quote'];q.update(stock_id=sid,channel=channel)
        if channel=='board':
            q['volume_shares']*=1000
            for level in q['asks']+q['bids']:level['shares']*=1000
        obs['hash']=j.digest(obs['body']);return obs
    def sleep(seconds):current[0]+=timedelta(seconds=seconds)
    assert a.run(root,clock=lambda:current[0],sleeper=sleep,fetcher=fetch)['status']=='ok'
    strategy=root/'strategy.sqlite3'
    stats=r.execution(sim.read(strategy));stock=next(x for x in stats if x['stock_id']=='2492')
    assert stock['filled']==200 and stock['unfilled']==799
    assert stock['average_price']=='99.1' and D(stock['fees_and_tax'])==28
    assert stock['adverse_vs_limit_bps']==-90
    # Same snapshots cannot produce a second fill.
    sim.match(strategy,fetch('tse','2492','odd'),lambda:current[0])
    assert p.summary(strategy,lambda:current[0])['fill_count']==1
    calls=[]
    monkeypatch.setattr(a,'command',lambda *args:calls.append('pipeline'))
    monkeypatch.setattr(a,'_signals',lambda *args:{'hash':'fixture-signal'})
    def seal_price(path, **kwargs):
        with j.connection(path) as con:
            return close(con,'2026-09-14',{'2492':'101','0050':'101'})
    monkeypatch.setattr(a.corporate,'capture_close',seal_price)
    # Freeze an empty next-day candidate input at the external signal boundary.
    reference=tmp_path/'reference'
    with j.connection(reference) as con:
        setup(con);close(con,'2026-09-14',{'2492':'101','0050':'101'})
        event=buy(con,n=1000,day='2026-09-15',signalday='2026-09-14')
    proof=deepcopy(event['body']['signal_proof']);proof['body']['signal_date']='2026-09-14';proof['body']['candidates']=[]
    proof['hash']=j.digest({k:v for k,v in proof.items() if k!='hash'})
    monkeypatch.setattr(cash.service,'source_signal',lambda *args:proof)
    current[0]=clock('2026-09-14',18)()
    blocked=a.run(root,clock=lambda:current[0])
    assert blocked['status']=='needs_attention'
    assert not any(x['body'].get('date')=='2026-09-14' for x in sim.read(strategy) if x['kind']=='close')
    assert p.summary(strategy,lambda:current[0])['reserved_cash']=='0'
    for role in ('strategy','benchmark'):
        base.approve(root/(role+'.sqlite3'),'TEST FIXTURE','測試人工確認，僅存在隔離測試帳本',clock=lambda:current[0])
    current[0]+=timedelta(minutes=16)
    assert a.run(root,clock=lambda:current[0])['status']=='ok'
    assert r.workflow(base._runs(root))[0]['completed']
    assert calls==['pipeline']
    head=sim.read(strategy)[-1]['hash']
    assert a.run(root,clock=lambda:current[0])['status']=='ok'
    assert sim.read(strategy)[-1]['hash']==head
    state=p.state(cash.verify(strategy,'strategy'))
    assert '0050' not in state['holdings']
    assert all(o['stock_id']!='0050' or o['closed'] for o in state['orders'].values())
    assert (sim.read(original),sim.read(benchmark))==before


def test_readiness_ui_reads_without_fetching_or_approving(tmp_path,monkeypatch):
    root, _, _=create(tmp_path)
    def never(*args,**kwargs):pytest.fail('opening panel must not refresh or approve')
    monkeypatch.setattr(r,'refresh',never)
    monkeypatch.setattr(base,'approve',never)
    before=sim.read(root/'strategy.sqlite3')
    app=AppTest.from_string('from pathlib import Path\nfrom app.forward_readiness_ui import render\nrender(Path('+repr(str(root))+'))').run()
    assert not app.exception
    assert any('0 天' in x.value for x in app.markdown)
    assert sim.read(root/'strategy.sqlite3')==before
