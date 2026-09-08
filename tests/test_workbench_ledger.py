from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.models import Base
from app.workbench_models import TABLES
from app.workbench_ledger import *


@pytest.fixture
def session():
    engine=create_engine('sqlite://')
    Base.metadata.create_all(engine,tables=TABLES)
    with Session(engine) as s:
        initialize_account(s,'paper',100000)
        yield s


def fill(s,side,qty,price,fee=0,tax=0,**kwargs):
    defaults=dict(account_id='paper',fill_id=uuid4().hex,stock_id='2330',side=side,qty=qty,
                  price=price,fee=fee,tax=tax,executed_at=datetime(2026,1,2,1))
    defaults.update(kwargs)
    return record_fill(s,**defaults)


def test_partial_fill_cancel_and_net_pnl(session):
    pid=create_plan(session,'paper','2330',100,90,300)
    fill(session,'buy',200,100,29,plan_id=pid)
    p=plans_for(session,'paper')[0]
    assert p.filled_qty==200
    assert reservation(p)>10000
    cancel_plan(session,'paper',pid)
    assert reservation(p)==0
    fill(session,'sell',100,110,20,33,executed_at=datetime(2026,1,3,1))
    state=ledger_state(100000,fills_for(session,'paper'),{'2330':{'close':100,'date':'2026-01-03'}})
    assert state['realized_pnl']==pytest.approx(932.5)
    assert state['unrealized_pnl']==pytest.approx(-14.5)
    assert state['net_pnl']==pytest.approx(918)
    assert state['cash']==90918
    assert state['fees_and_tax']==82


def test_missing_quote_is_unknown_not_zero_profit(session):
    fill(session,'buy',100,100)
    state=ledger_state(100000,fills_for(session,'paper'))
    assert state['net_pnl'] is None
    assert state['missing_quotes']==['2330']


def test_duplicate_fill_and_different_payload(session):
    fid=uuid4().hex
    fill(session,'buy',100,100,fill_id=fid)
    fill(session,'buy',100,100,fill_id=fid)
    assert len(fills_for(session,'paper'))==1
    with pytest.raises(ValueError,match='不同內容'):
        fill(session,'buy',101,100,fill_id=fid)


def test_oversell_cash_and_plan_mismatch(session):
    with pytest.raises(ValueError,match='超賣'):
        fill(session,'sell',1,100)
    with pytest.raises(ValueError,match='現金不足'):
        fill(session,'buy',1001,100)
    pid=create_plan(session,'paper','2330',100,90,100)
    with pytest.raises(ValueError,match='超過計畫'):
        fill(session,'buy',101,100,plan_id=pid)
    assert not fills_for(session,'paper')


def test_separate_accounts_and_reservations(session):
    initialize_account(session,'real',200000)
    create_plan(session,'paper','2330',100,90,600)
    with pytest.raises(ValueError,match='預留'):
        create_plan(session,'paper','2317',100,90,600)
    fill(session,'buy',100,100)
    assert not fills_for(session,'real')
    assert preview_plan(100000,100000,100,90,1,15,1000)['qty']==0
    with pytest.raises(ValueError):
        preview_plan(100000,100000,100,100,1,15)


def test_same_timestamp_keeps_cash_transition_order(session):
    # Simulate MySQL second-level timestamps and reverse lexical UUID ordering.
    fill(session,'buy',10,100,fill_id='f'*32)
    fill(session,'sell',10,110,fill_id='0'*32)
    for item in fills_for(session,'paper'):
        item.created_at=datetime(2026,1,2,2)
    session.flush()
    session.expire_all()
    assert [f.side for f in fills_for(session,'paper')]==['buy','sell']
    assert ledger_state(100000,fills_for(session,'paper'))['net_pnl']==100


def test_small_positions_include_minimum_fees_in_risk_budget():
    preview=preview_plan(1000,1000,100,90,1,100,1)
    assert preview['qty']==0
    preview=preview_plan(100000,100000,100,90,1,15,1)
    assert preview['estimated_loss']<=1000
    assert preview_plan(100000,100000,100,90,1,15,1,current_exposure=15000)['qty']==0
    with pytest.raises(ValueError):
        amount(1e308)
