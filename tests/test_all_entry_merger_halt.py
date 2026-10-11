from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import skills.all_entry_merger_halt as subject
from skills.mixed_odd_replay import MixedOrders
from skills.poc_range_execution import RangeOrders
from skills.residual_tick_replay import TickPlanning


class Planner(subject.MergerHaltBuy,RangeOrders,MixedOrders,TickPlanning):
    def prior(self,*args):return 11.8
    def official_halt(self,day,sid):return sid=='2888' and subject.START<=str(day.date())<subject.END


def planner():
    obj=object.__new__(Planner)
    obj.days=pd.to_datetime(['2025-07-11','2025-07-14']);obj.positions={d:i for i,d in enumerate(obj.days)}
    obj.day_plans={};obj.tick_plans=[];obj.base_tick_plans=[];obj.tick_attempts=set();obj.holdings={}
    obj.residual_budget=300000.;obj.benchmark=False;obj.orders=[]
    def no_limits(*_):raise AssertionError('Halted order must not read absent limits')
    obj.feeds=SimpleNamespace(get_limits=no_limits)
    return obj


def plan(obj,**changes):
    args=dict(day=pd.Timestamp(subject.START),sid='2888',side='buy',eid='halted-buy',signal='2025-07-11',
        qty=20000,budget=300000.,opening_cash=600000.,failure=None)
    args.update(changes);obj._plan(**args)
    return obj.day_plans[(args['eid'],args['side'])]


def test_real_planning_layers_reject_without_limits_and_keep_budget():
    obj=planner();p=plan(obj)
    assert p['planned_qty']==p['board_qty']==p['odd_qty']==0
    assert p['reserved_cash']==p['sizing_budget']==300000.
    assert p['rejection']==subject.REASON and p['prior_reference']==11.8
    assert obj.tick_plans==[p]
    assert obj._execute_order(pd.Timestamp(subject.START),'2888','buy',20000,'entry','halted-buy','2025-07-11')==0
    assert obj.orders==[dict(date=subject.START,stock_id='2888',side='buy',event_id='halted-buy',
        signal_date='2025-07-11',reason='entry',channel='event',requested_qty=0,filled_qty=0,
        failure='official_full_session_halt')]


@pytest.mark.parametrize('side,held',[('sell',0),('buy',1)])
def test_conversion_rights_cannot_be_discarded(side,held):
    obj=planner();obj.holdings={'2888':dict(qty=held)}
    with pytest.raises(ValueError,match='merger rights'):plan(obj,side=side)


def test_unverified_halt_keeps_strict_missing_limit_failure():
    obj=planner();obj.official_halt=lambda *_:False
    with pytest.raises(AssertionError,match='absent limits'):plan(obj)


def test_proof_rejects_positive_inventory_and_cash():
    obj=planner();p=plan(obj)
    case=dict(completed=True,account=dict(holdings=[],trades=[],tick_plans=[p],orders=[],cash_ledger=[]))
    assert subject.audit_halt(case)['plans_checked']==1
    for name,record in [('holdings',dict(stock_id='2888',date=subject.END,qty=1)),
                        ('trades',dict(stock_id='2888',date=subject.START,qty=1)),
                        ('cash_ledger',dict(date=subject.START,event_id='halted-buy',cash_change=-1))]:
        changed=deepcopy(case);changed['account'][name].append(record)
        with pytest.raises(ValueError):subject.audit_halt(changed)


def test_publication_rejects_unproved_suspension(monkeypatch):
    from scripts.summarize_all_entry_accounts import guard_certificate_inventory,UnverifiedCertificateInventory
    monkeypatch.setattr(subject,'load_halt',lambda *_:({},{}))
    p=plan(planner())
    case=dict(completed=True,account=dict(holdings=[],trades=[],tick_plans=[p],orders=[],cash_ledger=[]))
    with pytest.raises(UnverifiedCertificateInventory,match='2888'):guard_certificate_inventory(case)


def test_primary_notice_and_reader_preserve_frozen_terms(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    doc=root/subject.DOCUMENT
    if not (root/'.cache/all-entry-accounts-20261011/required-2888-halt-v1').exists():
        pytest.skip('Archived primary evidence unavailable')
    halt,refs=subject.load_halt(root,doc)
    assert halt['announcement_date']<halt['start'] and len(refs)==6
    source=dict(verified_halts=[],source_sha256={})
    runner=SimpleNamespace(read=lambda _:source,engine_types=lambda *_:(),write=lambda *_:None)
    subject.install_halt(runner,doc,root)
    assert runner.read(root/'docs/stock_universe_2019_corporate_terms.json')['verified_halts']==[halt]
    assert source==dict(verified_halts=[],source_sha256={})
