import numpy as np
import pandas as pd
import pytest
from skills.five_axis_replay import FiveAxisReplay,ARMS
from skills.cash_risk_replay import CashRiskReplay
from skills.scenario_exit_replay import ExitSignals
from skills.execution_stress import audit_stress
from test_cash_allocation_replay import fixture,ENTRY,SIZE


def execute(arm='control',stress='control',**options):
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+12,**options)
    engine=FiveAxisReplay(*args,exit_signals=ExitSignals(adjusted,days),arm=arm,stress_mode=stress,**kwargs)
    account=engine.run();audit_stress(account)
    return engine,account,days


def test_control_reproduces_full_cash_account():
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+12)
    expected=CashRiskReplay(*args,exit_signals=ExitSignals(adjusted,days),**kwargs).run()
    assert execute()[1]==expected


@pytest.mark.parametrize('arm',ARMS)
def test_every_arm_reconciles_and_keeps_idle_cash(arm):
    _,account,_=execute(arm)
    assert all(t['stock_id']!='0050' for t in account['trades'])
    assert all(d['cash']>=0 for d in account['daily'])


def test_resting_limit_does_not_assume_low_touch_fills():
    def prices(quotes,days):
        quotes.loc[(quotes.stock_id=='1101')&(quotes.date==days[ENTRY]),'high']=60
    engine,account,days=execute('limit3',mutate=prices)
    buys=[t for t in account['trades'] if t['side']=='buy']
    assert buys and buys[0]['date']==str(days[ENTRY+1].date())
    assert account['cohorts'][0]['entry_date']==str(days[ENTRY+1].date())
    assert any(d['reason']=='waiting_limit_all_day' for d in engine.policy_decisions)


def test_wait_expires_after_three_attempt_days():
    def prices(quotes,days):
        quotes.loc[(quotes.stock_id=='1101')&quotes.date.between(days[ENTRY],days[ENTRY+2]),'high']=60
    engine,account,_=execute('limit3',mutate=prices)
    assert not account['trades']
    assert len(engine.policy_decisions)==3


def test_broken_support_cancels_later_recovery():
    stock=np.full(SIZE,100.);stock[ENTRY]=90
    def prices(quotes,days):
        quotes.loc[(quotes.stock_id=='1101')&(quotes.date==days[ENTRY]),'high']=60
    engine,account,_=execute('limit3',stock=stock,mutate=prices)
    assert not account['trades']
    assert any(d['reason']=='cancel_broken_support' for d in engine.policy_decisions)


def test_staged_add_uses_prior_strength_without_resetting_exit():
    stock=np.full(SIZE,100.);stock[ENTRY+4:]=105.
    engine,account,days=execute('staged',stock=stock)
    adds=[t for t in account['trades'] if t['reason']=='staged_add']
    assert adds and adds[0]['date']==str(days[ENTRY+5].date())
    assert adds[0]['signal_date']==str(days[ENTRY+4].date())
    assert engine.exit_states['entry-'+str(ENTRY)]['entry_price']==100
    assert account['cohorts'][0]['bought_qty']==sum(t['qty'] for t in account['trades'] if t['side']=='buy')


def two_stock_account(arm):
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+3)
    quotes,companies,calendar,events,feeds,corporate=args
    extra=quotes[quotes.stock_id.eq('1101')].copy()
    extra.stock_id='1102';extra.volume*=2
    quotes=pd.concat([quotes,extra],ignore_index=True)
    companies=pd.concat([companies,pd.DataFrame([dict(stock_id='1102',name='1102',market='TWSE')])])
    adjusted['1102']=adjusted['1101']
    events[0]['group_members']=['1101','1102']
    events.append({**events[0],'event_id':'second','members':['1102']})
    from test_cash_allocation_replay import Feeds
    engine=FiveAxisReplay(quotes,companies,calendar,events,Feeds(quotes),corporate,
        exit_signals=ExitSignals(adjusted,days),arm=arm,**kwargs)
    account=engine.run();audit_stress(account)
    return engine,account


def test_capacity_orders_more_liquid_candidate_first():
    _,account=two_stock_account('capacity')
    assert account['trades'][0]['stock_id']=='1102'


def test_group_constraint_rejects_overlapping_peer_without_forced_sale():
    engine,account=two_stock_account('group_one')
    assert {t['stock_id'] for t in account['trades']}=={'1101'}
    assert any(d['reason']=='overlapping_signal_group' for d in engine.policy_decisions)
