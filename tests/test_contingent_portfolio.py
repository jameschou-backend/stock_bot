from copy import deepcopy
from decimal import Decimal
import pandas as pd
import pytest
from skills.contingent_portfolio import PortfolioReplay,sized_orders
from skills.contingent_replay import Tape
from skills.contingent_execution import OPEN


CAL=[str(d.date()) for d in pd.bdate_range('2022-01-03',periods=80)]


def event(index,sid='2330'):
    return dict(entry_date=CAL[index],signal_date=CAL[index-1],members=[sid],event_id=f'e{index}-{sid}')


def views(prices=None):
    prices=prices or {}
    def view(day,sids):
        return {s:dict(date=day,raw_cents=prices.get((day,s),10000),
            adjusted_close=str(prices.get((day,s),10000)),adv20_shares=100_000_000,
            amount20_cents=100_000_000_000) for s in sids}
    return view


def tape_provider(prices=None,missing=None,volumes=None):
    prices=prices or {};volumes=volumes or {}
    def tapes(day,keys):
        result={}
        for sid,channel in keys:
            if missing and (day,sid,channel) in missing:continue
            price=prices.get((day,sid),10000)
            # Both directions have post-order liquidity, with sufficient volume.
            rows=((OPEN+10,price-100,volumes.get((day,sid,channel),100_000_000),True),
                  (OPEN+20,price+100,volumes.get((day,sid,channel),100_000_000),True),
                  (OPEN+2_000_000,price-100,100_000_000,True))
            result[(sid,channel)]=Tape(sid,channel,'TWSE',day,rows,'synthetic fixture','a'*64,True)
        return result
    return tapes


def make(entries=None,prices=None,actions=None,tapes=None,end=5,**kwargs):
    return PortfolioReplay(CAL,entries or [event(1)],views(prices),
        actions or (lambda day,sids:[]),tapes or tape_provider(prices),start=CAL[1],end=CAL[end],**kwargs)


def test_crossday_uses_actual_position_and_no_repeat_buy():
    r=make().run()
    assert r['completed'] and r['synthetic'] and len(r['daily'])==5
    assert r['state']['holdings']==r['daily'][0]['holdings']
    assert sum(len(s['fills']) for s in r['sessions'])==2
    assert r['total_return']<0 and r['live_qualified'] is False


def test_missing_odd_stops_first_session_without_pretending_zero_fill():
    r=make(tapes=tape_provider(missing={(CAL[1],'2330','odd')})).run()
    assert not r['completed'] and r['total_return'] is None and not r['daily']
    assert r['state']['available_cents']==100_000_000 and not r['state']['holdings']
    assert r['blocked']['date']==CAL[1] and r['blocked']['missing'][0]['channel']=='odd'


def test_missing_end_mark_rolls_back_entire_day():
    engine=make(end=1)
    base=engine.view
    engine.view=lambda day,sids:{} if day==CAL[1] else base(day,sids)
    r=engine.run()
    assert not r['completed'] and r['blocked']['stage']=='valuation'
    assert not r['sessions'] and r['state']['holdings']=={}


def test_stop_uses_previous_close_and_latches_on_partial_sale():
    prices={(CAL[2],'2330'):8700,(CAL[3],'2330'):10000,(CAL[4],'2330'):10000}
    engine=make(prices=prices,end=4)
    base=tape_provider(prices)
    def provider(day,keys):
        if day==CAL[3]:
            return {k:Tape(k[0],k[1],'TWSE',day,((OPEN+10,10100,100000 if k[1]=='board' else 0,True),),'fixture','a'*64,True) for k in keys}
        return base(day,keys)
    engine.tapes=provider;r=engine.run()
    assert r['completed']
    assert r['plans'][2]['exit_reasons']=={'2330':'loss12'}
    assert r['plans'][3]['exit_reasons']=={'2330':'loss12'}
    assert all(p['signal_date']==CAL[2] for p in r['plans'][3]['spec']['plans'])
    assert not r['state']['holdings']


def test_time63_is_based_on_actual_entry_session():
    r=make(end=64).run()
    assert r['completed'] and not r['state']['holdings']
    exits=[p for p in r['plans'] if p['exit_reasons']]
    assert exits[0]['date']==CAL[64] and exits[0]['exit_reasons']=={'2330':'time63'}


def cash_action(day=2,pay=4):
    return dict(action_id='cash1',stock_id='2330',date=CAL[day],known_date=CAL[day-1],
        kind='cash_dividend',source_id='fixture',verified=True,cash_per_share_cents=100,
        reference_cents=9900,available_date=CAL[pay])


def test_dividend_entitlement_payment_preserves_nav_without_double_credit():
    a=cash_action();prices={(CAL[i],'2330'):9900 for i in range(2,6)}
    engine=make(prices=prices,actions=lambda day,sids:[a] if day==a['date'] else [])
    r=engine.run();qty=r['daily'][0]['holdings']['2330'];first=r['daily'][0]
    assert r['completed']
    assert r['daily'][1]['receivable_value_cents']==qty*100
    assert r['daily'][1]['nav_cents']==first['nav_cents']
    assert r['daily'][3]['available_cents']==first['available_cents']+qty*100
    assert r['daily'][3]['nav_cents']==first['nav_cents'] and not r['state']['pending']


def test_late_known_or_unsupported_action_stops_before_trading():
    for changes in ({'known_date':CAL[2]},{'kind':'unverified_merger'}):
        a=dict(cash_action(),**changes)
        r=make(actions=lambda day,sids:[a] if day==a['date'] else []).run()
        assert not r['completed'] and r['blocked']['stage']=='corporate_actions' and len(r['daily'])==1


def test_stock_rights_keep_slot_after_sale_and_deliver_old_exit_instruction():
    a=dict(cash_action(),action_id='stock1',kind='stock_dividend',new_shares_per_share='.1',
        fractional_cash_per_share_cents=10000,reference_cents=9000,available_date=CAL[5])
    prices={(CAL[i],'2330'):8700 for i in range(2,7)}
    engine=make(entries=[event(1),event(3,'2317'),event(4,'2317')],slots=1,prices=prices,
        actions=lambda day,sids:[a] if day==a['date'] else [],end=6)
    r=engine.run()
    assert r['completed']
    assert '2330' not in r['daily'][2]['holdings']
    assert not r['plans'][2]['candidates'] and not r['plans'][3]['candidates']
    assert r['plans'][4]['exit_reasons']=={'2330':'loss12'}
    assert not r['state']['holdings'] and not r['state']['pending']


def test_uncredited_sale_becomes_available_only_after_configured_lag():
    prices={(CAL[i],'2330'):8700 for i in range(2,7)}
    r=make(prices=prices,end=6,credit_delay_us=None).run()
    assert r['completed']
    assert r['daily'][2]['receivable_value_cents']>0
    assert r['daily'][4]['available_cents']==r['daily'][2]['available_cents']
    assert r['daily'][5]['receivable_value_cents']==0
    assert r['daily'][5]['available_cents']>r['daily'][4]['available_cents']


def test_future_prices_do_not_change_prior_plan():
    a=make(end=2).run();b=make(prices={(CAL[2],'2330'):5000},end=2).run()
    assert a['plans']==b['plans']


def test_compounding_uses_latest_nav_for_new_candidate_budget():
    prices={(CAL[i],'2330'):20000 for i in range(2,5)}
    r=make(entries=[event(1),event(3,'2317')],prices=prices,end=3).run()
    p=next(p for p in r['plans'][-1]['spec']['plans'] if p['stock_id']=='2317')
    assert p['qty']>r['plans'][0]['spec']['plans'][0]['qty']


def test_channel_sizing_handles_board_boundary_fee_discontinuity():
    for budget in (10000000,10030000,10060000,10100000):
        orders=sized_orders(CAL[1],'2330',budget,10000,CAL[0],45)
        assert sum(p['budget_cents'] for p in orders)<=budget


def test_verified_split_exchanges_shares_without_creating_free_nav():
    a=dict(cash_action(),kind='split',exchange_ratio='4',reference_cents=2500,available_date=CAL[2])
    prices={(CAL[i],'2330'):2500 for i in range(2,6)}
    e=make(prices=prices,actions=lambda day,sids:[a] if day==a['date'] else [])
    base=e.view
    def adjusted_view(day,sids):
        rows=base(day,sids)
        for row in rows.values():row['adjusted_close']='10000'
        return rows
    e.view=adjusted_view;r=e.run()
    assert r['completed'] and r['daily'][1]['holdings']['2330']==r['daily'][0]['holdings']['2330']*4
    assert r['daily'][1]['nav_cents']==r['daily'][0]['nav_cents']
    assert not any(p['exit_reasons'] for p in r['plans'])


def test_rights_gate_does_not_free_pending_member_when_physical_shares_sold():
    from skills.contingent_session import RightsGate
    from skills.contingent_execution import Plan
    plans=[Plan('s1','2330','sell','board',1000,10000,0,CAL[0]),
        Plan('s2','2317','sell','board',1000,10000,0,CAL[0]),
        Plan('b','2454','buy','board',1000,10000,10100000,CAL[0])]
    gate=RightsGate(CAL[1],plans,{'2330':1000,'2317':1000},20000000,2,reserved_members=['2330'])
    assert gate.submit_ready(OPEN)==['s1','s2']
    gate.confirm_fill('f1','s1',OPEN+1,1000,10000,9900000)
    assert gate.submit_ready(OPEN+1)==[]
    gate.confirm_fill('f2','s2',OPEN+2,1000,10000,9900000)
    assert gate.submit_ready(OPEN+2)==['b']


def test_benchmark_reinvests_dividend_only_when_available():
    a=dict(cash_action(),stock_id='0050',available_date=CAL[4])
    prices={(CAL[i],'0050'):9900 for i in range(2,6)}
    r=make(prices=prices,benchmark=True,actions=lambda day,sids:[a] if day==a['date'] else []).run()
    assert r['completed']
    first=r['daily'][0]['holdings']['0050']
    assert r['daily'][2]['holdings']['0050']==first
    assert r['daily'][3]['holdings']['0050']>first


def test_capital_reduction_exchanges_and_returns_cash_once():
    a=dict(cash_action(),kind='capital_reduction',exchange_ratio='.5',cash_return_per_old_share_cents=5000,
        fractional_cash_per_share_cents=10000,reference_cents=10000,available_date=CAL[3])
    e=make(actions=lambda day,sids:[a] if day==a['date'] else [])
    r=e.run();first=r['daily'][0];last=r['daily'][-1]
    qty=first['holdings']['2330']
    assert r['completed'] and last['holdings']['2330']==qty//2
    assert last['nav_cents']==first['nav_cents']
    assert not r['state']['pending']
