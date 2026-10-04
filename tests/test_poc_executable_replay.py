"""Causal allocation and independent reconstruction of precommitted orders."""
from copy import deepcopy
from collections import defaultdict
from types import SimpleNamespace
import math

import pandas as pd
import pytest

from skills import poc_executable_replay as m
from skills.poc_executable_odd import SCOPE


def tape(rows):
    return pd.DataFrame([(pd.Timedelta(t),p,n) for t,p,n in rows],columns=['time','price','shares'])


def test_print_price_is_fixed_when_capacity_arrives_not_by_later_extremes():
    early=tape([('09:01:00',90,900000),('09:02:00',101,99000),
                ('09:03:00',102,1000),('09:04:00',103,100000)])
    result=m.match_board_prints(early,'buy',110,2000,1_000_000)
    assert result['filled_qty']==2000 and result['reference_price']==102.5
    assert [(x['qty'],x['price']) for x in result['allocations']]==[(1000,102),(1000,103)]
    later=pd.concat([early,tape([('12:30:00',1,900000),('13:24:00',109,900000)])],ignore_index=True)
    changed=m.match_board_prints(later,'buy',110,2000,1_000_000)
    assert changed['reference_price']==result['reference_price']
    assert changed['allocations']==result['allocations']
    assert result['actual_fill_verified'] is False


@pytest.mark.parametrize('side,limit,prices',[('buy',100,[100,101]),('sell',100,[100,99])])
def test_equal_or_wrong_side_prints_get_no_queue_credit(side,limit,prices):
    data=tape([('09:02:00',p,1_000_000) for p in prices])
    r=m.match_board_prints(data,side,limit,1000,1_000_000)
    assert r['filled_qty']==0 and r['reference_price'] is None


def test_prior_adv_limits_capacity_and_unconfirmed_boundaries_are_excluded():
    data=tape([('09:01:00',105,1_000_000),('09:01:01',105,250000),('13:25:00',109,1_000_000)])
    r=m.match_board_prints(data,'sell',90,3000,199999)
    assert r['eligible_shares']==250000 and r['filled_qty']==1000
    assert r['reference_price']==105 and r['last_fill_time']=='09:01:01'


@pytest.mark.parametrize('field,value',[('price',float('nan')),('price',0),('shares',-1),('shares',1.5),('shares',True)])
def test_corrupt_tape_cannot_become_a_zero_fill(field,value):
    data=tape([('09:02:00',100,100000)])
    data[field]=value
    with pytest.raises(m.ReplayDataUnavailable):m.match_board_prints(data,'buy',110,1000,1_000_000)


def test_clock_order_and_unregistered_parameters_fail_closed():
    data=tape([('09:03:00',100,100000),('09:02:00',100,100000)])
    with pytest.raises(m.ReplayDataUnavailable):m.match_board_prints(data,'buy',110,1000,1_000_000)
    data=data.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):m.match_board_prints(data,'buy',110,999,1_000_000)
    with pytest.raises(ValueError):m.match_board_prints(data,'buy',110,1000,1_000_000,.02)


def test_benchmark_and_stock_sell_plans_use_same_prior_known_lower_bound():
    original=dict(date='2024-01-02',stock_id='0050',side='sell',event_id='a',planned_qty=1050,
                  limit_price=100,odd_limit=90)
    class Base:
        def _plan(self,day):self.tick_plans.append(deepcopy(original))
    class Account(m.ExecutableOrders,Base):pass
    obj=Account();obj.tick_plans=[];obj.day_plans={}
    obj.feeds=SimpleNamespace(get_limits=lambda sid:{'2024-01-02':dict(lower=90,upper=110)})
    obj._plan(pd.Timestamp('2024-01-02'))
    p=obj.day_plans[('a','sell')]
    assert p['limit_price']==90 and p['odd_limit']==90
    assert p['order_time']=='09:01:00' and p['odd_order_time']=='13:40:00'
    assert p==obj.tick_plans[-1] and p is not obj.tick_plans[-1]


@pytest.fixture
def audit_case():
    days=pd.bdate_range('2023-12-01',periods=22)
    day,previous=str(days[-1].date()),str(days[-2].date())
    quotes=pd.DataFrame([dict(date=d,stock_id='2330',close=100,volume=1_000_000) for d in days])
    budget=111000.
    qty=m.sized_quantity(math.floor((budget-40)/(100*1.005925)),110,budget,SimpleNamespace(_costs=m.costs),'2330')
    assert qty//1000==1 and qty%1000>0
    plan=dict(date=day,reference_date=previous,stock_id='2330',event_id='a',side='buy',signal_date=previous,
        order_time=m.OPEN,expires_at=m.END,odd_order_time=m.ODD_OPEN,odd_expires_at=m.ODD_END,
        planned_qty=qty,board_qty=1000,odd_qty=qty%1000,prior_reference=100,limit_price=110,odd_limit=110,
        sizing_budget=budget,reserved_cash=budget)
    data=tape([('09:02:00',100,100000),('12:00:00',101,100000)])
    rows=[];trades=[]
    auction=dict(after_hours=True,odd_high=101,odd_low=101,odd_shares=10000,auction_price=101,
                 volume_scope=SCOPE,volume_unit='shares',price_unit='TWD_per_share',auction_time=m.ODD_END)
    for channel in ['board','odd']:
        row=dict(date=day,stock_id='2330',event_id='a',side='buy',signal_date=previous,
            channel=channel,limit_price=110,requested_qty=plan[channel+'_qty'],
            order_time=m.OPEN if channel=='board' else m.ODD_OPEN,expires_at=m.END if channel=='board' else m.ODD_END,
            prior_avg_volume20=1_000_000,prior_avg_amount20=100_000_000)
        if channel=='board':row.update(m.match_board_prints(data,'buy',110,1000,1_000_000),ticks_sha256='source')
        else:row.update(m.match_after_hours(auction,'buy',110,plan['odd_qty']))
        rows.append(row)
        trades.append(dict(row,qty=row['filled_qty'],**m.costs(row['reference_price'],row['filled_qty'],'buy','2330')))
    account=dict(settings=dict(initial_cash=1_000_000),daily=[dict(date=day,cash=m.money(1_000_000+sum(t['cash_change'] for t in trades)))],
                 tick_plans=[plan],orders=rows,trades=trades,cash_ledger=[])
    ticks=SimpleNamespace(get=lambda *args:(data,'source'))
    odds=SimpleNamespace(get_odd=lambda *args:auction)
    corp=SimpleNamespace(reference_price=lambda sid,day,price:price)
    feeds=SimpleNamespace(get_limits=lambda sid:{day:dict(lower=90,upper=110)})
    return account,ticks,odds,{'2330':'TWSE'},quotes,days,corp,feeds


def test_independent_audit_rebuilds_prices_quantities_and_expenses(audit_case):
    r=m.audit_executable(*audit_case)
    assert r['fills_reconciled']==2 and r['chronological_board_allocations_rebuilt']==1
    assert r['afterhours_auctions_rebuilt']==1 and r['live_qualified'] is False


@pytest.mark.parametrize('change',['date','limit','quantity','clock','allocation','price','fee','missing_trade','reuse','cash'])
def test_independent_audit_rejects_noncausal_or_fabricated_results(audit_case,change):
    account=audit_case[0]
    if change=='date':account['tick_plans'][0]['signal_date']=account['tick_plans'][0]['date']
    elif change=='limit':account['tick_plans'][0]['limit_price']=109
    elif change=='quantity':account['tick_plans'][0]['board_qty']=2000
    elif change=='clock':account['orders'][0]['order_time']='08:59:00'
    elif change=='allocation':account['orders'][0]['allocations'][0]['price']=99
    elif change=='price':account['trades'][0]['reference_price']=99
    elif change=='fee':account['trades'][0]['commission']=0
    elif change=='missing_trade':account['trades'].pop()
    elif change=='reuse':account['orders'].append(deepcopy(account['orders'][0]))
    elif change=='cash':account['settings']['initial_cash']=1000
    with pytest.raises(ValueError):m.audit_executable(*audit_case)


def test_afterhours_sale_cannot_fund_morning_purchase():
    account=dict(settings=dict(initial_cash=100),daily=[dict(date='2024-01-02',cash=100)],cash_ledger=[],
        trades=[dict(date='2024-01-02',channel='odd',side='sell',cash_change=100),
                dict(date='2024-01-02',channel='board',side='buy',cash_change=-200),
                dict(date='2024-01-02',channel='odd',side='sell',cash_change=100)])
    with pytest.raises(ValueError,match='Later sale proceeds'):m.audit_cash_phases(account)


def test_deleting_every_fill_and_adjusting_cash_cannot_erase_planned_orders(audit_case):
    account=audit_case[0]
    account['orders']=[];account['trades']=[];account['daily'][0]['cash']=1_000_000
    with pytest.raises(ValueError,match='planned child order missing'):m.audit_executable(*audit_case)


def test_zero_sized_rejections_are_not_fabricated_halts(audit_case):
    account=audit_case[0]
    p=deepcopy(account['tick_plans'][0]);p.update(event_id='rejected',planned_qty=0,board_qty=0,odd_qty=0,reserved_cash=0)
    account['tick_plans'].append(p)
    account['orders'].append(dict(date=p['date'],event_id=p['event_id'],side='buy',stock_id=p['stock_id'],
        signal_date=p['signal_date'],channel='event',filled_qty=0,requested_qty=0,failure='slots_full'))
    assert m.audit_executable(*audit_case)['all_planned_children_reconciled']==2


def test_nonzero_plan_cannot_hide_under_unverified_halt(audit_case):
    account=audit_case[0];p=account['tick_plans'][0]
    account['trades']=[];account['daily'][0]['cash']=1_000_000
    account['orders']=[dict(date=p['date'],event_id=p['event_id'],side='buy',stock_id=p['stock_id'],
        signal_date=p['signal_date'],channel='event',filled_qty=0,requested_qty=p['planned_qty'],failure='official_full_session_halt')]
    with pytest.raises(ValueError,match='verified halt evidence'):m.audit_executable(*audit_case)


def test_real_execution_path_prices_board_and_odd_independently(audit_case):
    account,ticks,odds,routes,quotes,days,corp,feeds=audit_case
    plan=account['tick_plans'][0];day=days[-1]
    class Engine(m.ExecutableOrders):
        _costs=staticmethod(m.costs)
        def identity(self,*args):return dict(status='identified',category='股票',market='TWSE')
        def official_halt(self,*args):return False
        def require_prior_inputs(self,*args):pass
        def raw(self,day,sid,key='close'):return 1_000_000 if key=='volume' else 100.
        def cash_move(self,day,side,amount,**kwargs):self.cash=m.money(self.cash+amount)
    engine=Engine();engine.day_plans={('a','buy'):plan};engine.tick_attempts=set();engine.markets={}
    engine.feeds=feeds;engine.ticks=ticks;engine.ticks.audit_day=lambda *a:dict(own_order_fill_proven=False)
    engine.odd_feeds=odds;engine.volume20=quotes.pivot(index='date',columns='stock_id',values='volume')
    engine.amount20=engine.volume20*100;engine.names={};engine.used=defaultdict(int);engine.cash=1_000_000.
    engine.holdings={'2330':dict(qty=0)};engine.marks={};engine.day_cost=0;engine.day_basis=0
    engine.orders=[];engine.trades=[]
    assert engine._execute_order(day,'2330','buy',plan['planned_qty'],'entry','a',plan['signal_date'])==plan['planned_qty']
    assert [(t['channel'],t['reference_price']) for t in engine.trades]==[('board',100.),('odd',101.)]
    assert engine.trades[1]['commission']==20
    assert engine.cash==m.money(1_000_000+sum(t['cash_change'] for t in engine.trades))
