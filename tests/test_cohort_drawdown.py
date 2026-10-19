from copy import deepcopy
import json
from pathlib import Path

import pytest

from skills.cohort_drawdown import analyze, episodes, attributed_cash, boundary
from app.poc_drawdown_ui import validate


@pytest.fixture
def ledger():
    dates = [f'2024-01-0{i}' for i in range(1, 7)]
    a = dict(settings=dict(initial_cash=1000), corporate_actions=[], receivables=[], trades=[],
        cash_ledger=[dict(date=dates[0],kind='initial_deposit',cash_change=1000)], holdings=[], daily=[],
        cohorts=[dict(event_id='A',stock_id='1111',name='A',entry_date=dates[1]),
                 dict(event_id='B',stock_id='1111',name='B',entry_date=dates[3])])
    # Different entry cohorts intentionally share a stock code.
    for index,event,side,gross,cost in [(1,'A','buy',200,2),(3,'A','sell',160,3),(3,'B','buy',200,2)]:
        cash=(gross if side=='sell' else -gross)-cost
        common=dict(date=dates[index],event_id=event,stock_id='1111',channel='odd',cash_change=cash)
        a['trades'].append(dict(common,side=side,gross=gross,total_cost=cost,commission=cost,tax=0,slippage=0,qty=1,reason='three_black'))
        a['cash_ledger'].append(dict(common,kind=side))
    amounts=[{}, {'A':200}, {'A':400}, {'A':160,'B':150}, {'A':150,'B':100}, {'A':400,'B':300}]
    cash=1000;previous=1000
    for i,day in enumerate(dates):
        cash+=sum(r['cash_change'] for r in a['cash_ledger'] if r['date']==day and r['kind']!='initial_deposit')
        for event,value in amounts[i].items():
            a['holdings'].append(dict(date=day,event_id=event,stock_id='1111',market_value=value))
        mv=sum(amounts[i].values());cost=sum(t['total_cost'] for t in a['trades'] if t['date']==day);nav=cash+mv
        a['daily'].append(dict(date=day,nav=nav,cash=cash,market_value=mv,receivable=0,cost=cost,
            opening_nav=previous,market_pnl=nav-previous+cost,dividend_entitlement=0,execution_basis_pnl=0))
        previous=nav
    return a


def test_partial_sales_new_cohort_same_stock_and_costs_are_not_double_counted(ledger):
    e=analyze(ledger,ledger['daily'],top=1)['episodes'][0]
    assert e['components']==dict(winner_decline=-90,other_existing_decline=0,new_entry_decline=-100,gain_offset=0,cost=-5)
    assert e['net_change']==-195
    assert e['recovery_date']=='2024-01-06'
    rows={c['event_id']:c for c in e['cohorts']}
    assert rows['A']['net_pnl']==-93
    assert rows['B']['net_pnl']==-102
    assert rows['B']['group']=='new_entry'


def test_later_profit_does_not_reclassify_new_loser_or_change_trough(ledger):
    before=analyze(ledger,ledger['daily'],top=1)['episodes'][0]
    ledger['holdings'][-1]['market_value']+=10000
    for key in ('market_value','nav','market_pnl'):ledger['daily'][-1][key]+=10000
    assert analyze(ledger,ledger['daily'],top=1)['episodes'][0]==before


def test_dividend_receivable_and_payment_are_not_duplicate_profit(ledger):
    event=dict(event_id='A',stock_id='1111',action_id='cash-A')
    ledger['corporate_actions']=[dict(event,date='2024-01-03',kind='cash_dividend',entitlement_value=10),
        dict(event,date='2024-01-04',kind='payment',amount=10)]
    ledger['cash_ledger'].append(dict(date='2024-01-04',kind='dividend_payment',stock_id='1111',action_id='cash-A',cash_change=10))
    # Dividend already accrued at peak; payment merely moves receivable to cash.
    ledger['daily'][2]['receivable']=10;ledger['daily'][2]['nav']+=10
    ledger['daily'][3]['cash']+=10;ledger['daily'][3]['nav']+=10
    cash=attributed_cash(ledger)
    start=boundary(ledger,cash,'2024-01-03')['A'];end=boundary(ledger,cash,'2024-01-04')['A']
    assert end['net_pnl']-start['net_pnl']==-83
    assert end['receivable']-start['receivable']==-10


def test_pending_share_rights_cannot_be_silently_valued_zero(ledger):
    ledger['corporate_actions']=[dict(date='2024-01-03',event_id='A',stock_id='1111',action_id='stock-A',kind='stock_dividend')]
    with pytest.raises(ValueError,match='unpriced share'):boundary(ledger,attributed_cash(ledger),'2024-01-03')


def test_cost_or_boundary_corruption_fails(ledger):
    ledger['trades'][0]['commission']+=1
    with pytest.raises(ValueError,match='Trade cost'):analyze(ledger,ledger['daily'])
    ledger['trades'][0]['commission']-=1
    ledger['holdings'][1]['market_value']+=1
    with pytest.raises(ValueError,match='Boundary market_value'):analyze(ledger,ledger['daily'])


def test_high_watermark_episodes_do_not_overlap_and_include_unrecovered_tail():
    d=[dict(date=str(i),nav=v) for i,v in enumerate([100,120,100,110,90,120,120,100,130,110])]
    assert episodes(d)==[(1,4,5),(6,7,8),(8,9,None)]


def test_published_five_episodes_balance_and_preserve_qualification():
    root=Path(__file__).resolve().parents[1]
    report=json.loads((root/'artifacts/forward_simulation/poc_drawdown_attribution_20261019.json').read_text())
    assert validate(report) is report
    e=report['episodes'][0]
    assert e['net_change']==pytest.approx(-2255039.36)
    assert e['components']['winner_decline']==pytest.approx(-1247643.10)
    assert report['episodes'][4]['child_trade_count']==0
    assert report['episodes'][4]['components']['cost']==0
    for mutate in [lambda r:r.update(live_qualified=True),lambda r:r.update(backtests_run=1),
                   lambda r:r['episodes'][0]['components'].update(cost=0),
                   lambda r:r['episodes'][0]['cohorts'][0].update(net_pnl=1)]:
        bad=deepcopy(report);mutate(bad)
        with pytest.raises(ValueError):validate(bad)
