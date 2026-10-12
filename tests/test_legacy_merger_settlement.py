from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
import pytest
import pandas as pd

from skills.legacy_merger_settlement import (
    ACTION, EXCHANGE, ORDINARY, MergerSettlement, entitlement)
from skills.legacy_merger_audit import audit_exchange, corporate_auditor, _exchange_opening, _opening_sid
from skills.top3_validation import cost_function


@pytest.mark.parametrize('quantity,price,expected',[(13569,37.8,(3522,.5124,19)),(10000,37.8,(2596,0.,0)),(1250,25.,(324,.5,13))])
def test_exchange_integer_units_and_half_up_fraction(quantity,price,expected):
    assert entitlement(quantity,price)==expected


@pytest.mark.parametrize('quantity,price',[(0,37.8),(-1,37.8),(1.5,37.8),(True,37.8),(5,0)])
def test_bad_entitlement_fails(quantity,price):
    with pytest.raises(ValueError):entitlement(quantity,price)


def account_fixture(mode='baseline',q=13569):
    cost=cost_function(mode)(9.7, q, 'buy','2867'); cash=1_000_000+cost['cash_change']
    event='test-original';dates=['2026-08-19','2026-09-01','2026-09-08'];prices=[9.7,39.9,45.55]
    ledger=[dict(date=dates[0],kind='initial_deposit',cash_change=1_000_000.,cash_after=1_000_000.),
        dict(date=dates[0],kind='buy',stock_id='2867',event_id=event,channel='board',cash_change=cost['cash_change'],cash_after=cash)]
    # Use an integer board lot for the cash-auditor variant below.
    account=dict(settings=dict(initial_cash=1_000_000,slippage={'baseline':.0045,'fee2':.0045,'slip2':.009,'combined':.009}[mode]),
        cash_ledger=ledger,trades=[dict(date=dates[0],event_id=event,stock_id='2867',side='buy',qty=q,channel='board',
            reference_price=9.7,remaining_shares=q,sequence=1,requested_qty=q,capacity_qty=1000000,prior_avg_amount20=60e6,cash_after=cash,**cost)],
        cohorts=[dict(event_id=event,stock_id='2867',current_stock_id='2884',share_exchange_date=EXCHANGE,
            entry_date=dates[0],exit_date=None)],corporate_actions=[],receivables=[],holdings=[],daily=[])
    whole,fraction,gross=entitlement(q,37.8)
    a=dict(action_id=ACTION,date=EXCHANGE,kind='share_exchange',stock_id='2867',target_stock_id='2884',event_id=event,
        entitled_qty=q,qty_after=whole,ratio=.2596,fractional_right=fraction,fractional_gross_amount=gross,
        fractional_reference_date='2026-08-19',fractional_reference_price=37.8,fractional_rounding='half_up_ntd',
        fractional_cash_available=False,certificate_trading_modeled=False,certificate_delivery_date=EXCHANGE,
        ordinary_share_available_date=ORDINARY,valuation_basis='ordinary_close_proxy')
    account['corporate_actions']=[a]
    account['receivables']=[dict(action_id=ACTION+'-fractional-cash',amount=gross,pay_date=None,stock_id='2884')]
    previous=peak=1_000_000
    for day,price in zip(dates,prices):
        qty=q if day<EXCHANGE else whole;sid='2867' if day<EXCHANGE else '2884'
        asset=qty*price;right=gross if day>=EXCHANGE else 0;expense=cost['total_cost'] if day==dates[0] else 0
        nav=cash+asset+right;peak=max(peak,nav)
        row=dict(date=day,stock_id=sid,qty=qty,price=price,market_value=asset,mark_date=day,event_id=event)
        if day>=EXCHANGE:row['instrument_kind']='nontraded_merger_certificate' if day<ORDINARY else 'ordinary_share'
        account['holdings'].append(row)
        account['daily'].append(dict(date=day,opening_nav=previous,cash=cash,market_value=asset,receivable=right,nav=nav,
            market_pnl=nav-previous+expense,dividend_entitlement=0.,execution_basis_pnl=0.,cost=expense,
            daily_return=nav/previous-1,total_return=nav/1_000_000-1,drawdown=nav/peak-1))
        previous=nav
    quotes=pd.DataFrame([dict(stock_id='2884',date='2026-08-19',close=37.8)])
    return account,quotes


def test_independent_exchange_inventory_and_fractions():
    a,q=account_fixture();proof=audit_exchange(a,q)
    assert proof['events'][0]['new_units']==3522
    assert proof['events'][0]['fractional_gross']==19
    assert proof['events'][0]['daily'][0]['instrument_kind']=='nontraded_merger_certificate'


@pytest.mark.parametrize('mode',['baseline','fee2','slip2','combined'])
def test_private_cash_auditor_reconstructs_converted_units_all_costs(mode):
    a,q=account_fixture(mode,q=13000)
    assert audit_exchange(a,q)['events'][0]['new_units']==3374
    assert corporate_auditor(mode)(a)['daily_shares_reconstructed'] is True
    a['holdings'][1]['qty']+=1
    with pytest.raises(ValueError):corporate_auditor(mode)(a)


@pytest.mark.parametrize('fault',['ratio','qty_after','fractional_gross_amount','fractional_reference_price','fractional_rounding'])
def test_exchange_audit_rejects_bad_entitlement(fault):
    a,q=account_fixture();a['corporate_actions'][0][fault]='wrong'
    with pytest.raises(ValueError):audit_exchange(a,q)


@pytest.mark.parametrize('fault',['paid','wrong_target','early_ordinary','extra_share','early_sale'])
def test_exchange_audit_rejects_ownership_and_cash_errors(fault):
    a,q=account_fixture()
    if fault=='paid':a['cash_ledger'].append(dict(action_id=ACTION+'-fractional-cash'))
    elif fault=='wrong_target':a['holdings'][1]['stock_id']='2867'
    elif fault=='early_ordinary':a['holdings'][1]['instrument_kind']='ordinary_share'
    elif fault=='extra_share':a['holdings'][2]['qty']+=1
    elif fault=='early_sale':a['trades'].append(dict(event_id='test-original',date=EXCHANGE,stock_id='2884',side='sell',qty=1))
    with pytest.raises(ValueError):audit_exchange(a,q)


def test_one_slot_follows_the_exchanged_investment():
    a,_=account_fixture();cohort=a['cohorts'][0]
    assert _opening_sid(cohort,EXCHANGE)=='2867'
    assert _opening_sid(cohort,ORDINARY)=='2884'
    opening={'2867':dict(event_id='test-original',qty=13569),'1234':dict(event_id='other',qty=1000)}
    active=set(opening);_exchange_opening(a,EXCHANGE,opening,active,{})
    assert active=={'2884','1234'} and opening['2884']['qty']==3522


def test_same_day_opening_lock_maps_once_without_releasing_slot():
    obj=object.__new__(MergerSettlement);obj.merger_event_id='original'
    obj.opening_members={'2867','1234'}
    assert obj.opening_members=={'2884','1234'}


def test_unit_auditor_rejects_unexplained_exchange_units():
    from skills.legacy_merger_audit import _unit_exchange
    from collections import defaultdict
    units=defaultdict(int,{'2867':13569});action=dict(stock_id='2867',target_stock_id='2884',date=EXCHANGE,qty_after=3522)
    _unit_exchange(action,units);assert units['2867']==0 and units['2884']==3522
    with pytest.raises(ValueError):_unit_exchange(action,units)
