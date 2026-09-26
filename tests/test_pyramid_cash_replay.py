from copy import deepcopy

import numpy as np
import pytest

from skills.pyramid_cash_replay import PyramidCashReplay
from skills.pyramid_cash_audit import audit_pyramid, independent_capacity
from skills.pyramid_slot_audit import audit_pyramid_slots
from skills.support_risk_replay import SupportRiskReplay
from skills.technical_signals import TechnicalSignals
from test_cash_allocation_replay import fixture, ENTRY, SIZE, Corporate
from test_historical_selector_replay import identities


def run(mask=0,enabled=True,mutate=None,stock=None,corporate=None,stop=ENTRY+40):
    stock = stock if stock is not None else np.r_[np.full(ENTRY+4,100.),np.full(SIZE-ENTRY-4,112.)]
    def change(quotes,days):
        for i,day in enumerate(days):
            match = (quotes.stock_id=='1101')&(quotes.date==day)
            raw = stock[i]/2
            quotes.loc[match,['open','high','low','close']] = [raw,raw+1,raw-1,raw]
        # Original order only gets one board lot; addition has normal depth.
        quotes.loc[(quotes.stock_id=='1101')&(quotes.date==days[ENTRY+bool(mask&2)]),'volume'] = 100_000
        if mutate: mutate(quotes,days)
    days,adjusted,args,kw=fixture(entries=[ENTRY],stock=stock,end=stop,mutate=change,corporate=corporate)
    signals=TechnicalSignals(adjusted,args[0],days)
    kw.update(factor_mask=mask,exit_signals=signals,identity_report=identities(),
              technical_mode='support_risk2',technical_signals=signals)
    engine=PyramidCashReplay(*args,**kw,pyramid_enabled=enabled)
    account=engine.run()
    case=dict(completed=True,config=dict(factor_mask=mask),account=account,
        **{k:getattr(engine,k) for k in ('technical_entries','support_decisions','exit_decisions',
            'exit_states','slot_decisions','pyramid_decisions','resource_plans')})
    audit_pyramid_slots(account,engine.resource_plans,engine.slot_decisions,
                         engine.board_decisions,engine.residual_days,args[0])
    if enabled: audit_pyramid(case,signals,args[0],engine.prior,args[3],engine.pyramid_pending)
    return engine,account,case,days,args,kw


@pytest.mark.parametrize('mask',range(8))
def test_disabled_switch_reproduces_entire_original_account(mask):
    _,account,_,_,args,kw=run(mask,False)
    assert account==SupportRiskReplay(*args,**kw).run()


@pytest.mark.parametrize('mask',range(8))
def test_strong_add_uses_prior_signal_once_with_scenario_delay(mask):
    engine,account,_,days,_,_=run(mask)
    additions=[t for t in account['trades'] if t['reason']=='pyramid_add']
    assert len(additions)==1
    assert additions[0]['date']==str(days[ENTRY+5+bool(mask&2)].date())
    assert additions[0]['signal_date']==str(days[ENTRY+4].date())
    assert engine.exit_states['entry-130']['entry_index']==ENTRY+bool(mask&2)


def test_pending_add_cancelled_when_original_exit_latches():
    stock=np.full(SIZE,100.);stock[ENTRY+4]=112.;stock[ENTRY+5:]=80.
    _,account,case,_,_,_=run(2,stock=stock)
    assert not any(t['reason']=='pyramid_add' for t in account['trades'])
    assert any(r['pending_before'] and r['status']=='cancelled_exit_or_no_physical_shares'
               for r in case['pyramid_decisions'])


def test_missing_current_quote_cannot_create_fill_and_can_retry_later_signal():
    stock=np.full(SIZE,100.);stock[ENTRY+4]=112.;stock[ENTRY+5]=113.;stock[ENTRY+6:]=116.
    def missing(quotes,days):
        quotes.loc[(quotes.stock_id=='1101')&(quotes.date==days[ENTRY+5]),'volume']=0
    _,account,case,_,_,_=run(mutate=missing,stock=stock)
    assert any(r['status']=='unfilled' for r in case['pyramid_decisions'])
    assert sum(t['reason']=='pyramid_add' for t in account['trades'])==1


def test_future_ohlc_does_not_change_earlier_plan_or_account():
    end=ENTRY+8
    def future(quotes,days):
        quotes.loc[(quotes.stock_id=='1101')&(quotes.date>days[end]),['open','high','low','close']]=[1000,1200,900,1000]
    a=run(stop=end);b=run(stop=end,mutate=future)
    assert a[1]==b[1] and a[0].pyramid_decisions==b[0].pyramid_decisions


@pytest.mark.parametrize('field',['pyramid_decisions','resource_plans','trades'])
def test_omitted_addition_evidence_is_rejected(field):
    engine,account,case,_,args,kw=run()
    changed=deepcopy(case)
    if field=='trades': changed['account']['trades']=[t for t in account['trades'] if t['reason']!='pyramid_add']
    elif field=='resource_plans':changed[field]=[r for r in changed[field] if r.get('kind')!='pyramid_add']
    else:changed[field]=changed[field][1:]
    with pytest.raises(ValueError):
        audit_pyramid(changed,kw['technical_signals'],args[0],engine.prior,args[3],engine.pyramid_pending)


@pytest.mark.parametrize('field',['signal_date','original_qty','target_index','stop_ratio'])
def test_tampered_instruction_rejected(field):
    engine,_,case,_,args,kw=run(2)
    changed=deepcopy(case)
    row=next(r for r in changed['pyramid_decisions'] if r.get('created_instruction'))
    row['created_instruction'][field]='2099-01-01' if field=='signal_date' else 0
    with pytest.raises(ValueError):
        audit_pyramid(changed,kw['technical_signals'],args[0],engine.prior,args[3],engine.pyramid_pending)


def test_quantity_budget_includes_round_trip_scenario_costs():
    from skills.pyramid_cash_replay import addition_capacity
    for mask in (0,1):
        engine=run(mask)[0]
        for cash in (0., 50_000., 100_000.):
            actual=addition_capacity(engine._costs,40.,.92,1000,1_000_000.,cash,'1101')
            assert actual==independent_capacity(40.,.92,1000,1_000_000.,cash,.009 if mask else .0045)


def test_unsettled_stock_rights_block_addition():
    days,_,_,_=fixture()
    ex,pay=(str(days[ENTRY+i].date()) for i in (3,20))
    corp=Corporate({('1101',ex):[dict(stock_id='1101',action_id='stock',kind='stock_dividend',
        shares_per_share=1.,pay_date=pay,fractional_cash_per_share=0.)]})
    _,account,case,_,_,_=run(corporate=corp)
    assert not any(t['reason']=='pyramid_add' for t in account['trades'])
    assert any(r['status']=='cancelled_unsettled_or_nonboard_shares' for r in case['pyramid_decisions'])


def test_partial_addition_consumes_the_only_addition_and_no_remainder_is_retried():
    stock=np.full(SIZE,100.);stock[ENTRY+4]=112.;stock[ENTRY+5:]=115.
    def partial(quotes,days):
        quotes.loc[quotes.stock_id=='1101',['open','high','low','close']] /= 2
        quotes.loc[quotes.stock_id=='1101','volume'] *= 2
        quotes.loc[(quotes.stock_id=='1101')&quotes.date.isin([days[ENTRY],days[ENTRY+5]]),'volume']=100_000
    engine,account,case,_,_,_=run(mutate=partial,stock=stock)
    adds=[t for t in account['trades'] if t['reason']=='pyramid_add']
    assert len(adds)==1 and adds[0]['qty']==1000
    decision=next(r for r in case['pyramid_decisions'] if r['status']=='filled')
    assert decision['execution_capacity']['qty']>1000
    assert engine.pyramid_done=={'entry-130'}
    assert not engine.pyramid_pending


def test_addition_does_not_reset_the_original_time_exit():
    engine,account,_,days,_,_=run(stop=ENTRY+74)
    sales=[t for t in account['trades'] if t['side']=='sell']
    assert sales and sales[0]['date']==str(days[ENTRY+63].date())
    assert sales[0]['reason']=='time63'
    assert sales[0]['qty']==sum(t['qty'] for t in account['trades'] if t['side']=='buy')
    assert engine.exit_states['entry-130']['entry_index']==ENTRY
