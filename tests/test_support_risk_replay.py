from copy import deepcopy

import numpy as np
import pytest

from skills.residual_slot_replay import ResidualSlotReplay, audit_residual_slots
from skills.support_risk_replay import SupportRiskReplay, risk_quantity
from skills.support_risk_audit import audit_support_risk, independent_loss
from skills.technical_signals import TechnicalSignals
from test_cash_allocation_replay import fixture, ENTRY, SIZE, Corporate
from test_historical_selector_replay import identities


def run(mode,mask=0,stock=None,stop=ENTRY+74,mutate=None,lower_limits=(),corporate=None):
    days,adjusted,args,kw=fixture(entries=[ENTRY],stock=stock,end=stop,mutate=mutate,
                               lower_limits=lower_limits,corporate=corporate)
    signals=TechnicalSignals(adjusted,args[0],days)
    kw.update(factor_mask=mask,exit_signals=signals,identity_report=identities())
    engine=SupportRiskReplay(*args,**kw,technical_mode=mode,technical_signals=signals)
    account=engine.run()
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
                        engine.board_decisions,engine.residual_days,args[0])
    audit_support_risk(account,engine.technical_entries,engine.support_decisions,
        engine.exit_decisions,engine.exit_states,engine.slot_decisions,signals,mode,mask,engine.prior)
    return days,args,kw,engine,account


@pytest.mark.parametrize('mask',range(8))
def test_control_reproduces_full_sealed_account(mask):
    _,args,kw,_,account=run('control',mask)
    assert account==ResidualSlotReplay(*args,**kw,residual_policy='release').run()


@pytest.mark.parametrize('mode',['support20','risk2','support_risk2'])
@pytest.mark.parametrize('mask',range(8))
def test_each_arm_reconstructs_every_signal_and_quantity(mode,mask):
    stock=np.full(SIZE,100.);stock[ENTRY+3:ENTRY+7]=130.;stock[ENTRY+7:]=95.
    run(mode,mask,stock)


@pytest.mark.parametrize('mask',[0,4,7])
def test_latched_support_survives_lower_limit_and_rebound(mask):
    days,_,_,_=fixture()
    stock=np.full(SIZE,100.);stock[ENTRY+3]=97.;stock[ENTRY+4:]=120.
    first=ENTRY+4+(1 if mask&4 else 0)
    _,_,_,engine,account=run('support20',mask,stock,lower_limits=[('1101',days[first])])
    sale=next(t for t in account['trades'] if t['side']=='sell')
    assert sale['reason']=='support20' and sale['date']==str(days[first+1].date())
    assert sale['signal_date']==str(days[ENTRY+3].date())
    assert engine.exit_states['entry-130']['target_index']==first


def test_support_equality_and_original_signal_anchor_with_late_entry():
    stock=np.full(SIZE,100.);stock[ENTRY:]=98.
    _,_,_,engine,account=run('support_risk2',2,stock,stop=ENTRY+8)
    row=engine.technical_entries[0]
    assert row['context']['adjusted_close']==100.
    assert row['planned_stop_ratio']==.98
    assert not any(t['side']=='sell' for t in account['trades'])
    assert all(r['floor_after']==98. for r in engine.support_decisions)


def test_missing_initial_support_blocks_without_consuming_attempt():
    def missing(quotes,days):
        quotes.loc[(quotes.date==days[ENTRY-5])&(quotes.stock_id=='1101'),'low']=np.nan
    _,_,_,engine,account=run('support20',mutate=missing)
    assert not account['trades']
    assert engine.technical_entries[0]['allowed_qty']==0
    assert engine.slot_decisions[0]['attempted'] is False


def test_missing_later_support_keeps_floor_and_loss12_has_priority():
    def missing(quotes,days):
        quotes.loc[(quotes.date==days[ENTRY+1])&(quotes.stock_id=='1101'),'low']=np.nan
    stock=np.full(SIZE,100.);stock[ENTRY+3:]=87.
    _,_,_,engine,account=run('support20',stock=stock,mutate=missing)
    assert any(not r['support_available'] and r['floor_after']==98. for r in engine.support_decisions)
    assert next(t for t in account['trades'] if t['side']=='sell')['reason']=='loss12'


def test_future_ohlc_cannot_change_earlier_decisions():
    end=ENTRY+6
    def future(quotes,days):
        quotes.loc[(quotes.date>days[end])&(quotes.stock_id=='1101'),['open','high','low','close']]=[500,510,490,500]
    plain=run('support_risk2',stop=end)[3:]
    changed=run('support_risk2',stop=end,mutate=future)[3:]
    assert plain[1]==changed[1]
    assert plain[0].technical_entries==changed[0].technical_entries
    assert plain[0].support_decisions==changed[0].support_decisions


def test_double_slippage_can_lower_risk_quantity():
    a=run('risk2',0)[3];b=run('risk2',1)[3]
    # Around this boundary the extra round-trip execution cost removes a lot.
    cap=independent_loss(50.,44.,3000,.0045,.001425)
    assert risk_quantity(a._costs,50.,44.,4000,cap,'1101')==3000
    assert risk_quantity(b._costs,50.,44.,4000,cap,'1101')==2000


@pytest.mark.parametrize('field',['signal_date','planned_loss','filled_qty'])
def test_tampered_entry_evidence_rejected(field):
    days,_,_,engine,account=run('support_risk2')
    rows=deepcopy(engine.technical_entries)
    rows[0][field]=str(days[ENTRY+1].date()) if field=='signal_date' else rows[0][field]+1
    with pytest.raises(ValueError):
        audit_support_risk(account,rows,engine.support_decisions,engine.exit_decisions,
            engine.exit_states,engine.slot_decisions,engine.technical_signals,'support_risk2',0,engine.prior)


def test_omitted_or_tampered_support_trace_rejected():
    _,_,_,engine,account=run('support20')
    for rows in [engine.support_decisions[1:],deepcopy(engine.support_decisions)]:
        if len(rows)==len(engine.support_decisions):rows[0]['floor_after']-=1
        with pytest.raises(ValueError):
            audit_support_risk(account,engine.technical_entries,rows,engine.exit_decisions,
                engine.exit_states,engine.slot_decisions,engine.technical_signals,'support20',0,engine.prior)


@pytest.mark.parametrize('mask',[0,4])
def test_late_corporate_delivery_inherits_same_support_instruction(mask):
    days,_,_,_=fixture()
    ex,pay=(str(days[ENTRY+i].date()) for i in (2,8))
    corporate=Corporate({('1101',ex):[dict(stock_id='1101',action_id='stock',kind='stock_dividend',
        shares_per_share=1.,pay_date=pay,fractional_cash_per_share=0.)]})
    stock=np.full(SIZE,100.);stock[ENTRY+3]=97.;stock[ENTRY+4:]=120.
    _,_,_,_,account=run('support20',mask,stock,corporate=corporate)
    sales=[t for t in account['trades'] if t['side']=='sell']
    assert [t['date'] for t in sales]==[str(days[ENTRY+4+(1 if mask&4 else 0)].date()),pay]
    assert all(t['reason']=='support20' and t['signal_date']==str(days[ENTRY+3].date()) for t in sales)


def test_ratcheted_support_never_descends_after_new_high():
    stock=np.full(SIZE,100.);stock[ENTRY:ENTRY+25]=120.;stock[ENTRY+25:]=117.6
    _,_,_,engine,account=run('support20',stock=stock)
    floors=[r['floor_after'] for r in engine.support_decisions]
    assert floors==sorted(floors) and max(floors)==pytest.approx(117.6)
    assert next(t for t in account['trades'] if t['side']=='sell')['reason']=='time63'


def test_risk_cap_zero_does_not_reserve_a_slot():
    def expensive(quotes,days):
        quotes.loc[quotes.stock_id=='1101',['open','high','low','close']]=[180.,181.,179.,180.]
    _,_,_,engine,account=run('risk2',mutate=expensive)
    assert engine.technical_entries[0]['allowed_qty']==0
    assert engine.slot_decisions[0]['attempted'] is False and not account['trades']


def test_execution_gap_still_obeys_cash_and_original_quantity_cap():
    def gap(quotes,days):
        quotes.loc[(quotes.stock_id=='1101')&(quotes.date==days[ENTRY]),['open','high','low','close']]=[100.,101.,99.,100.]
    _,_,_,engine,account=run('risk2',mutate=gap)
    buy=next(t for t in account['trades'] if t['side']=='buy')
    assert buy['qty']<=engine.technical_entries[0]['allowed_qty']
    assert -buy['cash_change']<=200_000 and all(d['cash']>=0 for d in account['daily'])
