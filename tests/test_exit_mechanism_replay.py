from copy import deepcopy

import numpy as np
import pytest

from skills.exit_mechanism_replay import ExitMechanismReplay, audit_exits
from skills.exit_policy import MODES
from skills.residual_slot_replay import ResidualSlotReplay, audit_residual_slots
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY, SIZE
from test_historical_selector_replay import identities


def run(mode, mask=0, stock=None, lower_limits=(), stop=ENTRY+74):
    days, adjusted, args, kw = fixture(entries=[ENTRY],stock=stock,end=stop,lower_limits=lower_limits)
    kw.update(factor_mask=mask,exit_signals=ExitSignals(adjusted,days),identity_report=identities())
    engine = ExitMechanismReplay(*args,**kw,exit_mode=mode)
    account = engine.run()
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
        engine.board_decisions,engine.residual_days,args[0])
    audit_exits(account,engine.exit_decisions,engine.exit_states,adjusted,days,mode,mask)
    return days,adjusted,args,kw,engine,account


@pytest.mark.parametrize('mask',range(8))
def test_loss12_full_account_reproduces_sealed_rules(mask):
    _,_,args,kw,_,account=run('loss12',mask)
    assert account==ResidualSlotReplay(*args,**kw,residual_policy='release').run()


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('mask',[0,7])
def test_each_mode_rebuilds_decisions_and_delay(mode,mask):
    stock=np.full(SIZE,100.);stock[ENTRY+3:ENTRY+7]=130.;stock[ENTRY+7:]=87.
    run(mode,mask,stock)


def test_latched_exit_survives_rebound_and_lower_limit():
    days,_,_,_=fixture()
    stock=np.full(SIZE,100.);stock[ENTRY+2]=87.;stock[ENTRY+3:]=120.
    _,_,_,_,engine,account=run('loss12',stock=stock,lower_limits=[('1101',days[ENTRY+3])])
    sale=next(t for t in account['trades'] if t['side']=='sell')
    assert sale['date']==str(days[ENTRY+4].date()) and sale['reason']=='loss12'
    assert sale['signal_date']==str(days[ENTRY+2].date())


def test_tampered_signal_and_omitted_decisions_are_rejected():
    days,adjusted,_,_,engine,account=run('adaptive')
    for rows in [engine.exit_decisions[1:],deepcopy(engine.exit_decisions)]:
        if len(rows)==len(engine.exit_decisions):rows[0]['signal_close']+=1
        with pytest.raises(ValueError):
            audit_exits(account,rows,engine.exit_states,adjusted,days,'adaptive',0)


def test_future_prices_cannot_change_earlier_decisions():
    _,_,_,_,before,_=run('adaptive')
    prices=np.full(SIZE,100.);prices[ENTRY+10:]=150.
    days,_,_,_,after,_=run('adaptive',stock=prices)
    cutoff=str(days[ENTRY+10].date())
    assert [r for r in before.exit_decisions if r['date']<=cutoff]==[r for r in after.exit_decisions if r['date']<=cutoff]


@pytest.mark.parametrize('mask',[0,4])
def test_strong_stock_extension_still_exits_at_hard_deadline(monkeypatch,mask):
    import test_cash_allocation_replay as fixtures
    monkeypatch.setattr(fixtures,'SIZE',300)
    prices=np.linspace(80,300,300)
    days,_,_,_,_,account=run('trend126',mask,stock=prices,stop=ENTRY+130)
    sale=next(t for t in account['trades'] if t['side']=='sell')
    assert sale['date']==str(days[ENTRY+126+bool(mask&4)].date())
    assert sale['reason']=='hard_time126'
