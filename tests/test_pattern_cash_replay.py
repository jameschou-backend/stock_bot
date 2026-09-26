from copy import deepcopy

import numpy as np
import pytest

from skills.pattern_cash_replay import PatternCashReplay, audit_pattern_entries
from skills.support_risk_replay import SupportRiskReplay
from skills.support_risk_audit import audit_support_risk
from skills.residual_slot_replay import audit_residual_slots
from skills.technical_signals import TechnicalSignals
from test_cash_allocation_replay import fixture, ENTRY, SIZE
from test_historical_selector_replay import identities


def run(mode='control',mask=0,enabled=True,passing=True,missing=False,future=False):
    def modify(quotes,days):
        recent=(quotes.date>=days[ENTRY-11])&(quotes.date<days[ENTRY-1])&(quotes.stock_id=='1101')
        quotes.loc[recent,['high','low']]=[50.5,49.5]
        signal=(quotes.date==days[ENTRY-1])&(quotes.stock_id=='1101')
        quotes.loc[signal,'volume']=3_000_000 if passing else 2_000_000
        if missing:quotes.loc[(quotes.date==days[ENTRY-5])&(quotes.stock_id=='1101'),'low']=np.nan
        if future:quotes.loc[(quotes.date>days[ENTRY+9])&(quotes.stock_id=='1101'),['open','high','low','close']]=[500.,510.,490.,500.]
    stock=np.full(SIZE,100.);stock[ENTRY-1]=105.
    days,adjusted,args,kw=fixture(entries=[ENTRY],stock=stock,end=ENTRY+8,mutate=modify)
    technical=TechnicalSignals(adjusted,args[0],days)
    kw.update(factor_mask=mask,exit_signals=technical,technical_signals=technical,
              identity_report=identities(),technical_mode=mode)
    engine=PatternCashReplay(*args,**kw,pattern_filter=enabled)
    account=engine.run()
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
                        engine.board_decisions,engine.residual_days,args[0])
    audit_support_risk(account,engine.technical_entries,engine.support_decisions,
        engine.exit_decisions,engine.exit_states,engine.slot_decisions,technical,mode,mask,engine.prior)
    if enabled:audit_pattern_entries(account,engine.pattern_entries,engine.slot_decisions,technical,args[3],mask)
    return args,kw,engine,account


@pytest.mark.parametrize('mode',['control','support_risk2'])
@pytest.mark.parametrize('mask',range(8))
def test_disabled_pattern_preserves_complete_parent_account(mode,mask):
    args,kw,engine,account=run(mode,mask,enabled=False)
    assert account==SupportRiskReplay(*args,**kw).run() and not engine.pattern_entries


@pytest.mark.parametrize('mode',['control','support_risk2'])
@pytest.mark.parametrize('mask',range(8))
def test_original_pattern_passes_even_after_one_extra_entry_day(mode,mask):
    _,_,engine,account=run(mode,mask)
    assert engine.pattern_entries[0]['reason']=='pattern_pass'
    assert any(t['side']=='buy' for t in account['trades'])


@pytest.mark.parametrize('mode',['control','support_risk2'])
@pytest.mark.parametrize('missing',[False,True])
def test_failed_or_missing_pattern_has_no_trades_or_reserved_attempt(mode,missing):
    _,_,engine,account=run(mode,passing=False,missing=missing)
    assert engine.pattern_entries[0]['reason']==('pattern_unavailable' if missing else 'pattern_rejected')
    assert not account['trades'] and not engine.slot_decisions[0]['attempted']


def test_future_bars_do_not_change_prior_pattern_or_account():
    _,_,a,x=run('support_risk2')
    _,_,b,y=run('support_risk2',future=True)
    assert x==y and a.pattern_entries==b.pattern_entries


def test_reanchoring_signal_to_delayed_entry_is_detected():
    args,_,engine,account=run(mask=2)
    rows=deepcopy(engine.pattern_entries);rows[0]['signal_date']=args[3][0]['entry_date']
    with pytest.raises(ValueError):
        audit_pattern_entries(account,rows,engine.slot_decisions,engine.technical_signals,args[3],2)
