from copy import deepcopy

import pandas as pd
import pytest

from skills.execution_factorial import FactorialReplay
from skills.residual_slot_replay import ResidualSlotReplay, residual_values, audit_residual_slots
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, Corporate, Feeds, ENTRY
from test_historical_selector_replay import identities


@pytest.mark.parametrize('mask', [0, 7])
def test_keep_policy_preserves_full_original_account(mask):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+74)
    kwargs.update(factor_mask=mask, exit_signals=ExitSignals(adjusted, days), identity_report=identities())
    assert ResidualSlotReplay(*args, **kwargs, residual_policy='keep').run() == FactorialReplay(*args, **kwargs).run()


def six_stocks(dividend=.01, next_offset=64, delivery_offset=4):
    days, adjusted, args, kwargs = fixture(end=ENTRY+74)
    quotes, _, _, _, _, _ = args
    ids = [str(1101+i) for i in range(6)]
    parts = [quotes[quotes.stock_id.eq('0050')]]
    for sid in ids:
        part = quotes[quotes.stock_id.eq('1101')].copy(); part['stock_id'] = sid; parts.append(part)
        adjusted[sid] = adjusted['1101']
    quotes = pd.concat(parts, ignore_index=True)
    events = [dict(event_id='entry-'+sid, members=[sid], priority=.1,
        signal_date=str(days[index-1].date()), entry_date=str(days[index].date()))
        for sid,index in zip(ids, [ENTRY]*5+[ENTRY+next_offset])]
    corporate = Corporate({(sid, str(days[ENTRY+2].date())): [dict(stock_id=sid,
        date=str(days[ENTRY+2].date()), action_id='stock-'+sid, kind='stock_dividend',
        shares_per_share=dividend, pay_date=str(days[ENTRY+delivery_offset].date()), fractional_cash_per_share=0.)]
        for sid in ids[:5]})
    args = (quotes, pd.DataFrame([dict(stock_id=sid,name=sid,market='TWSE') for sid in ids]),
            days, events, Feeds(quotes), corporate)
    kwargs.update(factor_mask=0, exit_signals=ExitSignals(adjusted,days), identity_report=identities(tuple(ids)))
    return days, args, kwargs


def test_release_retains_dividend_remainders_and_allows_next_day_new_position():
    days, args, kwargs = six_stocks()
    old = ResidualSlotReplay(*args, **kwargs, residual_policy='keep').run()
    engine = ResidualSlotReplay(*args, **kwargs, residual_policy='release')
    account = engine.run()
    assert not any(t['stock_id']=='1106' for t in old['trades'])
    assert any(t['stock_id']=='1106' and t['side']=='buy' for t in account['trades'])
    day = str(days[ENTRY+64].date())
    rows = [r for r in account['holdings'] if r['date']==day]
    assert len(rows)==6 and sum(r['qty']<1000 for r in rows)==5
    assert all(t['qty'] % 1000 == 0 for t in account['trades'])
    assert audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,
        engine.board_decisions,engine.residual_days,args[0])['active_slots_rebuilt']
    corrupted = deepcopy(engine.residual_days)
    next(r for r in corrupted if r['date']==day)['released'] = {}
    with pytest.raises(ValueError,match='classification'):
        audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,corrupted,args[0])


@pytest.mark.parametrize('dividend,offset,reason', [(.01,63,'slots_full'),(.3,64,'residual_exposure_cap')])
def test_no_same_day_release_and_no_new_buy_above_residual_risk_cap(dividend,offset,reason):
    days,args,kwargs=six_stocks(dividend,next_offset=offset)
    engine=ResidualSlotReplay(*args,**kwargs,residual_policy='release')
    account=engine.run()
    assert not any(t['stock_id']=='1106' for t in account['trades'])
    assert any(r['stock_id']=='1106' and r['failure']==reason for r in account['orders'])
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,args[0])


def test_late_stock_delivery_is_valued_after_physical_shares_have_sold():
    days,args,kwargs=six_stocks(delivery_offset=70)
    args[0].loc[args[0].stock_id.ne('0050') & args[0].date.eq(days[ENTRY+66]), 'close'] = 30.
    engine=ResidualSlotReplay(*args,**kwargs,residual_policy='release')
    account=engine.run()
    snapshot=next(r for r in engine.residual_days if r['date']==str(days[ENTRY+67].date()))
    assert snapshot['released']['1101']['qty']==0
    assert snapshot['released']['1101']['prior_price']==30.
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,args[0])
    corrupted=args[0].copy()
    corrupted.loc[corrupted.stock_id.eq('1101') & corrupted.date.eq(days[ENTRY+66]),'close']=31.
    with pytest.raises(ValueError,match='classification'):
        audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,engine.residual_days,corrupted)


def test_pending_shares_and_unexited_positions_keep_their_slots():
    holdings={'1101':dict(qty=400,event_id='old'), '1102':dict(qty=100,event_id='new')}
    marks={s:dict(price=20.,date='2026-01-02') for s in holdings}
    rights=[dict(kind='shares',stock_id='1101',qty=600)]
    assert residual_values(holdings,rights,marks,{'old'}) == {}
    rights[0]['qty']=599
    assert residual_values(holdings,rights,marks,{'old'})['1101']['value']==999*20.
