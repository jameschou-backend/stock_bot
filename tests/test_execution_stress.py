from copy import deepcopy
import numpy as np
import pytest
from skills.execution_stress import StressReplay, StressBenchmark, audit_stress
from skills.technical_replay import TechnicalReplay
from skills.million_replay import Replay
from test_technical_replay import fixture, Signals, ENTRY, SIZE


def run(mode='control',**kwargs):
    days,adjusted,args,options=fixture(entries=[ENTRY],**kwargs)
    engine=StressReplay(*args,technical_signals=Signals(adjusted,days),stress_mode=mode,**options)
    account=engine.run()
    audit_stress(account)
    return engine,account,days


def test_neutral_engine_and_benchmark_exactly_match_sealed_primitives():
    days,adjusted,args,options=fixture(entries=[ENTRY])
    expected=TechnicalReplay(*args,technical_signals=Signals(adjusted,days),mode='control',**options).run()
    assert run()[1]==expected
    days,adjusted,args,options=fixture(entries=[ENTRY])
    expected=Replay(*args,benchmark=True,**options).run()
    days,adjusted,args,options=fixture(entries=[ENTRY])
    assert StressBenchmark(*args,**options).run()==expected


def test_high_slippage_independently_audited_and_no_global_fee_change():
    _,a,_=run('slip90')
    assert a['settings']['slippage']==.009
    bad=deepcopy(a);bad['trades'][0]['slippage']-=1
    with pytest.raises(ValueError,match='fee/cash'):
        audit_stress(bad)
    _,neutral,_=run()
    assert neutral['settings']['slippage']==.0045


@pytest.mark.parametrize('quantity',[7,None])
def test_depth_is_cumulative_and_missing_depth_cannot_fill(quantity):
    days,adjusted,args,options=fixture(entries=[ENTRY])
    original=args[4].get_odd
    args[4].get_odd=lambda *a:dict(original(*a),ask_qty=quantity,bid_qty=quantity)
    e=StressReplay(*args,technical_signals=Signals(adjusted,days),stress_mode='depth',**options)
    a=e.run();audit_stress(a)
    used={}
    for t in a['trades']:
        if t['channel']=='odd':
            key=(t['date'],t['stock_id']);used[key]=used.get(key,0)+t['qty']
            assert quantity is not None and used[key]<=quantity
    if quantity is None:
        assert not any(t['channel']=='odd' for t in a['trades'])
        assert any(t['failure']=='missing_opposing_depth' for t in a['orders'])


def test_quote_uses_opposing_side_not_last_trade():
    _,account,_=run('quote',end=ENTRY+66)
    odd=[t for t in account['trades'] if t['channel']=='odd']
    assert {t['side'] for t in odd}=={'buy','sell'}
    assert all(t['reference_price']==t['odd_ask' if t['side']=='buy' else 'odd_bid'] for t in odd)


def test_entry_delay_keeps_original_signal_and_uses_actual_entry_for_exit():
    _,a,days=run('entry_delay',end=ENTRY+66)
    cohort=a['cohorts'][0]
    assert cohort['signal_date']==str(days[ENTRY-1].date())
    assert cohort['entry_date']==str(days[ENTRY+1].date())
    assert cohort['due_index']==ENTRY+1+63


def test_exit_delay_latches_once_and_waits_for_next_session():
    stock=np.full(SIZE,100.);stock[ENTRY+2:]=80.
    _,base,days=run(stock=stock)
    engine,a,_=run('exit_delay',stock=stock)
    sells=lambda a:[t for t in a['trades'] if t['stock_id']=='1101' and t['side']=='sell']
    assert sells(base)[0]['date']==str(days[ENTRY+3].date())
    assert sells(a)[0]['date']==str(days[ENTRY+4].date())
    assert len(engine.delayed_exits)==1
