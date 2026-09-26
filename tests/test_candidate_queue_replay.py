from copy import deepcopy

import pytest

from skills.candidate_queue_replay import CandidateQueueReplay, audit_queue
from skills.residual_slot_replay import ResidualSlotReplay, audit_residual_slots
from test_residual_slot_replay import six_stocks
from test_cash_allocation_replay import ENTRY
from skills.million_replay import PARTICIPATION


def run(validity=2, *, offset=63, mask=0, actions=()):
    days, args, kw = six_stocks(next_offset=offset)
    kw.update(factor_mask=mask, action_dates=actions)
    engine=CandidateQueueReplay(*args, **kw, validity_sessions=validity)
    account=engine.run()
    result=audit_queue(account,engine.queue_decisions,args[3],days,args[0],actions,validity,mask)
    audit_residual_slots(account,engine.resource_plans,engine.slot_decisions,engine.board_decisions,
                         engine.residual_days,args[0])
    return days,args,kw,engine,account,result


@pytest.mark.parametrize('mask', range(8))
def test_one_session_reproduces_full_sealed_account(mask):
    _,args,kw,_,account,_=run(1,mask=mask)
    assert account==ResidualSlotReplay(*args,**kw,residual_policy='release').run()


def test_retry_uses_next_day_slot_and_cash_never_same_day():
    days,_,_,engine,account,result=run()
    buy=[t for t in account['trades'] if t['stock_id']=='1106' and t['side']=='buy']
    assert len(buy)==1 and buy[0]['date']==str(days[ENTRY+64].date())
    assert buy[0]['signal_date']==str(days[ENTRY+62].date())
    assert any(r['stock_id']=='1106' and r['failure']=='slots_full' for r in account['orders'])
    assert next(c for c in account['cohorts'] if c['stock_id']=='1106')['attempt_number']==2
    assert sum(r['outcome']=='filled' for r in result['candidate_outcomes'])==6
    assert any(r['reason']=='already_filled' for r in engine.queue_decisions)


def test_expiry_and_corporate_action_use_only_occurred_events():
    days,_,_,_,account,result=run(offset=60)
    assert not any(t['stock_id']=='1106' for t in account['trades'])
    assert result['candidate_outcomes'][-1]['outcome']=='expired_unfilled'
    _,_,_,_,account,result=run(actions=[('1106',days[ENTRY+64])])
    assert not any(t['stock_id']=='1106' for t in account['trades'])
    assert result['candidate_outcomes'][-1]['outcome']=='cancelled'
    _,_,_,_,account,_=run(actions=[('1106',days[ENTRY+65])])
    assert any(t['stock_id']=='1106' and t['side']=='buy' for t in account['trades'])


def test_audit_rejects_tampered_rank_and_repeated_fill():
    days,args,_,engine,account,_=run()
    bad=deepcopy(engine.queue_decisions);bad[0]['rank']=99
    with pytest.raises(ValueError,match='ranking'):
        audit_queue(account,bad,args[3],days,args[0],(),2,0)
    bad=deepcopy(account)
    fill=deepcopy(next(t for t in bad['trades'] if t['stock_id']=='1106' and t['side']=='buy'))
    fill['date']=str(days[ENTRY+65].date());bad['trades'].append(fill)
    with pytest.raises(ValueError,match='repeatedly'):
        audit_queue(bad,engine.queue_decisions,args[3],days,args[0],(),2,0)


def test_future_prices_cannot_change_prior_queue_decisions():
    days,args,kw,engine,account,_=run()
    boundary=days[ENTRY+64]
    quotes=args[0].copy()
    quotes.loc[quotes.date>boundary,['open','close','high','low','volume']]*=2
    changed=CandidateQueueReplay(quotes,*args[1:],**kw,validity_sessions=2)
    other=changed.run()
    prefix=lambda rows:[r for r in rows if r['date']<=str(boundary.date())]
    assert prefix(changed.queue_decisions)==prefix(engine.queue_decisions)
    assert prefix(other['daily'])==prefix(account['daily'])


def test_no_infinite_retry_or_hidden_parameter_sweep():
    _,args,kw=six_stocks()
    for validity in (0,3,True):
        with pytest.raises(ValueError,match='preregistered'):
            CandidateQueueReplay(*args,**kw,validity_sessions=validity)


def test_partial_fill_ends_candidate_and_sample_end_is_not_expiry():
    days,args,kw=six_stocks(next_offset=60)
    quotes=args[0].copy()
    quotes.loc[quotes.date.eq(days[ENTRY]) & quotes.stock_id.eq('1101'),'volume']=1000/PARTICIPATION
    engine=CandidateQueueReplay(quotes,*args[1:],**kw,validity_sessions=2)
    account=engine.run()
    fills=[t for t in account['trades'] if t['stock_id']=='1101' and t['side']=='buy']
    assert len(fills)==1 and fills[0]['qty']==1000
    assert any(r['event_id']=='entry-1101' and r['attempt_number']==2 and r['reason']=='already_filled'
               for r in engine.queue_decisions)
    kw['end']=str(days[ENTRY+60].date())
    engine=CandidateQueueReplay(*args,**kw,validity_sessions=2)
    account=engine.run()
    result=audit_queue(account,engine.queue_decisions,args[3],days,args[0],(),2,0)
    assert result['candidate_outcomes'][-1]['outcome']=='sample_end_truncated'
