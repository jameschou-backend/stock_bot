from types import SimpleNamespace
import pandas as pd
import pytest
from skills.waiting_exit import WaitingExit, waiting_context, waiting_choice


def context(values, benchmark=None):
    c=pd.DataFrame({'1234':values,'0050':benchmark or [100.]*len(values)})
    return waiting_context(c,c.rolling(2,min_periods=2).mean(),'1234',0,len(c)-1)


def test_complete_five_closes_and_ten_day_comparison():
    assert not waiting_choice(context([100.]*4),'stall5')
    assert waiting_choice(context([100.]*5),'stall5')
    assert not waiting_choice(context([100.]*5),'stall10')
    assert waiting_choice(context([100.]*10),'stall10')


def test_started_winner_and_three_percent_progress_are_protected():
    assert not waiting_choice(context([100.,106.,100.,100.,100.]),'stall5')
    assert not waiting_choice(context([100.,101.,102.,102.,103.]),'stall5')
    assert not waiting_choice(context([100.,105.,100.,100.,100.]),'stall5')


def test_missing_path_cannot_be_assumed_stagnant():
    c=context([100.,float('nan'),100.,100.,100.])
    assert c['known'] is False and not waiting_choice(c,'stall5')


def test_weak_rule_requires_both_market_lag_and_below_average():
    assert waiting_choice(context([100.,100.,100.,100.,99.],[100.,101.,102.,103.,104.]),'stall5weak')
    assert not waiting_choice(context([100.,100.,100.,100.,101.],[100.,101.,102.,103.,104.]),'stall5weak')
    assert not waiting_choice(context([100.,100.,100.,100.,99.],[100.,99.,98.,97.,96.]),'stall5weak')


class Base:
    def __init__(self, close):
        self.days=close.index;self.positions={d:i for i,d in enumerate(self.days)}
        self.exit_signals=SimpleNamespace(adjusted_close=close,ma20=close.rolling(20).mean())
        self.exit_states={'entry':dict(entry_index=20,trigger_reason=None)}
        self.holdings={'1234':dict(event_id='entry',qty=1234,due_index=83)}
        self.opening_limit=100000;self.plans=[]
    def corporate_day(self,day):return 0
    def _plan(self,*args):self.plans.append(args)


class Example(WaitingExit,Base):pass


def test_sells_sixth_session_using_fifth_close_and_latches():
    days=pd.bdate_range('2020-01-01',periods=30)
    c=pd.DataFrame({'1234':100.,'0050':100.},index=days)
    future=c.copy();future.loc[days[25]:,'1234']=10000.
    engines=[Example(f,waiting_mode='stall5') for f in (c,future)]
    for e in engines:
        e.corporate_day(days[24]);assert e.plans==[]
        e.corporate_day(days[25]);assert len(e.plans)==1
        assert e.plans[0][0]==days[25] and e.plans[0][4]==str(days[24].date())
        e.corporate_day(days[26]);assert len(e.plans)==1
    assert engines[0].waiting_decisions==engines[1].waiting_decisions


def test_existing_stop_has_priority_and_control_keeps_account_unchanged():
    c=pd.DataFrame({'1234':100.,'0050':100.},index=pd.bdate_range('2020-01-01',periods=30))
    for mode in ('control','stall5','stall10','stall5weak'):
        e=Example(c,waiting_mode=mode);e.exit_states['entry']['trigger_reason']='loss12'
        e.corporate_day(c.index[25]);assert not e.plans and not e.waiting_decisions
    with pytest.raises(ValueError):waiting_choice(context([100.]*5),'arbitrary')


def test_independent_audit_rejects_shifted_decision_and_preserves_future_invariance():
    from copy import deepcopy
    from skills.waiting_exit import audit_waiting
    c=pd.DataFrame({'1234':100.,'0050':100.},index=pd.bdate_range('2020-01-01',periods=30))
    e=Example(c,waiting_mode='stall5');e.corporate_day(c.index[25])
    account=dict(settings={'waiting_mode':'stall5'},cohorts=[dict(event_id='entry',entry_date=str(c.index[20].date()))],
                 waiting_decisions=e.waiting_decisions,trades=[])
    signals=SimpleNamespace(adjusted_close=c,days=c.index)
    assert audit_waiting(account,signals)['waiting_exits']==1
    future=c.copy();future.iloc[25:]=5000
    assert audit_waiting(account,SimpleNamespace(adjusted_close=future,days=future.index))['waiting_exits']==1
    changed=deepcopy(account);changed['waiting_decisions'][0]['signal_date']=str(c.index[25].date())
    with pytest.raises(ValueError,match='same-day'):audit_waiting(changed,signals)
    changed=deepcopy(account);changed['waiting_decisions'][0]['peak_return']=.04
    with pytest.raises(ValueError,match='historical prefix'):audit_waiting(changed,signals)
