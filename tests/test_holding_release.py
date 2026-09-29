from copy import deepcopy
from types import SimpleNamespace
import pandas as pd
import pytest
from skills.holding_release import HoldingRelease, release_choices


def context(sid='1111', age=20, own=.01, relative=-.02):
    return dict(stock_id=sid, event_id=sid+'-entry', age=age, return20=own, relative20=relative)


def test_stagnation_needs_age_and_absolute_and_relative_weakness():
    rows=[context(),context('2222',age=19),context('3333',own=.04),
          context('4444',relative=.01),context('5555',own=float('nan'))]
    assert [r['stock_id'] for r,_ in release_choices(rows,None,[],False,3,'stagnant')]==['1111']


def test_stronger_exits_only_weakest_and_requires_full_slots_no_pending():
    rows=[context(),context('2222',relative=-.05),context('3333',relative=.04)]
    openings=['1111','2222','3333'];candidate={'priority':.15}
    assert release_choices(rows,candidate,openings,False,3,'stronger')==[(rows[1],'stronger_signal')]
    assert release_choices(rows,candidate,openings,True,3,'stronger')==[]
    assert release_choices(rows,candidate,openings,False,5,'stronger')==[]
    assert release_choices([context(age=9)],candidate,openings,False,3,'stronger')==[]
    assert release_choices([context(relative=.01)],candidate,openings,False,3,'stronger')==[]
    assert release_choices(rows,{'priority':.01},openings,False,3,'stronger')==[]


class Base:
    def __init__(self, close):
        self.exit_signals=SimpleNamespace(adjusted_close=close,relative20=(close/close.shift(20)-1).sub(close['0050']/close['0050'].shift(20)-1,axis=0))
        self.days=close.index;self.positions={d:i for i,d in enumerate(self.days)}
        self.events={};self.slots=3;self.opening_members={'1111'};self.opening_limit=100000
        self.holdings={'1111':dict(qty=1234,event_id='1111-entry',due_index=63)}
        self.exit_states={'1111-entry':dict(entry_index=0,trigger_reason=None)};self.plans=[]
    def corporate_day(self,day):return 0
    def _plan(self,*args):self.plans.append(args)


class Example(HoldingRelease,Base):pass


def test_decision_reads_only_previous_close_and_latches_no_buy():
    days=pd.bdate_range('2020-01-01',periods=24)
    close=pd.DataFrame({'1111':[100.]*24,'0050':[100+i for i in range(24)]},index=days)
    before=close.copy();before.loc[days[21]:,'1111']=10000
    engines=[Example(c,release_mode='stagnant') for c in (close,before)]
    for e in engines:
        e.corporate_day(days[21]);assert e.plans[0][2]=='sell'
        assert e.plans[0][4]==str(days[20].date())
        e.corporate_day(days[22]);assert len(e.plans)==1
    assert engines[0].release_decisions==engines[1].release_decisions
    assert engines[0].holdings==engines[1].holdings


def test_existing_stop_is_not_overwritten_and_control_is_noop():
    days=pd.bdate_range('2020-01-01',periods=24)
    close=pd.DataFrame({'1111':100.,'0050':range(100,124)},index=days)
    for mode in ('control','stagnant','stronger'):
        e=Example(close,release_mode=mode);e.exit_states['1111-entry']['trigger_reason']='loss12'
        before=deepcopy(e.exit_states);e.corporate_day(days[21])
        assert e.exit_states==before and e.plans==[]


def test_unknown_policy_fails():
    with pytest.raises(ValueError):release_choices([],None,[],False,3,'unknown')


def test_missing_other_holding_context_cannot_break_serializable_decision():
    import json
    days=pd.bdate_range('2020-01-01',periods=24)
    close=pd.DataFrame({'1111':100.,'2222':float('nan'),'0050':range(100,124)},index=days)
    e=Example(close,release_mode='stagnant')
    e.holdings['2222']=dict(qty=1000,event_id='2222-entry',due_index=63)
    e.exit_states['2222-entry']=dict(entry_index=0,trigger_reason=None)
    e.corporate_day(days[21])
    assert [r['stock_id'] for r in e.release_decisions]==['1111']
    assert [r['stock_id'] for r in e.release_decisions[0]['eligible_contexts']]==['1111']
    json.dumps(e.release_decisions,allow_nan=False)
