from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.poc_trailing import TrailingExit, REASON, audit_trailing, trailing_decision


@pytest.mark.parametrize('entry,current,peak,want', [
    (100., 105.6, 120., True), (100., 105.61, 120., False),
    (100., 100., 119.99, False), (100., 110., 125., True),
    (100., None, 125., False), (None, 110., 125., False),
    (100., float('nan'), 125., False),
])
def test_activation_and_drawdown_boundaries(entry, current, peak, want):
    assert trailing_decision(entry, current, peak) is want


class Base:
    def __init__(self, prices, original=None):
        self.days = pd.bdate_range('2026-01-01', periods=len(prices))
        self.positions = {d:i for i,d in enumerate(self.days)}
        matrix = pd.DataFrame({'1234':prices}, index=self.days)
        self.exit_signals = SimpleNamespace(adjusted_close=matrix,
            price=lambda i,sid: float(matrix.iloc[i][sid]) if pd.notna(matrix.iloc[i][sid]) else None)
        self.exit_states = {'e':dict(stock_id='1234',entry_index=0,entry_price=100.,peak_price=100.,
            trigger_reason=None,signal_date=None,target_date=None,target_index=None)}
        self.holdings = {'1234':dict(event_id='e',qty=1200,due_index=63)}
        self.cohorts = [dict(event_id='e',entry_date=str(self.days[0].date()))]
        self.original = original
        self.opening_limit = 12345.
        self.plans = []

    def corporate_day(self, day):
        state = self.exit_states['e']; i = self.positions[day]
        current = self.exit_signals.price(i-1,'1234')
        if current is not None:
            state['peak_price'] = max(state['peak_price'], current)
        if self.original and not state['trigger_reason']:
            state.update(trigger_reason=self.original,signal_date=str(self.days[i-1].date()),
                target_date=str(day.date()),target_index=i)
        return 17.

    def _plan(self, *args):
        self.plans.append(args)

    def run(self):
        for day in self.days[1:]:
            assert self.corporate_day(day) == 17.
        return dict(settings={},cohorts=self.cohorts,trades=[])


class Engine(TrailingExit, Base):
    pass


def test_t_plus_one_peak_latch_and_partial_sale_retry():
    engine = Engine([100.,125.,110.,200.,210.])
    account = engine.run()
    assert account['trailing_audit']['triggers'] == 1
    assert len(engine.plans) == 1
    assert engine.plans[0][0] == engine.days[3]
    assert engine.plans[0][4] == str(engine.days[2].date())
    assert engine.exit_states['e']['trigger_reason'] == REASON
    assert engine.exit_states['e']['target_index'] == 3
    # Today's 200 and tomorrow's 210 never enter the 110-close decision.
    assert account['trailing_records'][2]['before']['peak_price'] == 125.


@pytest.mark.parametrize('original',['loss12','time63','three_black'])
def test_original_exits_are_never_replaced(original):
    engine=Engine([100.,125.,110.,200.], original=original)
    result=engine.run()
    assert result['trailing_audit']['triggers']==0
    assert engine.exit_states['e']['trigger_reason']==original
    assert not engine.plans


def test_unknown_close_cannot_trigger_and_auditor_detects_tampering():
    engine=Engine([100.,125.,float('nan'),110.,200.])
    account=engine.run()
    assert account['trailing_records'][2]['triggered'] is False
    assert account['trailing_audit']['first_trigger']==str(engine.days[4].date())
    changed=deepcopy(account)
    changed['trailing_records'][3]['before']['peak_price']=130.
    with pytest.raises(ValueError,match='source close'):
        audit_trailing(changed,engine.exit_signals.adjusted_close)
