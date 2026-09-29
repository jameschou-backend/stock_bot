from types import SimpleNamespace
import pandas as pd
import pytest

from skills.close_confirmed_exit import CloseConfirmedExit, update_peak
from skills.close_confirmed_audit import audit_close_stops
from skills.scenario_exit_replay import ExitSignals
from skills.high_return_audit import audit_high_return_resources
from skills.replay_market_feeds import ReplayDataUnavailable
from test_cash_allocation_replay import fixture, ENTRY
from test_historical_selector_replay import identities
from test_mixed_odd_replay import Odds


def simulate(*, huge_entry_high=False, future_high=61., close_breach=True, no_odd=False):
    def mutate(q,days):
        if huge_entry_high:
            q.loc[q.stock_id.eq('1101') & q.date.eq(days[ENTRY]),'high'] = 65.
        bars = [(1,(50.,60.,48.,59.)),(2,(58.,60.,48.,50. if close_breach else 59.)),
                (3,(58.,future_high,57.,59.)),(4,(59.,60.,58.,59.))]
        for n,values in bars:
            q.loc[q.stock_id.eq('1101') & q.date.eq(days[ENTRY+n]),['open','high','low','close']] = values
    days,adjusted,args,kwargs = fixture(entries=[ENTRY],end=ENTRY+4,mutate=mutate)
    args[4].get_limits = lambda sid:{str(d.date()):dict(lower=40.,upper=90.) for d in days}
    class OddFeed(Odds):
        def get_odd(self,day,sid,market):
            r = super().get_odd(day,sid,market)
            if no_odd and day==str(days[ENTRY+3].date()):
                r['odd_shares'] = 0
            return r
    events = pd.DataFrame(columns=['stock_id','event_date','ratio'])
    e = CloseConfirmedExit(*args,**kwargs,stop_events=events,ordering='original',position_count=3,
        factor_mask=0,residual_policy='release',identity_report=identities(),liquidity_identity=identities(),
        exit_signals=ExitSignals(adjusted,days),ticks=None,odd_feeds=OddFeed(high=54.,low=48.))
    a = e.run()
    audit_high_return_resources(a,e.resource_plans,e.slot_decisions,e.board_decisions,e.residual_days,args[0])
    audit_close_stops(a,e.exit_states,SimpleNamespace(days=days,quotes=args[0],features=e.exit_signals,
        events=events,end=kwargs['end']),e)
    return e,a,days


def test_intraday_breach_recovery_does_not_sell_but_close_breach_sells_next_session():
    _,a,days = simulate()
    sells = [t for t in a['trades'] if t['side']=='sell']
    assert {t['channel'] for t in sells}=={'board','odd'}
    assert {t['date'] for t in sells}=={str(days[ENTRY+3].date())}
    assert {t['signal_date'] for t in sells}=={str(days[ENTRY+2].date())}
    assert all(t['reason']=='close_confirmed_peak15' for t in sells)
    _,recovered,_ = simulate(close_breach=False)
    assert not [t for t in recovered['trades'] if t['side']=='sell']


def test_entry_high_is_excluded_and_future_bar_cannot_change_trigger_or_order_size():
    e,a,_ = simulate()
    changed,b,_ = simulate(huge_entry_high=True,future_high=80.)
    assert e.close_evidence==changed.close_evidence
    assert changed.close_evidence[0]['peak']==60.
    assert [(t['date'],t['qty'],t['signal_date']) for t in a['trades']]==[(t['date'],t['qty'],t['signal_date']) for t in b['trades']]


def test_unfilled_odd_exit_stays_latched_after_rebound():
    _,a,days = simulate(no_odd=True)
    odd = [t for t in a['trades'] if t['side']=='sell' and t['channel']=='odd']
    assert odd and odd[0]['date']==str(days[ENTRY+4].date())
    assert odd[0]['signal_date']==str(days[ENTRY+2].date())


def test_adjusted_peak_and_exact_threshold():
    assert update_peak(100.,51.,50.,.5)==(51.,False)
    assert update_peak(100.,90.,85.)==(100.,True)
    with pytest.raises(ReplayDataUnavailable):
        update_peak(100.,float('nan'),85.)
