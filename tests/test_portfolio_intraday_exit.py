from copy import deepcopy
import pandas as pd
import pytest

from skills.portfolio_intraday_exit import PortfolioIntradayExit
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.scenario_exit_replay import ExitSignals
from skills.high_return_audit import audit_high_return_resources
from skills.portfolio_intraday_audit import audit_sources
from scripts.research_portfolio_intraday import validate_intraday_account
from test_cash_allocation_replay import fixture, ENTRY
from test_historical_selector_replay import identities
from test_mixed_odd_replay import Odds


def simulate(intraday=True, future_low=48., huge_entry_high=False):
    def mutate(q, days):
        if huge_entry_high:
            q.loc[q.stock_id.eq('1101') & q.date.eq(days[ENTRY]),'high'] = 65.
        for offset, values in [(1,(50.,60.,49.,59.)),(2,(58.,59.,future_low,53.))]:
            q.loc[q.stock_id.eq('1101') & q.date.eq(days[ENTRY+offset]),['open','high','low','close']] = values
    days, adjusted, args, kwargs = fixture(entries=[ENTRY],end=ENTRY+5,mutate=mutate)
    args[4].get_limits = lambda sid: {str(d.date()):dict(upper=70.,lower=40.) for d in days}
    class Tape:
        def get(self,sid,day,market):
            if day==str(days[ENTRY+1].date()):
                # The low came BEFORE the new high: OHLC alone is ambiguous.
                prices,shares = [50.,60.,59.],[1000,1000,1000]
            elif day==str(days[ENTRY+2].date()):
                prices,shares = [58.,50.,49.],[1000,1000,1000000]
            else:
                prices,shares = [50.,50.,50.],[1000,1000,1000000]
            return pd.DataFrame(dict(time=pd.to_timedelta(['09:10:00','10:10:00','11:10:00']),
                price=prices,shares=shares)),'synthetic-tape'
    cls = PortfolioIntradayExit if intraday else MidpointExitReplay
    e=cls(*args,**kwargs,ordering='original',position_count=3,factor_mask=0,residual_policy='release',
        identity_report=identities(),liquidity_identity=identities(),exit_signals=ExitSignals(adjusted,days),
        ticks=Tape(),odd_feeds=Odds(high=54.,low=48.),
        **(dict(stop_events=pd.DataFrame(columns=['stock_id','event_date','ratio'])) if intraday else {}))
    a=e.run()
    audit_high_return_resources(a,e.resource_plans,e.slot_decisions,e.board_decisions,e.residual_days,args[0])
    validate_intraday_account(a,[str(d.date()) for d in days],kwargs['start'],kwargs['end'])
    if intraday:
        audit_sources(a,args[0],days,e.stop_events,e.ticks,e.odd_feeds,e.feeds,e.markets)
    return e,a,days


def test_full_account_uses_post_trigger_board_fills_and_next_session_odds():
    e,a,days=simulate()
    sells=[t for t in a['trades'] if t['side']=='sell']
    board=[t for t in sells if t['channel']=='board']
    odd=[t for t in sells if t['channel']=='odd']
    assert board and odd
    assert all(t['date']==str(days[ENTRY+2].date()) and t['reference_price']==49. for t in board)
    assert all(t['date']==str(days[ENTRY+3].date()) for t in odd)
    assert all(t['fill_time']>t['order_time'] for t in board)
    first = next(r for r in e.intraday_evidence if r['date']==str(days[ENTRY+1].date()))
    assert first['status']=='tape_no_crossing'
    assert a['daily'][-1]['market_value']==0


def test_exit_rule_does_not_change_prior_buys_or_create_daily_low_fills():
    _,control,_=simulate(False)
    _,candidate,_=simulate()
    _,changed,_=simulate(future_low=45.)
    assert [t for t in control['trades'] if t['side']=='buy']==[t for t in candidate['trades'] if t['side']=='buy']
    assert candidate['trades']==changed['trades']


def test_entry_day_high_is_not_used_as_a_known_post_fill_peak():
    e,a,days=simulate(huge_entry_high=True)
    first=next(r for r in e.intraday_evidence if r['date']==str(days[ENTRY+1].date()))
    buys=[t['reference_price'] for t in a['trades'] if t['side']=='buy']
    assert first['prior_peak']==max(50.,*buys) and first['prior_peak']<65.


@pytest.mark.parametrize('mutation',['future','same_day_odd','pre_trigger','limit'])
def test_account_validator_rejects_noncausal_or_locked_down_exits(mutation):
    _,a,days=simulate(); bad=deepcopy(a)
    t=next(t for t in bad['trades'] if t['side']=='sell' and t['channel']==('odd' if mutation=='same_day_odd' else 'board'))
    if mutation=='future':t['signal_date']='2099-01-01'
    elif mutation=='same_day_odd':t['signal_date']=t['date']
    elif mutation=='pre_trigger':t['fill_time']=t['order_time']
    else:t['reference_price']=t['limit_price']
    with pytest.raises(ValueError):
        validate_intraday_account(bad,[str(d.date()) for d in days],bad['daily'][0]['date'],bad['daily'][-1]['date'])
