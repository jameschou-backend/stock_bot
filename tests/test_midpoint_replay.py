from copy import deepcopy
import pytest
from skills.midpoint_replay import MidpointStock,midpoint_match
from skills.midpoint_audit import audit_midpoint
from skills.high_return_audit import audit_high_return_resources
from skills.scenario_exit_replay import ExitSignals
from skills.replay_market_feeds import ReplayDataUnavailable
from test_cash_allocation_replay import fixture,ENTRY
from test_historical_selector_replay import identities
from test_mixed_odd_replay import Odds


class NoTicks:
    def get(self,*args):raise AssertionError('HL2 must not fetch ticks')


def run(high=54.,low=48.):
    days,adjusted,args,kwargs=fixture(entries=[ENTRY],end=ENTRY+65)
    args[4].get_limits=lambda sid:{str(d.date()):dict(upper=55.,lower=45.) for d in days}
    e=MidpointStock(*args,**kwargs,ordering='capacity',factor_mask=0,residual_policy='release',
        identity_report=identities(),liquidity_identity=identities(),exit_signals=ExitSignals(adjusted,days),
        ticks=NoTicks(),odd_feeds=Odds(high=high,low=low))
    a=e.run()
    audit_high_return_resources(a,e.resource_plans,e.slot_decisions,e.board_decisions,e.residual_days,args[0])
    audit_midpoint(a,e.ticks,e.odd_feeds,e.markets,args[0],days,e.corporate,e.feeds)
    return e,a,args[0],days


def test_midpoint_buys_and_sells_use_separate_channel_prices():
    _,a,q,_=run()
    assert {t['side'] for t in a['trades']}=={'buy','sell'}
    assert {t['channel'] for t in a['trades']}=={'board','odd'}
    for t in a['trades']:
        assert t['reference_price']==(t['source_high']+t['source_low'])/2
        if t['channel']=='odd':assert t['reference_price']==51.
        assert t['commission']>=20 and t['signal_date']<t['date']
    assert not a['settings']['actual_fill_verified']


def test_future_range_does_not_change_frozen_shares():
    _,a,_,_=run(high=54.,low=48.)
    _,b,_,_=run(high=54.,low=52.)
    assert a['tick_plans'][0]==b['tick_plans'][0]
    assert a['trades'][1]['reference_price']!=b['trades'][1]['reference_price']


def test_volume_limit_and_price_boundary_are_not_guaranteed_fills():
    args=(54.,48.,250.,1e6,999,'buy',55.,45.,55.,'odd')
    assert midpoint_match(*args)['filled_qty']==2
    assert midpoint_match(55.,55.,1e6,1e6,1000,'buy',55.,45.,55.,'board')['filled_qty']==0
    assert midpoint_match(45.,45.,1e6,1e6,1000,'sell',45.,45.,55.,'board')['filled_qty']==0
    assert midpoint_match(54.,48.,500e3,100e3,9000,'buy',55.,45.,55.,'board')['filled_qty']==1000
    assert midpoint_match(None,None,0,1e6,999,'buy',55.,45.,55.,'odd')['filled_qty']==0
    with pytest.raises(ReplayDataUnavailable):midpoint_match(None,48.,250.,1e6,999,'buy',55.,45.,55.,'odd')


@pytest.mark.parametrize('mutation',['price','capacity','qualification'])
def test_audit_rejects_wrong_midpoint_capacity_or_claim(mutation):
    e,a,q,days=run();bad=deepcopy(a)
    if mutation=='price':bad['trades'][0]['reference_price']+=.1
    elif mutation=='capacity':
        row=next(o for o in bad['orders'] if o['channel']=='odd' and o['filled_qty'])
        row['capacity_qty']+=1
    else:bad['settings']['actual_fill_verified']=True
    with pytest.raises(ValueError):audit_midpoint(bad,e.ticks,e.odd_feeds,e.markets,q,days,e.corporate,e.feeds)
