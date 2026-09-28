from copy import deepcopy
import pytest
from skills.midpoint_exit_replay import legal_exit_floor
from skills.midpoint_exit_audit import legacy_resource_plans
from skills.midpoint_replay import midpoint_match
from skills.replay_market_feeds import ReplayDataUnavailable


def test_globalwafers_exit_can_fill_below_prior_close():
    args=dict(high=948.,low=864.,volume=4996192.,adv=9434349.75,qty=1000,side='sell',lower=864.,upper=1050.,channel='board')
    assert midpoint_match(**args,limit=959.)['filled_qty']==0
    out=midpoint_match(**args,limit=864.)
    assert out['filled_qty']==1000 and out['reference_price']==906.


def test_lower_limit_does_not_remove_capacity_or_lock_down_checks():
    args=dict(high=948.,low=864.,volume=50000.,adv=9434349.75,qty=1000,side='sell',lower=864.,upper=1050.,channel='board',limit=864.)
    assert midpoint_match(**args)['filled_qty']==0
    assert midpoint_match(**dict(args,high=864.,volume=1000000.))['filled_qty']==0
    assert midpoint_match(**dict(args,volume=0.))['filled_qty']==0


def test_audit_normalization_rejects_unauthorized_sell_price_without_mutating_account():
    base=dict(date='2026-07-29',stock_id='6488',event_id='x',side='sell',limit_price=959.)
    plan=dict(base,limit_price=864.,odd_limit=864.)
    account=dict(base_tick_plans=[base],tick_plans=[plan]); original=deepcopy(account)
    class Feed:
        def get_limits(self,sid):return {'2026-07-29':dict(lower=864.,upper=1050.)}
    assert legacy_resource_plans(account,Feed())[0]['limit_price']==959.
    assert account==original
    plan['limit_price']=865.
    with pytest.raises(ValueError,match='official'):
        legacy_resource_plans(account,Feed())


@pytest.mark.parametrize('limits',[None,{},dict(lower=float('nan'),upper=100),dict(lower=0,upper=100),dict(lower=101,upper=100)])
def test_unknown_legal_limits_fail_closed(limits):
    with pytest.raises(ReplayDataUnavailable):legal_exit_floor(limits)
