from copy import deepcopy
import pytest
from skills.high_return_replay import HighReturnReplay
from skills.high_return_audit import audit_high_return_resources
from skills.mixed_odd_replay import factory
from skills.mixed_odd_audit import audit_mixed_execution
from skills.strict_tick_inputs import StrictResidualReplay
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY
from test_historical_selector_replay import identities
from test_opening_entry_replay import Ticks
from test_mixed_odd_replay import Odds


def setup(cls=HighReturnReplay, **policy):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+65)
    args[4].get_limits = lambda sid: {str(d.date()): dict(upper=55., lower=45.) for d in days}
    engine = cls(*args, **kwargs, **policy, residual_policy='release', factor_mask=0,
        identity_report=identities(), liquidity_identity=identities(),
        exit_signals=ExitSignals(adjusted, days), ticks=Ticks(), odd_feeds=Odds())
    account = engine.run()
    return engine, account, args[0], days


@pytest.mark.parametrize('ordering', ['original', 'capacity', 'capacity_vol'])
def test_three_slot_cash_and_fills_reconstruct(ordering):
    e, a, q, days = setup(ordering=ordering)
    assert a['settings']['slots'] == 3
    assert all(p['budget'] <= p['opening_cash']/3+.01 for p in e.resource_plans)
    assert not any(t['stock_id'] == '0050' for t in a['trades'])
    assert all(t['signal_date'] < t['date'] for t in a['trades'])
    audit_high_return_resources(a,e.resource_plans,e.slot_decisions,e.board_decisions,e.residual_days,q)
    audit_mixed_execution(a,e.ticks,e.odd_feeds,e.markets,q,days,e.corporate,e.feeds)
    tampered = deepcopy(e.residual_days)
    tampered[0]['new_position_budget'] += 1
    with pytest.raises(ValueError):
        audit_high_return_resources(a,e.resource_plans,e.slot_decisions,e.board_decisions,tampered,q)


def test_neutral_five_slot_account_identical():
    _, old, _, _ = setup(factory(StrictResidualReplay))
    _, new, _, _ = setup(ordering='capacity', position_count=5)
    for key in ('research_policy', 'ordering', 'idle_capital'):
        del new['settings'][key]
    assert old == new


@pytest.mark.parametrize('count', [True, 3., 1, 7])
def test_unregistered_slot_count_rejected(count):
    with pytest.raises(ValueError):
        setup(ordering='capacity', position_count=count)


@pytest.mark.parametrize('ordering', ['original', 'capacity', 'capacity_vol'])
def test_future_data_cannot_change_preopen_candidates_or_budgets(ordering):
    import numpy as np
    from test_reservation_replay import multi_stock
    plans = []
    for mutate_future in (False, True):
        days, adjusted, args, kwargs = multi_stock()
        quotes = args[0].copy()
        for i, sid in enumerate(('1101','1102','1103','1104')):
            quotes.loc[quotes.stock_id.eq(sid), 'volume'] *= i+1
            adjusted[sid] *= 1 + np.sin(np.arange(len(days))) * (.001*(i+1))
        if mutate_future:
            quotes.loc[quotes.date.ge(days[ENTRY]), 'volume'] *= 9
            adjusted.loc[days[ENTRY]:] *= 4
        args = (quotes, *args[1:])
        args[4].get_limits = lambda sid: {str(d.date()): dict(upper=50.,lower=.001) for d in days}
        identity = identities(('1101','1102','1103','1104'))
        e = HighReturnReplay(*args, **kwargs, ordering=ordering, factor_mask=0,
            residual_policy='release', identity_report=identity, liquidity_identity=identity,
            exit_signals=ExitSignals(adjusted,days), ticks=Ticks(), odd_feeds=Odds())
        # Build pre-open plans only: future changes must not affect selection.
        e.corporate_day(days[ENTRY])
        plans.append(e.tick_plans)
    assert plans[0] == plans[1]
    ids = [p['stock_id'] for p in plans[0]]
    assert ids == (['1102','1103','1104','1101'] if ordering=='original'
                   else ['1104','1103','1102','1101'])
    assert sum(p['reserved_cash'] > 0 for p in plans[0]) == 3
