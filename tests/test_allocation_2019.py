"""Exercise the research composition through fills and independent cash/slot audits."""
from copy import deepcopy
import pytest
from skills.allocation_2019 import POLICIES, allocation_options
from skills.candidate_quality import CandidateQuality
from skills.historical_odd_regime import HistoricalOddEra, normalized_era_account
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.midpoint_exit_audit import audit_midpoint_exit
from skills.high_return_audit import audit_high_return_resources
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY
from test_historical_selector_replay import identities
from test_midpoint_replay import NoTicks
from test_mixed_odd_replay import Odds


class Replay(CandidateQuality, HistoricalOddEra, MidpointExitReplay):
    pass


@pytest.mark.parametrize('arm', POLICIES)
def test_registered_composition_preserves_cash_slots_and_next_day_execution(arm):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+65)
    args[4].get_limits = lambda sid: {str(d.date()): dict(upper=55., lower=45.) for d in days}
    engine = Replay(*args, **kwargs, **allocation_options(arm), candidate_arm='original',
        factor_mask=0, residual_policy='release', identity_report=identities(),
        liquidity_identity=identities(), exit_signals=ExitSignals(adjusted, days),
        ticks=NoTicks(), odd_feeds=Odds())
    account = engine.run()
    assert account['settings']['slots'] == POLICIES[arm][1]
    assert account['settings']['ordering'] == POLICIES[arm][0]
    assert all(t['signal_date'] < t['date'] for t in account['trades'])
    assert not any(t['stock_id'] == '0050' for t in account['trades'])
    audit_high_return_resources(account, engine.resource_plans, engine.slot_decisions,
        engine.board_decisions, engine.residual_days, args[0])
    view, _ = normalized_era_account(account)
    audit_midpoint_exit(view, engine.ticks, engine.odd_feeds, engine.markets,
        args[0], days, engine.corporate, engine.feeds)
    bad = deepcopy(engine.residual_days)
    bad[0]['new_position_budget'] += 1
    with pytest.raises(ValueError):
        audit_high_return_resources(account, engine.resource_plans, engine.slot_decisions,
            engine.board_decisions, bad, args[0])
