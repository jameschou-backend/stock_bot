from collections import defaultdict
from copy import deepcopy

import pytest

from skills.board_only_verified_replay import (BoardOnlyVerifiedReplay, BoardOnlyVerifiedBenchmark,
                                              audit_verified_board_only)
from skills.board_only_replay import BoardOnlyReplay
from skills.replay_market_feeds import ReplayDataUnavailable
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY
from test_board_only_replay import no_odd


def test_individual_missing_limit_date_blocks_even_when_stock_source_exists():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY])
    original = args[4].get_limits
    args[4].get_limits = lambda sid: {key: value for key, value in original(sid).items()
                                    if key != str(days[ENTRY].date())}
    args[4].get_odd = no_odd
    with pytest.raises(ReplayDataUnavailable, match='missing price-limit date: 1101'):
        BoardOnlyVerifiedReplay(*args, **kwargs, exit_signals=ExitSignals(adjusted, days)).run()
    # The sealed predecessor intentionally remains unchanged for reproducibility.
    old = BoardOnlyReplay(*args, **kwargs, exit_signals=ExitSignals(adjusted, days)).run()
    assert any(row.get('failure') == 'missing_price_limits' for row in old['orders'])


def test_sub_lot_rejection_does_not_require_any_limit_or_odd_feed():
    days, _, args, kwargs = fixture()
    args[4].get_limits = no_odd
    engine = BoardOnlyVerifiedBenchmark(*args, **kwargs)
    day = days[ENTRY]
    engine.holdings['0050'] = dict(qty=999, event_id='benchmark', due_index=None)
    assert engine.order(day, '0050', 'sell', 999, 'test_exit', 'benchmark', str(days[ENTRY-1].date())) == 0
    assert engine.holdings['0050']['qty'] == 999
    assert engine.board_decisions[-1]['trade_sequences'] == []


def test_negative_net_sale_updates_final_decision_and_corruption_is_rejected():
    def cheap(quotes, days):
        quotes.loc[quotes.stock_id.eq('0050'), ['open', 'high', 'low', 'close']] = [.01, .02, .005, .01]
    days, _, args, kwargs = fixture(mutate=cheap)
    args[4].get_odd = no_odd
    engine = BoardOnlyVerifiedBenchmark(*args, **kwargs)
    engine.cash = 100.
    engine.holdings['0050'] = dict(qty=1000, event_id='benchmark', due_index=None)
    engine.used = defaultdict(int)
    engine.day_cost = engine.day_basis = 0.
    assert engine.order(days[ENTRY], '0050', 'sell', 1000, 'test_exit', 'benchmark',
                        str(days[ENTRY-1].date())) == 1000
    assert engine.trades[0]['negative_proceeds_settlement']
    assert engine.cash == 89. and engine.holdings['0050']['qty'] == 0
    assert engine.board_decisions[0]['filled_qty'] == 1000
    account = dict(settings={'execution_policy': 'board_only'}, trades=engine.trades, orders=engine.orders)
    assert audit_verified_board_only(account, engine.board_decisions, [])['final_fill_decisions_exact']
    damaged = deepcopy(engine.board_decisions)
    damaged[0]['filled_qty'] = 0
    with pytest.raises(ValueError, match='final filled quantity'):
        audit_verified_board_only(account, damaged, [])


@pytest.mark.parametrize('stress', ['control', 'combined'])
def test_verified_account_preserves_parent_when_sources_are_complete(stress):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+68)
    args[4].get_odd = no_odd
    kwargs.update(stress_mode=stress, exit_signals=ExitSignals(adjusted, days))
    original = BoardOnlyReplay(*args, **kwargs).run()
    engine = BoardOnlyVerifiedReplay(*args, **kwargs)
    actual = engine.run()
    assert actual == original
    assert audit_verified_board_only(actual, engine.board_decisions, engine.resource_plans)['final_fill_decisions_exact']
