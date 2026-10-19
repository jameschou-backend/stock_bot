from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.poc_market_sizing import VARIANT, market_state, audit_market_sizing, install
from skills.top3_validation import cost_function
from skills.mixed_odd_replay import sized_quantity


def prices(last=90.):
    dates = pd.bdate_range(end='2024-01-31', periods=125)
    return pd.DataFrame({'0050': [100.]*124+[last]}, index=dates)


def test_market_boundary_causality_valid_observations_and_missing_current():
    frame = prices()
    assert market_state(frame, '2024-01-31')['state'] == 'weak'
    frame.loc[pd.Timestamp('2024-02-01')] = 999999
    assert market_state(frame, '2024-01-31')['state'] == 'weak'
    assert market_state(prices(100), '2024-01-31')['state'] == 'normal'
    frame = prices(110)
    frame.iloc[30:35] = np.nan
    result = market_state(frame, '2024-01-31')
    assert result['state'] == 'normal' and result['observations'] == 120
    frame.iloc[-1] = np.nan
    assert market_state(frame, '2024-01-31')['state'] == 'unknown'
    assert market_state(prices().iloc[-119:], '2024-01-31')['state'] == 'unknown'
    assert market_state(prices(), '2024-02-01')['state'] == 'unknown'
    # Uniform adjustment scale cannot create a bearish split signal.
    assert market_state(prices(110)/4, '2024-01-31')['state'] == 'normal'


def fixture(frame, side='buy', mode='baseline', budget=60000.):
    n = sized_quantity(int((budget-40)/(100*1.005925)),110.,budget,SimpleNamespace(_costs=cost_function(mode)),'2330')
    before = dict(date='2024-02-01', signal_date='2024-01-31', reference_date='2024-01-31',
        event_id='event', side=side, stock_id='2330', prior_reference=100., limit_price=110.,
        odd_limit=110., planned_qty=n, board_qty=n//1000*1000, odd_qty=n%1000,
        reserved_cash=budget, sizing_budget=budget, rejection=None)

    class Parent:
        def __init__(self):
            self.day_plans = {}; self.tick_plans = []
            self.exit_signals = SimpleNamespace(adjusted_close=frame)
            self.feeds = SimpleNamespace(get_limits=lambda sid: {'2024-02-01': dict(upper=110., lower=90.)})
            self._costs = cost_function(mode)

        def _plan(self, *args):
            self.day_plans[('event', side)] = deepcopy(before)
            self.tick_plans.append(deepcopy(before))

        def run(self):
            return dict(settings=dict(top3_cost_mode=mode), tick_plans=self.tick_plans)

    runner = SimpleNamespace(engine_types=lambda: (Parent, object), write=lambda p, v: v)
    install(runner, VARIANT, {})
    stock, benchmark = runner.engine_types()
    assert benchmark is object
    engine = stock()
    engine._plan(pd.Timestamp('2024-02-01'), '2330', side, 'event', '2024-01-31', 500, budget, budget, None)
    return engine.run(), before


@pytest.mark.parametrize('mode', ['baseline','fee2','slip2','combined'])
@pytest.mark.parametrize('budget', [60000.,600000.,240.])
def test_weak_budget_half_preserves_reservation_and_independent_cost_sizing(mode,budget):
    frame = prices()
    account, before = fixture(frame, mode=mode, budget=budget)
    p = account['tick_plans'][0]
    assert p['sizing_budget'] == before['sizing_budget']/2
    assert p['reserved_cash'] == before['reserved_cash']
    assert 0 <= p['planned_qty'] < before['planned_qty']
    assert account['market_sizing_audit']['reduced_buy_plans'] == 1
    assert account['market_sizing_audit']['first_trigger'] == '2024-02-01'
    for row in (p, account['market_sizing_records'][0]['after']):
        row['planned_qty'] += 1; row['odd_qty'] += 1
    with pytest.raises(ValueError, match='unrelated plan'):
        audit_market_sizing(account, frame)


def test_normal_and_existing_sell_unchanged_and_unknown_blocks_new_buy():
    for side, frame in [('buy', prices(100)), ('sell', prices())]:
        account, before = fixture(frame, side)
        assert account['tick_plans'][0] == dict(before, market_sizing_applied=side == 'buy')
        assert account['market_sizing_audit']['first_trigger'] is None
    frame = prices(); frame.iloc[-1] = np.nan
    account, before = fixture(frame)
    p = account['tick_plans'][0]
    assert p['planned_qty'] == 0 and p['sizing_budget'] == 0
    assert p['reserved_cash'] == before['reserved_cash']
    assert p['rejection'] == 'market_history_unknown'
    assert account['market_sizing_audit']['unknown_buy_plans'] == 1


@pytest.mark.parametrize('fault', ['window','missing','reservation','sell'])
def test_audit_rejects_source_evidence_and_unrelated_changes(fault):
    frame = prices()
    account, _ = fixture(frame, side='sell' if fault == 'sell' else 'buy')
    record = account['market_sizing_records'][0]
    if fault == 'window': record['market']['average'] += 1
    if fault == 'missing': account['market_sizing_records'] = []
    if fault in ('reservation','sell'):
        field = 'reserved_cash' if fault == 'reservation' else 'planned_qty'
        record['after'][field] += 1; account['tick_plans'][0][field] += 1
    with pytest.raises(ValueError): audit_market_sizing(account, frame)
