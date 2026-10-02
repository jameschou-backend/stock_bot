from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skills.midpoint_replay import MidpointStock
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.scenario_exit_replay import ExitSignals
from skills.verified_volume_midpoint import VerifiedVolumeMidpointOrders
from test_cash_allocation_replay import fixture, ENTRY
from test_historical_selector_replay import identities
from test_midpoint_replay import NoTicks
from test_mixed_odd_replay import Odds


class SafeReplay(VerifiedVolumeMidpointOrders, MidpointStock):
    pass


class SafeExitReplay(VerifiedVolumeMidpointOrders, MidpointExitReplay):
    pass


def run(*, policy='strict', ordinary=800_000., missing=None, end=ENTRY+65, replay_class=SafeReplay):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=end)
    args[4].get_limits = lambda sid: {str(d.date()): dict(upper=55., lower=45.) for d in days}
    frame = pd.DataFrame(ordinary, index=days, columns=['1101', '0050'])
    if missing is not None:
        frame.loc[days[missing], '1101'] = np.nan
    engine = replay_class(*args, **kwargs, ordering='capacity', factor_mask=0, residual_policy='release',
        identity_report=identities(), liquidity_identity=identities(), exit_signals=ExitSignals(adjusted, days),
        ticks=NoTicks(), odd_feeds=Odds(), ordinary_volumes={'TWSE': frame}, volume_policy=policy)
    original_volume = engine.fields['volume'].copy(deep=True)
    original_average = engine.volume20.copy(deep=True)
    account = engine.run()
    pd.testing.assert_frame_equal(engine.fields['volume'], original_volume)
    pd.testing.assert_frame_equal(engine.volume20, original_average)
    return engine, account, days


def test_strict_executes_with_ordinary_but_keeps_total_sizing_and_odd_scope():
    _, account, _ = run()
    assert {r['side'] for r in account['trades']} == {'buy', 'sell'}
    board = [r for r in account['trades'] if r['channel'] == 'board']
    assert board
    for row in board:
        assert row['source_volume'] == 800_000
        assert row['capacity_prior_ordinary_volume20'] == 800_000
        assert row['prior_avg_volume20'] == row['source_total_volume'] == 2_000_000
        assert row['capacity_qty'] == 8_000
        assert row['volume_scope'] == 'ordinary_session'
    odds = [r for r in account['trades'] if r['channel'] == 'odd']
    assert odds and all(r['source_volume'] == 100_000 for r in odds)
    assert account['ordinary_volume_evidence']['all_requested_board_capacity_observed']
    assert account['settings']['actual_fill_verified'] is False


@pytest.mark.parametrize('missing', [ENTRY, ENTRY-1, ENTRY-20])
def test_missing_current_or_prior_volume_blocks_board_without_total_fallback(missing):
    _, account, days = run(missing=missing)
    orders = [r for r in account['orders'] if r['channel'] == 'board' and r['side'] == 'buy']
    assert orders and orders[0]['failure'] == 'ordinary_volume_evidence_missing'
    row = orders[0]
    assert row['filled_qty'] == row['capacity_qty'] == 0
    assert row['reference_price'] is None
    assert row['source_volume'] == (None if missing == ENTRY else 800_000)
    assert row['prior_avg_volume20'] == 2_000_000
    assert any(g['date'] == str(days[missing].date()) for g in row['ordinary_volume_gaps'])
    assert not any(t['channel'] == 'board' and t['side'] == 'buy' for t in account['trades'])
    assert account['ordinary_volume_evidence']['blocked_board_children'] > 0
    assert account['ordinary_volume_evidence']['all_requested_board_capacity_observed'] is False
    # An independent odd-lot child can still execute from its own official row.
    assert any(t['channel'] == 'odd' and t['side'] == 'buy' for t in account['trades'])


def test_new_day_ordinary_data_does_not_change_the_precommitted_buy_plan():
    _, low, _ = run(ordinary=100_000., end=ENTRY+1)
    _, high, _ = run(ordinary=2_000_000., end=ENTRY+1)
    assert low['tick_plans'][0] == high['tick_plans'][0]
    assert next(t['qty'] for t in low['trades'] if t['channel'] == 'board') < next(t['qty'] for t in high['trades'] if t['channel'] == 'board')


def test_legacy_policy_is_explicit_and_never_certifies_ordinary_volume():
    _, account, _ = run(policy='legacy_total_research', missing=ENTRY)
    row = next(t for t in account['trades'] if t['channel'] == 'board')
    assert row['source_volume'] == 2_000_000
    assert row['capacity_prior_ordinary_volume20'] is None
    assert row['volume_scope'] == 'all_daily_sessions_research_proxy'
    assert row['ordinary_capacity_verified'] is False
    assert account['ordinary_volume_evidence']['all_requested_board_capacity_observed'] is False
    assert account['settings']['volume_policy'] == 'legacy_total_research'


def test_legacy_control_preserves_frozen_midpoint_accounting():
    from test_midpoint_replay import run as legacy_run
    _, expected, _, _ = legacy_run(high=51., low=49.)
    _, actual, _ = run(policy='legacy_total_research')
    assert actual['daily'] == expected['daily']
    assert actual['cash_ledger'] == expected['cash_ledger']
    assert actual['tick_plans'] == expected['tick_plans']
    fields = ('date','stock_id','side','channel','qty','reference_price','cash_change','cash_after')
    assert [tuple(r[k] for k in fields) for r in actual['trades']] == [tuple(r[k] for k in fields) for r in expected['trades']]


@pytest.mark.parametrize('value', [-1., .5, float('inf'), float(2**54), True])
def test_invalid_ordinary_matrix_is_rejected_before_base_initialization(value):
    frame = pd.DataFrame(value,index=pd.bdate_range('2025-01-01',periods=21),columns=['5314'])
    with pytest.raises(ValueError, match='[Oo]rdinary'):
        SafeReplay(ordinary_volumes={'TPEX':frame})


def test_ordinary_matrix_is_not_optional_in_strict_mode():
    with pytest.raises(ValueError, match='requires explicit'):
        SafeReplay()


def test_end_inventory_is_marked_not_synthetically_sold():
    _, account, _ = run(end=ENTRY+1)
    final = account['ending_inventory']
    assert final['remaining_positions'] == 1
    assert final['valuation'] == 'mark_to_market'
    assert final['automatically_liquidated'] is False
    assert not any(r['side'] == 'sell' for r in account['trades'])


def test_full_replay_retains_unsold_board_holding_after_missing_sell_evidence():
    _, baseline, days = run(replay_class=SafeExitReplay)
    exit_day = next(r['date'] for r in baseline['trades'] if r['side']=='sell' and r['channel']=='board')
    _, account, _ = run(missing=days.get_loc(pd.Timestamp(exit_day)), replay_class=SafeExitReplay)
    blocked = [r for r in account['orders'] if r.get('failure')=='ordinary_volume_evidence_missing' and r['side']=='sell']
    assert blocked and all(r['filled_qty']==0 for r in blocked)
    assert not any(r['side']=='sell' and r['channel']=='board' for r in account['trades'])
    assert account['ending_inventory']['remaining_positions'] == 1
    assert account['ending_inventory']['automatically_liquidated'] is False
    assert account['daily'][-1]['market_value'] > 0


def isolated(*, ordinary=800_000., planned=9_000, side='buy'):
    """Small ledger harness for exact capacity and blocked-sell invariants."""
    engine = object.__new__(VerifiedVolumeMidpointOrders)
    days = pd.bdate_range('2025-03-13', periods=21)
    day, sid = days[-1], '5314'
    engine.days, engine.positions = days, {d:i for i,d in enumerate(days)}
    engine.ordinary_market_resolver = lambda d,s: 'TPEX'
    engine.ordinary_volumes = {'TPEX': pd.DataFrame(ordinary, index=days, columns=[sid])}
    engine.volume_policy = 'strict'; engine.volume_evidence_blocks = []
    engine.day_plans = {('evt',side): dict(stock_id=sid, signal_date=str(days[-2].date()), planned_qty=planned,
        board_qty=planned, odd_qty=0, limit_price=11. if side == 'buy' else 9.)}
    engine.tick_attempts = set(); engine.markets = {sid:'TPEX'}; engine.names = {sid:sid}
    engine.official_halt = lambda d,s: False; engine.require_prior_inputs = lambda d,s: None
    engine.fields = {k:pd.DataFrame(v,index=days,columns=[sid]) for k,v in
        dict(high=10.5,low=9.5,volume=2_000_000.).items()}
    engine.volume20 = pd.DataFrame(2_000_000.,index=days,columns=[sid])
    engine.amount20 = pd.DataFrame(100_000_000.,index=days,columns=[sid])
    engine.feeds = SimpleNamespace(get_limits=lambda s:{str(day.date()):dict(lower=9.,upper=11.)})
    engine.cash = 1_000_000.; engine.holdings = {sid:dict(qty=planned if side == 'sell' else 0)}
    engine.trades = []; engine.orders = []; engine.used = defaultdict(int); engine.day_cost=engine.day_basis=0.
    engine.marks = {}; engine.raw = lambda d,s: 10.
    engine._costs = lambda p,n,side,s: dict(cash_change=(-1 if side=='buy' else 1)*p*n,total_cost=0.)
    def cash_move(d,k,change,**extra):engine.cash += change
    engine.cash_move = cash_move
    return engine,day,sid,str(days[-2].date())


@pytest.mark.parametrize('planned,cap,volumes', [
    # Exact official 5314 current+20-prior arrays from the 2026-10-02 audit,
    # including the seven independently evidenced suspension zeroes.
    (8_000,7_000,[291000,266000,97000,169000,80000,166000,192000,0,0,0,0,0,0,0,5337000,3698000,2517000,145000,454000,1835000,1918000]),
    (9_000,8_000,[266000,97000,169000,80000,166000,192000,0,0,0,0,0,0,0,5337000,3698000,2517000,145000,454000,1835000,1918000,22088000]),
    (20_000,19_000,[97000,169000,80000,166000,192000,0,0,0,0,0,0,0,5337000,3698000,2517000,145000,454000,1835000,1918000,22088000,21078000]),
])
def test_5314_conflict_capacities_are_7000_8000_19000(planned, cap, volumes):
    engine,day,sid,signal = isolated(planned=planned)
    engine.ordinary_volumes['TPEX'][sid] = volumes
    assert engine._execute_order(day,sid,'buy',planned,'entry','evt',signal) == cap
    assert engine.trades[0]['qty'] == cap
    assert engine.orders[0]['capacity_qty'] == cap
    assert engine.orders[0]['failure'] == 'partial_midpoint_daily_capacity'


def test_blocked_sell_keeps_cash_and_holding_then_can_sell_next_market_day():
    engine,day,sid,signal = isolated(side='sell')
    engine.ordinary_volumes['TPEX'].at[day,sid] = np.nan
    cash,holding = engine.cash,deepcopy(engine.holdings)
    assert engine._execute_order(day,sid,'sell',9_000,'exit','evt',signal) == 0
    assert engine.cash == cash and engine.holdings == holding and not engine.trades
    assert engine.volume_evidence_blocks[0]['side'] == 'sell'
    # Scheduling remains the outer replay's responsibility; a fresh day/plan
    # with valid evidence may consume the still-held shares, without fake cash.
    next_day = day+pd.offsets.BDay()
    engine.days = engine.days.append(pd.DatetimeIndex([next_day])); engine.positions[next_day]=21
    for frame in [*engine.fields.values(),engine.volume20,engine.amount20]:frame.loc[next_day]=frame.iloc[-1]
    engine.ordinary_volumes['TPEX'].loc[day,sid] = 800_000.
    engine.ordinary_volumes['TPEX'].loc[next_day,sid] = 800_000.
    engine.tick_attempts.clear(); engine.feeds.get_limits=lambda s:{str(next_day.date()):dict(lower=9.,upper=11.)}
    engine.day_plans[('evt','sell')]['signal_date']=str(day.date())
    assert engine._execute_order(next_day,sid,'sell',9_000,'exit','evt',str(day.date())) == 8_000
    assert engine.holdings[sid]['qty'] == 1_000


def test_each_prior_day_uses_its_historical_market_and_future_is_not_read():
    engine,day,sid,signal = isolated()
    switch = engine.days[10]
    engine.ordinary_market_resolver = lambda d,s: 'TPEX' if d<switch else 'TWSE'
    engine.ordinary_volumes['TWSE'] = pd.DataFrame(2_000_000.,index=engine.days,columns=[sid])
    engine.ordinary_volumes['TPEX'].loc[switch:]=np.nan
    result = engine.ordinary_capacity_inputs(day,sid)
    assert not result['gaps']
    assert result['prior_average'] == 1_400_000
    assert result['current'] == 2_000_000
    engine.ordinary_volumes['TWSE'].loc[day+pd.offsets.BDay(),sid] = 99_999_999
    assert engine.ordinary_capacity_inputs(day,sid) == result


def test_known_zero_ordinary_history_does_not_become_missing_or_fallback():
    engine,day,sid,signal = isolated(ordinary=0.)
    assert engine._execute_order(day,sid,'buy',9_000,'entry','evt',signal) == 0
    assert engine.orders[0]['failure'] == 'official_ordinary_history_zero'
    assert engine.orders[0]['ordinary_capacity_verified'] is True
    assert not engine.volume_evidence_blocks


def test_known_zero_current_ordinary_volume_without_price_serializes_as_no_fill():
    import json
    engine,day,sid,signal = isolated()
    engine.ordinary_volumes['TPEX'].at[day,sid] = 0.
    engine.fields['high'].at[day,sid] = engine.fields['low'].at[day,sid] = np.nan
    assert engine._execute_order(day,sid,'buy',9_000,'entry','evt',signal) == 0
    row = engine.orders[0]
    assert row['source_volume'] == 0 and row['source_high'] is None
    assert row['failure'] == 'official_zero_volume'
    json.dumps(row,allow_nan=False)
