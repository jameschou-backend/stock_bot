"""Intraday daily odd-lot estimates cannot claim tick prices or matching times."""
from copy import deepcopy

import pytest

from skills import poc_range_execution as m
from test_poc_executable_replay import tape
from test_poc_gap_execution import (  # noqa: F401
    engine as gap_engine, Engine, DAY, NEXT, SID, EID, execute, tracked, unavailable,
)


def intraday(**changes):
    row = dict(odd_high=103., odd_low=99., odd_shares=10000,
        source_date=str(DAY.date()), market='twse', after_hours=False,
        volume_scope=m.SCOPE, volume_unit='shares', price_unit='TWD_per_share',
        evidence_status='official_intraday_daily_table', intraday_tick_verified=False,
        actual_fill_verified=False)
    row.update(changes)
    return row


@pytest.fixture
def engine(gap_engine):
    class IntradayEngine(m.RangeGapOrders, Engine):
        pass
    obj = object.__new__(IntradayEngine)
    obj.__dict__.update(deepcopy(gap_engine.__dict__))
    obj.buy_fraction, obj.sell_fraction = .5, .5
    obj.odd_feeds.get_odd = lambda *args: intraday()
    obj.day_plans[(EID, 'buy')].update(order_time=m.OPEN, expires_at=m.END, odd_order_time=m.ODD_OPEN, odd_expires_at=m.ODD_END)
    obj.tick_plans = [deepcopy(obj.day_plans[(EID, 'buy')])]
    return obj


def test_independent_daily_midpoint_and_one_percent_capacity_are_explicit():
    result = m.match_range_odd(intraday(odd_shares=59999), 'buy', 110., 700)
    assert result['filled_qty'] == result['capacity_qty'] == 599
    assert result['reference_price'] == result['proxy_price'] == 101.
    assert result['failure'] == 'partial_range_capacity'
    assert result['participation_limit'] == .01
    assert result['source_volume'] == 59999
    assert result['execution_evidence'] == m.ODD_EVIDENCE
    assert result['last_fill_time'] is result['actual_fill_time'] is None
    for key in ('intraday_tick_verified', 'odd_tick_verified', 'within_window_execution_verified',
                'price_level_volume_verified', 'actual_fill_verified', 'live_qualified'):
        assert result[key] is False


@pytest.mark.parametrize('side,limit,filled', [('buy', 101., 0), ('sell', 101., 0),
    ('buy', 100., 0), ('sell', 102., 0), ('buy', 102., 99), ('sell', 100., 99)])
def test_midpoint_must_strictly_cross_preplanned_limit(side, limit, filled):
    result = m.match_range_odd(intraday(odd_shares=9999), side, limit, 200)
    assert result['filled_qty'] == filled
    assert result['proxy_price'] == 101.
    assert result['reference_price'] == (101. if filled else None)


def test_capacity_counts_whole_shares_without_rounding_up_or_reusing_fills():
    assert m.match_range_odd(intraday(odd_shares=99), 'buy', 110., 5)['filled_qty'] == 0
    result = m.match_range_odd(intraday(odd_shares=201), 'buy', 110., 5, used_shares=1)
    assert result['daily_capacity_qty'] == 2 and result['filled_qty'] == 1
    with pytest.raises(ValueError, match='Previously used'):
        m.match_range_odd(intraday(odd_shares=201), 'buy', 110., 5, used_shares=3)


@pytest.mark.parametrize('changes', [dict(volume_scope='after_hours_odd_auction'),
    dict(after_hours=True), dict(volume_unit='lots'), dict(price_unit='unknown'),
    dict(evidence_status='unverified'), dict(intraday_tick_verified=True),
    dict(actual_fill_verified=True), dict(auction_time='14:30:00'),
    dict(odd_shares=True), dict(odd_shares=1.5), dict(odd_shares=-1),
    dict(odd_shares=2**53), dict(odd_high=float('nan')), dict(odd_low=0),
    dict(odd_high=True), dict(odd_low=104), dict(odd_shares=0)])
def test_wrong_session_or_corrupt_data_cannot_be_used_as_intraday_volume(changes):
    with pytest.raises(m.ReplayDataUnavailable):
        m.match_range_odd(intraday(**changes), 'buy', 110., 10)


def test_missing_data_and_verified_zero_volume_are_distinct():
    with pytest.raises(m.ReplayDataUnavailable, match='Missing independent'):
        m.match_range_odd(None, 'buy', 110., 10)
    result = m.match_range_odd(intraday(odd_shares=0, odd_high=None, odd_low=None), 'buy', 110., 10)
    assert result['filled_qty'] == 0 and result['failure'] == 'official_zero_intraday_odd_volume'
    assert result['reference_price'] is result['proxy_price'] is None


@pytest.mark.parametrize('kwargs', [dict(side='bad'), dict(quantity=1000), dict(quantity=True),
    dict(quantity=0), dict(limit_price=float('nan')), dict(limit_price=True),
    dict(participation=.05), dict(participation=.005), dict(used_shares=-1)])
def test_unregistered_order_or_rate_raises_program_error(kwargs):
    args = dict(side='buy', limit_price=110., quantity=5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        m.match_range_odd(intraday(), **args)


def test_range_prices_preserve_cash_mutations_without_invented_clocks(engine, gap_engine):
    # This fixture's constant regular price gives equal financial cash movements.
    assert execute(engine) == execute(gap_engine) == 1003
    assert engine.cash == gap_engine.cash
    assert engine.holdings == gap_engine.holdings
    assert engine.cash_ledger == gap_engine.cash_ledger
    assert engine.orders[0]['source_volume'] == 100000
    for trade in engine.trades:
        assert trade['order_time'] == '09:00:00' and trade['expires_at'] == '13:30:00'
        assert trade['last_fill_time'] is trade['actual_fill_time'] is None
        assert trade['price_fraction'] == .5
    assert engine.trades[0]['allocations'] == []


def test_both_daily_plan_clocks_change_and_copies_remain_frozen(engine):
    original = deepcopy(engine.tick_plans[0])
    class Base:
        def _plan(self, day):
            self.tick_plans.append(deepcopy(original))
    class Account(m.RangeOrders, Base):
        pass
    obj = Account()
    obj.tick_plans = []
    obj.day_plans = {}
    obj._plan(DAY)
    expected = dict(original, order_time=m.OPEN, expires_at=m.END,
                    odd_order_time=m.ODD_OPEN, odd_expires_at=m.ODD_END)
    assert obj.tick_plans[-1] == obj.day_plans[(EID, 'buy')] == expected
    assert obj.tick_plans[-1] is not obj.day_plans[(EID, 'buy')]


@pytest.mark.parametrize('source', ['board', 'odd_missing', 'odd_scope', 'odd_date', 'odd_market', 'odd_range'])
def test_whole_order_gap_still_rolls_back_both_channels_and_keeps_correct_stage(engine, source):
    if source == 'board':
        engine.ticks.get = unavailable
    elif source == 'odd_missing':
        engine.odd_feeds.get_odd = lambda *a: None
    elif source == 'odd_scope':
        engine.odd_feeds.get_odd = lambda *a: intraday(after_hours=True)
    elif source == 'odd_date':
        engine.odd_feeds.get_odd = lambda *a: intraday(source_date='2024-01-04')
    elif source == 'odd_market':
        engine.odd_feeds.get_odd = lambda *a: intraday(market='tpex')
    else:
        # The midpoint itself is legal: reject the invalid daily range anyway.
        engine.odd_feeds.get_odd = lambda *a: intraday(odd_high=111., odd_low=91.)
    before = tracked(engine)
    holding_ref = engine.holdings[SID]
    assert execute(engine) == 0
    after = tracked(engine)
    assert {k:v for k,v in before.items() if k != 'orders'} == {k:v for k,v in after.items() if k != 'orders'}
    assert engine.holdings[SID] is holding_ref and holding_ref['qty'] == 0
    gap = engine.data_gap_exclusions[0]
    stage = 'board' if source == 'board' else 'odd'
    assert gap['failure_stage'] == engine.orders[-1]['failure_stage'] == stage
    assert gap['excluded_channels'] == ['board', 'odd']
    assert engine.tick_attempts == {SID}
    assert engine.orders[-1]['requested_qty'] == 1003
    assert engine.orders[-1]['filled_qty'] == 0


def test_verified_zero_odd_session_keeps_successful_ordinary_fill(engine):
    engine.odd_feeds.get_odd = lambda *a: intraday(odd_shares=0, odd_high=None, odd_low=None)
    assert execute(engine) == 1000
    assert len(engine.trades) == 1 and engine.trades[0]['channel'] == 'board'
    assert engine.orders[-1]['failure'] == 'official_zero_intraday_odd_volume'
    assert engine.data_gap_exclusions == []


def test_scheduled_sale_gap_retains_shares_and_retries_unchanged_exit_signal(engine):
    plan = engine.day_plans.pop((EID, 'buy'))
    plan.update(side='sell', limit_price=90., odd_limit=90.)
    engine.day_plans[(EID, 'sell')] = plan
    engine.tick_plans = [deepcopy(plan)]
    engine.holdings[SID]['qty'] = 1003
    engine.exit_states[EID] = dict(trigger_reason='loss12', signal_date='2024-01-02', target_date=str(DAY.date()))
    engine.odd_feeds.get_odd = lambda *a: None
    assert execute(engine, side='sell', reason='scheduled_exit', signal_date=None) == 0
    assert engine.holdings[SID]['qty'] == 1003 and engine.cash == 1_000_000.
    assert engine.orders[-1]['reason'] == 'loss12' and engine.orders[-1]['failure_stage'] == 'odd'
    retry = dict(plan, date=str(NEXT.date()), reference_date=str(DAY.date()))
    engine.day_plans[(EID, 'sell')] = retry
    engine.tick_plans.append(deepcopy(retry))
    engine.tick_attempts.clear()
    engine.used.clear()
    engine.odd_feeds.get_odd = lambda *a: intraday(source_date=str(NEXT.date()))
    assert execute(engine, day=NEXT, side='sell', reason='scheduled_exit', signal_date=None) == 1003
    assert engine.holdings[SID]['qty'] == 0 and engine.cash > 1_000_000.
    assert all(t['reason'] == 'loss12' and t['signal_date'] == '2024-01-02' for t in engine.trades)
    assert len(engine.data_gap_exclusions) == 1


def test_accounting_and_program_errors_remain_fatal(engine):
    def fail(*args):
        raise ValueError('program invariant')
    engine.odd_feeds.get_odd = fail
    with pytest.raises(ValueError, match='program invariant'):
        execute(engine)
    assert not engine.data_gap_exclusions


def test_run_labels_proxy_and_preserves_disclosed_missing_order_counts(engine):
    engine.odd_feeds.get_odd = lambda *a: None
    execute(engine)
    class Result:
        def run(self):
            return dict(settings={}, orders=self.orders)
    class Account(m.RangeGapOrders, Result):
        pass
    obj = Account()
    obj.data_gap_exclusions = engine.data_gap_exclusions
    obj.orders = engine.orders
    result = obj.run()
    settings = result['settings']
    assert settings['execution'] == m.MODEL
    assert settings['odd_execution_evidence'] == m.ODD_EVIDENCE
    assert settings['odd_participation'] == settings['board_participation'] == .01
    assert settings['buy_fraction'] == settings['sell_fraction'] == .5
    assert settings['board_tick_verified'] is False
    assert settings['odd_order_time'] == '09:00:00'
    assert settings['odd_expires_at'] == '13:30:00'
    assert settings['odd_tick_verified'] is False and settings['live_qualified'] is False
    assert result['ordinary_volume_evidence']['excluded_positive_board_children'] == 1


@pytest.mark.parametrize('side,fraction,expected', [('buy',.5,100.),('sell',.5,100.),('buy',.7,104.),('sell',.3,96.)])
def test_each_side_and_channel_uses_its_registered_range_fraction(side, fraction, expected):
    data = tape([('09:00:00',90.,100000),('13:30:00',110.,100000)])
    limit = 120. if side == 'buy' else 80.
    board = m.match_range_board(data,side,limit,2000,1_000_000,fraction=fraction)
    odd = m.match_range_odd(intraday(odd_high=110.,odd_low=90.),side,limit,100,fraction=fraction)
    assert board['reference_price'] == odd['reference_price'] == expected
    assert board['filled_qty'] == 2000 and odd['filled_qty'] == 100
    assert board['last_fill_time'] is None and board['allocations'] == []


def test_regular_capacity_includes_delayed_close_but_never_fixed_price_or_zero_volume_extremes():
    data = tape([('08:59:59',1.,9000000),('09:00:00',99.,50000),('13:30:00',101.,49999),
                 ('13:33:59.999999',103.,100001),('13:34:00',110.,9000000),
                 ('14:30:00',120.,9000000)])
    # Zero-volume quotes cannot stretch the range or increase the quantity.
    import pandas as pd
    data = pd.concat([data,tape([('15:00:00',1000.,0)])],ignore_index=True)
    result = m.match_range_board(data,'buy',110.,4000,1_000_000)
    assert result['source_volume'] == 200000
    assert result['source_low'] == 99. and result['source_high'] == 103.
    assert result['filled_qty'] == result['daily_capacity_qty'] == 2000
    capped = m.match_range_board(data,'buy',110.,4000,199999)
    assert capped['filled_qty'] == capped['prior_adv_capacity_qty'] == 1000


@pytest.mark.parametrize('side,limit', [('buy',100.),('sell',100.)])
def test_board_equal_proxy_price_gets_no_queue_credit(side, limit):
    result=m.match_range_board(tape([('09:00:00',100.,200000)]),side,limit,1000,1_000_000)
    assert result['filled_qty'] == result['capacity_qty'] == 0
    assert result['proxy_price'] == 100. and result['reference_price'] is None
    assert result['failure'] == 'range_limit_not_crossed'


def test_board_daily_lot_capacity_rounds_down_without_treating_fixed_volume_as_regular():
    result=m.match_range_board(tape([('09:00:00',100.,99999),('14:30:00',100.,900000)]),'buy',110.,1000,1_000_000)
    assert result['filled_qty'] == 0 and result['failure'] == 'range_capacity_zero'
    result=m.match_range_board(tape([('09:00:00',100.,0)]),'buy',110.,1000,1_000_000)
    assert result['source_high'] is result['source_low'] is result['reference_price'] is None
    assert result['failure'] == 'zero_regular_session_volume'


@pytest.mark.parametrize('buy,sell', [(.3,.3),(.7,.7),(.3,.7),(.5,.3),(True,.5),(.5,float('nan'))])
def test_constructor_rejects_unregistered_price_pairs(buy,sell):
    with pytest.raises(ValueError,match='registered'):
        m.RangeOrders(buy_fraction=buy,sell_fraction=sell)


@pytest.mark.parametrize('side,fraction', [('buy',.3),('sell',.7),('buy',True),('sell',float('nan'))])
def test_matcher_rejects_wrong_side_fraction(side,fraction):
    with pytest.raises(ValueError):
        m.match_range_board(tape([('09:00:00',100.,100000)]),side,110.,1000,1_000_000,fraction=fraction)
    with pytest.raises(ValueError):
        m.match_range_odd(intraday(),side,110.,10,fraction=fraction)


def test_board_source_conflict_preserves_whole_order_and_original_failure(engine):
    engine.ticks.audit_day=unavailable
    assert execute(engine)==0
    assert engine.data_gap_exclusions[0]['failure_stage']=='board'
    assert engine.holdings[SID]['qty']==0 and engine.cash==1_000_000.
    assert not engine.trades
