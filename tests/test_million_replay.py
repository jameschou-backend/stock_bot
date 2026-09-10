"""Integer-account algebra and execution tests, with no external data providers."""
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.million_replay import Replay, UnresolvedAction, affordable, costs


class Feeds:
    def __init__(self, quotes, *, odd_volume=100_000, missing_limits=False, odd_override=None):
        self.quotes = quotes.set_index(['date', 'stock_id'])
        self.days = quotes.date.unique()
        self.odd_volume = odd_volume
        self.missing_limits = missing_limits
        self.odd_override = odd_override

    def get_limits(self, sid):
        if self.missing_limits:
            return {}
        return {str(pd.Timestamp(day).date()): {'upper': 100_000., 'lower': .001} for day in self.days}

    def get_odd(self, day, sid, market):
        if self.odd_override is not None:
            return deepcopy(self.odd_override)
        price = float(self.quotes.loc[(pd.Timestamp(day), sid), 'close'])
        return {'odd_shares': self.odd_volume, 'odd_last': price,
                'odd_bid': price * .999, 'odd_ask': price * 1.001}


class Corporate:
    def __init__(self, events=None):
        self.events = events or {}
        self.prepared = []

    def prepare(self, sid):
        self.prepared.append(sid)

    def on_date(self, sid, day):
        return deepcopy(self.events.get((sid, day), []))


def inputs(n=32, stock_price=50., etf_price=100., stock_volume=2_000_000):
    days = pd.bdate_range('2021-11-01', periods=n)
    quotes = pd.DataFrame([{'stock_id': sid, 'date': day, 'open': price, 'high': price + 1.,
                            'low': price - 1., 'close': price, 'volume': volume}
                           for day in days for sid, price, volume in
                           [('0050', etf_price, 10_000_000), ('1101', stock_price, stock_volume)]])
    companies = pd.DataFrame([{'stock_id': '1101', 'name': 'A', 'market': 'TWSE'}])
    return days, quotes, companies


def event(days, entry=21, sid='1101', identity='leader-1'):
    return {'event_id': identity, 'signal_date': str(days[entry - 1].date()),
            'entry_date': str(days[entry].date()), 'members': [sid], 'priority': .1}


def replay(*, n=32, events=True, horizon=63, corporate=None, mutate=None, feeds_kwargs=None,
           start_index=20, end_index=None, initial_cash=1_000_000.):
    days, quotes, companies = inputs(n=n)
    if mutate:
        mutate(quotes, days)
    feeds = Feeds(quotes, **(feeds_kwargs or {}))
    engine = Replay(quotes, companies, days, [event(days)] if events else [], feeds, corporate or Corporate(),
                    start=str(days[start_index].date()), end=str(days[-1 if end_index is None else end_index].date()),
                    initial_cash=initial_cash, horizon=horizon)
    return engine, days, quotes


def assert_accounts(result, initial_cash=1_000_000.):
    cash = initial_cash
    for movement in result['cash_ledger'][1:]:
        cash = round(cash + movement['cash_change'], 2)
        assert cash == pytest.approx(movement['cash_after'], abs=.011)
        assert cash >= 0
    assert cash == pytest.approx(result['daily'][-1]['cash'])
    previous = initial_cash
    for row in result['daily']:
        assert row['nav'] == pytest.approx(row['cash'] + row['market_value'] + row['receivable'])
        assert row['opening_nav'] == pytest.approx(previous)
        expected = previous + row['market_pnl'] + row['dividend_entitlement'] + row['execution_basis_pnl'] - row['cost']
        assert row['nav'] == pytest.approx(expected, abs=.06)
        previous = row['nav']
    assert all(type(t['qty']) is int and t['qty'] > 0 for t in result['trades'])
    for trade in result['trades']:
        assert trade['qty'] <= trade['requested_qty']
        assert trade['gross'] == pytest.approx(trade['reference_price'] * trade['qty'], abs=.005)
        assert trade['total_cost'] == trade['commission'] + trade['tax'] + trade['slippage']
        assert trade['cash_change'] == pytest.approx((trade['gross'] if trade['side'] == 'sell' else -trade['gross']) - trade['total_cost'], abs=.011)


def test_integer_account_reconciles_all_cash_movements_and_keeps_final_open_position():
    engine, _, _ = replay()
    result = engine.run()
    assert_accounts(result)
    assert result['cohorts'][0]['exit_date'] is None
    last = result['daily'][-1]
    assert last['holdings'] == 1 and last['market_value'] > 0
    assert not any(o['reason'] == 'terminal_liquidation' for o in result['orders'])


def test_board_and_odd_orders_pay_their_own_minimum_fees():
    engine, _, _ = replay()
    result = engine.run()
    first = [t for t in result['trades'] if t['event_id'] == 'leader-1' and t['side'] == 'buy' and t['stock_id'] == '1101']
    assert {t['channel'] for t in first} == {'board', 'odd'}
    assert all(t['commission'] >= 20 for t in first)
    assert all(t['qty'] % 1000 == 0 for t in first if t['channel'] == 'board')
    assert all(0 < t['qty'] < 1000 for t in first if t['channel'] == 'odd')


def test_cost_rounding_and_benchmark_tax_are_exact():
    tiny = costs(10., 1, 'sell', '1101')
    assert tiny == {'gross': 10., 'commission': 20., 'tax': 0., 'slippage': 1., 'total_cost': 21., 'cash_change': -11.}
    stock = costs(100., 1000, 'sell', '1101')
    etf = costs(100., 1000, 'sell', '0050')
    assert stock['commission'] == etf['commission'] == 143.
    assert stock['tax'] == 300. and etf['tax'] == 100.
    assert stock['slippage'] == 450.
    assert affordable(999, 1, 10., 100., '1101') == 7


@pytest.mark.parametrize('failure', ['zero_volume', 'single_price', 'missing_limits', 'no_odd_trade', 'crossed_odd_quote'])
def test_execution_refuses_missing_volume_single_price_and_invalid_odd_counterparty(failure):
    def mutate(q, d):
        mask = q.stock_id.eq('1101') & q.date.eq(d[21])
        if failure == 'zero_volume': q.loc[mask, 'volume'] = 0
        elif failure == 'single_price': q.loc[mask, ['high', 'low']] = 50.
    feed = {'missing_limits': True} if failure == 'missing_limits' else {}
    if failure == 'no_odd_trade': feed['odd_override'] = {'odd_shares': 0, 'odd_last': 50.}
    elif failure == 'crossed_odd_quote': feed['odd_override'] = {'odd_shares': 100_000, 'odd_last': 50., 'odd_bid': 51., 'odd_ask': 49.}
    engine, days, _ = replay(mutate=mutate, feeds_kwargs=feed)
    result = engine.run()
    failures = [o for o in result['orders'] if o['stock_id'] == '1101' and o['date'] == str(days[21].date())]
    if failure in ('no_odd_trade', 'crossed_odd_quote'):
        assert not any(t['stock_id'] == '1101' and t['channel'] == 'odd' for t in result['trades'])
        assert any(o['channel'] == 'odd' and o['failure'] in ('no_odd_lot_trade', 'invalid_odd_lot_quote') for o in failures)
    else:
        assert not any(t['stock_id'] == '1101' for t in result['trades'])
        assert any(o['failure'] == {'zero_volume': 'missing_or_zero_quote_volume', 'single_price': 'single_price_session', 'missing_limits': 'missing_price_limits'}[failure] for o in failures)
    assert_accounts(result)


def test_missing_first_day_benchmark_execution_does_not_crash_next_day_on_zero_holdings():
    engine, days, _ = replay(events=False, feeds_kwargs={'missing_limits': True})
    result = engine.run()
    assert result['trades'] == []
    assert all(row['nav'] == 1_000_000 and row['cash'] == 1_000_000 for row in result['daily'])


def test_same_day_capacity_is_cumulative_and_does_not_promote_unfilled_board_shares_to_odd():
    engine, days, _ = replay(events=False, feeds_kwargs={'odd_volume': 100})
    day = days[21]
    engine.used = defaultdict(int); engine.day_cost = engine.day_basis = 0.
    engine.holdings['1101'] = {'qty': 0, 'event_id': 'manual', 'due_index': 99}
    engine.fields['volume'].at[day, '1101'] = 100_000  # Board capacity 1000 shares.
    first = engine.order(day, '1101', 'buy', 1500, 'test', 'manual')
    second = engine.order(day, '1101', 'buy', 1500, 'test', 'manual')
    assert first == 1005 and second == 0
    assert engine.holdings['1101']['qty'] == 1005
    assert any(o['failure'] == 'partial_capacity_or_cash' for o in engine.orders)
    assert sum(t['qty'] for t in engine.trades if t['channel'] == 'odd') == 5


def test_today_price_decline_does_not_enlarge_presized_share_quantity():
    original, days, _ = replay()
    first = original.run()
    def cheaper(q, d):
        mask = q.stock_id.eq('1101') & q.date.ge(d[21])
        q.loc[mask, ['open', 'close']] = 40.; q.loc[mask, 'high'] = 41.; q.loc[mask, 'low'] = 39.
    adjusted, _, _ = replay(mutate=cheaper)
    second = adjusted.run()
    request = lambda result: sum(o['requested_qty'] for o in result['orders'] if o['stock_id'] == '1101' and o['reason'] == 'leader_entry')
    assert request(first) == request(second)
    assert_accounts(second)


def test_entry_requires_past_liquidity_and_does_not_use_today_volume_to_rescue_it():
    def low_history(q, days):
        q.loc[q.stock_id.eq('1101') & q.date.lt(days[21]), 'volume'] = 100_000
        q.loc[q.stock_id.eq('1101') & q.date.eq(days[21]), 'volume'] = 100_000_000
    engine, _, _ = replay(mutate=low_history)
    result = engine.run()
    assert not any(t['stock_id'] == '1101' for t in result['trades'])
    assert any(o['failure'] == 'prior_liquidity_below_50m_or_missing' for o in result['orders'])


def test_horizon_starts_at_actual_entry_and_blocked_exit_retries_without_fake_sale():
    def block_due(q, days):
        q.loc[q.stock_id.eq('1101') & q.date.eq(days[23]), 'volume'] = 0
    engine, days, _ = replay(horizon=2, mutate=block_due)
    result = engine.run()
    cohort = result['cohorts'][0]
    assert cohort['due_date'] == str(days[23].date())
    assert cohort['exit_date'] == str(days[24].date())
    assert not any(t['stock_id'] == '1101' and t['side'] == 'sell' and t['date'] <= str(days[23].date()) for t in result['trades'])
    assert any(o['stock_id'] == '1101' and o['date'] == str(days[23].date()) and o['failure'] == 'missing_or_zero_quote_volume' for o in result['orders'])
    assert_accounts(result)


def test_cash_dividend_ex_rights_and_payment_stay_separate_even_after_position_sale():
    days, _, _ = inputs()
    ex, pay = str(days[22].date()), str(days[25].date())
    corp = Corporate({('1101', ex): [{'kind': 'cash_dividend', 'stock_id': '1101', 'action_id': 'cash-1',
                                    'cash_per_share': 1., 'pay_date': pay}]})
    def drop(q, d):
        mask = q.stock_id.eq('1101') & q.date.ge(d[22])
        q.loc[mask, ['open', 'close']] = 49.; q.loc[mask, 'high'] = 50.; q.loc[mask, 'low'] = 48.
    engine, _, _ = replay(horizon=2, corporate=corp, mutate=drop)
    result = engine.run()
    qty = result['cohorts'][0]['bought_qty']
    ex_row = next(r for r in result['daily'] if r['date'] == ex)
    assert ex_row['dividend_entitlement'] == qty
    assert ex_row['receivable'] == qty
    assert ex_row['market_pnl'] == pytest.approx(-qty)
    payments = [r for r in result['cash_ledger'] if r['kind'] == 'dividend_payment']
    assert len(payments) == 1 and payments[0]['date'] == pay and payments[0]['cash_change'] == qty
    assert result['cohorts'][0]['exit_date'] < pay
    assert_accounts(result)


def test_buy_on_ex_dividend_date_does_not_receive_new_entitlement():
    days, _, _ = inputs()
    corp = Corporate({('1101', str(days[21].date())): [{'kind': 'cash_dividend', 'action_id': 'not-owned',
                                                    'cash_per_share': 5., 'pay_date': str(days[22].date())}]})
    engine, _, _ = replay(corporate=corp)
    result = engine.run()
    assert not result['corporate_actions'] and not result['receivables']
    assert_accounts(result)


def test_missing_dividend_pay_date_keeps_receivable_unspendable():
    days, _, _ = inputs()
    corp = Corporate({('1101', str(days[22].date())): [{'kind': 'cash_dividend', 'action_id': 'unknown-pay', 'cash_per_share': 1., 'pay_date': None}]})
    engine, _, _ = replay(horizon=2, corporate=corp)
    result = engine.run()
    assert len(result['receivables']) == 1 and result['receivables'][0]['pay_date'] is None
    assert not any(row['kind'] == 'dividend_payment' for row in result['cash_ledger'])
    assert result['daily'][-1]['receivable'] == result['cohorts'][0]['bought_qty']
    assert_accounts(result)


def test_split_changes_integer_units_without_creating_profit_or_multiplying_cash_receivable():
    days, _, _ = inputs()
    ex, split = str(days[21].date()), str(days[22].date())
    corp = Corporate({('0050', ex): [{'kind': 'cash_dividend', 'stock_id': '0050', 'action_id': 'etf-cash',
                                    'cash_per_share': 1., 'pay_date': None}],
                      ('0050', split): [{'kind': 'split', 'stock_id': '0050', 'action_id': 'etf-split', 'multiplier': 4}]})
    def prices(q, d):
        mask = q.stock_id.eq('0050') & q.date.ge(d[21])
        q.loc[mask, ['open', 'close']] = 99.; q.loc[mask, 'high'] = 100.; q.loc[mask, 'low'] = 98.
        mask = q.stock_id.eq('0050') & q.date.ge(d[22])
        q.loc[mask, ['open', 'close']] = 24.75; q.loc[mask, 'high'] = 25.; q.loc[mask, 'low'] = 24.
    engine, _, _ = replay(events=False, corporate=corp, mutate=prices)
    result = engine.run()
    holding = lambda day: next(h for h in result['holdings'] if h['date'] == day and h['stock_id'] == '0050')
    assert holding(split)['qty'] == 4 * holding(ex)['qty']
    daily = {r['date']: r for r in result['daily']}
    assert daily[split]['nav'] == pytest.approx(daily[ex]['nav'])
    assert daily[split]['market_pnl'] == pytest.approx(0.)
    assert daily[split]['receivable'] == daily[ex]['receivable']
    assert_accounts(result)


def test_stale_price_values_held_asset_but_cannot_fill_scheduled_exit():
    def missing(q, d):
        mask = q.stock_id.eq('1101') & q.date.ge(d[23])
        q.loc[mask, ['open', 'high', 'low', 'close']] = np.nan
        q.loc[mask, 'volume'] = 0
    engine, days, _ = replay(horizon=2, mutate=missing)
    result = engine.run()
    assert result['cohorts'][0]['exit_date'] is None
    last = [h for h in result['holdings'] if h['date'] == str(days[-1].date()) and h['stock_id'] == '1101'][0]
    assert last['stale'] and last['mark_date'] == str(days[22].date())
    assert result['daily'][-1]['stale_holdings'] == 1
    assert_accounts(result)


def test_future_quotes_do_not_change_earlier_transactions_or_nav():
    engine, days, _ = replay()
    original = engine.run()
    def future(q, d):
        mask = q.stock_id.eq('1101') & q.date.ge(d[28])
        q.loc[mask, ['open', 'close']] = 65.; q.loc[mask, 'high'] = 66.; q.loc[mask, 'low'] = 64.
        q.loc[mask, 'volume'] = 100
    changed, _, _ = replay(mutate=future)
    second = changed.run()
    cutoff = str(days[28].date())
    for field in ('daily', 'trades', 'orders', 'cash_ledger', 'holdings'):
        assert [r for r in original[field] if r['date'] < cutoff] == [r for r in second[field] if r['date'] < cutoff]


def test_unknown_corporate_entitlement_stops_instead_of_fabricating_units():
    days, _, _ = inputs()
    corporate = Corporate({('1101', str(days[22].date())): [{'kind': 'capital_reduction_unknown', 'action_id': 'unknown'}]})
    engine, _, _ = replay(corporate=corporate)
    with pytest.raises(UnresolvedAction, match='Unresolved corporate action'):
        engine.run()


def test_ex_dividend_day_sale_keeps_locked_entitlement_and_payment_after_sale():
    days, _, _ = inputs()
    ex, pay = str(days[22].date()), str(days[25].date())
    corp = Corporate({('1101', ex): [{'kind': 'cash_dividend', 'stock_id': '1101', 'action_id': 'cash-on-sale',
                                    'cash_per_share': 1.25, 'pay_date': pay}]})
    engine, _, _ = replay(horizon=1, corporate=corp)
    result = engine.run()
    assert result['cohorts'][0]['exit_date'] == ex
    qty = result['cohorts'][0]['bought_qty']
    entitled = next(a for a in result['corporate_actions'] if a['kind'] == 'cash_dividend')
    assert entitled['entitled_qty'] == qty and entitled['entitlement_value'] == pytest.approx(qty * 1.25)
    payments = [c for c in result['cash_ledger'] if c['kind'] == 'dividend_payment']
    assert payments[0]['date'] == pay and payments[0]['cash_change'] == pytest.approx(qty * 1.25)
    assert_accounts(result)


def test_past_due_stock_dividend_delivery_is_sold_on_delivery_not_before():
    days, _, _ = inputs()
    ex, deliver = str(days[22].date()), str(days[26].date())
    corp = Corporate({('1101', ex): [{'kind': 'stock_dividend', 'stock_id': '1101', 'action_id': 'stock-delivery',
                                    'shares_per_share': 1., 'pay_date': deliver}]})
    def prices(q, d):
        mask = q.stock_id.eq('1101') & q.date.ge(d[22])
        q.loc[mask, ['open', 'close']] = 25.; q.loc[mask, 'high'] = 26.; q.loc[mask, 'low'] = 24.
    engine, _, _ = replay(horizon=2, corporate=corp, mutate=prices)
    result = engine.run()
    cohort = result['cohorts'][0]
    sales = [t for t in result['trades'] if t['stock_id'] == '1101' and t['side'] == 'sell']
    assert {t['date'] for t in sales} == {str(days[23].date()), deliver}
    assert sum(t['qty'] for t in sales) == cohort['bought_qty'] * 2
    assert sum(t['qty'] for t in sales if t['date'] == deliver) == cohort['bought_qty']
    assert cohort['exit_date'] == deliver and not result['receivables']
    assert_accounts(result)


def test_fractional_stock_right_with_known_cash_settlement_does_not_follow_market_price():
    engine, _, _ = replay(events=False)
    engine.receivables = [{'stock_id': '1101', 'kind': 'shares', 'qty': 10, 'fraction': .25,
                           'fractional_cash_per_share': 8., 'pay_date': '2021-12-15',
                           'action_id': 'fraction', 'event_id': 'leader-1'}]
    engine.marks['1101'] = {'price': 50., 'date': '2021-12-01'}
    assert engine.receivable_value() == 502.
    engine.marks['1101']['price'] = 80.
    assert engine.receivable_value() == 802.


def test_fractional_cash_settlement_has_no_trading_commission_or_slippage():
    engine, days, _ = replay(events=False)
    day = days[22]
    engine.receivables = [{'stock_id': '1101', 'kind': 'shares', 'qty': 2, 'fraction': .4,
                           'fractional_cash_per_share': 12.5, 'pay_date': str(day.date()),
                           'action_id': 'fraction', 'event_id': 'leader-1'}]
    engine.holdings['1101'] = {'qty': 10, 'event_id': 'leader-1', 'due_index': 99}
    engine.marks['1101'] = {'price': 50., 'date': str(days[21].date())}
    before = engine.cash + engine.holdings['1101']['qty'] * 50. + engine.receivable_value()
    engine.corporate_day(day)
    after = engine.cash + engine.holdings['1101']['qty'] * 50. + engine.receivable_value()
    assert before == after
    assert engine.holdings['1101']['qty'] == 12 and engine.cash == 1_000_005.
    assert not engine.trades and not engine.orders and not engine.receivables
    payment = engine.cash_ledger[-1]
    assert payment['kind'] == 'fractional_share_payment' and payment['cash_change'] == 5.


def test_undefined_fractional_stock_settlement_is_explicitly_unresolved():
    days, _, _ = inputs()
    corp = Corporate({('1101', str(days[22].date())): [{'kind': 'stock_dividend', 'action_id': 'missing-fraction-rate',
        'shares_per_share': .0000001, 'pay_date': str(days[26].date())}]})
    engine, _, _ = replay(corporate=corp)
    with pytest.raises(UnresolvedAction, match='Fractional share settlement'):
        engine.run()


def test_buy_and_sell_share_the_same_day_channel_participation_budget():
    engine, days, _ = replay(events=False, feeds_kwargs={'odd_volume': 100})
    day = days[21]
    engine.used = defaultdict(int); engine.day_cost = engine.day_basis = 0.
    engine.holdings['1101'] = {'qty': 2000, 'event_id': 'manual', 'due_index': 99}
    engine.fields['volume'].at[day, '1101'] = 100_000
    assert engine.order(day, '1101', 'sell', 1003, 'test', 'manual') == 1003
    assert engine.order(day, '1101', 'buy', 1004, 'test', 'manual') == 2
    assert sum(t['qty'] for t in engine.trades if t['channel'] == 'board') == 1000
    assert sum(t['qty'] for t in engine.trades if t['channel'] == 'odd') == 5
    assert len(engine.trades) == 3 and all(t['commission'] >= 20. for t in engine.trades)


def test_partial_entry_does_not_become_a_future_day_catchup_order():
    engine, days, _ = replay(feeds_kwargs={'odd_volume': 20})
    result = engine.run()
    bought = [t for t in result['trades'] if t['stock_id'] == '1101' and t['side'] == 'buy']
    assert {t['date'] for t in bought} == {str(days[21].date())}
    assert sum(t['qty'] for t in bought if t['channel'] == 'odd') <= 1
    assert any(o['stock_id'] == '1101' and o['channel'] == 'odd' and o['failure'] == 'partial_capacity_or_cash' for o in result['orders'])
    assert_accounts(result)


def test_nonconsecutive_calendar_dates_still_count_sessions_not_calendar_days():
    days, quotes, companies = inputs()
    days = days.delete(23)  # Explicit market closure, not a tradable session.
    quotes = quotes[quotes.date.isin(days)]
    trade_event = event(days)
    engine = Replay(quotes, companies, days, [trade_event], Feeds(quotes), Corporate(),
                    start=str(days[20].date()), end=str(days[-1].date()), horizon=2)
    result = engine.run()
    assert result['cohorts'][0]['due_date'] == str(days[23].date())
    assert result['cohorts'][0]['exit_date'] == str(days[23].date())
    assert_accounts(result)


def test_fully_sold_etf_repurchase_uses_new_raw_mark_after_price_change():
    days, quotes, companies = inputs()
    extra = [quotes[quotes.stock_id.eq('1101')].assign(stock_id=sid) for sid in ('1102', '1103')]
    quotes = pd.concat([quotes, *extra], ignore_index=True)
    companies = pd.DataFrame([dict(stock_id=sid, name=sid, market='TWSE')
                              for sid in ('1101', '1102', '1103')])
    mask = quotes.stock_id.eq('0050') & quotes.date.ge(days[22])
    quotes.loc[mask, ['open', 'close']] = 120.
    quotes.loc[mask, 'high'] = 121.
    quotes.loc[mask, 'low'] = 119.
    events = [event(days, sid=sid, identity=sid) for sid in ('1101', '1102', '1103')]
    engine = Replay(quotes, companies, days, events, Feeds(quotes), Corporate(),
                    start=str(days[20].date()), end=str(days[24].date()), horizon=1)
    result = engine.run()
    sold_day, repurchase_day = str(days[21].date()), str(days[22].date())
    sales = [t for t in result['trades'] if t['date'] == sold_day and t['stock_id'] == '0050' and t['side'] == 'sell']
    assert sales and sales[-1]['remaining_shares'] == 0
    purchases = [t for t in result['trades'] if t['date'] == repurchase_day and t['stock_id'] == '0050' and t['side'] == 'buy']
    assert purchases
    holding = next(h for h in result['holdings'] if h['date'] == repurchase_day and h['stock_id'] == '0050')
    assert holding['price'] == 120. and holding['mark_date'] == repurchase_day
    assert not holding['stale']
    assert_accounts(result)
