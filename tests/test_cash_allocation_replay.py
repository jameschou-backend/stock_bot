"""Idle-capital policies use synthetic prices and providers without network I/O."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.replay_million import audit
from skills.cash_allocation_replay import ALLOCATION_MODES, CashAllocationReplay
from skills.scenario_exit_replay import ExitSignals, ScenarioExitReplay


ENTRY = 130
SIZE = 220


class Feeds:
    def __init__(self, quotes, lower_limit_days=(), odd_volume=100_000):
        self.quotes = quotes.set_index(['date', 'stock_id'])
        self.days = sorted(quotes.date.unique())
        self.lower_limit_days = set(lower_limit_days)
        self.odd_volume = odd_volume

    def get_limits(self, sid):
        return {str(pd.Timestamp(day).date()): dict(upper=100_000.,
                lower=float(self.quotes.loc[(day, sid), 'close'])
                if (sid, pd.Timestamp(day)) in self.lower_limit_days else .001)
                for day in self.days}

    def get_odd(self, day, sid, market):
        price = float(self.quotes.loc[(pd.Timestamp(day), sid), 'close'])
        return dict(odd_shares=self.odd_volume, odd_last=price, odd_bid=price-.01,
                    odd_ask=price+.01, bid_qty=10_000, ask_qty=10_000)


class Corporate:
    def __init__(self, events=None):
        self.events = events or {}
        self.prepared = []

    def prepare(self, sid):
        self.prepared.append(sid)

    def on_date(self, sid, day):
        return deepcopy(self.events.get((sid, day), []))


def fixture(*, market=None, stock=None, entries=None, end=ENTRY+8, start=ENTRY-1,
            mutate=None, corporate=None, lower_limits=(), odd_volume=100_000):
    days = pd.bdate_range('2021-01-04', periods=SIZE)
    adjusted = pd.DataFrame({'0050': 100.+np.arange(SIZE)*.1,
                             '1101': np.full(SIZE, 100.)}, index=days)
    if market is not None:
        adjusted['0050'] = market
    if stock is not None:
        adjusted['1101'] = stock
    quotes = pd.DataFrame([dict(date=day, stock_id=sid, open=price,
        high=price+1, low=price-1, close=price, volume=2_000_000)
        for day in days for sid, price in [('0050', 100.), ('1101', 50.)]])
    if mutate:
        mutate(quotes, days)
    events = [dict(event_id='entry-'+str(index), members=['1101'], priority=.1,
                   signal_date=str(days[index-1].date()), entry_date=str(days[index].date()))
              for index in (entries or [])]
    companies = pd.DataFrame([dict(stock_id='1101', name='1101', market='TWSE')])
    args = (quotes, companies, days, events,
            Feeds(quotes, lower_limits, odd_volume), corporate or Corporate())
    kwargs = dict(start=str(days[start].date()), end=str(days[end].date()))
    return days, adjusted, args, kwargs


def run(mode, **options):
    days, adjusted, args, kwargs = fixture(**options)
    replay = CashAllocationReplay(*args, exit_signals=ExitSignals(adjusted, days),
                                  allocation_mode=mode, **kwargs)
    account = replay.run()
    assert all(value for key, value in audit(account).items() if isinstance(value, bool))
    return replay, account, days


def etf_sales(account, reason='parking_trend_off'):
    return [t for t in account['trades'] if t['stock_id']=='0050'
            and t['side']=='sell' and t['reason']==reason]


@pytest.mark.parametrize('with_stock', [False, True])
def test_always_0050_reproduces_entire_loss12_account_exactly(with_stock):
    stock = np.full(SIZE, 100.); stock[ENTRY+2:] = 87.
    days, adjusted, args, kwargs = fixture(stock=stock, entries=[ENTRY] if with_stock else [])
    baseline = ScenarioExitReplay(*args, exit_signals=ExitSignals(adjusted, days),
                                  mode='loss12', **kwargs).run()
    replay = CashAllocationReplay(*args, exit_signals=ExitSignals(adjusted, days), **kwargs)
    actual = replay.run()
    assert actual == baseline
    assert 'allocation_decisions' not in actual
    assert replay.allocation_decisions
    assert audit(actual)['all_daily_nav_reconciled']


def test_cash_without_stock_signal_stays_exactly_one_million_and_never_prepares_etf():
    corporate = Corporate()
    replay, account, _ = run('cash', corporate=corporate)
    assert not account['trades'] and not account['orders'] and not account['holdings']
    assert not corporate.prepared
    assert all(row['cash']==row['nav']==1_000_000. and row['cost']==0
               for row in account['daily'])
    assert {r['action'] for r in replay.allocation_decisions} == {'hold_cash'}


@pytest.mark.parametrize('mode', ALLOCATION_MODES)
def test_stock_loss_trigger_and_next_session_sale_are_unchanged(mode):
    stock = np.full(SIZE, 100.); stock[ENTRY+2:] = 87.
    replay, account, days = run(mode, stock=stock, entries=[ENTRY])
    sales = [t for t in account['trades'] if t['stock_id']=='1101' and t['side']=='sell']
    assert sales and {t['reason'] for t in sales} == {'loss12'}
    assert {t['date'] for t in sales} == {str(days[ENTRY+3].date())}
    assert {t['signal_date'] for t in sales} == {str(days[ENTRY+2].date())}
    assert replay.exit_states['entry-'+str(ENTRY)]['target_index'] == ENTRY+3
    if mode=='cash':
        assert not any(t['stock_id']=='0050' for t in account['trades'])
        assert account['daily'][-1]['market_value'] == 0


def test_market_break_is_used_only_on_next_session_and_sales_include_today_price_move():
    market = 100.+np.arange(SIZE)*.1; market[ENTRY:] = 80.
    def mutate(quotes, days):
        rows = quotes.stock_id.eq('0050') & quotes.date.ge(days[ENTRY+1])
        for field, price in [('open', 80.), ('high', 81.), ('low', 79.), ('close', 80.)]:
            quotes.loc[rows, field] = price
    replay, account, days = run('trend_0050', market=market, mutate=mutate)
    sales = etf_sales(account)
    assert sales and {t['date'] for t in sales} == {str(days[ENTRY+1].date())}
    assert {t['signal_date'] for t in sales} == {str(days[ENTRY].date())}
    assert {t['reference_price'] for t in sales} == {80.}
    assert {t['event_id'] for t in sales} == {'benchmark'}
    quantity = next(h['qty'] for h in account['holdings']
                    if h['date']==str(days[ENTRY].date()) and h['stock_id']=='0050')
    sale_day = next(r for r in account['daily'] if r['date']==str(days[ENTRY+1].date()))
    assert sale_day['market_pnl'] == pytest.approx(quantity*(80.-100.))
    assert sale_day['market_value'] == 0
    assert replay.allocation_decisions[-1]['action'] == 'sell_0050'


def test_off_start_keeps_cash_instead_of_buying_etf():
    _, account, _ = run('trend_0050', market=np.full(SIZE, 100.))
    assert not account['trades']
    assert account['daily'][-1]['nav'] == 1_000_000.


def test_no_prior_market_row_is_unknown_not_the_last_future_row():
    replay, account, _ = run('trend_0050', start=0, end=0)
    assert not account['trades']
    assert all(r['signal_date'] is None and r['market_state']=='UNKNOWN'
               for r in replay.allocation_decisions)


def test_missing_previous_close_keeps_existing_etf_and_does_not_buy_or_sell():
    market = 100.+np.arange(SIZE)*.1; market[ENTRY] = np.nan
    replay, account, days = run('trend_0050', market=market)
    unknown = [r for r in replay.allocation_decisions if r['date']==str(days[ENTRY+1].date())]
    assert unknown and all(r['action']=='retain_unknown' and r['market_state']=='UNKNOWN'
                           and r['etf_qty_before']==r['etf_qty_after']>0
                           and r['cash_before']==r['cash_after'] and r['filled_qty']==0
                           for r in unknown)
    assert not any(t['date']==str(days[ENTRY+1].date()) for t in account['trades'])


def test_unknown_state_still_allows_normal_etf_sale_to_fund_frozen_stock_entry():
    market = 100.+np.arange(SIZE)*.1; market[ENTRY] = np.nan
    replay, account, days = run('trend_0050', market=market, entries=[ENTRY+1])
    funding = etf_sales(account, 'fund_stock')
    assert funding and {t['date'] for t in funding} == {str(days[ENTRY+1].date())}
    assert len(account['cohorts']) == 1
    assert not etf_sales(account)
    decision = next(r for r in replay.allocation_decisions if r['date']==str(days[ENTRY+1].date()))
    assert decision['market_state']=='UNKNOWN' and decision['filled_qty']==0


@pytest.mark.parametrize('after_off', ['ON', 'OFF', 'UNKNOWN'])
def test_partial_etf_sale_reconsiders_next_day_instead_of_latching(after_off):
    market = 100.+np.arange(SIZE)*.1
    market[ENTRY] = 80.
    market[ENTRY+1] = {'ON': 150., 'OFF': 80., 'UNKNOWN': np.nan}[after_off]
    def mutate(quotes, days):
        quotes.loc[quotes.stock_id.eq('0050') & quotes.date.eq(days[ENTRY+1]), 'volume'] = 100_000
    replay, account, days = run('trend_0050', market=market, mutate=mutate, end=ENTRY+2)
    partial = next(r for r in replay.allocation_decisions if r['date']==str(days[ENTRY+1].date()))
    assert 0 < partial['filled_qty'] < partial['requested_qty']
    assert partial['etf_qty_after'] > 0
    following = replay.allocation_decisions[-1]
    assert following['market_state'] == after_off
    if after_off=='OFF':
        assert following['action']=='sell_0050' and following['etf_qty_after']==0
    elif after_off=='ON':
        assert following['action']=='buy_0050' and following['etf_qty_after']>following['etf_qty_before']
    else:
        assert following['action']=='retain_unknown' and following['filled_qty']==0
        assert following['etf_qty_after']==following['etf_qty_before']


def test_lower_limit_blocks_etf_liquidation_until_a_later_off_session():
    market = 100.+np.arange(SIZE)*.1; market[ENTRY:] = 80.
    days = pd.bdate_range('2021-01-04', periods=SIZE)
    _, account, _ = run('trend_0050', market=market, lower_limits=[('0050', days[ENTRY+1])])
    assert {t['date'] for t in etf_sales(account)} == {str(days[ENTRY+2].date())}
    blocked = [o for o in account['orders'] if o['reason']=='parking_trend_off'
               and o['date']==str(days[ENTRY+1].date())]
    assert blocked and all(o['filled_qty']==0 and o['failure']=='at_lower_limit' for o in blocked)


def test_stock_funding_and_later_off_sale_share_the_same_daily_etf_capacity():
    market = 100.+np.arange(SIZE)*.1; market[ENTRY-1:] = 80.
    def mutate(quotes, days):
        quotes.loc[quotes.stock_id.eq('0050') & quotes.date.eq(days[ENTRY]), 'volume'] = 300_000
    _, account, days = run('trend_0050', market=market, mutate=mutate,
                           entries=[ENTRY], end=ENTRY, odd_volume=1_000)
    day = str(days[ENTRY].date())
    board = [t for t in account['trades'] if t['date']==day and t['stock_id']=='0050'
             and t['channel']=='board']
    assert board and sum(t['qty'] for t in board)==3_000
    assert board[0]['reason']=='fund_stock'
    parking = [o for o in account['orders'] if o['date']==day
               and o['reason']=='parking_trend_off' and o['channel']=='board']
    funded = sum(t['qty'] for t in board if t['reason']=='fund_stock')
    assert parking and sum(o['filled_qty'] for o in parking)==3_000-funded
    assert all(o['filled_qty']<o['requested_qty'] and o['failure']=='partial_capacity_or_cash' for o in parking)
    assert account['cohorts']


def test_split_is_applied_before_off_sale_without_fabricating_market_loss():
    market = 100.+np.arange(SIZE)*.1; market[ENTRY:] = 80.
    days = pd.bdate_range('2021-01-04', periods=SIZE)
    split_day = str(days[ENTRY+1].date())
    corporate = Corporate({('0050', split_day): [dict(action_id='split', stock_id='0050',
                                                      kind='split', multiplier=4.)]})
    def mutate(quotes, dates):
        mask = quotes.stock_id.eq('0050') & quotes.date.ge(dates[ENTRY+1])
        for field, price in [('open', 25.), ('high', 26.), ('low', 24.), ('close', 25.)]:
            quotes.loc[mask, field] = price
        quotes.loc[mask, 'volume'] = 10_000_000
    _, account, _ = run('trend_0050', market=market, corporate=corporate, mutate=mutate)
    action = next(r for r in account['corporate_actions'] if r['kind']=='split')
    assert action['qty_after'] == 4*action['entitled_qty']
    assert sum(t['qty'] for t in etf_sales(account)) == action['qty_after']
    assert all(t['reference_price']==25. for t in etf_sales(account))
    assert all(r['market_pnl']==pytest.approx(0.) for r in account['daily'])


def test_etf_dividend_entitlement_survives_off_sale_and_pays_to_cash():
    market = 100.+np.arange(SIZE)*.1; market[ENTRY:] = 80.
    days = pd.bdate_range('2021-01-04', periods=SIZE)
    ex, pay = str(days[ENTRY+1].date()), str(days[ENTRY+4].date())
    corporate = Corporate({('0050', ex): [dict(action_id='dividend', stock_id='0050',
        kind='cash_dividend', cash_per_share=1., pay_date=pay)]})
    _, account, _ = run('trend_0050', market=market, corporate=corporate)
    entitlement = next(a for a in account['corporate_actions'] if a['kind']=='cash_dividend')
    payment = next(m for m in account['cash_ledger'] if m['kind']=='dividend_payment')
    assert entitlement['entitled_qty'] == sum(t['qty'] for t in etf_sales(account))
    assert payment['date']==pay and payment['cash_change']==entitlement['entitlement_value']
    assert not account['receivables']
    assert account['daily'][-1]['market_value']==0


@pytest.mark.parametrize('arguments', [dict(allocation_mode='guess'), dict(mode='fixed63'),
                                      dict(mode='adaptive'), dict(horizon=126), dict(benchmark=True)])
def test_invalid_or_changed_stock_policy_is_rejected(arguments):
    days, adjusted, args, kwargs = fixture()
    with pytest.raises(ValueError):
        CashAllocationReplay(*args, exit_signals=ExitSignals(adjusted, days), **kwargs, **arguments)
