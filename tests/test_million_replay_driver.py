"""Independent journal auditing uses only synthetic prices and provider stubs."""
from copy import deepcopy

import pandas as pd
import pytest

from scripts.replay_million import audit
from skills.million_replay import Replay


def account(*, rights=False, split=False):
    days = pd.bdate_range('2021-11-01', periods=30)
    quotes = pd.DataFrame([
        dict(date=day, stock_id=sid, open=price, high=price+1, low=price-1,
             close=price, volume=2_000_000)
        for day in days for sid, price in [('0050',100.),('1101',50.)]
    ])

    class Feeds:
        def get_limits(self, sid):
            return {str(day.date()): dict(upper=10000., lower=.001) for day in days}

        def get_odd(self, day, sid, market):
            price = float(quotes.loc[quotes.stock_id.eq(sid)&quotes.date.eq(pd.Timestamp(day)), 'close'].iloc[0])
            return dict(odd_shares=100_000, odd_last=price, odd_bid=price-.01, odd_ask=price+.01)

    class Corporate:
        def prepare(self, sid):
            pass

        def on_date(self, sid, day):
            if rights and sid == '1101' and day == str(days[22].date()):
                return [dict(stock_id=sid, date=day, action_id='right', kind='stock_dividend',
                             shares_per_share=.5, pay_date=str(days[26].date()), fractional_cash_per_share=10.)]
            if split and sid == '0050' and day == str(days[23].date()):
                return [dict(stock_id=sid, date=day, action_id='split', kind='split', multiplier=4)]
            return []

    if split:
        mask = quotes.stock_id.eq('0050') & quotes.date.ge(days[23])
        quotes.loc[mask, ['open','high','low','close']] /= 4
    events = [dict(event_id='event', members=['1101'], priority=.1,
                   signal_date=str(days[20].date()), entry_date=str(days[21].date()))]
    companies = pd.DataFrame([dict(stock_id='1101', name='Synthetic', market='TWSE')])
    return Replay(quotes,companies,days,events,Feeds(),Corporate(),
                  start=str(days[20].date()),end=str(days[-1].date()),horizon=2).run()


@pytest.mark.parametrize('rights,split', [(False,False),(True,False),(False,True),(True,True)])
def test_complete_synthetic_account_rebuilds_units_fees_and_cash(rights, split):
    result = audit(account(rights=rights, split=split))
    assert result['fees_recomputed'] and result['daily_shares_reconstructed']
    assert result['trade_cash_reconciled'] and result['opening_nav_continuity']


@pytest.mark.parametrize('field', ['gross','commission','tax','slippage','total_cost','cash_change','cash_after'])
def test_individual_trade_financial_fields_cannot_be_rewritten(field):
    result = account()
    result['trades'][0][field] += 123.
    with pytest.raises(ValueError):
        audit(result)


def test_balanced_holding_amount_cannot_hide_wrong_integer_shares():
    result = account()
    result['holdings'][0]['qty'] *= 2
    result['holdings'][0]['price'] /= 2
    with pytest.raises(ValueError, match='holdings differ'):
        audit(result)


def test_opening_nav_cannot_change_even_when_pnl_offsets_it():
    result = account()
    result['daily'][1]['opening_nav'] += 100.
    result['daily'][1]['market_pnl'] -= 100.
    with pytest.raises(ValueError, match='Opening NAV'):
        audit(result)


def test_daily_cost_must_equal_actual_trade_costs():
    result = account()
    result['daily'][0]['cost'] += 100.
    result['daily'][0]['market_pnl'] += 100.
    with pytest.raises(ValueError, match='Daily cost'):
        audit(result)


def test_every_trade_requires_the_identical_cash_journal_identity():
    result = account()
    next(row for row in result['cash_ledger'] if row['kind']=='buy')['event_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        audit(result)


def test_stock_delivery_cannot_exceed_locked_entitlement():
    result = account(rights=True)
    next(row for row in result['corporate_actions'] if row['kind']=='share_delivery')['qty'] += 1
    with pytest.raises(ValueError, match='delivery differs'):
        audit(result)


def test_split_ratio_cannot_disagree_with_recorded_resulting_shares():
    result = account(split=True)
    next(row for row in result['corporate_actions'] if row['kind']=='split')['multiplier'] = 2
    with pytest.raises(ValueError, match='Split share'):
        audit(result)


@pytest.mark.parametrize('field', ['daily_return','total_return','drawdown'])
def test_daily_derived_metrics_are_recomputed(field):
    result = account()
    result['daily'][1][field] += .1
    with pytest.raises(ValueError):
        audit(result)


def test_duplicate_or_outside_daily_rows_are_invalid():
    result = account()
    result['daily'].append(deepcopy(result['daily'][-1]))
    with pytest.raises(ValueError, match='unique'):
        audit(result)


def test_nonfinite_trade_values_fail_closed():
    result = account()
    result['trades'][0]['commission'] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        audit(result)
