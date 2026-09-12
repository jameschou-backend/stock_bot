from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from skills.cash_allocation_replay import CashAllocationReplay
from skills.cash_risk_replay import CashRiskReplay, risk_schedule
from skills.execution_stress import audit_stress
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY, SIZE


def execute(stress='control', risk='none', **options):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], **options)
    engine = CashRiskReplay(*args, exit_signals=ExitSignals(adjusted, days),
                           stress_mode=stress, risk_mode=risk, **kwargs)
    account = engine.run()
    audit_stress(account)
    return engine, account, days


def test_neutral_full_account_matches_cash_engine():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY])
    expected = CashAllocationReplay(*args, exit_signals=ExitSignals(adjusted, days),
                                    allocation_mode='cash', **kwargs).run()
    assert execute()[1] == expected


@pytest.mark.parametrize('mode', ['control', 'depth', 'quote', 'slip90', 'entry_delay', 'exit_delay', 'combined'])
def test_cash_stress_preserves_accounting_and_never_buys_etf(mode):
    stock = np.full(SIZE, 100.); stock[ENTRY+2:] = 80.
    _, account, _ = execute(mode, stock=stock)
    assert not any(t['stock_id'] == '0050' for t in account['trades'])
    assert all(r['cash'] >= 0 for r in account['daily'])


def test_market_event_can_reduce_existing_position_without_stock_stop():
    market = 100. + np.arange(SIZE)*.1
    market[ENTRY+2:] = 80.
    engine, account, days = execute(risk='shock', market=market)
    sells = [t for t in account['trades'] if t['side'] == 'sell']
    assert sells and sells[0]['reason'] == 'account_risk_trim'
    assert sells[0]['date'] == str(days[ENTRY+3].date())
    assert sells[0]['signal_date'] == str(days[ENTRY+2].date())
    assert not engine.exit_states['entry-'+str(ENTRY)]['trigger_reason']


def test_unfilled_risk_sale_retries_without_fabricated_fill():
    market = 100. + np.arange(SIZE)*.1; market[ENTRY+2:] = 80.
    days, _, _, _ = fixture()
    engine, account, _ = execute(risk='shock', market=market,
                               lower_limits=[('1101', days[ENTRY+3])])
    first = [r for r in engine.risk_decisions if r['action'] == 'reduce_existing'][0]
    assert first['filled_qty'] == 0 and first['remaining_qty'] == first['requested_qty']
    assert any(t['side'] == 'sell' and t['date'] > first['date'] for t in account['trades'])


def test_gap_loss_is_not_removed_by_account_trim():
    market = 100. + np.arange(SIZE)*.1; market[ENTRY+2:] = 80.
    def gap(quotes, days):
        mask = (quotes.stock_id == '1101') & (quotes.date >= days[ENTRY+3])
        for field in ('open', 'close', 'high', 'low'):
            quotes.loc[mask, field] *= .8
    _, account, days = execute(risk='shock', market=market, mutate=gap)
    before = next(r for r in account['holdings'] if r['date'] == str(days[ENTRY+2].date()))
    on_gap = next(r for r in account['daily'] if r['date'] == str(days[ENTRY+3].date()))
    assert on_gap['market_pnl'] == pytest.approx(-10 * before['qty'])


@pytest.mark.parametrize('mode', ['trend60', 'shock'])
def test_future_mutation_does_not_change_risk_decisions(mode):
    days = pd.bdate_range('2020-01-01', periods=150)
    close = pd.Series(100.+np.arange(150)*.1, index=days)
    close.iloc[90:110] *= .8
    full = risk_schedule(close, mode)
    mutated = close.copy(); mutated.iloc[111:] *= 12
    pd.testing.assert_frame_equal(full.iloc[:111], risk_schedule(mutated, mode).iloc[:111])
    pd.testing.assert_frame_equal(full.iloc[:111], risk_schedule(close.iloc[:111], mode))


def test_missing_day_blocks_buys_without_inventing_recovery():
    market = 100.+np.arange(SIZE)*.1; market[ENTRY-1] = np.nan
    engine, account, _ = execute(risk='shock', market=market)
    assert not account['trades']
    assert engine.risk_decisions[0]['allowed_qty'] == 0


def test_recovery_requires_five_observations_and_does_not_auto_add():
    market = 100.+np.arange(SIZE)*.1
    market[ENTRY+2:ENTRY+5] = 80.
    engine, account, days = execute(risk='trend60', market=market)
    assert engine.risk.loc[str(days[ENTRY+8].date()), 'cap'] == .5
    assert engine.risk.loc[str(days[ENTRY+9].date()), 'cap'] == 1.
    assert {t['date'] for t in account['trades'] if t['side'] == 'buy'} == {str(days[ENTRY].date())}
