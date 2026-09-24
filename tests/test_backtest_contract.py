from copy import deepcopy
import pytest

from skills.backtest_contract import validate_signals, validate_completed_account, validate_comparison


CALENDAR = ['2026-09-04', '2026-09-07', '2026-09-08', '2026-09-09']


def signal():
    liquidity = dict(as_of='2026-09-04', complete_20_sessions=True, observations=20,
                     adv20_shares=100000, mean_turnover20_twd=50000000)
    return dict(event_id='example', signal_date='2026-09-04', entry_date='2026-09-07',
                group_cutoff_date='2026-09-04', trend_decision_date='2026-09-04',
                liquidity_at_signal=deepcopy(liquidity), liquidity_before_entry=deepcopy(liquidity))


def test_signal_uses_next_market_day_over_a_weekend():
    assert validate_signals([signal()], CALENDAR)['next_market_day']


@pytest.mark.parametrize('field', ['entry_date', 'group_cutoff_date', 'trend_decision_date'])
def test_future_dates_and_skipped_entry_are_rejected(field):
    entry = signal()
    entry[field] = '2026-09-08'
    with pytest.raises(ValueError):
        validate_signals([entry], CALENDAR)


def test_liquidity_as_of_next_day_is_not_available_to_the_signal():
    entry = signal()
    entry['liquidity_before_entry']['as_of'] = entry['entry_date']
    with pytest.raises(ValueError, match='future'):
        validate_signals([entry], CALENDAR)


def test_truncated_account_cannot_be_published_as_full_period():
    account = dict(daily=[{'date': day} for day in CALENDAR[:-1]], trades=[])
    with pytest.raises(ValueError, match='Incomplete account'):
        validate_completed_account(account, CALENDAR, CALENDAR[0], CALENDAR[-1])


def test_same_day_buy_and_missing_signal_are_rejected():
    account = dict(daily=[{'date': day} for day in CALENDAR], trades=[dict(side='buy', date=CALENDAR[1])])
    with pytest.raises(ValueError, match='prior signal'):
        validate_completed_account(account, CALENDAR, CALENDAR[0], CALENDAR[-1])
    account['trades'][0]['signal_date'] = CALENDAR[1]
    with pytest.raises(ValueError, match='same-day'):
        validate_completed_account(account, CALENDAR, CALENDAR[0], CALENDAR[-1])


@pytest.mark.parametrize('field', ['initial_cash', 'slippage', 'participation'])
def test_benchmark_requires_identical_cost_capital_and_capacity(field):
    settings = dict(initial_cash=1000000, commission=.001425, minimum_fee=20, participation=.01,
                    odd_participation=.05, slippage=.0045)
    strategy = dict(config=dict(benchmark=False, stress='control', board_only=False),
                    account=dict(daily=[{'date': day} for day in CALENDAR], settings=settings))
    benchmark = deepcopy(strategy)
    benchmark['config']['benchmark'] = True
    assert validate_comparison(strategy, benchmark)['same_dates']
    benchmark['account']['settings'][field] *= 2
    with pytest.raises(ValueError, match='capital/cost'):
        validate_comparison(strategy, benchmark)
