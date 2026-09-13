from copy import deepcopy
import pandas as pd
import pytest

from skills.reservation_replay import ReservedCapacityReplay, ReservedBenchmark, audit_reservations
from skills.five_axis_replay import FiveAxisReplay
from skills.execution_stress import StressBenchmark
from skills.scenario_exit_replay import ExitSignals
from scripts.research_reservation_bridge import inspect_ledger
from test_cash_allocation_replay import fixture, ENTRY


@pytest.mark.parametrize('stress', ['control', 'combined'])
@pytest.mark.parametrize('benchmark', [False, True])
def test_neutral_preserves_entire_old_account(stress, benchmark):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+5)
    kwargs['stress_mode'] = stress
    if benchmark:
        original = StressBenchmark(*args, **kwargs).run()
        actual = ReservedBenchmark(*args, reserve_before_open=False, **kwargs).run()
    else:
        kwargs['exit_signals'] = ExitSignals(adjusted, days)
        original = FiveAxisReplay(*args, arm='capacity', **kwargs).run()
        actual = ReservedCapacityReplay(*args, reserve_before_open=False, **kwargs).run()
    assert original == actual


def multi_stock():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+2)
    quotes, companies, calendar, events, feeds, corporate = args
    for sid in ('1102', '1103', '1104'):
        extra = quotes[quotes.stock_id.eq('1101')].copy(); extra.stock_id = sid
        quotes = pd.concat([quotes, extra], ignore_index=True)
        companies = pd.concat([companies, pd.DataFrame([dict(stock_id=sid, name=sid, market='TWSE')])])
        adjusted[sid] = adjusted['1101']
        events.append(dict(events[0], members=[sid], event_id=sid))
    return days, adjusted, (quotes, companies, calendar, events, feeds, corporate), kwargs


def test_unfilled_buys_do_not_release_reservations_or_fourth_slot():
    days, adjusted, args, kwargs = multi_stock()
    # Every price is deliberately at the upper limit: none may fill.
    args[4].get_limits = lambda sid: {str(d.date()):dict(upper=50., lower=.001) for d in days}
    engine = ReservedCapacityReplay(*args, exit_signals=ExitSignals(adjusted, days), **kwargs)
    account = engine.run()
    assert not account['trades']
    assert sum(p['planned_qty'] > 0 for p in engine.plans) == 3
    assert engine.plans[-1]['stock_id'] == '1104' and engine.plans[-1]['planned_qty'] == 0
    audit_reservations(account, engine.plans)


def test_board_and_odd_share_budget_even_when_execution_price_gaps():
    def change(quotes, days):
        at = quotes.stock_id.eq('1101') & quotes.date.ge(days[ENTRY])
        quotes.loc[at, ['open', 'high', 'low', 'close']] = [60., 61., 59., 60.]
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+2, mutate=change)
    engine = ReservedCapacityReplay(*args, exit_signals=ExitSignals(adjusted, days), **kwargs)
    account = engine.run(); audit_reservations(account, engine.plans)
    plan = next(p for p in engine.plans if p['planned_qty'])
    buys = [t for t in account['trades'] if t['side']=='buy']
    assert {t['channel'] for t in buys} == {'board', 'odd'}
    assert sum(-t['cash_change'] for t in buys) <= plan['reserved_cash']
    assert sum(t['qty'] for t in buys) < plan['planned_qty']
    broken = deepcopy(engine.plans); broken[0]['reserved_cash'] = 1.
    with pytest.raises(ValueError, match='frozen budget'):
        audit_reservations(account, broken)


def test_benchmark_dividend_payment_waits_until_following_day():
    from test_cash_allocation_replay import Corporate
    days, adjusted, args, kwargs = fixture(end=ENTRY+4)
    ex = str(days[ENTRY+1].date()); pay = str(days[ENTRY+2].date())
    args = (*args[:-1], Corporate({('0050', ex):[dict(kind='cash_dividend', cash_per_share=10.,
        pay_date=pay, action_id='dividend', stock_id='0050')]}))
    engine = ReservedBenchmark(*args, **kwargs)
    account = engine.run(); audit_reservations(account, engine.plans)
    pay_buys = [t for t in account['trades'] if t['date']==pay and t['side']=='buy']
    prior = next(d['cash'] for d in account['daily'] if d['date']==str(days[ENTRY+1].date()))
    assert sum(-t['cash_change'] for t in pay_buys) <= prior
    assert any(t['date']==str(days[ENTRY+3].date()) for t in account['trades'])


def test_dependency_report_does_not_fabricate_strict_return():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+2)
    engine = ReservedCapacityReplay(*args, exit_signals=ExitSignals(adjusted, days), **kwargs)
    account = engine.run()
    report = inspect_ledger(account, engine.markets)
    assert report['cash_dependent_days'] == 0
    assert report['first_odd_trade']['channel'] == 'odd'
    assert report['strict_intraday']['total_return'] is None
    assert not report['strict_intraday']['completed']


def test_preparation_budget_survives_restarts(tmp_path):
    from scripts.research_reservation_bridge import PreparationBudget
    from skills.replay_market_feeds import ReplayDataUnavailable
    for _ in range(100):
        PreparationBudget(tmp_path).take('official')
    with pytest.raises(ReplayDataUnavailable, match='ceiling'):
        PreparationBudget(tmp_path).take('official')
    PreparationBudget(tmp_path).take('finmind')


def test_entry_plan_is_unchanged_by_order_day_price():
    plans = []
    for price in (50., 60.):
        def mutate(quotes, days):
            at = quotes.stock_id.eq('1101') & quotes.date.eq(days[ENTRY])
            quotes.loc[at, ['high','low','close']] = [price+1, price-1, price]
        days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+1, mutate=mutate)
        engine = ReservedCapacityReplay(*args, exit_signals=ExitSignals(adjusted, days), **kwargs)
        engine.run(); plans.append(engine.plans)
    assert plans[0] == plans[1]
