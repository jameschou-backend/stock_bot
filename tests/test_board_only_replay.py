"""Synthetic policy, funding, corporate residual and sealed-control checks."""
from copy import deepcopy

import pytest

from skills.board_only_replay import BoardOnlyReplay, BoardOnlyBenchmark, audit_board_only, execution_summary
from skills.conservative_diversification import ConservativeDiversification
from skills.execution_resources import audit_resources
from skills.slot_reuse_replay import audit_slots
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, Corporate, Feeds, ENTRY
from test_reservation_replay import multi_stock


def no_odd(*args):
    raise AssertionError('Board-only account requested an odd-lot feed')


def audited(engine):
    account = engine.run()
    if account['settings']['benchmark']:
        audit_resources(account, engine.resource_plans, opening_cash_only=True, lock_unused=True, lock_slots=False)
    else:
        audit_slots(account, engine.resource_plans, engine.slot_decisions, opening_cash_only=True,
                    lock_unused=True, lock_opening_slots=True, lock_failed_slots=True)
    audit_board_only(account, engine.board_decisions, engine.resource_plans)
    return account


@pytest.mark.parametrize('stress', ['control', 'combined'])
@pytest.mark.parametrize('benchmark', [False, True])
def test_policy_never_queries_or_fills_odd_lots(stress, benchmark):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+68)
    args[4].get_odd = no_odd
    kwargs['stress_mode'] = stress
    engine = (BoardOnlyBenchmark(*args, **kwargs) if benchmark else
              BoardOnlyReplay(*args, **kwargs, exit_signals=ExitSignals(adjusted, days)))
    account = audited(engine)
    assert account['trades'] and all(t['channel'] == 'board' and t['qty'] % 1000 == 0 for t in account['trades'])
    assert any(r['rejected_odd_qty'] for r in engine.board_decisions)


def test_rounding_follows_resource_sizing_and_gap_affordability_stays_full_lot():
    def gap(quotes, days):
        at = quotes.stock_id.eq('1101') & quotes.date.ge(days[ENTRY])
        quotes.loc[at, ['open', 'high', 'low', 'close']] = [80., 81., 79., 80.]
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], mutate=gap)
    args[4].get_odd = no_odd
    engine = BoardOnlyReplay(*args, **kwargs, exit_signals=ExitSignals(adjusted, days))
    account = audited(engine)
    plan, decision = engine.resource_plans[0], engine.board_decisions[0]
    assert decision['requested_qty'] == plan['planned_qty'] and plan['planned_qty'] % 1000
    assert decision['board_qty'] == 3000
    assert account['trades'][0]['qty'] == 2000
    assert plan['spent'] <= 200_000 and plan['locked_after'] == round(200_000-plan['spent'], 2)


def test_below_lot_attempts_lock_budget_and_failed_slots_until_close():
    days, adjusted, args, kwargs = multi_stock()
    quotes, companies, calendar, events, feeds, corp = args
    quotes.loc[quotes.stock_id.ne('0050'), ['open', 'high', 'low', 'close']] = [400., 401., 399., 400.]
    feeds = Feeds(quotes)
    feeds.get_odd = no_odd
    engine = BoardOnlyReplay(quotes, companies, calendar, events, feeds, corp,
                             **kwargs, exit_signals=ExitSignals(adjusted, days))
    account = audited(engine)
    assert not account['trades'] and not account['cohorts']
    assert account['daily'][-1]['cash'] == 1_000_000
    assert engine.resource_plans[-1]['locked_after'] == 800_000
    assert engine.slot_decisions[-1]['unfilled_before'] == ['1101', '1102', '1103']
    assert all(d['failure'] == 'board_only_below_one_lot' for d in engine.board_decisions)


def test_stock_dividend_residual_stays_valued_occupies_slot_and_has_no_fake_sale():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+68)
    ex, pay = (str(days[ENTRY+n].date()) for n in (1, 2))
    corp = Corporate({('1101', ex): [dict(kind='stock_dividend', shares_per_share=.05,
        pay_date=pay, action_id='stock-rights', stock_id='1101', fractional_cash_per_share=0.)]})
    args = (*args[:-1], corp)
    args[4].get_odd = no_odd
    engine = BoardOnlyReplay(*args, **kwargs, exit_signals=ExitSignals(adjusted, days))
    account = audited(engine)
    assert engine.holdings['1101']['qty'] == 150
    assert account['cohorts'][0]['exit_date'] is None
    assert account['holdings'][-1]['qty'] == 150 and account['holdings'][-1]['market_value'] == 7500
    assert account['daily'][-1]['holdings'] == 1
    assert sum(t['qty'] for t in account['trades'] if t['side'] == 'sell') == 3000
    assert len([r for r in account['cash_ledger'] if r['kind'] == 'sell']) == 1
    assert any(d['side'] == 'sell' and d['failure'] == 'board_only_below_one_lot' for d in engine.board_decisions)
    summary = execution_summary(account, engine.board_decisions)
    assert summary['residual_only_slot_days'] > 0 and summary['final_residual_market_value'] == 7500
    # Changing that residual to a fake zero would fail independent share accounting.
    broken = deepcopy(account)
    broken['holdings'][-1]['qty'] = 0
    with pytest.raises(ValueError):
        audit_resources(broken, engine.resource_plans, opening_cash_only=True, lock_unused=True, lock_slots=True)


def test_mixed_parent_is_unchanged_by_creating_board_account():
    days, adjusted, args, kwargs = fixture(entries=[ENTRY])
    kw = dict(kwargs, exit_signals=ExitSignals(adjusted, days))
    before = ConservativeDiversification(*args, **kw, position_count=5).run()
    BoardOnlyReplay(*args, **kw).run()
    assert ConservativeDiversification(*args, **kw, position_count=5).run() == before


def test_missing_corporate_source_is_blocked_without_network(tmp_path):
    from types import SimpleNamespace
    from scripts.research_board_only import run_case, NoNetwork
    import pandas as pd
    days, adjusted, args, kwargs = fixture(entries=[ENTRY])
    quotes, companies, calendar, entries, _, _ = args
    data = SimpleNamespace(quotes=quotes, companies=companies, days=calendar, entries=entries,
        events=pd.DataFrame(columns=['stock_id', 'event_date']), features=ExitSignals(adjusted, days), **kwargs)
    result = run_case(data, dict(stress='control', benchmark=False, board_only=True, position_count=5), tmp_path, NoNetwork())
    assert not result['completed']
    assert result['reason'].startswith('Frozen dividend source missing:')
    assert 'partial_account' in result and 'summary' not in result


def test_network_preparation_is_explicitly_forbidden():
    from scripts.research_board_only import NoNetwork
    with pytest.raises(RuntimeError, match='prohibits network'):
        NoNetwork().http('https://example.invalid')
