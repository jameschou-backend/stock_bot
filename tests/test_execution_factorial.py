from copy import deepcopy

import pytest

from skills.execution_factorial import FactorialReplay, flags, shapley, stock_pnl, CapitalReturnActions, load_capital_terms, CORPORATE_DOCUMENT
from skills.historical_selector_replay import HistoricalBoardReplay
from skills.scenario_exit_replay import ExitSignals
from test_cash_allocation_replay import fixture, ENTRY
from test_cash_allocation_replay import Corporate
from test_historical_selector_replay import identities


@pytest.mark.parametrize('mask,mode', [(0, 'control'), (1, 'slip90'), (2, 'entry_delay'),
                                      (4, 'exit_delay'), (7, 'combined')])
def test_factors_reproduce_each_existing_account_and_preserve_input(mask, mode):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+74)
    before = deepcopy(args[3])
    kwargs.update(exit_signals=ExitSignals(adjusted, days), identity_report=identities())
    old = HistoricalBoardReplay(*args, **kwargs, stress_mode=mode).run()
    new = FactorialReplay(*args, **kwargs, factor_mask=mask).run()
    assert new == old
    assert args[3] == before
    assert all(t['signal_date'] < t['date'] for t in new['trades'])


@pytest.mark.parametrize('mask', [3, 5, 6])
def test_paired_factors_preserve_next_day_timing_and_expected_entry_delay(mask):
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+74)
    engine = FactorialReplay(*args, **kwargs, factor_mask=mask,
        exit_signals=ExitSignals(adjusted, days), identity_report=identities())
    account = engine.run()
    buys = [t for t in account['trades'] if t['side'] == 'buy']
    assert buys[0]['date'] == str(days[ENTRY + int(flags(mask)['entry_delay'])].date())
    assert account['settings']['slippage'] == (.009 if flags(mask)['slippage'] else .0045)
    assert all(t['signal_date'] < t['date'] for t in account['trades'])


def test_shapley_splits_interaction_and_preserves_total():
    values = {m: 100 - (10 if m & 1 else 0) - (20 if m & 2 else 0)
        - (30 if m & 4 else 0) - (12 if m == 7 else 0) for m in range(8)}
    result = shapley(values)
    assert result == pytest.approx(dict(slippage=-14, entry_delay=-24, exit_delay=-34))
    with pytest.raises(ValueError, match='eight'):
        shapley({0: 1.})


def test_pnl_preserves_cash_dividends_and_unpaid_stock_rights():
    account = dict(settings={'initial_cash': 1000}, daily=[{'date': '2026-09-09', 'nav': 1120., 'receivable': 20.}],
        cash_ledger=[{'kind': 'initial_deposit', 'cash_change': 1000},
            {'kind': 'buy', 'stock_id': '1101', 'cash_change': -100.},
            {'kind': 'dividend_payment', 'stock_id': '1101', 'cash_change': 10.}],
        holdings=[{'date': '2026-09-09', 'stock_id': '1101', 'market_value': 190.}],
        receivables=[{'stock_id': '1101', 'kind': 'shares', 'qty': 2, 'fraction': 0., 'fractional_cash_per_share': 0.}])
    assert stock_pnl(account, {'1101': {'price': 10.}})['1101']['profit'] == 120.
    account['daily'][-1]['nav'] = 1121.
    with pytest.raises(ValueError, match='P&L'):
        stock_pnl(account, {'1101': {'price': 10.}})


def test_cash_reduction_entitles_old_shares_and_releases_cash_only_on_payment_date():
    import pandas as pd
    from skills.million_replay import Replay
    engine = Replay.__new__(Replay)
    engine.holdings = {'1808': dict(qty=400, event_id='entry')}
    engine.marks = {'1808': dict(price=34.45)}
    engine.cash, engine.receivables, engine.actions, engine.cash_ledger = 1000., [], [], []
    engine.raw = lambda day, sid: 37.15
    action = dict(kind='capital_reduction', stock_id='1808', date='2025-11-24',
        action_id='1808-2025-11-24', cash_per_share=1., cash_rounding='floor_ntd',
        multiplier=.9, pay_date='2025-11-28')
    provider = Corporate({('1808', '2025-11-24'): [action]})
    engine.corporate = CapitalReturnActions(provider)
    engine.corporate_day(pd.Timestamp('2025-11-24'))
    assert engine.holdings['1808']['qty'] == 360
    assert engine.cash == 1000. and engine.receivables[0]['amount'] == 400.
    assert engine.actions[0]['distribution_type'] == 'capital_return'
    assert provider.events[('1808', '2025-11-24')][0] == action
    engine.corporate_day(pd.Timestamp('2025-11-27'))
    assert engine.cash == 1000.
    engine.corporate_day(pd.Timestamp('2025-11-28'))
    engine.corporate_day(pd.Timestamp('2025-12-01'))
    assert engine.cash == 1400. and not engine.receivables
    assert len(engine.cash_ledger) == 1


def test_capital_reduction_full_ledger_reconciles_and_fractional_exchange_blocks():
    from skills.execution_stress import audit_stress
    from skills.million_replay import UnresolvedAction
    days, adjusted, args, kwargs = fixture(entries=[ENTRY], end=ENTRY+8)
    ex, pay = str(days[ENTRY+2].date()), str(days[ENTRY+6].date())
    action = dict(kind='capital_reduction', stock_id='1101', date=ex, action_id='1101-'+ex,
        cash_per_share=1., cash_rounding='floor_ntd', multiplier=.9, pay_date=pay)
    args = list(args)
    args[-1] = Corporate({('1101', ex): [action]})
    engine = FactorialReplay(*args, **kwargs, factor_mask=0,
        exit_signals=ExitSignals(adjusted, days), identity_report=identities())
    account = engine.run()
    audit_stress(account)
    movement = next(r for r in account['cash_ledger'] if r.get('cash_flow_nature') == 'capital_return')
    assert movement['date'] == pay
    assert movement['cash_change'] == account['trades'][0]['qty']
    action['multiplier'] = .90001
    with pytest.raises(UnresolvedAction, match='Fractional split'):
        FactorialReplay(*args, **kwargs, factor_mask=0,
            exit_signals=ExitSignals(adjusted, days), identity_report=identities()).run()


def test_capital_source_change_or_future_terms_are_rejected(tmp_path):
    import hashlib
    import json
    path = tmp_path / CORPORATE_DOCUMENT
    path.parent.mkdir()
    (tmp_path / 'primary.pdf').write_bytes(b'primary fixture')
    doc = dict(schema=1, finmind_requests=0, secondary_sources_used_in_overrides=False,
        evidence_sha256={'primary.pdf': hashlib.sha256(b'primary fixture').hexdigest()},
        overrides={'1808-2025-11-24': dict(kind='capital_reduction', known_date='2025-10-13',
            pay_date='2025-11-28', cash_rounding='floor_ntd', fractional_policy='block_noninteger_conversion',
            evidence_files=['primary.pdf'], multiplier=.9, cash_per_share=1.)})
    path.write_text(json.dumps(doc))
    assert load_capital_terms(tmp_path)[0] == doc['overrides']
    doc['overrides']['1808-2025-11-24']['known_date'] = '2025-11-25'
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='settlement'):
        load_capital_terms(tmp_path)
    (tmp_path / 'primary.pdf').write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence changed'):
        load_capital_terms(tmp_path)
