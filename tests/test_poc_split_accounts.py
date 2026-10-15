from copy import deepcopy
from types import SimpleNamespace

import pytest

from skills.poc_split_accounts import audit_funding, install
from skills.top3_validation import cost_function
from skills.mixed_odd_replay import sized_quantity
from scripts.run_poc_split_matrix import account_output


def account():
    return dict(settings=dict(initial_cash=500000),
        cash_ledger=[dict(kind='initial_deposit', cash_change=500000)],
        daily=[dict(cash=500000)])


def test_funding_is_changed_before_constructor_and_original_is_untouched():
    class Engine:
        def __init__(self, **kwargs):
            self.initial = kwargs['initial_cash']
        def run(self):
            result = account()
            result['settings']['initial_cash'] = self.initial
            return result
    saved = []
    runner = SimpleNamespace(engine_types=lambda: (Engine, Engine), write=lambda p, v: saved.append(deepcopy(v)))
    install(runner, 'split50', dict(initial_cash=500000))
    stock, benchmark = runner.engine_types()
    assert stock(initial_cash=1000000).run()['split_funding_audit']['independent_cash']
    assert benchmark is Engine and Engine(initial_cash=1000000).initial == 1000000
    with pytest.raises(ValueError, match='seam changed'):
        stock(initial_cash=500000)
    runner.write('/account/report.json', dict(initial_cash=1000000))
    assert saved[0]['initial_cash'] == 500000


@pytest.mark.parametrize('mutation', ['wrong_deposit', 'repeat_deposit', 'transfer', 'negative_cash'])
def test_independent_funding_rejects_cross_account_money(mutation):
    value = account()
    if mutation == 'wrong_deposit': value['cash_ledger'][0]['cash_change'] = 1000000
    if mutation == 'repeat_deposit': value['cash_ledger'] *= 2
    if mutation == 'transfer': value['cash_ledger'].append(dict(kind='transfer', cash_change=1))
    if mutation == 'negative_cash': value['daily'][0]['cash'] = -1
    with pytest.raises(ValueError): audit_funding(value)


def test_half_capital_has_integer_shares_and_its_own_minimum_fees():
    cost = cost_function('baseline')
    broker = SimpleNamespace(_costs=cost)
    whole = sized_quantity(10000, 501, 1000000/3, broker, '2330')
    half = sized_quantity(10000, 501, 500000/3, broker, '2330')
    assert type(half) is int and half != whole/2
    assert cost(10, 1, 'buy', '2330')['commission'] == 20


def test_sleeves_have_separate_run_locks():
    paths = [account_output('/base', 'split50', mode, arm).parent
             for mode in ('baseline', 'fee2', 'slip2', 'combined')
             for arm in ('poc_up_red', 'poc_priority')]
    assert len(set(paths)) == 8
