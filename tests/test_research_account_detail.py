from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from app.research_account_detail import load_comparison, scenario_label
from scripts.replay_million import summarize


@pytest.fixture
def package(tmp_path):
    days = [str(d.date()) for d in pd.bdate_range('2022-01-03', '2023-04-12')]
    rows = [dict(date=d, opening_nav=1_000_000., nav=1_000_000., cash=1_000_000.,
        market_value=0., receivable=0., daily_return=0., total_return=0., drawdown=0.,
        stale_holdings=0) for d in days]
    account = dict(settings=dict(initial_cash=1_000_000, commission=.001425, minimum_fee=20,
        slippage=.0045, participation=.01, odd_participation=.01), daily=rows, trades=[],
        cohorts=[], holdings=[], orders=[], receivables=[])
    case = dict(completed=True, live_qualified=False, config=dict(factor_mask=0, benchmark=False),
        account=account, summary=summarize(account))
    benchmark = deepcopy(case); benchmark['config']['benchmark'] = True
    def save(name, value):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def publication():
        ref = save('benchmark.json', benchmark)
        original = save('original.json', dict(cases={'benchmark_control': dict(result=ref)}))
        case_ref = save('case.json', case)
        return dict(start=days[0], end=days[-1], initial_cash=1_000_000,
            original_publication=original, cases={'rule_0': dict(completed=True,
                config=case['config'], summary=case['summary'], result=case_ref,
                metrics=dict(benchmark_return=0.))})
    return tmp_path, case, benchmark, publication


def test_exact_account_and_cost_matched_benchmark(package):
    root, case, benchmark, publish = package
    a, b = load_comparison(publish(), 'rule_0', root)
    assert a == case['account'] and b == benchmark['account']
    assert scenario_label(7) == '滑價加倍、進場多晚一天、出場多晚一天'


@pytest.mark.parametrize('mutation', ['cash', 'calendar', 'summary', 'cost', 'future_signal'])
def test_re_signed_but_inconsistent_accounts_are_rejected(package, mutation):
    root, case, benchmark, publish = package
    if mutation == 'cash':
        case['account']['daily'][10]['opening_nav'] += 1000
    elif mutation == 'calendar':
        benchmark['account']['daily'].pop(10)
        benchmark['summary'] = summarize(benchmark['account'])
    elif mutation == 'summary':
        case['summary']['final_nav'] += 1000
    elif mutation == 'cost':
        benchmark['account']['settings']['slippage'] = .009
    else:
        case['account']['trades'].append(dict(side='buy', date='2022-01-04', signal_date='2022-01-04',
            commission=0, tax=0, slippage=0, total_cost=0))
        case['summary'] = summarize(case['account'])
    with pytest.raises(ValueError):
        load_comparison(publish(), 'rule_0', root)


def test_changed_file_or_incomplete_account_cannot_be_displayed(package):
    root, _, _, publish = package
    report = publish()
    (root / 'case.json').write_text('{}')
    with pytest.raises(ValueError, match='已變動'):
        load_comparison(report, 'rule_0', root)
    report = publish(); report['cases']['rule_0']['completed'] = False
    with pytest.raises(ValueError, match='未完成'):
        load_comparison(report, 'rule_0', root)
