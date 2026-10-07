from copy import deepcopy

from scripts.summarize_strategy_account_comparison import (
    LABELS, PARITY_FIELDS, describe_benchmark, describe_cohorts, markdown, parity,
    pending_cash_claims,
)


def test_closed_win_rate_excludes_open_positions_and_unpaid_receivables():
    events = [dict(event_id=k, stock_id=s, name=s, entry_date='2024-01-02', exit_date=None)
              for k, s in [('win', '1101'), ('loss', '1102'), ('open', '1103'), ('owed', '1104')]]
    ledger = [dict(kind='buy', event_id=e['event_id'], stock_id=e['stock_id'], cash_change=-100.) for e in events]
    ledger += [dict(kind='sell', event_id=k, stock_id=s, cash_change=c)
               for k, s, c in [('win', '1101', 130.), ('loss', '1102', 80.), ('owed', '1104', 120.)]]
    rights = [dict(kind='cash', event_id='owed', stock_id='1104', amount=10.)]
    value = dict(account=dict(cohorts=events, cash_ledger=ledger, corporate_actions=[], receivables=rights),
                 summary=dict(initial_cash=1000., cash=930., market_value=140., receivable=10., profit=80.,
                              final_holdings=[dict(event_id='open', stock_id='1103', market_value=140.)],
                              final_receivables=rights))
    metrics = describe_cohorts(value)
    assert metrics['funded'] == 4 and metrics['settled'] == 2 and metrics['unresolved'] == 2
    assert metrics['win_rate'] == .5 and metrics['profit_factor'] == 1.5


def test_parity_requires_full_journal_not_just_equal_final_nav():
    old = dict(completed=True, summary={'final_nav': 1200}, account={k: [] for k in PARITY_FIELDS})
    assert parity(old, deepcopy(old))['all_exact']
    changed = deepcopy(old)
    changed['account']['trades'] = [dict(date='2024-01-03')]
    assert not parity(old, changed)['all_exact']
    assert parity(old, {'completed': False})['reason'] == 'incomplete_account'


def test_buy_and_hold_is_not_a_closed_trade_win_rate():
    value = dict(account=dict(cohorts=[], trades=[dict(side='buy', event_id='benchmark'),
                                                 dict(side='buy', event_id='benchmark')]))
    result = describe_benchmark(value)
    assert result['funded'] == 1 and result['settled'] == 0 and result['win_rate'] is None
    assert result['cohort_profit_reconciled'] is False


def test_failed_arm_remains_visible_without_partial_period_return():
    value = dict(cases={arm: dict(label=label, completed=False, reason='missing necessary source',
                                 last_date='2024-01-10') for arm, label in LABELS.items()},
                 original_poc_parity={'all_exact': False}, source_report={'path': 'report.json', 'sha256': 'abc'})
    text = markdown(value)
    assert text.count('| 未完成 | — | — | — | — | — |') == 8
    assert 'missing necessary source' in text
    assert 'live_qualified=false' in text
    assert '2024/1/2～2026/10/2' in text


def test_undated_gross_claim_is_not_presented_as_confirmed_net_cash():
    value = dict(summary=dict(initial_cash=1000., final_nav=1207.,
                             final_receivables=[dict(kind='cash', amount=7., pay_date=None),
                                                dict(kind='cash', amount=10., pay_date='2026-10-15')]))
    original = deepcopy(value)
    result = pending_cash_claims(value)
    assert result['gross_unavailable_cash'] == 7.
    assert result['available_cash_increment'] == 0
    assert result['has_unconfirmed_payment_claims'] is True
    assert result['nav_if_these_claims_pay_zero'] == 1200.
    assert result['nav_using_recorded_gross_claims'] == 1207.
    assert value == original
