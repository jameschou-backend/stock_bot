from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import summarize_poc_broker_account as module


def fixture_case():
    days = [('2024-01-02', 1100.), ('2024-12-31', 990.),
            ('2025-12-31', 1188.), ('2026-10-02', 1069.2)]
    previous, peak, rows = 1000., 1000., []
    for day, nav in days:
        peak = max(peak, nav)
        rows.append(dict(date=day, nav=nav, opening_nav=previous, cash=nav,
                         market_value=0., receivable=0., cost=0., holdings=0,
                         stale_holdings=0, daily_return=nav/previous-1,
                         total_return=nav/1000-1, drawdown=nav/peak-1))
        previous = nav
    annual = [dict(year='2024', start_nav=1000., end_nav=990., profit=-10., total_return=-.01,
                   max_drawdown=-.1, partial_year=False),
              dict(year='2025', start_nav=990., end_nav=1188., profit=198., total_return=.2,
                   max_drawdown=0., partial_year=False),
              dict(year='2026', start_nav=1188., end_nav=1069.2, profit=-118.8, total_return=-.1,
                   max_drawdown=-.1, partial_year=True)]
    summary = dict(start='2024-01-02', end='2026-10-02', initial_cash=1000.,
                   final_nav=1069.2, profit=69.2, total_return=.0692,
                   max_drawdown=-.1, cash=1069.2, market_value=0., receivable=0.,
                   trading_days=4, trade_count=0, buy_count=0, sell_count=0,
                   stock_cohorts=0, minimum_cash=990.,
                   costs=dict(commission=0., tax=0., slippage=0., total_cost=0.), annual=annual)
    return dict(completed=True, summary=summary,
                account=dict(settings=dict(initial_cash=1000.), daily=rows, trades=[], cohorts=[]),
                broker_gate_decisions=[], entry_gate_decisions=[], profile_queries=[])


def test_annual_chaining_mdd_and_final_nav_are_recomputed():
    got = module.account_metrics(fixture_case())
    assert [r['total_return'] for r in got['annual']] == pytest.approx([-.01, .2, -.1])
    assert got['annual'][1]['start_nav'] == 990.
    assert got['total_return'] == pytest.approx(.0692)
    assert got['max_drawdown'] == pytest.approx(-.1)
    assert got['funded_events'] == 0


@pytest.mark.parametrize('field,value', [('daily_return', .01), ('drawdown', 0.), ('opening_nav', 1000.), ('cost', 3.)])
def test_tampered_daily_evidence_stops(field, value):
    case = fixture_case()
    case['account']['daily'][2][field] = value
    if field == 'drawdown':
        case['account']['daily'][1][field] = value
    with pytest.raises(ValueError, match='mismatch'):
        module.account_metrics(case)


def test_incomplete_account_has_no_performance_even_if_journal_exists():
    case = fixture_case(); case['completed'] = False
    with pytest.raises(ValueError, match='Incomplete'):
        module.account_metrics(case)


def test_child_fills_costs_and_funded_cohorts_are_different_counts():
    case = fixture_case()
    trades = [dict(date='2024-01-02', side='buy', qty=1000, event_id='a', channel='board',
                   commission=2, tax=0, slippage=3, total_cost=5),
              dict(date='2024-01-02', side='buy', qty=1, event_id='a', channel='odd',
                   commission=1, tax=0, slippage=1, total_cost=2)]
    case['account']['trades'] = trades
    case['account']['cohorts'] = [dict(event_id='a')]
    case['account']['daily'][0]['cost'] = 7
    case['summary'].update(trade_count=2, buy_count=2, stock_cohorts=1,
                           costs=dict(commission=3, tax=0, slippage=4, total_cost=7))
    got = module.account_metrics(case)
    assert (got['trade_count'], got['funded_events']) == (2, 1)
    assert got['costs']['total_cost'] == 7
    assert got['child_fills_by_channel'] == dict(board=1, odd=1)
    case['account']['trades'][0]['qty'] = 1000.5
    with pytest.raises(ValueError, match='Noninteger'):
        module.account_metrics(case)


def diagnostic(eid='old', known=True, persistent=False, peak=.25, status='closed'):
    return dict(event_id=eid, stock_id='2330', signal_date='2024-01-02', name='test',
                branch=dict(known=known, concentrated_directional=True),
                persistence5=dict(known=known, passed=persistent),
                outcome=dict(status=status, peak_close_return=peak, net_return=-.1, exit_date='2024-01-30'))


def decision(known=True, kept=False, source='old', persistent=False):
    return dict(event_id='new', stock_id='2330', signal_date='2024-01-02', entry_date='2024-01-03',
                known5=known, kept=kept, source_event_id=source,
                persistent5=persistent if known else None, combined=persistent if known else None,
                reason='branch_condition_failed' if known else 'outside_matched_coverage',
                unknown_reason=None if known else 'missing')


def test_excluded_peak_winner_is_separate_from_account_pnl_and_uses_stock_date():
    case = fixture_case(); case['broker_gate_decisions'] = [decision()]
    got = module.gate_analysis('poc_persist_guard', case, [diagnostic()], {'new'})
    assert got['diagnostic_population']['excluded_closed_peak20'] == 1
    assert got['excluded_big_winners'][0]['diagnostic_net_return'] == -.1
    assert got['excluded_big_winners'][0]['baseline_funded'] is True
    assert 'profit' not in got
    assert got['all']['known_condition_failed'] == 1


def test_open_diagnostic_not_promoted_to_closed_big_winner():
    case = fixture_case(); case['broker_gate_decisions'] = [decision()]
    got = module.gate_analysis('poc_persist_guard', case, [diagnostic(status='open')], set())
    assert got['diagnostic_population']['statuses'] == {'open': 1}
    assert got['diagnostic_population']['excluded_closed_peak20'] == 0


def test_unknown_never_false_and_policy_differs_between_guard_and_matched():
    case = fixture_case()
    row = decision(known=False, kept=True, source=None)
    row['reason'] = 'unknown_kept_by_guard_policy'
    case['broker_gate_decisions'] = [row]
    got = module.gate_analysis('poc_persist_guard', case, [], set())
    assert got['all']['unknown_kept'] == 1
    with pytest.raises(ValueError, match='unknown/condition'):
        module.gate_analysis('poc_known5_filter', case, [], set())
    row['kept'] = False; row['reason'] = 'outside_matched_coverage'
    got = module.gate_analysis('poc_known5_filter', case, [], set())
    assert got['all']['excluded'] == 1
    assert got['all']['known_condition_failed'] == 0


def test_diagnostic_identity_cannot_be_joined_by_stock_only():
    case = fixture_case(); case['broker_gate_decisions'] = [decision()]
    wrong = diagnostic(); wrong['signal_date'] = '2024-01-01'
    with pytest.raises(ValueError, match='join'):
        module.gate_analysis('poc_persist_guard', case, [wrong], set())


def test_future_outcome_change_does_not_change_coverage_or_rejection():
    case = fixture_case(); case['broker_gate_decisions'] = [decision()]
    old = module.gate_analysis('poc_persist_guard', case, [diagnostic()], set())
    other = diagnostic(peak=-.1); other['outcome']['net_return'] = -1.
    new = module.gate_analysis('poc_persist_guard', case, [other], set())
    assert old['all'] == new['all']
    assert old['funded'] == new['funded']
    assert old['diagnostic_population']['excluded_closed_peak20'] == 1
    assert new['diagnostic_population']['excluded_closed_peak20'] == 0


def test_funded_rejected_or_unlisted_event_fails():
    case = fixture_case(); case['broker_gate_decisions'] = [decision()]
    case['account']['trades'] = [dict(side='buy', event_id='new')]
    with pytest.raises(ValueError, match='Funded candidate'):
        module.gate_analysis('poc_persist_guard', case, [diagnostic()], set())
    case['account']['trades'][0]['event_id'] = 'not-listed'
    with pytest.raises(ValueError, match='lacks branch'):
        module.gate_analysis('poc_persist_guard', case, [diagnostic()], set())


def test_source_snapshots_are_explicit_and_only_for_code(tmp_path):
    (tmp_path/'code.py').write_text('new')
    (tmp_path/'snapshot.py').write_text('old')
    digest = module.sha(tmp_path/'snapshot.py')
    sources = module.Sources(tmp_path)
    report = dict(source_sha256={'code.py': digest},
                  source_snapshots={'code.py': dict(path='snapshot.py', sha256=digest)})
    sources.closure(report, 'r')
    assert sources.used == {'snapshot.py': digest}
    assert sources.snapshot_resolutions[0]['original'] == 'code.py'
    (tmp_path/'data.json').write_text('new')
    report = dict(source_sha256={'data.json': digest},
                  source_snapshots={'data.json': dict(path='snapshot.py', sha256=digest)})
    with pytest.raises(ValueError, match='hash mismatch'):
        sources.closure(report, 'r')


def test_cached_verification_rechecks_changed_bytes(tmp_path):
    path = tmp_path/'data'; path.write_text('old')
    sources = module.Sources(tmp_path); digest = module.sha(path)
    sources.bind('data', digest); path.write_text('different-size')
    with pytest.raises(ValueError, match='hash mismatch'):
        sources.bind('data', digest)
    with pytest.raises(ValueError):
        sources.path('../outside')


def test_comparison_is_percentage_points_not_return_ratio():
    base = module.account_metrics(fixture_case())
    changed = deepcopy(base); changed['total_return'] += .10; changed['final_nav'] += 100
    got = module.compare_metrics(changed, base, base)
    assert got['poc_red']['total_return_percentage_points'] == pytest.approx(10)
    assert got['known5_control'] is None


def make_bundle(tmp_path, monkeypatch, *, incomplete=False):
    def put(name, value):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return module.sha(path)
    case = fixture_case()
    # Same hand-selected NAV path with the study's fixed one-million capital.
    for row in case['account']['daily']:
        for key in ('nav', 'opening_nav', 'cash', 'market_value', 'receivable', 'cost'):
            row[key] *= 1000
    case['account']['settings']['initial_cash'] *= 1000
    for key in ('initial_cash', 'final_nav', 'profit', 'cash', 'market_value', 'receivable', 'minimum_cash'):
        case['summary'][key] *= 1000
    for row in case['summary']['annual']:
        for key in ('start_nav', 'end_nav', 'profit'):
            row[key] *= 1000
    source = put('input.json', {'immutable': True})
    base_sha = put('base.json', case)
    monkeypatch.setattr(module, 'BASE_CASE', 'base.json')
    monkeypatch.setattr(module, 'BASE_SHA', base_sha)
    monkeypatch.setattr(module, 'DIAG', 'diag')
    rows_sha = put('diag/rows.json', [])
    diag_sha = put('diag/report.json', dict(rows_sha256=rows_sha, source_sha256={'input.json': source}))
    monkeypatch.setattr(module, 'DIAG_REPORT_SHA', diag_sha)
    put('scripts/summarize_poc_broker_account.py', {'helper': 'fixture'})
    put('tests/test_summarize_poc_broker_account.py', {'test': 'fixture'})
    benchmark = dict(path='base.json', sha256=base_sha, summary=case['summary'])
    cases = {}
    for arm in module.ARMS:
        value = deepcopy(case)
        if incomplete and arm != 'poc_red':
            value = dict(completed=False, summary=None, reason='missing_raw', partial_journal=case['account'])
        digest = put('run/'+arm+'.json', value)
        cases[arm] = dict(path='run/'+arm+'.json', sha256=digest,
                          completed=value['completed'], summary=value['summary'])
    report = dict(start=module.START, end=module.END, initial_cash=1_000_000,
                   source_sha256={'input.json': source}, benchmark=benchmark, cases=cases)
    digest = put('run/report.json', report)
    (tmp_path/'run/report.sha256').write_text(digest)
    return tmp_path/'run/report.json'


def test_bound_report_end_to_end_is_offline_deterministic(tmp_path, monkeypatch):
    report = make_bundle(tmp_path, monkeypatch)
    first = module.build_report([report], tmp_path)
    second = module.build_report([report], tmp_path)
    assert first == second
    assert first['all_six_completed'] is True
    assert first['arms']['poc_red']['comparison']['poc_red']['total_return_percentage_points'] == 0
    a = module.save_report(first, tmp_path/'analysis-a')
    b = module.save_report(second, tmp_path/'analysis-b')
    assert a == b
    assert len(module.table_rows(first)) == 7
    with pytest.raises(ValueError, match='new empty'):
        module.save_report(first, tmp_path/'analysis-a')


def test_bound_incomplete_report_does_not_publish_partial_results(tmp_path, monkeypatch):
    report = make_bundle(tmp_path, monkeypatch, incomplete=True)
    result = module.build_report([report], tmp_path)
    assert result['all_six_completed'] is False
    assert result['arms']['poc_known5_control']['metrics'] is None
    assert result['arms']['poc_red']['comparison']['known5_control'] is None
    row = next(r for r in module.table_rows(result) if r['arm'] == 'poc_known5_control')
    assert row['reason'] == 'missing_raw'
    assert 'total_return_pct' not in row


def test_case_hash_tamper_stops_report(tmp_path, monkeypatch):
    report = make_bundle(tmp_path, monkeypatch)
    path = tmp_path/'run/poc_known5_filter.json'
    value = json.loads(path.read_text()); value['summary']['final_nav'] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Source hash mismatch'):
        module.build_report([report], tmp_path)
