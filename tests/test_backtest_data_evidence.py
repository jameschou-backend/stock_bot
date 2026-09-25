import copy
import json
from pathlib import Path

import pytest

from skills.backtest_data_evidence import (
    case_evidence, channel_evidence, checked, digest, historical_value_available,
    inspect_tape, parse_twse_industry_change, requirements, verify_report)


def case(*, board_only=False, completed=True):
    account = dict(orders=[dict(stock_id='2330', date='2022-01-04', channel='board',
        side='buy', requested_qty=1000, filled_qty=0, failure='limit_blocked')],
        daily=[dict(date='2022-01-04')], holdings=[], trades=[])
    return dict(config=dict(board_only=board_only, benchmark=False), completed=completed,
                **{'account' if completed else 'partial_account': account})


def test_unfilled_orders_and_both_sides_need_one_full_session():
    result = case()
    result['account']['orders'].append(dict(result['account']['orders'][0], side='sell', filled_qty=1000))
    rows, excluded = requirements(result)
    assert excluded == {}
    assert len(rows) == 1 and rows[0]['sides'] == ['buy', 'sell']
    assert rows[0]['order_count'] == 2 and rows[0]['unfilled_order_count'] == 1
    assert channel_evidence(rows, {}, 'board')['missing_sessions'] == 1


def test_resource_rejection_is_not_an_order_even_with_planned_quantity():
    result = case()
    result['account']['orders'].append(dict(result['account']['orders'][0],
        channel='event', failure='resource_slots_locked', requested_qty=7000))
    rows, excluded = requirements(result)
    assert len(rows) == 1 and excluded['no_order_event'] == 1


def test_board_only_odd_remainder_is_not_a_requested_odd_execution():
    result = case(board_only=True)
    odd = dict(result['account']['orders'][0], channel='odd', requested_qty=150,
               failure='board_only_below_one_lot')
    result['account']['orders'].append(odd)
    rows, excluded = requirements(result)
    assert excluded['policy_forbidden_odd_remainder'] == 1
    assert channel_evidence(rows, {}, 'odd')['status'] == 'not_required_by_policy'
    odd['filled_qty'] = 1
    with pytest.raises(ValueError, match='actual odd-lot'):
        requirements(result)


def test_mixed_odd_attempt_and_partial_path_remain_blocked():
    result = case(completed=False)
    result['partial_account']['orders'].append(dict(result['partial_account']['orders'][0],
        channel='odd', requested_qty=100, failure='no_quote'))
    pit = dict(components=[dict(code='publication', status='blocked')], episodes=[dict(
        stock_id='2330', start='1994-09-05', end=None, market='TWSE', category='股票')])
    output = case_evidence('example', result, {}, pit)
    assert output['ordinary']['missing_sessions'] == 1
    assert output['odd_lot']['missing_sessions'] == 1
    assert not output['complete_path'] and not output['all_possible_paths_covered']
    assert 'unobserved_path_after_blocked_session' in output['missing_codes']
    assert output['pit']['components'][0]['issue_count'] == 0
    assert output['strict_data_ready'] is False and 'total_return' not in output


def test_current_case_identity_cannot_borrow_an_old_case_pass():
    result = case()
    pit = dict(components=[], episodes=[])
    output = case_evidence('newcase', result, {}, pit)
    assert output['pit']['components'][0]['issue_count'] == 1
    assert 'case_dated_market_identity' in output['missing_codes']


def test_benchmark_does_not_invent_an_industry_selection_dependency():
    result = case()
    result['config']['benchmark'] = True
    result['account']['orders'][0]['stock_id'] = '0050'
    pit = dict(components=[dict(code='historical_industry_membership', status='blocked')],
               episodes=[dict(stock_id='0050', start='2003-06-30', end=None,
                              market='TWSE', category='ETF')])
    output = case_evidence('benchmark', result, {}, pit)
    assert 'historical_industry_membership' not in output['missing_codes']
    component = output['pit']['components'][1]
    assert component['status'] == 'not_required_by_case'
    assert component['global_evidence_status'] == 'blocked'
    assert output['pit']['complete_historical_industry_membership'] is False


def tape_file(tmp_path, **overrides):
    document = dict(schema='normalized_auction_v1', stock_id='2330', date='2022-01-04',
        market='TWSE', timezone='Asia/Taipei', channel='odd', quantity_unit='shares',
        price_unit='TWD_cents', session_complete=True,
        rows=[dict(record_type='trade', time_us=33060000000, price_cents=60000, shares=10)],
        source_url='https://www.twse.com.tw/example', synthetic=False)
    document.update(overrides)
    path = tmp_path / 'tape.json'
    path.write_text(json.dumps(document))
    item = dict(path='tape.json', sha256=digest(path), format='normalized_auction_v1',
                stock_id='2330', date='2022-01-04', market='TWSE', channel='odd')
    return item


def test_declared_complete_format_is_not_independent_source_certification(tmp_path):
    item = tape_file(tmp_path)
    refs = {}
    observed = inspect_tape(item, tmp_path, refs)
    assert observed['format_valid'] and observed['actual_trade_rows'] == 1
    assert not observed['source_authenticated'] and not observed['accepted_for_strict_replay']
    assert not observed['independently_verified_session_complete']
    rows = [dict(date='2022-01-04', stock_id='2330', channel='odd', sides=['buy'])]
    key = ('2022-01-04', '2330', 'odd')
    evidence = channel_evidence(rows, {key:observed}, 'odd')
    assert evidence['missing_sessions'] == 0 and evidence['unverified_sessions'] == 1
    observed['accepted_for_strict_replay'] = True
    with pytest.raises(ValueError, match='Unsupported source certification'):
        channel_evidence(rows, {key:observed}, 'odd')


@pytest.mark.parametrize('changes,match', [
    ({'synthetic':True}, 'Synthetic'),
    ({'session_complete':False}, 'complete session'),
    ({'quantity_unit':'lots'}, 'units'),
    ({'rows':[]}, 'no-trade'),
])
def test_bad_or_incomplete_tape_never_satisfies_evidence(tmp_path, changes, match):
    item = tape_file(tmp_path, **changes)
    with pytest.raises((ValueError, RuntimeError), match=match):
        inspect_tape(item, tmp_path, {})


def test_importer_hash_and_identity_mismatch_are_errors(tmp_path):
    item = tape_file(tmp_path)
    item['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        inspect_tape(item, tmp_path, {})
    item['sha256'] = digest(tmp_path / 'tape.json')
    item['date'] = '2022-01-05'
    with pytest.raises(ValueError, match='identity mismatch'):
        inspect_tape(item, tmp_path, {})


def test_first_observation_and_future_revision_cannot_be_backdated():
    assert not historical_value_available(dict(create_time='2026-04-21'), '2022-01-04T09:00:00+08:00')
    row = dict(official_published_at='2022-01-03T18:00:00+08:00',
        version_available_at='2022-03-01T18:00:00+08:00', payload_sha256='a' * 64,
        publication_source_sha256='b' * 64, version_id='revision-2')
    assert not historical_value_available(row, '2022-01-04T09:00:00+08:00')
    assert historical_value_available(row, '2022-03-02T09:00:00+08:00')
    row['version_available_at'] = '2022-03-01T18:00:00'
    with pytest.raises(ValueError, match='Timezone-aware'):
        historical_value_available(row, '2022-03-02T09:00:00+08:00')


def test_report_consumer_rejects_source_and_receipt_mutation(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('original')
    report = tmp_path / 'report.json'
    report.write_text(json.dumps(dict(schema='backtest_data_completion_v1', live_qualified=False, strict_data_ready=False,
        input_sha256={'source.txt':digest(source)}, code_sha256={})))
    report.with_suffix('.sha256').write_text(digest(report))
    assert verify_report(report, tmp_path)['live_qualified'] is False
    source.write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_report(report, tmp_path)
    source.write_text('original')
    report.write_text(report.read_text() + ' ')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_report(report, tmp_path)


def test_evidence_cannot_escape_its_declared_root(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('outside')
    root = tmp_path / 'root'
    root.mkdir()
    with pytest.raises(ValueError, match='escapes'):
        checked(source, digest(source), {}, root)


def test_industry_parser_requires_whole_event_and_handles_wrapped_old_category():
    lines = ['調整至「新類別」：共計47家']
    for n in range(46):
        lines.append(f'{n+1} {1000+n} 公司{n} 其他')
    lines += ['電腦及週邊', '47 2442 新美齊股份有限公司', '設備業']
    rows = parse_twse_industry_change('\n'.join(lines))
    assert rows[-1]['old_industry'] == '電腦及週邊設備業'
    assert rows[-1]['official_publication_time'] is None
    assert rows[-1]['prior_interval_start'] is None
    assert rows[-1]['finmind_supply_chain_membership_equivalent'] is False
    with pytest.raises(ValueError, match='incomplete or duplicated'):
        parse_twse_industry_change('\n'.join(lines[:-3]))


def test_tick_request_plan_deduplicates_cases_and_requires_dated_market():
    from scripts.prepare_backtest_board_ticks import requests_from_report
    request = dict(date='2022-01-04', stock_id='2330')
    report = dict(cases={name:dict(ordinary=dict(missing=[request])) for name in ('a', 'b')})
    episodes = [dict(stock_id='2330', start='1994-09-05', end=None, market='TWSE', category='股票')]
    assert requests_from_report(report, episodes) == [dict(request, market='TWSE')]
    with pytest.raises(ValueError, match='Missing dated market'):
        requests_from_report(report, [])


def test_tick_budget_rejects_oversized_plan_before_fetch(tmp_path, monkeypatch):
    from scripts import prepare_backtest_board_ticks as preparer
    root = tmp_path
    source = root / '.cache/listing-continuation-20260924/report.json'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(dict(episodes=[])))
    report = dict(input_sha256={str(source.relative_to(root)):digest(source)})
    monkeypatch.setattr(preparer, 'ROOT', root)
    monkeypatch.setattr(preparer, 'verify_report', lambda *args: report)
    monkeypatch.setattr(preparer, 'requests_from_report', lambda *args: [{}, {}])
    calls = []
    monkeypatch.setattr(preparer, 'TickCache', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match='exceed explicit budget'):
        preparer.prepare(root/'plan.json', root/'.cache/new', maximum=1, fetch=True)
    assert not calls and not (root/'.cache/new').exists()


def test_failed_tick_attempt_is_durable_and_never_implicitly_retried(tmp_path, monkeypatch):
    from scripts import prepare_backtest_board_ticks as preparer
    from app.finmind import FinMindError
    root = tmp_path
    for name in ('scripts/prepare_backtest_board_ticks.py', 'scripts/research_intraday_limit.py',
                 'skills/intraday_limit_replay.py', 'app/finmind.py'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture code')
    plan = root/'plan.json'
    plan.write_text('{}')
    source = root/'.cache/listing-continuation-20260924/report.json'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(dict(episodes=[])))
    report = dict(input_sha256={str(source.relative_to(root)):digest(source)})
    monkeypatch.setattr(preparer, 'ROOT', root)
    monkeypatch.setattr(preparer, 'verify_report', lambda *args: report)
    monkeypatch.setattr(preparer, 'requests_from_report', lambda *args: [
        dict(date='2022-01-04', stock_id='2330', market='TWSE')])
    calls = []
    class FailingCache:
        def __init__(self, cache_root, **kwargs):
            self.root, self.calls = Path(cache_root), 0
            self.root.mkdir(parents=True, exist_ok=True)
        def get(self, *args):
            calls.append(args)
            self.calls += 1
            preparer.write(self.root/'budget.json', dict(reserved=1))
            raise FinMindError('simulated provider failure')
    monkeypatch.setattr(preparer, 'TickCache', FailingCache)
    output = root/'.cache/new'
    with pytest.raises(RuntimeError, match='Tick preparation stopped: FinMindError'):
        preparer.prepare(plan, output, maximum=1, fetch=True)
    receipt = json.loads((output/'summary.json').read_text())
    assert receipt['all_completed'] is False and receipt['adapter_attempts_lifetime'] == 1
    with pytest.raises(RuntimeError, match='requires review before retry'):
        preparer.prepare(plan, output, maximum=1, fetch=True)
    assert len(calls) == 1
