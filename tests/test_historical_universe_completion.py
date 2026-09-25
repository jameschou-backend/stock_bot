"""Regression checks for listing-date contraction, venue changes, and halt boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from skills.historical_universe_completion import (
    apply_completion, parse_tpex_halts, parse_twse_halts, resolve_completion,
    validate_exclusions)

ROOT = Path(__file__).resolve().parents[1]


def episode(sid='1780', start='2010-04-29', end=None, venue='TPEx', category='股票'):
    return dict(stock_id=sid, market=venue, start=start, end=end, category=category,
                start_evidence='current_official_ISIN')


def report(episodes=None, exclusions=None):
    return dict(episodes=episodes or [episode()], trading_exclusions=exclusions or [], coverage_end='2026-09-09')


def correction(sid='1780', old='2010-04-29', new='2026-06-17'):
    return dict(stock_id=sid, market='TPEx', start=new, replaces_snapshot_start=old)


def test_new_mainboard_listing_contracts_prior_emerging_period_without_mutating_base():
    base = report()
    fixed = apply_completion(base, [correction()], [])
    assert base['episodes'][0]['start'] == '2010-04-29'
    view = report(fixed)
    assert resolve_completion(view, '1780', '2026-06-16')['status'] == 'outside_verified_intervals'
    assert resolve_completion(view, '1780', '2026-06-17')['status'] == 'identified'
    assert resolve_completion(view, '1780', '2026-06-18')['status'] == 'identified'


def test_industry_record_can_move_back_to_verified_original_ipo():
    base = report([episode('3629', '2020-06-01')])
    fixed = apply_completion(base, [correction('3629', '2020-06-01', '2010-05-27')], [])
    assert fixed[0]['start'] == '2010-05-27'


@pytest.mark.parametrize('change', ['wrong_prior', 'already_reviewed', 'duplicate'])
def test_date_changes_require_exact_prior_episode(change):
    base, rows = report(), [correction()]
    if change == 'wrong_prior':
        rows[0]['replaces_snapshot_start'] = '2010-04-28'
    elif change == 'already_reviewed':
        base['episodes'][0]['start_evidence'] = 'reviewed_primary_archive'
    else:
        rows *= 2
    with pytest.raises(ValueError):
        apply_completion(base, rows, [])


def test_old_venue_open_halt_does_not_block_new_venue_or_replace_outside_status():
    episodes = [episode('1234', '2000-01-01', '2022-01-05'),
                episode('1234', '2022-01-05', venue='TWSE')]
    view = report(episodes, [dict(stock_id='1234', market='TPEx', start='2022-01-04', end=None, kind='trading_suspension')])
    assert resolve_completion(view, '1234', '2022-01-04')['tradable'] is False
    assert resolve_completion(view, '1234', '2022-01-05')['status'] == 'identified'
    assert resolve_completion(view, '1234', '1999-12-31')['status'] == 'outside_verified_intervals'


def test_pre_delisting_suspension_end_is_exclusive():
    view = report([episode('1507', '1989-11-09', '2022-04-21', 'TWSE')],
                  [dict(stock_id='1507', market='TWSE', start='2022-04-14', end='2022-04-21', kind='trading_suspension')])
    assert resolve_completion(view, '1507', '2022-04-13')['status'] == 'identified'
    assert resolve_completion(view, '1507', '2022-04-14')['status'] == 'official_trading_suspension'
    assert resolve_completion(view, '1507', '2022-04-20')['status'] == 'official_trading_suspension'
    assert resolve_completion(view, '1507', '2022-04-21')['status'] == 'outside_verified_intervals'


def test_reviewed_8080_intermediate_resumption_is_not_extended_to_next_halt():
    recipe = json.loads((ROOT / 'docs/evidence_historical_universe_completion_20260925.json').read_text())
    exclusions = [r for r in recipe['manual_exclusions'] if r['stock_id'] == '8080']
    view = report([episode('8080', '2004-05-10')], exclusions)
    assert resolve_completion(view, '8080', '2022-05-04')['status'] == 'official_trading_suspension'
    assert resolve_completion(view, '8080', '2022-05-05')['status'] == 'identified'
    assert resolve_completion(view, '8080', '2022-05-06')['status'] == 'identified'
    assert resolve_completion(view, '8080', '2023-10-10')['status'] == 'identified'
    assert resolve_completion(view, '8080', '2023-10-11')['status'] == 'official_trading_suspension'
    assert resolve_completion(view, '8080', '2024-01-24')['status'] == 'identified'


def test_reviewed_5364_managed_board_has_exact_restoration_boundary():
    recipe = json.loads((ROOT / 'docs/evidence_historical_universe_completion_20260925.json').read_text())
    rows = [r for r in recipe['manual_exclusions'] if r['stock_id'] == '5364']
    view = report([episode('5364', '1998-12-07')], rows)
    assert resolve_completion(view, '5364', '2013-08-25')['status'] == 'identified'
    assert resolve_completion(view, '5364', '2013-08-26')['status'] == 'not_general_board'
    assert resolve_completion(view, '5364', '2016-01-05')['status'] == 'not_general_board'
    assert resolve_completion(view, '5364', '2016-01-06')['status'] == 'identified'


def twse_payload():
    return dict(stat='OK', title='暫停交易證券 查詢範圍：全部上市證券 期間：111/01/01 到 111/12/31',
                fields=['編號', '證券代號', '證券名稱', '暫停交易日期', '暫停交易時間', '恢復交易日期', '恢復交易時間'],
                data=[[1, '3037', '欣興', '111/02/22', '8:00', '111/02/23', '8:00']])


def test_twse_halt_parser_checks_query_year_and_count():
    data = twse_payload()
    assert parse_twse_halts(data, 'fixture', '2026-09-09', 2022)[0]['end'] == '2022-02-23'
    with pytest.raises(ValueError, match='year/range'):
        parse_twse_halts(data, 'fixture', '2026-09-09', 2021)
    data['total'] = 2
    with pytest.raises(ValueError, match='Incomplete'):
        parse_twse_halts(data, 'fixture', '2026-09-09', 2022)


def test_tdr_omission_requires_verified_category_not_four_digit_inference():
    data = twse_payload()
    data['data'][0] = [1, '9188', '精熙-DR', '111/01/03', '9:00', '111/01/05', '8:00']
    with pytest.raises(ValueError, match='Intraday'):
        parse_twse_halts(data, 'fixture', '2026-09-09', 2022)
    assert parse_twse_halts(data, 'fixture', '2026-09-09', 2022, {'9188'}) == []


def tpex_payload():
    return dict(stat='ok', date='2022', tables=[dict(date='2022', totalCount=2,
        fields=['編號', '有價證券類別', '有價證券代號', '有價證券名稱', '暫停交易日期', '暫停交易時間', '恢復交易日期', '恢復交易時間'],
        data=[[1, '上櫃股票', '4120', '友華', '111/05/13', '8:00', '-', '-'],
              [2, '上櫃股票', '4120', '友華', '-', '-', '111/05/16', '8:00']])])


def test_tpex_pairs_distinct_rows_and_rejects_partial_or_wrong_market_table():
    payload = tpex_payload()
    rows = parse_tpex_halts([('fixture', payload)], '2026-09-09')
    assert rows[0]['start'] == '2022-05-13' and rows[0]['end'] == '2022-05-16'
    payload['tables'][0]['data'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        parse_tpex_halts([('fixture', payload)], '2026-09-09')
    payload = tpex_payload()
    payload['tables'][0]['data'][0][1] = '興櫃股票'
    with pytest.raises(ValueError, match='Non-mainboard'):
        parse_tpex_halts([('fixture', payload)], '2026-09-09')


def test_empty_or_duplicate_exclusion_is_rejected():
    row = dict(stock_id='1234', market='TWSE', start='2022-01-03', end='2022-01-03', kind='share_exchange')
    with pytest.raises(ValueError):
        validate_exclusions([row])
    row['end'] = '2022-01-04'
    with pytest.raises(ValueError, match='Duplicate'):
        validate_exclusions([row, deepcopy(row)])


def test_open_exclusion_is_not_extended_beyond_verified_cutoff():
    view = report([episode('4804', '2000-01-01')], [dict(stock_id='4804', market='TPEx',
        start='2026-04-14', end=None, kind='trading_suspension')])
    assert resolve_completion(view, '4804', '2026-09-09')['tradable'] is False
    assert resolve_completion(view, '4804', '2026-09-10')['tradable'] is None
    assert resolve_completion(view, '4804', '2026-09-10')['status'] == 'outside_verified_coverage'
    from scripts.audit_historical_universe_completion import audit_records
    view['coverage_start'] = '2021-01-01'
    for stamp in ('2020-12-31', '2026-09-10'):
        result = audit_records(view, [dict(stock_id='4804', date=stamp)])
        assert not result['passed']
        assert result['issues'][0]['status'] == 'outside_verified_coverage'


def test_official_halt_events_must_belong_to_their_queried_source_year():
    data = twse_payload()
    data['data'][0][3] = '110/12/31'
    with pytest.raises(ValueError, match='outside the queried source year'):
        parse_twse_halts(data, 'fixture', '2026-09-09', 2022)
    data = twse_payload()
    data['data'][0][5] = '112/01/03'
    assert parse_twse_halts(data, 'fixture', '2026-09-09', 2022)[0]['end'] == '2023-01-03'
    data = tpex_payload()
    data['tables'][0]['data'][0][4] = '110/12/31'
    with pytest.raises(ValueError, match='outside its source year'):
        parse_tpex_halts([('fixture', data)], '2026-09-09')


def test_offline_report_rejects_changed_seal_reconstruction_or_source(tmp_path, monkeypatch):
    import scripts.audit_historical_universe_completion as audit
    import scripts.audit_identity_continuation as evidence
    from scripts.research_exit_scenarios import sha

    source_path = tmp_path / 'official.html'
    source_path.write_text('exact archived response')
    report_path = tmp_path / 'report.json'
    data = dict(schema='fixture', source_sha256={'official.html': sha(source_path)})
    report_path.write_text(json.dumps(data))
    report_path.with_suffix('.sha256').write_text(sha(report_path))
    monkeypatch.setattr(evidence, 'ROOT', tmp_path)
    monkeypatch.setattr(audit, 'build', lambda: deepcopy(data))
    assert audit.verify_report(report_path) == data
    source_path.write_text('altered archive')
    with pytest.raises(ValueError, match='Evidence missing or changed'):
        audit.verify_report(report_path)
    source_path.write_text('exact archived response')
    monkeypatch.setattr(audit, 'build', lambda: dict(data, forged_passed=True))
    with pytest.raises(ValueError, match='offline reconstruction'):
        audit.verify_report(report_path)
    report_path.write_text(json.dumps(dict(data, forged_passed=True)))
    with pytest.raises(ValueError, match='hash differs'):
        audit.verify_report(report_path)


@pytest.mark.parametrize('change', ['status', 'url', 'hash'])
def test_primary_source_receipt_cannot_substitute_a_different_response(tmp_path, monkeypatch, change):
    import scripts.audit_historical_universe_completion as audit
    import scripts.audit_identity_continuation as evidence
    from scripts.research_exit_scenarios import sha

    raw = tmp_path / 'official.json'
    raw.write_text('{"stat":"ok"}')
    url = 'https://www.tpex.org.tw/www/zh-tw/bulletin/sprcHis?date=2022'
    receipt = dict(status=200, sha256=sha(raw), url=url)
    receipt[change if change != 'hash' else 'sha256'] = {
        'status': 428, 'url': 'https://www.tpex.org.tw/wrong-document', 'hash': '0' * 64}[change]
    meta = tmp_path / 'receipt.json'
    meta.write_text(json.dumps(receipt))
    row = dict(source_path='official.json', source_sha256=sha(raw), source_url=url,
               source_meta_path='receipt.json', source_meta_sha256=sha(meta), verification={'kind': 'json'})
    monkeypatch.setattr(evidence, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Primary origin, receipt, URL, or bytes differ'):
        audit.source(row, {})
