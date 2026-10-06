import hashlib
import json

import pytest

from app import research_terminal_archive as archive


def write_sealed(root, identifier, value):
    path = root / 'artifacts' / 'forward_simulation' / (identifier + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_archive_missing_or_corrupt_evidence_never_reuses_returns(tmp_path):
    identifier = 'poc_range_first_account_20261005'
    path = write_sealed(tmp_path, identifier, dict(live_qualified=False, start='2024-01-02', end='2026-10-02',
        cases={'poc_range50_all': dict(completed=True, summary={'total_return': 2.0})}))
    result = archive.overview(tmp_path)
    row = next(x for x in result['items'] if x['id'] == identifier)
    assert row['status'] == 'verified_publication'
    assert row['cases'][0]['total_return'] == 2.0
    assert '事前未知' in row['execution_note']
    assert '首日' in row['note']
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name)
    assert archive.publication_bytes(identifier, tmp_path) == path.read_bytes()
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest() + '  unrelated.json')
    with pytest.raises(ValueError, match='指紋'):
        archive.publication_bytes(identifier, tmp_path)
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_text(path.read_text() + ' ')
    row = next(x for x in archive.overview(tmp_path)['items'] if x['id'] == identifier)
    assert row['status'] == 'unavailable'
    assert row['cases'] == []
    with pytest.raises(ValueError, match='未知研究紀錄'):
        archive.publication_bytes('../../.env', tmp_path)


def test_case_detail_verifies_account_summary_and_file_before_exposing_curve(tmp_path):
    identifier, case_id = 'poc_range_first_account_20261005', 'poc_range50_all'
    summary = {'total_return': .1, 'final_nav': 1100000}
    path = tmp_path / 'case.json'
    report = dict(completed=True, live_qualified=False, summary=summary,
        account={'daily': [{'date': '2024-01-02', 'nav': 1100000}],
                 'trades': [{'date': '2024-01-02', 'stock_id': '2330', 'side': 'buy', 'qty': 10}]})
    path.write_text(json.dumps(report))
    publication = dict(live_qualified=False, initial_cash=1000000, cases={case_id: dict(
        path='case.json', sha256=hashlib.sha256(path.read_bytes()).hexdigest(), summary=summary)})
    write_sealed(tmp_path, identifier, publication)
    result = archive.case_detail(identifier, case_id, tmp_path)
    assert result['daily'][0]['nav'] == 1100000
    assert result['trades'][0]['stock_id'] == '2330'
    assert result['actual_fill_verified'] is False
    report['summary'] = {'total_return': 9}
    path.write_text(json.dumps(report))
    publication['cases'][case_id]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_sealed(tmp_path, identifier, publication)
    with pytest.raises(ValueError, match='摘要不同'):
        archive.case_detail(identifier, case_id, tmp_path)
    with pytest.raises(ValueError, match='曲線'):
        archive.case_detail(identifier, '../case', tmp_path)


def test_nested_scanner_period_and_benchmark_account_label(tmp_path):
    assert archive._period({'summary': {'start': '2026-09-29', 'end': '2026-10-05'}}) == {
        'start': '2026-09-29', 'end': '2026-10-05'}
    assert archive._period({'study': {'start': '2024-01-02', 'end': '2026-10-02'}})['start'] == '2024-01-02'
    summary = {'total_return': .1}
    path = tmp_path / 'benchmark.json'
    path.write_text(json.dumps(dict(completed=True, live_qualified=False, summary=summary,
        account={'daily': [], 'trades': [{'slippage': 20, 'gross': 2000, 'cash_change': -2020}]})))
    write_sealed(tmp_path, 'poc_range_first_account_20261005', dict(live_qualified=False, initial_cash=1000000,
        cases={'benchmark_range50': dict(path=path.name, summary=summary,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())}))
    result = archive.case_detail('poc_range_first_account_20261005', 'benchmark_range50', tmp_path)
    assert result['position_count'] == 1
    assert result['note'].startswith('0050 基準帳戶')
    assert result['trades'][0]['slippage'] == 20
    assert result['trades'][0]['cash_change'] == -2020
