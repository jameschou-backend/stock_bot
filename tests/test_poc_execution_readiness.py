from copy import deepcopy
import json
from pathlib import Path
import pytest

from app.poc_execution_readiness import validate, IDENTIFIER
from app.research_terminal_archive import overview

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report():
    return json.loads((ROOT/'artifacts/forward_simulation'/(IDENTIFIER+'.json')).read_text())


def test_sealed_execution_gaps_remain_distinct_from_historical_returns(report):
    validate(report, report['source_publication']['sha256'])
    assert report['rerun_accounts'] == 0
    assert report['finmind']['actual_api_requests'] == 3
    assert report['odd']['required_coordinates'] == 557
    assert len(report['cases']) == 12


@pytest.mark.parametrize('field', ['whole_account_replayed','execution_validation_completed',
    'causal_fill_verified','paper_enabled','actual_fill_verified','live_qualified'])
def test_missing_data_cannot_be_promoted(report, field):
    report[field] = True
    with pytest.raises(ValueError): validate(report)


@pytest.mark.parametrize('mutation', ['return','excess','benchmark','missing_case','missing_data','different_source'])
def test_no_invented_return_or_mismatched_controls(report, mutation):
    row = report['cases']['poc_priority:original:combined']
    if mutation == 'return': row['executable_total_return'] = row['proxy_total_return']
    if mutation == 'excess': row['executable_excess_return'] = 0
    if mutation == 'benchmark': row['benchmark_proxy_return'] = report['cases']['benchmark:original:baseline']['proxy_total_return']
    if mutation == 'missing_case': del report['cases']['benchmark:original:fee2']
    if mutation == 'missing_data': report['odd']['verified_chronological_coordinates'] = 557
    with pytest.raises(ValueError):
        validate(report, '0'*64 if mutation=='different_source' else report['source_publication']['sha256'])


def test_missing_or_corrupt_new_audit_does_not_hide_completed_accounts(tmp_path, report):
    folder = tmp_path/'artifacts/forward_simulation'
    folder.mkdir(parents=True)
    source = ROOT/'artifacts/forward_simulation'
    for suffix in ('.json','.sha256'):
        name = 'poc_followup_20261014'+suffix
        (folder/name).write_bytes((source/name).read_bytes())
    (folder/(IDENTIFIER+'.json')).write_text('{}')
    (folder/(IDENTIFIER+'.sha256')).write_text('0'*64)
    item = next(r for r in overview(tmp_path)['items'] if r['id']=='poc_followup_20261014')
    assert item['status'] == 'verified_publication'
    assert item['followup']['completed_new_accounts'] == 24
    assert item['execution_readiness']['status'] == 'unavailable'
