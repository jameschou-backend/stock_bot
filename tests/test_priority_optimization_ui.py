import hashlib
import json

import pytest

from app.priority_optimization_ui import comparison_rows, load


def seal(path, value):
    path.write_text(json.dumps(value))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_research_view_checks_source_integrity_and_cannot_promote_live(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{}')
    path = tmp_path / 'report.json'
    value = dict(schema='priority_optimization_v1', live_qualified=False, unseen_validation=False,
        evidence_sha256={'source.json': hashlib.sha256(source.read_bytes()).hexdigest()})
    seal(path, value)
    assert load(path, tmp_path) == value
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='證據'):
        load(path, tmp_path)
    value['live_qualified'] = True
    seal(path, value)
    with pytest.raises(ValueError, match='資格'):
        load(path, tmp_path)


def test_incomplete_account_never_displays_partial_profit():
    cases = {}
    for kind in ('capacity', 'benchmark'):
        for stress in ('control', 'combined'):
            for policy in ('mixed', 'board_only'):
                cases[f'{kind}_{stress}_{policy}'] = dict(completed=True,
                    summary=dict(total_return=.5, max_drawdown=-.1))
    cases['capacity_combined_board_only'] = dict(completed=False,
        partial_account={'daily': [{'nav': 9_000_000}]})
    rows = comparison_rows(cases)
    assert rows[1]['成交壓力累積淨報酬'] == '未完成'
    assert rows[1]['壓力最大回撤'] == '未完成'
    assert rows[3]['成交壓力累積淨報酬'] == '50.00%'
