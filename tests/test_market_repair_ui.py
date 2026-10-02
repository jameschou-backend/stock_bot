from copy import deepcopy
import json
import os

import pytest
from streamlit.testing.v1 import AppTest

from app import market_repair_ui as ui
from skills.ordinary_volume_bundle import digest


def save(root, value):
    path = root/ui.REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.with_suffix('.sha256').write_text(digest(path))


def fixture(root):
    source = root/'source.json'
    source.write_text('original evidence')
    summary = dict(total_return=.25, max_drawdown=-.15, final_nav=1_250_000)
    value = dict(schema='market_input_repair_replay_v1', start='2019-01-02', end='2026-09-09',
        live_qualified=False, actual_fill_verified=False, unseen_validation=False, complete_verified_data=False,
        repairs=dict(quotes_added=403, original_candidates=29930, repaired_candidates=29945),
        research=dict(completed=True, repeat_identical=True, volume_policy='legacy_total_research',
            cases={k:dict(summary=deepcopy(summary)) for k in ('three_black', 'benchmark')}),
        strict=dict(completed=False, capacity_complete=False, blocked_board_orders=7,
                    cases={k:dict(completed=False,ordinary_capacity_complete=False) for k in ('three_black','benchmark')}),
        limitations=['普通盤資料未完整'],
        source_sha256={source.name:digest(source)})
    save(root, value)
    return value, source


def test_repaired_research_is_displayed_separately_from_strict_capacity(tmp_path):
    fixture(tmp_path)
    app = AppTest.from_string('from pathlib import Path\nfrom app.market_repair_ui import render\n'
                             f'render(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    assert any('不能視為已驗證可成交' in r.value for r in app.warning)
    assert any('全日量估計普通盤容量' in r.value for r in app.caption)
    assert len(app.dataframe) == 1


def test_repair_cache_invalidates_changed_source_even_with_restored_mtime(tmp_path):
    _, source = fixture(tmp_path)
    assert ui.overview(tmp_path)['available']
    before = source.stat()
    source.write_text('modified evidence')
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not ui.overview(tmp_path)['available']


def test_finished_replay_without_capacity_evidence_still_warns(tmp_path):
    value, _ = fixture(tmp_path)
    value['strict'].update(completed=True, blocked_board_orders=0)
    for case in value['strict']['cases'].values(): case['completed'] = True
    save(tmp_path, value)
    app = AppTest.from_string('from pathlib import Path\nfrom app.market_repair_ui import render\n'
                             f'render(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    assert any('容量仍未完整認證' in r.value for r in app.warning)


@pytest.mark.parametrize('field', ['live_qualified', 'actual_fill_verified', 'complete_verified_data', 'repeat_identical', 'volume_policy'])
def test_promoted_or_inconsistent_repaired_results_are_rejected(tmp_path, field):
    value, _ = fixture(tmp_path)
    if field == 'repeat_identical': value['research'][field] = False
    elif field == 'volume_policy': value['research'][field] = 'strict'
    else: value[field] = True
    save(tmp_path, value)
    assert not ui.overview(tmp_path)['available']
