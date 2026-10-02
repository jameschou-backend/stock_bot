from copy import deepcopy
import hashlib
import json
import os

import pytest
from streamlit.testing.v1 import AppTest

from app import market_input_validation_ui as ui
from skills.market_input_validation import CHECK_NAMES, MarketEvidenceError


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def fixture_report(tmp_path):
    raw = tmp_path/'source.json'
    raw.write_text('original')
    coverage = dict(required=561,verified=36,missing=525,complete=False,fraction=36/561)
    result = dict(schema='market_input_validation_v1',live_qualified=False,actual_fill_verified=False,
        unseen_validation=False,return_recomputed=False,complete_verified_data=False,
        checks=dict.fromkeys(CHECK_NAMES,False),start='2019-01-02',end='2026-09-09',
        source_sha256={'source.json':hashlib.sha256(raw.read_bytes()).hexdigest()},
        counts={},source_days=185,request_plan=[{}]*3866,requests_lower_bound=3684,repair_summary=None,
        identity=dict(candidate_count=29930,candidate_issues=[]),price_conflicts=[],
        coverage={k:deepcopy(coverage) for k in ('ordinary_fill_prices','traded_stock_day_prices',
                                               'holding_marks','candidate_signal_prices')})
    save(tmp_path/ui.REPORT,result)
    return result,raw


def test_cached_source_change_rejects_even_restored_mtime(tmp_path):
    value,raw = fixture_report(tmp_path)
    first = ui.load(tmp_path)
    first['live_qualified'] = True
    assert ui.load(tmp_path)['live_qualified'] is False
    previous = raw.stat()
    raw.write_text('modified')
    os.utime(raw,ns=(previous.st_atime_ns,previous.st_mtime_ns))
    with pytest.raises(MarketEvidenceError,match='原始來源已變動'):
        ui.load(tmp_path)
    assert not ui.overview(tmp_path)['available']


@pytest.mark.parametrize('mutation',['qualified','missing_check','count','complete','escape'])
def test_malformed_or_promoted_evidence_cannot_be_shown(tmp_path,mutation):
    value,_ = fixture_report(tmp_path)
    if mutation == 'qualified': value['live_qualified'] = True
    elif mutation == 'missing_check': value['checks'].pop(CHECK_NAMES[0])
    elif mutation == 'count': value['coverage']['holding_marks']['verified'] = 900
    elif mutation == 'complete': value['complete_verified_data'] = True
    else: value['source_sha256'] = {'../outside.json':'a'*64}
    save(tmp_path/ui.REPORT,value)
    assert not ui.overview(tmp_path)['available']


def test_read_only_panel_keeps_incomplete_scope_visible(tmp_path):
    fixture_report(tmp_path)
    app = AppTest.from_string(
        'from pathlib import Path\nfrom app.market_input_validation_ui import render\n'
        f'render(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    assert any('尚未完成全期間驗證' in w.value for w in app.warning)
    assert app.metric[0].value == '36 / 561'
    assert app.metric[1].value == '0 / 29,930'


def test_missing_sources_are_unavailable_not_an_empty_success(tmp_path):
    assert not ui.overview(tmp_path)['available']
