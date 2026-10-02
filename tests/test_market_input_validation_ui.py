from copy import deepcopy
from datetime import date, timedelta
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
        counts={},source_days=185,request_plan=[
            dict(market=market,date=stamp,scope='full_daily_prices_and_roster',status=status)
            for stamp,status in [('2019-01-02','cached'),('2019-01-03','source_missing')]
            for market in ('TWSE','TPEX')],requests_lower_bound=2,repair_summary=None,
        identity=dict(candidate_count=29930,candidate_issues=[],in_scope_official_presence_issues=[]),
        price_conflicts=[],missing_positive_quotes_in_scope=[],positive_local_absent_official=[],
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


@pytest.mark.parametrize('mutation',['qualified','missing_check','count','complete','escape',
    'missing_market_days','duplicate_market_day','market_day_complete'])
def test_malformed_or_promoted_evidence_cannot_be_shown(tmp_path,mutation):
    value,_ = fixture_report(tmp_path)
    if mutation == 'qualified': value['live_qualified'] = True
    elif mutation == 'missing_check': value['checks'].pop(CHECK_NAMES[0])
    elif mutation == 'count': value['coverage']['holding_marks']['verified'] = 900
    elif mutation == 'complete': value['complete_verified_data'] = True
    elif mutation == 'escape': value['source_sha256'] = {'../outside.json':'a'*64}
    elif mutation == 'missing_market_days': value['requests_lower_bound'] = 0
    elif mutation == 'duplicate_market_day': value['request_plan'].append(value['request_plan'][0])
    else: value['checks']['all_historical_market_days_observed'] = True
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
    assert app.metric[2].label == '官方市場日表覆蓋'
    assert app.metric[2].value == '2 / 4'


def test_missing_sources_are_unavailable_not_an_empty_success(tmp_path):
    assert not ui.overview(tmp_path)['available']


def test_supplement_counts_render_without_promoting_qualification(tmp_path):
    value, _ = fixture_report(tmp_path)
    value.update(schema='market_input_validation_v2', supplement=dict(
        added_source_days=3, source_count=3, legacy_status_unknown=1))
    save(tmp_path/ui.REPORT, value)
    app = AppTest.from_string(
        'from pathlib import Path\nfrom app.market_input_validation_ui import render\n'
        f'render(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    assert any('新增核對 3 張' in item.value for item in app.info)
    assert any('不能因此視為可實戰' in item.value for item in app.warning)
    value['supplement']['added_source_days'] = 4
    save(tmp_path/ui.REPORT, value)
    assert not ui.overview(tmp_path)['available']


def test_completed_market_days_keep_other_data_failures_separate(tmp_path):
    value, _ = fixture_report(tmp_path)
    value.update(schema='market_input_validation_v2', supplement=dict(
        added_source_days=3745, source_count=3745, legacy_status_unknown=1),
        source_days=3930, requests_lower_bound=0,
        request_plan=[dict(market=market,date=(date(2019,1,2)+timedelta(days=i)).isoformat(),
            scope='full_daily_prices_and_roster',status='cached')
            for i in range(1964) for market in ('TWSE','TPEX')])
    value['checks']['all_historical_market_days_observed'] = True
    save(tmp_path/ui.REPORT,value)
    overview = ui.overview(tmp_path)
    assert overview['available']
    assert overview['source_days'] == 3930
    assert overview['verified_market_days'] == overview['required_market_days'] == 3928
    assert overview['missing_market_days'] == 0
    assert overview['live_qualified'] is False
    assert overview['complete_verified_data'] is False
    app = AppTest.from_string(
        'from pathlib import Path\nfrom app.market_input_validation_ui import render\n'
        f'render(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    assert app.metric[2].value == '3,928 / 3,928'
    assert any('官方市場日表已補齊：3,928 / 3,928 張' in item.value for item in app.success)
    assert any('尚有行情／身分／成交資料問題' in item.value for item in app.warning)
    assert any('不能因此視為可實戰' in item.value for item in app.warning)
    assert not any('尚缺' in item.value or '尚未完成全期間驗證' in item.value for item in app.warning)
    assert not any('報告已記錄缺漏日期' in item.value for item in app.markdown)


def test_no_required_day_can_be_hidden_by_extra_source_days(tmp_path):
    value, _ = fixture_report(tmp_path)
    value['source_days'] = 3930
    save(tmp_path/ui.REPORT,value)
    overview = ui.overview(tmp_path)
    assert overview['verified_market_days'] == 2
    assert overview['required_market_days'] == 4
    assert overview['missing_market_days'] == 2


@pytest.mark.parametrize('schema',['market_input_validation_v1','market_input_validation_v2'])
@pytest.mark.parametrize('field',list(ui.ISSUE_LABELS))
@pytest.mark.parametrize('mutation',['missing','not_list','incomplete_row'])
def test_issue_evidence_cannot_silently_become_zero(tmp_path,schema,field,mutation):
    value,_ = fixture_report(tmp_path)
    value['schema'] = schema
    if schema == 'market_input_validation_v2':
        value['supplement'] = dict(added_source_days=3,source_count=3,legacy_status_unknown=1)
    owner = value['identity'] if field == 'in_scope_official_presence_issues' else value
    if mutation == 'missing':
        owner.pop(field)
    elif mutation == 'not_list':
        owner[field] = {'count':0}
    else:
        owner[field] = [{'stock_id':'6873','market':'TWSE'}]
    save(tmp_path/ui.REPORT,value)
    assert not ui.overview(tmp_path)['available']


def test_all_checks_and_open_issue_counts_are_visible_with_no_candidate_identity_errors(tmp_path):
    value,_ = fixture_report(tmp_path)
    value.update(schema='market_input_validation_v2',supplement=dict(
        added_source_days=3,source_count=3,legacy_status_unknown=1))
    row = dict(stock_id='6873',market='TWSE',date='2024-04-30')
    value['missing_positive_quotes_in_scope'] = [dict(row,positive_official_price=True)] * 399
    value['identity']['in_scope_official_presence_issues'] = [dict(row,issue='dated_identity')] * 1576
    value['positive_local_absent_official'] = [row] * 183
    value['checks']['no_observed_price_conflicts'] = True
    save(tmp_path/ui.REPORT,value)
    overview = ui.overview(tmp_path)
    assert overview['issue_counts'] == dict(missing_positive_quotes_in_scope=399,
        in_scope_official_presence_issues=1576,positive_local_absent_official=183)
    assert overview['candidate_identity_issues'] == 0
    assert overview['live_qualified'] is False
    app = AppTest.from_string(
        'from pathlib import Path\nfrom app.market_input_validation_ui import render\n'
        f'render(Path({str(tmp_path)!r}))').run()
    assert not app.exception
    assert app.metric[1].value == '0 / 29,930'
    checks = app.dataframe[0].value
    assert checks['核對項目'].tolist() == [ui.CHECK_LABELS[key] for key in CHECK_NAMES]
    assert checks['結果'].tolist() == ['通過' if value['checks'][key] else '待核對' for key in CHECK_NAMES]
    issues = app.dataframe[1].value
    assert issues['待核對項目'].tolist() == list(ui.ISSUE_LABELS.values())
    assert issues['股日數'].tolist() == [399,1576,183]
