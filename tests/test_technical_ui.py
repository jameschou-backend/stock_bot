"""Synthetic sealed reports exercise display, ledger choices, and cache safety."""
from copy import deepcopy
import hashlib
import json
import os
import platform

import pytest
from streamlit.testing.v1 import AppTest

from app import technical_research as ui


def fixture():
    dates = ['2022-01-03', '2022-01-04', '2022-12-30', '2023-12-29',
             '2024-12-31', '2025-12-31', '2026-09-09']

    def case(mode, increment):
        final = 1_000_000 + increment
        daily, holdings = [], []
        for index, day in enumerate(dates):
            nav = 1_000_000 + increment*index/6
            cash = nav*.7 if mode == 'cash' else 100.
            market = nav-cash-1000.
            etf = 0. if mode == 'cash' else market*.7
            daily.append(dict(date=day, nav=nav, cash=cash, market_value=market, receivable=1000.))
            holdings.append(dict(date=day, stock_id='2330', market_value=market-etf))
            if etf:
                holdings.append(dict(date=day, stock_id='0050', market_value=etf))
        trades = [dict(date=dates[0], stock_id='2330' if mode=='cash' else '0050',
            name='測試公司' if mode=='cash' else '元大台灣50', side='buy', channel='board',
            qty=1000, reference_price=100., commission=143., tax=0., slippage=450., total_cost=593.,
            cash_change=-100593., cash_after=899407., reason='leader_entry' if mode=='cash' else 'initial_allocation',
            signal_date=None, event_id='event'),
            dict(date=dates[1], stock_id='2330', name='測試公司', side='sell', channel='odd',
            qty=110+int(increment/10000), reference_price=80., commission=1., tax=3., slippage=4., total_cost=8.,
            cash_change=80., cash_after=899487., reason='loss12', signal_date=dates[0], event_id='event',
            odd_bid_qty=10, odd_ask_qty=1000)]
        summary = dict(start=ui.START, end=ui.END, initial_cash=1_000_000, final_nav=final,
            profit=increment, total_return=increment/1e6, cagr=.03, max_drawdown=-.22,
            cash=daily[-1]['cash'], market_value=daily[-1]['market_value'], receivable=1000.,
            trading_days=len(dates), trade_count=len(trades), costs={'total_cost':601.},
            annual=[dict(year=str(year), total_return=.03) for year in range(2022, 2027)])
        return dict(schema=1, mode=mode, research_kind='technical', control_exit_mode='loss12', account=dict(daily=daily, trades=trades, holdings=holdings),
            summary=summary, exit_decisions=[], exit_states={}, sizing_decisions=[], add_decisions=[], pattern_decisions=[], audit={})

    return dict(schema=1, mode_order=list(ui.MODES),
        cases={mode:case(mode, 100000+i*10000) for i, mode in enumerate(ui.MODES)},
        benchmark=case('benchmark',80000), live_qualified=False, unseen_validation=False,
        auto_promote=False, research_kind='technical', control_exit_mode='loss12', candidate_count=458, execution_signal_lag_market_sessions=1,
        limitations=['同一段歷史已反覆研究，沒有未見驗證。'],
        candidate_features=dict(schema=1, counts_are_orders_or_fills=False,
            summary=dict(candidate_count=458, scope='complete_original_candidates_before_portfolio_eligibility',
                pattern_pass_true_count=0, pattern_pass_false_count=458, pattern_pass_unknown_count=0),
            rows=[dict(event_id='candidate-'+str(index), stock_id='2330', entry_date='2022-01-04',
                signal_date='2022-01-03', original_signal_date='2022-01-03', pattern_pass=False)
                for index in range(458)]))


def displayed_fixture():
    report = ui._normalize(fixture())
    report['source_verification'] = {'status':'verified', 'file_count':40}
    return report


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    ui._verified.cache_clear()
    report_path = tmp_path/ui.CACHE/'report.json'
    write(report_path, fixture())
    names = [ui.DRIVER, ui.SPEC, ui.INPUT_MANIFEST, ui.INPUT_PREPARER, 'raw/parent-prices.parquet',
             str(ui.CACHE/'cases/control.json'), str(report_path.relative_to(tmp_path))]
    for name in names[:-1]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic source')
    for name in ui.VERIFICATION_DIRECTORIES:
        (tmp_path/name).mkdir(parents=True, exist_ok=True)
    manifest_path = report_path.with_name('manifest.json')
    write(manifest_path, dict(schema=1, offline_identical=True, live_qualified=False,
        context={'runtime_versions':{'python':platform.python_version()}},
        verification_files_sha256={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in names},
        verification_directories=list(ui.VERIFICATION_DIRECTORIES)))
    calls = []

    def verify(output):
        calls.append(str(output))
        manifest = ui._read(output/'manifest.json')
        for name, expected in manifest['verification_files_sha256'].items():
            if hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Changed source '+name)
        return manifest

    monkeypatch.setattr(ui, '_verify_driver', verify)
    yield dict(root=tmp_path, report=report_path, manifest=manifest_path, calls=calls)
    ui._verified.cache_clear()


def reseal(sealed, report):
    write(sealed['report'], report)
    manifest = ui._read(sealed['manifest'])
    manifest['verification_files_sha256'][str(sealed['report'].relative_to(sealed['root']))] = hashlib.sha256(sealed['report'].read_bytes()).hexdigest()
    write(sealed['manifest'], manifest)


def decision_fixture(raw):
    for mode, case in raw['cases'].items():
        if mode == 'control':
            continue
        case['sizing_decisions'] = [dict(date='2022-01-04', signal_date='2022-01-03', stock_id='2330',
            mode=mode, event_id='event', prior_nav=1e6, previous_price=100., adjusted_close=50.,
            support20=45., planned_stop=45., raw_planned_stop=90., risk_cap=20000., planned_risk=15000.,
            requested_qty=1000, filled_qty=500, failure='partial_or_unfilled_execution', diagnostics=[])]
        if mode != 'risk2':
            case['exit_decisions'] = [dict(date='2022-01-04', signal_date='2022-01-03', stock_id='2330',
                mode=mode, signal_close=44., support20=43., support_floor=45., support_failure=True,
                reason='support20', first_signal_date='2022-01-03', technical_diagnostics=[], event_id='event')]
    case = raw['cases']['support_risk2_add']
    case['add_decisions'] = [dict(date='2022-01-04', signal_date='2022-01-03', stock_id='2330',
        event_id='event', requested_qty=100, filled_qty=0, breakout20=True, planned_risk=19000.,
        risk_cap=20000., entry_price=50., adjusted_close=56., failure='partial_or_unfilled_execution'),
        dict(date='2022-12-30', signal_date='2022-01-04', stock_id='2330', event_id='event',
        requested_qty=100, filled_qty=50, breakout20=True, planned_risk=19000., risk_cap=20000.,
        entry_price=50., adjusted_close=57., failure='partial_or_unfilled_execution')]
    raw['cases']['support_risk2_pattern']['pattern_decisions'] = [
        dict(date='2022-01-04', signal_date='2022-01-03', stock_id='2330', event_id='event',
             pattern_available=True, pattern_pass=False, breakout20=True, contraction10=False,
             volume_expansion=True, diagnostics=[]),
        dict(date='2022-12-30', signal_date='2022-01-04', stock_id='2330', event_id='event2',
             pattern_available=False, pattern_pass=None, breakout20=True, contraction10=True,
             volume_expansion=None, diagnostics=['volume_history_incomplete'])]
    return raw


def test_all_six_cases_required_and_unchanged_sources_verified_once(sealed):
    for _ in range(3):
        result = ui.overview()
        assert result['available'] and set(result['methods']) == set(ui.MODES)
        assert result['source_verification']['file_count'] == 7
    assert len(sealed['calls']) == 1
    raw = fixture()
    del raw['cases']['support_risk2_pattern']
    reseal(sealed, raw)
    assert not ui.overview()['available']


@pytest.mark.parametrize('target', [ui.DRIVER, ui.SPEC, ui.INPUT_MANIFEST, ui.INPUT_PREPARER,
    'raw/parent-prices.parquet', str(ui.CACHE/'cases/control.json'), str(ui.CACHE/'report.json')])
def test_any_sealed_source_change_hides_results(sealed, target):
    assert ui.overview()['available']
    path = sealed['root']/target
    path.write_text(path.read_text()+' ')
    result = ui.overview()
    assert not result['available'] and result['source_verification']['status'] == 'invalid'
    assert 'methods' not in result and 'benchmark' not in result


def test_same_size_mtime_modification_invalidates_cache_via_ctime(sealed):
    assert ui.overview()['available']
    source = sealed['root']/'raw/parent-prices.parquet'
    stamp = source.stat()
    source.write_bytes(b'X'+source.read_bytes()[1:])
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert not ui.overview()['available'] and len(sealed['calls']) == 2


def test_new_source_rechecks_full_inventory_and_missing_watch_is_rejected(sealed):
    assert ui.overview()['available']
    (sealed['root']/ui.VERIFICATION_DIRECTORIES[0]/'9999.parquet').write_text('new source')
    assert ui.overview()['available'] and len(sealed['calls']) == 2
    manifest = ui._read(sealed['manifest'])
    manifest['verification_directories'] = []
    write(sealed['manifest'], manifest)
    assert not ui.overview()['available']


def test_unfinished_offline_reproduction_stays_hidden(sealed):
    manifest = ui._read(sealed['manifest'])
    manifest['offline_identical'] = False
    write(sealed['manifest'], manifest)
    assert not ui.overview()['available']


@pytest.mark.parametrize('field,value', [('live_qualified', True), ('unseen_validation', True),
    ('auto_promote', True), ('control_exit_mode', 'fixed63'), ('research_kind', 'allocation'),
    ('candidate_count', 459), ('execution_signal_lag_market_sessions', 0), ('mode_order', list(reversed(ui.MODES)))])
def test_incompatible_research_claims_are_hidden(sealed, field, value):
    raw = fixture()
    raw[field] = value
    reseal(sealed, raw)
    assert not ui.overview()['available']


def test_in_read_source_mutation_is_hidden(sealed, monkeypatch):
    original = ui._verify_driver
    def changed(output):
        manifest = original(output)
        (sealed['root']/ui.SPEC).write_text('changed during verification')
        return manifest
    monkeypatch.setattr(ui, '_verify_driver', changed)
    assert not ui.overview()['available']


def test_absent_report_is_pending_without_zero_return(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    result = ui.overview()
    assert result['source_verification']['status'] == 'pending' and 'methods' not in result


def test_comparison_delta_uses_preregistered_base_never_best_result():
    report = displayed_fixture()
    rows = ui._comparison_rows(report)
    assert len(rows) == 7
    assert rows.iloc[1]['本項比較對象'] == '原策略對照'
    assert rows.iloc[1]['比比較對象增減（百分點）'] == '+1.00'
    assert rows.iloc[4]['本項比較對象'] == '支撐出場＋2%配置'
    assert rows.iloc[4]['比比較對象增減（百分點）'] == '+1.00'
    assert rows.iloc[5]['比比較對象增減（百分點）'] == '+2.00'


def test_normalization_preserves_decisions_and_nav_weights_including_rights():
    raw = decision_fixture(fixture())
    before = deepcopy(raw)
    result = ui._normalize(raw)
    for mode, account in result['methods'].items():
        assert account['sizing_decisions'] == raw['cases'][mode]['sizing_decisions']
        for day in account['exposure']:
            assert sum(day[key] for key in ('cash', 'etf', 'stocks', 'receivable')) == pytest.approx(1)
            assert day['receivable'] > 0 and day['etf'] > 0
    assert raw == before


@pytest.mark.parametrize('problem', ['current_signal', 'negative_qty', 'overfilled_qty', 'bad_mode', 'bad_risk', 'false_pattern'])
def test_malformed_decision_or_causality_is_rejected(problem):
    raw = decision_fixture(fixture())
    row = raw['cases']['support_risk2']['sizing_decisions'][0]
    if problem == 'current_signal': row['signal_date'] = row['date']
    elif problem == 'negative_qty': row['filled_qty'] = -1
    elif problem == 'overfilled_qty': row['filled_qty'] = 1001
    elif problem == 'bad_mode': row['mode'] = 'control'
    elif problem == 'bad_risk': row['planned_risk'] = float('nan')
    else: raw['cases']['support_risk2_pattern']['pattern_decisions'][0]['pattern_pass'] = True
    with pytest.raises(ValueError):
        ui._normalize(raw)


def test_support_and_trade_price_scales_are_explicit_and_csv_is_selected_account():
    report = ui._normalize(decision_fixture(fixture()))
    account = report['methods']['support_risk2']
    account['trades'][1]['reason'] = 'support20'
    before = deepcopy(account)
    support = ui._support_rows(account)
    sizing = ui._sizing_rows(account)
    assert support.iloc[0]['只升不降支撐（還原尺度）'] == 45.
    assert support.iloc[0]['20日低點（還原尺度）'] == 43.
    assert sizing.iloc[0]['計畫停損（原始尺度）'] == 90.
    frame = ui._ledger(account)
    assert frame.iloc[1]['原因'] == '跌破只升不降的支撐'
    assert frame.iloc[1]['零股對手量檢查'] == '高於：僅日量假設'
    payload = frame.to_csv(index=False).encode('utf-8-sig')
    assert payload.startswith(b'\xef\xbb\xbf') and '跌破只升不降的支撐' in payload.decode('utf-8-sig')
    assert account == before


def run_app():
    return AppTest.from_string('from app.technical_research import render\nrender()').run(timeout=15)


@pytest.mark.parametrize('status', ['pending', 'invalid'])
def test_unavailable_report_never_exposes_returns_or_ledger(monkeypatch, status):
    monkeypatch.setattr(ui, 'overview', lambda: {'available': False, 'source_verification': {'status': status}, 'note': '完整封存尚未完成'})
    app = run_app()
    assert not app.exception and not app.dataframe and not app.selectbox
    assert (app.info if status == 'pending' else app.warning)[0].value == '完整封存尚未完成'


def test_six_modes_have_matched_comparison_curve_and_own_ledger(monkeypatch):
    report = ui._normalize(decision_fixture(fixture()))
    report['source_verification'] = {'status': 'verified', 'file_count': 40}
    charts = []
    monkeypatch.setattr(ui, 'overview', lambda: report)
    monkeypatch.setattr(ui.st, 'line_chart', lambda data, **kwargs: charts.append(data.copy()))
    app = run_app()
    assert not app.exception
    for mode in ui.MODES:
        app.selectbox(key='technical_method').set_value(mode).run()
        assert not app.exception
        expected = [ui.MODE_LABELS[mode]] + ([] if mode == ui.BASES[mode] else [ui.MODE_LABELS[ui.BASES[mode]]]) + ['0050持有']
        assert list(charts[-1]) == expected
        assert charts[-1].iloc[-1, 0] == report['methods'][mode]['summary']['final_nav']
        ledger = next(item.value for item in app.dataframe if '成交參考價' in item.value)
        assert ledger.iloc[1]['股數'] == report['methods'][mode]['trades'][1]['qty']
        if mode == 'support_risk2_add':
            assert any('有成交的加碼共 1 次，共 50 股' in item.value for item in app.markdown)
            additions = next(item.value for item in app.dataframe if '加碼成交股數' in item.value)
            assert additions['加碼成交股數'].tolist() == [0, 50]
        if mode == 'support_risk2_pattern':
            patterns = next(item.value for item in app.dataframe if '三條件全部通過' in item.value and '執行日' in item.value)
            assert patterns['三條件全部通過'].tolist() == ['不符合', '資料不足']
            assert any('沒有辨識W底' in item.value for item in app.caption)
            candidates = next(item.value for item in app.dataframe if '原候選進場日' in item.value)
            assert len(candidates) == 458
            assert any('不是下單或成交次數' in item.value for item in app.caption)
    assert any('不是實際損失保證或全帳戶風險上限' in item.value for item in app.caption)
    assert any('不會啟動回測或抓取資料' in item.value for item in app.caption)


def test_available_flag_without_verified_sources_is_not_enough(monkeypatch):
    report = displayed_fixture()
    report['source_verification']['status'] = 'invalid'
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    assert not app.exception and not app.dataframe and not app.selectbox


@pytest.mark.parametrize('problem', ['missing_row', 'duplicate', 'wrong_count', 'called_fills'])
def test_incomplete_or_mislabelled_candidate_audit_is_hidden(problem):
    raw = fixture()
    candidates = raw['candidate_features']
    if problem == 'missing_row': candidates['rows'].pop()
    elif problem == 'duplicate': candidates['rows'][1]['event_id'] = candidates['rows'][0]['event_id']
    elif problem == 'wrong_count': candidates['summary']['pattern_pass_true_count'] = 1
    else: candidates['counts_are_orders_or_fills'] = True
    with pytest.raises(ValueError):
        ui._normalize(raw)


def test_fee_funded_sales_use_actual_cash_deficit_without_duplicate_cost(monkeypatch):
    report = displayed_fixture()
    account = report['methods']['control']
    account['trades'][1].update(cash_change=-15.25, total_cost=20.)
    zero_sale = deepcopy(account['trades'][1])
    zero_sale['cash_change'] = 0.
    account['trades'].append(zero_sale)
    before = deepcopy(account)
    assert ui._exit_cash_costs(account) == dict(negative_count=1, zero_count=1, cash_paid=15.25)
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    assert not app.exception
    assert any('共 NT$15.25' in item.value and '不是再加扣一次' in item.value for item in app.caption)
    assert any('淨現金收付為0，股數仍已賣出' in item.value for item in app.caption)
    assert account == before


def test_rejection_distinguishes_fee_shortfall_from_cash_unavailable():
    account = dict(orders=[dict(date='2022-01-04', stock_id='2330', requested_qty=1, filled_qty=0,
        failure='proceeds_below_costs_insufficient_cash', event_id='tiny-exit')])
    rows = ui._rejected_rows(account)
    assert rows.iloc[0]['原因'] == '賣出所得不足支付成本，現金也不足補差額'
    assert rows.iloc[0]['成交股數'] == 0


def test_fee_funded_metadata_must_match_actual_sale_cash():
    account = dict(trades=[dict(side='sell', cash_change=-15.25)],
        summary={'technical': {'fee_funded_exit_cash_paid': 20.}})
    with pytest.raises(ValueError, match='differs from actual trades'):
        ui._exit_cash_costs(account)
    account['summary']['technical']['fee_funded_exit_cash_paid'] = 15.25
    assert ui._exit_cash_costs(account)['cash_paid'] == 15.25
