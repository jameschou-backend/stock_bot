"""Synthetic sealed reports exercise display, ledger choices, and cache safety."""
from copy import deepcopy
import hashlib
import json
import os
import platform

import pytest
from streamlit.testing.v1 import AppTest

from app import cash_allocation_research as ui


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
        return dict(schema=1, mode=mode, account=dict(daily=daily, trades=trades, holdings=holdings),
            summary=summary, exit_decisions=[], exit_states={}, allocation_decisions=[], audit={})

    return dict(schema=1, mode_order=list(ui.MODES),
        cases={mode:case(mode, 100000+i*10000) for i, mode in enumerate(ui.MODES)},
        benchmark=case('benchmark',80000), live_qualified=False, unseen_validation=False,
        auto_promote=False, exit_mode='loss12', candidate_count=458, execution_signal_lag_market_sessions=1,
        limitations=['同一段歷史已反覆研究，沒有未見驗證。'])


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
             str(ui.CACHE/'cases/cash.json'), str(report_path.relative_to(tmp_path))]
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


def test_unchanged_sources_are_verified_once_and_full_comparison_is_required(sealed):
    for _ in range(3):
        result = ui.overview()
        assert result['available'] and set(result['methods']) == set(ui.MODES)
        assert result['source_verification']['file_count'] == 7
    assert len(sealed['calls']) == 1
    raw = fixture()
    del raw['cases']['trend_0050']
    reseal(sealed, raw)
    assert not ui.overview()['available']


@pytest.mark.parametrize('target', [ui.DRIVER, ui.SPEC, ui.INPUT_MANIFEST, ui.INPUT_PREPARER,
    'raw/parent-prices.parquet', str(ui.CACHE/'cases/cash.json'), str(ui.CACHE/'report.json')])
def test_every_declared_source_change_hides_results(sealed, target):
    assert ui.overview()['available']
    path = sealed['root']/target
    path.write_text(path.read_text()+' ')
    result = ui.overview()
    assert not result['available'] and result['source_verification']['status']=='invalid'
    assert 'methods' not in result and 'benchmark' not in result


def test_preserved_file_size_and_mtime_cannot_reuse_cached_verification(sealed):
    assert ui.overview()['available']
    source = sealed['root']/'raw/parent-prices.parquet'
    stamp = source.stat()
    source.write_bytes(b'X'+source.read_bytes()[1:])
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert not ui.overview()['available'] and len(sealed['calls'])==2


def test_dividend_source_addition_rechecks_inventory_and_missing_watch_is_rejected(sealed):
    assert ui.overview()['available']
    (sealed['root']/ui.VERIFICATION_DIRECTORIES[0]/'9999.parquet').write_text('new source')
    assert ui.overview()['available'] and len(sealed['calls'])==2
    manifest = ui._read(sealed['manifest'])
    manifest['verification_directories'] = []
    write(sealed['manifest'], manifest)
    assert not ui.overview()['available']


def test_unfinished_offline_reproduction_is_not_a_publishable_report(sealed):
    assert ui.overview()['available']
    manifest = ui._read(sealed['manifest'])
    manifest['offline_identical'] = False
    write(sealed['manifest'], manifest)
    assert not ui.overview()['available']


@pytest.mark.parametrize('field, value', [('live_qualified',True), ('unseen_validation',True),
    ('auto_promote',True), ('exit_mode','fixed63'), ('execution_signal_lag_market_sessions',0),
    ('mode_order',['cash','always_0050','trend_0050'])])
def test_incompatible_experiment_or_causal_claim_is_hidden(sealed, field, value):
    raw = fixture()
    raw[field] = value
    reseal(sealed, raw)
    assert not ui.overview()['available']


def test_missing_source_and_required_inventory_are_rejected(sealed):
    assert ui.overview()['available']
    (sealed['root']/ui.SPEC).unlink()
    assert not ui.overview()['available']
    manifest = ui._read(sealed['manifest'])
    del manifest['verification_files_sha256'][ui.SPEC]
    write(sealed['manifest'], manifest)
    assert not ui.overview()['available']


def test_source_mutation_during_read_is_rejected(sealed, monkeypatch):
    original = ui._verify_driver

    def changing(output):
        result = original(output)
        (sealed['root']/'raw/parent-prices.parquet').write_text('changed during read')
        return result

    monkeypatch.setattr(ui, '_verify_driver', changing)
    assert not ui.overview()['available']


def test_absent_report_is_pending_without_a_zero_return(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    result = ui.overview()
    assert result['source_verification']['status']=='pending' and 'methods' not in result


def test_nav_weights_include_receivables_and_follow_selected_ledger():
    raw = fixture()
    before = deepcopy(raw)
    result = ui._normalize(raw)
    for mode, account in result['methods'].items():
        for day in account['exposure']:
            assert sum(day[key] for key in ('cash','etf','stocks','receivable')) == pytest.approx(1)
            assert day['receivable'] > 0
            if mode=='cash':
                assert day['etf']==0 and day['cash']==pytest.approx(.7)
    assert raw==before


@pytest.mark.parametrize('failure', ['missing_holding', 'duplicate_holding', 'negative_cash', 'nav_mismatch'])
def test_exposure_cannot_show_incomplete_or_nonreconciling_account(failure):
    raw = fixture()
    account = raw['cases']['cash']['account']
    if failure=='missing_holding': account['holdings'].pop()
    elif failure=='duplicate_holding': account['holdings'].append(account['holdings'][0])
    elif failure=='negative_cash': account['daily'][0]['cash']=-1
    else: account['daily'][0]['receivable']+=100
    with pytest.raises(ValueError): ui._normalize(raw)


def run_app():
    return AppTest.from_string('from app.cash_allocation_research import render\nrender()').run(timeout=15)


@pytest.mark.parametrize('status', ['pending','invalid'])
def test_unavailable_report_does_not_expose_any_returns_or_ledger(monkeypatch, status):
    monkeypatch.setattr(ui, 'overview', lambda: {'available':False, 'source_verification':{'status':status}, 'note':'完整封存尚未完成'})
    app = run_app()
    assert not app.exception and not app.dataframe and not app.selectbox
    assert (app.info if status=='pending' else app.warning)[0].value=='完整封存尚未完成'


def test_available_flag_without_explicit_verification_is_not_enough(monkeypatch):
    report = displayed_fixture()
    report['source_verification']['status']='invalid'
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    assert not app.exception and not app.dataframe and not app.selectbox


def test_three_modes_show_own_ledger_cash_exposure_and_matched_curves(monkeypatch):
    report, charts, exposures = displayed_fixture(), [], []
    monkeypatch.setattr(ui, 'overview', lambda: report)
    monkeypatch.setattr(ui.st, 'line_chart', lambda data, **kwargs: charts.append(data.copy()))
    monkeypatch.setattr(ui.st, 'area_chart', lambda data, **kwargs: exposures.append(data.copy()))
    app = run_app()
    assert not app.exception
    table = next(item.value for item in app.dataframe if '累積淨報酬' in item.value)
    assert len(table)==4 and table.iloc[-1]['方法']=='0050持有'
    assert table.iloc[0]['相對0050（百分點）']=='+2.00'
    for mode in ui.MODES:
        app.selectbox(key='cash_allocation_method').set_value(mode).run()
        assert not app.exception
        ledger = [item.value for item in app.dataframe if '成交參考價' in item.value][-1]
        assert ledger.iloc[1]['股數']==report['methods'][mode]['trades'][1]['qty']
        assert ledger.iloc[1]['零股對手量檢查']=='高於：僅日量假設'
        expected = [ui.MODE_LABELS[mode]] + ([] if mode=='always_0050' else [ui.MODE_LABELS['always_0050']])+['0050持有']
        assert list(charts[-1])==expected and len(charts[-1])==7
        assert charts[-1].iloc[-1,0]==report['methods'][mode]['summary']['final_nav']
        assert list(exposures[-1])==['現金','0050','個股','應收款與新股權利']
        assert exposures[-1].sum(axis=1).tolist()==pytest.approx([100]*7)
        if mode=='cash':
            assert (exposures[-1]['0050']==0).all()
            assert any('沒有0050成交' in item.value for item in app.info)
        annual = next(item.value for item in app.dataframe if '2026截至09/09' in item.value)
        assert annual.iloc[0]['方法']==ui.MODE_LABELS[mode]
    assert any('不是避險' in item.value for item in app.markdown)
    assert any('不因資料缺失強制賣出' in item.value for item in app.caption)
    assert any('不會啟動回測或抓取資料' in item.value for item in app.caption)


def test_readable_etf_switch_reasons_and_csv_preserve_original_account():
    report = displayed_fixture()
    account = report['methods']['trend_0050']
    account['trades'][0]['reason']='parking_trend_off'
    before = deepcopy(account)
    frame = ui._allocation_rows(account)
    assert len(frame)==1 and frame.iloc[0]['原因']=='大盤趨勢轉弱，賣出0050'
    csv = ui._ledger(account).to_csv(index=False).encode('utf-8-sig')
    assert csv.startswith(b'\xef\xbb\xbf') and '0050' in csv.decode('utf-8-sig')
    assert '大盤趨勢轉弱，賣出0050' in csv.decode('utf-8-sig')
    assert account==before


def test_pending_etf_sale_and_unknown_keep_are_distinct_from_executed_trades(monkeypatch):
    report = displayed_fixture()
    selected = report['methods']['trend_0050']
    selected['allocation_decisions'] = [dict(date='2022-01-04', signal_date='2022-01-03',
        market_state='OFF', allocation_mode='trend_0050', hook_reason='idle_cash', action='sell_0050',
        cash_before=100., cash_after=100., etf_qty_before=1000, etf_qty_after=1000,
        requested_qty=1000, filled_qty=0), dict(date='2022-12-30', signal_date='2022-01-04',
        market_state='UNKNOWN', allocation_mode='trend_0050', hook_reason='idle_cash', action='retain_unknown',
        cash_before=100., cash_after=100., etf_qty_before=1000, etf_qty_after=1000,
        requested_qty=0, filled_qty=0)]
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    app.selectbox(key='cash_allocation_method').set_value('trend_0050').run()
    assert not app.exception
    decisions = next(item.value for item in app.dataframe if '配置指令' in item.value)
    assert decisions['配置指令'].tolist()==['嘗試賣0050', '訊號不足，維持配置']
    assert decisions['成交股數'].tolist()==[0, 0]
    assert any('成交0股不代表已完成配置' in item.value for item in app.caption)
