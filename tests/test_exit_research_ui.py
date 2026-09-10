"""Small sealed fixtures and Streamlit UI tests; no historical replay or API."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform

import pytest
from streamlit.testing.v1 import AppTest

from app import exit_research as ui


def fixture():
    dates = ['2022-01-03', '2022-01-04', '2022-12-30', '2023-12-29',
             '2024-12-31', '2025-12-31', '2026-09-09']

    def case(mode, increment):
        final = 1_000_000+increment
        daily = [{'date': day, 'nav': 1_000_000 + increment*i/6} for i, day in enumerate(dates)]
        reason = 'time63' if mode == 'fixed63' else 'loss12'
        trades = [dict(date=dates[0], stock_id='0050', name='元大台灣50', side='buy', channel='board',
            qty=1000, reference_price=100., commission=143., tax=0., slippage=450., total_cost=593.,
            cash_change=-100593., cash_after=899407., reason='initial_allocation', signal_date=None, event_id='benchmark'),
            dict(date=dates[1], stock_id='2330', name='測試公司', side='sell', channel='odd',
            qty=110+int(increment/10000), reference_price=80., commission=1., tax=3., slippage=4., total_cost=8.,
            cash_change=80., cash_after=899487., reason=reason, signal_date=dates[0], event_id='event',
            odd_bid_qty=10, odd_ask_qty=1000)]
        annual = [dict(year=year, total_return=(int(year)-2021)/100 + increment/1e7) for year in map(str, range(2022,2027))]
        waits = [] if mode == 'benchmark' else [dict(event_id='event', stock_id='2330', reason=reason,
            signal_date=dates[0], target_date=dates[1], first_fill_date=dates[1], complete_exit_date=None,
            target_to_first_fill_sessions=0)]
        summary = dict(start=ui.START, end=ui.END, initial_cash=1_000_000, final_nav=final,
            profit=increment, total_return=increment/1e6, cagr=.03, max_drawdown=-.22,
            cash=100., market_value=final-1100., receivable=1000., trading_days=len(dates), trade_count=2,
            costs={'total_cost':601.}, annual=annual, reason_counts={} if mode=='benchmark' else {reason:1}, exit_waits=waits,
            depth_audit=dict(odd_trade_count=1, exceeds_last_opposing_depth_count=1, missing_opposing_depth_count=0))
        return dict(schema=1, mode=mode, account=dict(daily=daily,trades=trades), summary=summary,
                    exit_decisions=[],exit_states={},audit={})

    return dict(schema=1, mode_order=list(ui.MODES),
        cases={mode:case(mode, 100000+i*10000) for i,mode in enumerate(ui.MODES)},
        benchmark=case('benchmark',80000), live_qualified=False, unseen_validation=False,
        auto_promote=False, remaining_cash_asset='0050', candidate_count=458,execution_signal_lag_market_sessions=1,
        limitations=['已反覆使用歷史；這裡沒有未見驗證。'])


def displayed_fixture():
    result = ui._normalize(fixture())
    result['source_verification'] = {'status':'verified','file_count':40,'checked_at':'2026-09-10T00:00:00+00:00'}
    return result


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'ROOT', tmp_path)
    ui._verified.cache_clear()
    report_path = tmp_path / ui.CACHE / 'report.json'
    _write(report_path, fixture())
    names = [ui.DRIVER, ui.SPEC, ui.INPUT_MANIFEST, 'raw/parent-prices.parquet',
             str(ui.CACHE/'cases/adaptive.json'), str(report_path.relative_to(tmp_path))]
    for name in names[:-1]:
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('small synthetic source')
    manifest_path = report_path.with_name('manifest.json')
    for name in ui.VERIFICATION_DIRECTORIES:
        (tmp_path/name).mkdir(parents=True,exist_ok=True)
    inventory = {name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in names}
    _write(manifest_path,dict(schema=1, offline_identical=True,live_qualified=False,
        context={'runtime_versions':{'python':platform.python_version()}}, verification_files_sha256=inventory,
        verification_directories=list(ui.VERIFICATION_DIRECTORIES)))
    calls = []
    def verify(output):
        calls.append(str(output))
        manifest = ui._read(output/'manifest.json')
        for name, expected in manifest['verification_files_sha256'].items():
            if hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Changed source '+name)
        return manifest
    monkeypatch.setattr(ui,'_verify_driver',verify)
    yield dict(root=tmp_path,report=report_path,manifest=manifest_path,calls=calls)
    ui._verified.cache_clear()


def reseal_report(sealed, value):
    _write(sealed['report'], value)
    manifest = ui._read(sealed['manifest'])
    manifest['verification_files_sha256'][str(sealed['report'].relative_to(sealed['root']))] = hashlib.sha256(sealed['report'].read_bytes()).hexdigest()
    _write(sealed['manifest'],manifest)


def test_complete_actual_schema_is_accepted_and_verified_once_when_unchanged(sealed):
    first = ui.overview()
    assert first['available'] and set(first['methods'])==set(ui.MODES)
    assert first['source_verification']['file_count']==6
    assert first['methods']['loss12']['exit_reason_counts']=={'loss12':1}
    for _ in range(4):
        assert ui.overview()['available']
    assert len(sealed['calls'])==1


@pytest.mark.parametrize('target',[ui.DRIVER,ui.SPEC,ui.INPUT_MANIFEST,'raw/parent-prices.parquet',
                                 str(ui.CACHE/'cases/adaptive.json'),str(ui.CACHE/'report.json')])
def test_any_named_code_spec_parent_case_or_output_change_hides_all_returns(sealed,target):
    assert ui.overview()['available']
    path = sealed['root']/target
    path.write_text(path.read_text()+' ')
    result = ui.overview()
    assert result['available'] is False and result['source_verification']['status']=='invalid'
    assert 'methods' not in result and 'benchmark' not in result


def test_same_length_source_mutation_and_restored_mtime_still_invalidates_by_ctime(sealed):
    assert ui.overview()['available']
    source = sealed['root']/'raw/parent-prices.parquet'
    stamp = source.stat()
    old = source.read_bytes()
    source.write_bytes(b'X'+old[1:])
    os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert ui.overview()['available'] is False
    assert len(sealed['calls'])==2


def test_manifest_change_rechecks_even_if_named_sources_are_unchanged(sealed):
    assert ui.overview()['available']
    manifest = ui._read(sealed['manifest'])
    manifest['offline_identical'] = False
    _write(sealed['manifest'],manifest)
    assert ui.overview()['available'] is False
    assert len(sealed['calls'])==2


def test_source_directory_addition_cannot_reuse_cached_verification(sealed):
    assert ui.overview()['available']
    (sealed['root']/'raw/new-source.parquet').write_text('new evidence file')
    assert ui.overview()['available']
    assert len(sealed['calls'])==2


def test_explicit_dividend_directory_addition_and_missing_inventory_invalidate(sealed):
    assert ui.overview()['available']
    (sealed['root']/ui.VERIFICATION_DIRECTORIES[0]/'9999.parquet').write_text('new dividend evidence')
    assert ui.overview()['available']
    assert len(sealed['calls'])==2
    manifest=ui._read(sealed['manifest'])
    manifest['verification_directories']=[]
    _write(sealed['manifest'],manifest)
    assert ui.overview()['available'] is False


@pytest.mark.parametrize('field',['cases','mode_order'])
def test_incomplete_seven_method_comparison_is_not_displayed_even_when_hashes_match(sealed,field):
    report = fixture()
    if field=='cases': del report[field]['adaptive']
    else: report[field].append('fixed63')
    reseal_report(sealed,report)
    assert ui.overview()['available'] is False


@pytest.mark.parametrize('field,value',[('live_qualified',True),('unseen_validation',True),('auto_promote',True),
                                      ('remaining_cash_asset','cash'),('execution_signal_lag_market_sessions',0)])
def test_incompatible_research_or_causal_claims_are_rejected(sealed,field,value):
    report = fixture();report[field]=value
    reseal_report(sealed,report)
    assert ui.overview()['available'] is False


def test_missing_file_or_incomplete_inventory_does_not_reuse_success(sealed):
    assert ui.overview()['available']
    (sealed['root']/ui.SPEC).unlink()
    assert ui.overview()['available'] is False
    manifest = ui._read(sealed['manifest'])
    del manifest['verification_files_sha256'][ui.SPEC]
    _write(sealed['manifest'],manifest)
    assert ui.overview()['available'] is False


def test_absent_report_is_pending_not_a_zero_return(tmp_path,monkeypatch):
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    value = ui.overview()
    assert value['source_verification']['status']=='pending'
    assert not value['available'] and 'methods' not in value


def test_json_rejects_duplicate_keys_and_nonfinite_values():
    for payload in ('{"a":1,"a":2}', '{"return":NaN}', '{"return":Infinity}'):
        with pytest.raises(ValueError): ui._decode(payload)


def run_app():
    return AppTest.from_string('from app.exit_research import render\nrender()').run(timeout=15)


@pytest.mark.parametrize('status',['pending','invalid'])
def test_unavailable_ui_has_no_returns_chart_details_or_selector(monkeypatch,status):
    monkeypatch.setattr(ui,'overview',lambda:{'available':False,'source_verification':{'status':status},'note':'驗證尚未完成'})
    app = run_app()
    assert not app.exception and not app.dataframe and not app.selectbox and not app.metric
    assert (app.info if status=='pending' else app.warning)[0].value=='驗證尚未完成'


def test_ui_requires_explicit_source_verification_even_if_available_flag_is_true(monkeypatch):
    report=displayed_fixture();report['source_verification']['status']='invalid'
    monkeypatch.setattr(ui,'overview',lambda:report)
    app=run_app()
    assert not app.exception and not app.dataframe and not app.selectbox


def test_all_seven_modes_show_their_own_ledger_rules_and_fair_curves(monkeypatch):
    report=displayed_fixture();charts=[]
    monkeypatch.setattr(ui,'overview',lambda:report)
    monkeypatch.setattr(ui.st,'line_chart',lambda data,**kwargs:charts.append(data.copy()))
    app=run_app()
    assert not app.exception
    table=next(f.value for f in app.dataframe if '累積淨報酬' in f.value)
    assert len(table)==8 and table.iloc[-1]['方法']=='0050持有'
    assert table.iloc[0]['相對0050（百分點）']=='+2.00'
    for mode in ui.MODES:
        app.selectbox(key='exit_method').set_value(mode).run()
        assert not app.exception
        ledger=next(f.value for f in app.dataframe if '成交參考價' in f.value)
        assert ledger.iloc[0]['代號']=='0050'
        assert ledger.iloc[0]['交易別']=='整張'
        assert ledger.iloc[1]['股數']==report['methods'][mode]['trades'][1]['qty']
        assert ledger.iloc[1]['當日期末資產']==report['methods'][mode]['daily'][1]['nav']
        assert ledger.iloc[1]['零股對手量檢查']=='高於：僅日量假設'
        expected=[ui.MODE_LABELS[mode]] + ([] if mode=='fixed63' else [ui.MODE_LABELS['fixed63']])+['0050持有']
        assert list(charts[-1])==expected and len(charts[-1])==7
        assert charts[-1].iloc[-1,0]==report['methods'][mode]['summary']['final_nav']
        annual=next(f.value for f in app.dataframe if '2026截至09/09' in f.value)
        assert annual.iloc[0]['方法']==ui.MODE_LABELS[mode]
        waits=next(f.value for f in app.dataframe if '部位含配股全數結束' in f.value)
        assert waits.iloc[0]['部位含配股全數結束']=='期末仍未結束'
    assert any('不會' in c.value and '回測' in c.value for c in app.caption)
    assert any('不是盤中成交瞬間資產' in c.value for c in app.caption)
    assert any('不是保證成交價' in c.value for c in app.caption)
    assert any('買入日還原收盤為起點' in c.value for c in app.caption)
    assert any('每一項都要符合' in c.value for c in app.caption)
    app.selectbox(key='exit_method').set_value('market_weak').run()
    assert any('0050連續兩天未站上120日均線' in m.value for m in app.markdown)


def test_csv_uses_readable_columns_original_stock_text_and_no_input_mutation():
    report=displayed_fixture();before=deepcopy(report)
    frame=ui._trades(report['methods']['adaptive'])
    csv=frame.to_csv(index=False).encode('utf-8-sig')
    assert csv.startswith(b'\xef\xbb\xbf') and b'0050' in csv
    assert '當日期末資產' in csv.decode('utf-8-sig') and '滑價現金成本' in frame
    assert report==before


@pytest.mark.parametrize('reason,label',[('leader_entry','族群領先訊號進場'),
    ('fund_stock','賣0050準備買股'),('idle_cash','閒置資金投入0050')])
def test_actual_engine_board_channel_and_reasons_are_translated_in_app_and_csv(monkeypatch,reason,label):
    report=displayed_fixture()
    report['methods']['fixed63']['trades'][0].update(channel='board',reason=reason)
    monkeypatch.setattr(ui,'overview',lambda:report)
    app=run_app()
    assert not app.exception
    ledger=next(f.value for f in app.dataframe if '成交參考價' in f.value)
    assert ledger.iloc[0]['交易別']=='整張' and ledger.iloc[0]['原因']==label
    csv=ui._trades(report['methods']['fixed63']).to_csv(index=False)
    assert label in csv and reason not in csv and 'board' not in csv


def test_source_change_during_verification_is_rejected(sealed,monkeypatch):
    original=ui._verify_driver
    def racing(output):
        result=original(output)
        path=sealed['root']/'raw/parent-prices.parquet'
        path.write_text('changed while verifying')
        return result
    monkeypatch.setattr(ui,'_verify_driver',racing)
    assert ui.overview()['available'] is False


def test_certificate_restriction_warning_follows_selected_case_and_survives_normalization(monkeypatch):
    raw = fixture()
    action = dict(stock_id='2880', date='2024-08-13', kind='stock_dividend',
        action_id='synthetic-2880-stock', certificate_restriction=dict(
            certificate_delivery_date='2024-08-30', ordinary_share_available_date='2024-09-30',
            valuation_basis='ordinary_share_close_proxy', certificate_trading_modeled=False))
    raw['cases']['weak20']['summary']['restricted_certificate_actions'] = [action]
    report = ui._normalize(raw)
    report['source_verification'] = {'status':'verified', 'file_count':40}
    assert report['methods']['weak20']['summary']['restricted_certificate_actions']==[action]
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    assert not app.exception and not app.warning
    app.selectbox(key='exit_method').set_value('weak20').run()
    assert not app.exception
    warning, = app.warning
    for text in ('已發放增資權利證書', '等換成普通股才賣', '普通股價代理估值',
                 '占用持股名額', '不是實際交易的保守下限'):
        assert text in warning.value
    app.selectbox(key='exit_method').set_value('fixed63').run()
    assert not app.exception and not app.warning


@pytest.mark.parametrize('sid', ['2881', '2880', '2736'])
def test_fractional_cash_caption_is_issuer_independent_and_not_available_cash(monkeypatch, sid):
    report = displayed_fixture()
    report['methods']['weak20']['summary'].update(unverified_fractional_cash_amount=3.,
        unverified_fractional_cash_receivables=[dict(stock_id=sid, amount=3., kind='cash',
            pay_date=None, action_id=f'{sid}-stock-fractional-cash')])
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    assert not app.exception
    assert not any('畸零股毛額應收' in c.value for c in app.caption)
    app.selectbox(key='exit_method').set_value('weak20').run()
    assert not app.exception
    text = next(c.value for c in app.caption if '畸零股毛額應收' in c.value)
    assert 'NT$3' in text and '付款日與淨額未核實' in text and '沒有當成可用現金' in text
