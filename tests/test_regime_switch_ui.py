"""Synthetic UI fixtures verify displayed scenarios and lazy ledger semantics."""
from copy import deepcopy

from streamlit.testing.v1 import AppTest

from app import regime_switch_ui as ui


def row(rule, basis, scenario, delay=0):
    i = list(ui.NAMES).index(rule)
    value = .4 + i / 10 + (1 if basis == 'snapshot' else 0) + (.2 if scenario == 'base' else 0) + delay / 100
    return {'rule': rule, 'name': ui.NAMES[rule], 'basis': basis, 'scenario': scenario,
            'delay': delay, 'case_file': f'case-{rule}-{basis}-{scenario}-{delay}.json',
            'summary': {'total_return': value, 'cagr': value / 5,
                'max_drawdown': -.3 if rule == 'exit_cash' else -.2,
                'mean_cash_weight': .25 if rule in ('idle_cash', 'exit_cash') else 0,
                'forced_exit_requested_cohorts': 3 if rule == 'exit_cash' else 0,
                'forced_exit_completed_cohorts': 2 if rule == 'exit_cash' else 0,
                'final_liquidation_complete': True, 'entered_cohorts': 4, 'completed_cohorts': 4,
                'annual_returns': {'2022': value / 2, '2023': value / 4},
                'total_cost': .03, 'turnover': 7.2, 'trade_count': 10},
            'valuation_audit': {'finding_count': 0, 'findings': []}}


def fixture():
    results = [row(rule, basis, scenario, delay)
               for basis, scenario, delay in ui.SCENARIOS.values() for rule in ui.NAMES if rule != 'benchmark']
    baselines = [row('benchmark', basis, scenario) for basis in ('official', 'snapshot') for scenario in ('base', 'stress')]
    return {'available': True, 'start': '2022-01-03', 'end': '2026-06-23', 'signal_end': '2025-12-31',
            'results': results, 'baselines': baselines,
            'charts': {rule: [{'date': '2022-01-03', 'nav': 1.}, {'date': '2026-06-23', 'nav': 1.5}]
                       for rule in ui.NAMES},
            'states': {'elapsed_seconds': .03, 'stats': {
                basis: {'state_days': {'ON': 4, 'OFF': 2, 'UNKNOWN': 1},
                        'observed_transitions': [{'date': '2022-01-03', 'state': 'ON', 'close': 100., 'ma120': 99.},
                                                 {'date': '2022-01-06', 'state': 'OFF', 'close': 90., 'ma120': 99.}]}
                for basis in ('official', 'snapshot')}},
            'elapsed_seconds': 1.2, 'finmind_requests': 0,
            'limitations': ['同一歷史已使用，不是未見測試。']}


def app():
    return AppTest.from_string('from app.regime_switch_ui import render\nrender()').run(timeout=15)


def table(rendered):
    return next(frame.value for frame in rendered.dataframe if '累積試算淨報酬' in frame.value)


def test_missing_report_is_friendly_without_loading_case(monkeypatch):
    calls = []
    monkeypatch.setattr(ui, 'overview', lambda: {'available': False, 'note': '研究尚未完成，請先執行研究。'})
    monkeypatch.setattr(ui, 'load_case', lambda *args: calls.append(args))
    rendered = app()
    assert not rendered.exception
    assert rendered.info[0].value == '研究尚未完成，請先執行研究。'
    assert not rendered.metric and not rendered.dataframe and not calls


def test_all_seven_rows_switch_scenarios_and_keep_ledger_lazy(monkeypatch):
    report, calls = fixture(), []
    monkeypatch.setattr(ui, 'overview', lambda: report)
    monkeypatch.setattr(ui, 'load_case', lambda *args: calls.append(args))
    rendered = app()
    assert not rendered.exception
    assert len(table(rendered)) == 7
    assert {'最大跌幅', '平均現金比重', '估值疑點筆數', '期末', '轉弱要求／已退出事件'}.issubset(table(rendered))
    assert any('未勝過' in info.value for info in rendered.info)
    assert any('最大跌幅仍大於' in warning.value for warning in rendered.warning)
    assert any('之後不再平衡' in caption.value for caption in rendered.caption)
    assert any('廣度方案先前因資料不足' in caption.value for caption in rendered.caption)
    assert any('沒有未見測試' in warning.value for warning in rendered.warning)
    for choice, key in ui.SCENARIOS.items():
        rendered.selectbox(key='regime_switch_scenario').set_value(choice).run()
        assert not rendered.exception
        expected = next(result for result in report['results'] if result['rule'] == 'exit_cash'
                        and (result['basis'], result['scenario'], result['delay']) == key)
        assert rendered.metric[0].value == ui.pct(expected['summary']['total_return'])
        primary = table(rendered).set_index('方法').loc[ui.NAMES['exit_cash']]
        assert primary['累積試算淨報酬'] == ui.pct(expected['summary']['total_return'])
        assert primary['最大跌幅'] == '30.00%'
        annual = next(frame.value for frame in rendered.dataframe if '2022' in frame.value)
        assert annual.set_index('方法').loc[ui.NAMES['exit_cash'], '2022'] == ui.pct(expected['summary']['annual_returns']['2022'])
    assert any('延遲 2 個交易日' in caption.value for caption in rendered.caption)
    assert not calls


def test_unresolved_valuation_prevents_positive_claim_and_marked_assets_warn(monkeypatch):
    report = fixture()
    primary = next(result for result in report['results'] if result['rule'] == 'exit_cash'
                   and (result['basis'], result['scenario'], result['delay']) == ('official', 'stress', 0))
    primary['summary'].update(total_return=100., max_drawdown=0., final_liquidation_complete=False)
    primary['valuation_audit']['finding_count'] = 2
    monkeypatch.setattr(ui, 'overview', lambda: report)
    rendered = app()
    assert not rendered.exception
    assert any('2 筆持有估值疑點' in error.value for error in rendered.error)
    assert any('未平倉估值' in warning.value for warning in rendered.warning)
    assert not any('較有利' in info.value for info in rendered.info)
    primary_row = table(rendered).set_index('方法').loc[ui.NAMES['exit_cash']]
    assert primary_row['期末'] == '含未平倉估值' and primary_row['估值疑點筆數'] == 2


def test_case_is_loaded_only_on_request_and_translates_recorded_exits(monkeypatch):
    report, calls = fixture(), []
    monkeypatch.setattr(ui, 'overview', lambda: report)
    def load(saved_report, selected):
        calls.append(selected['rule'])
        assert saved_report is report
        return {**deepcopy(selected), 'available': True,
                'cohorts': [{'event_id': 'a', 'members': ['2330'], 'entry_date': '2022-01-04',
                             'exit_date': '2022-01-07', 'exit_reason': 'forced_exit',
                             'forced_exit_requested_on': '2022-01-07', 'blocked_exit_sessions': 0}],
                'gate_rejections': [{'event_id': 'b', 'signal_date': '2022-01-06', 'reason': 'trend_unknown'}],
                'executions': [{'date': '2022-01-07', 'event_id': 'a', 'stock_id': '2330',
                                'side': 'sell', 'reason': 'forced_exit', 'price': 100., 'units': .2}]}
    monkeypatch.setattr(ui, 'load_case', load)
    rendered = app()
    assert not calls
    rendered.checkbox(key='regime_switch_details').set_value(True).run()
    assert not rendered.exception and calls == ['exit_cash']
    cohorts = next(frame.value for frame in rendered.dataframe if '退出原因' in frame.value)
    assert cohorts.iloc[0]['退出原因'] == '轉弱要求退出'
    rejected = next(frame.value for frame in rendered.dataframe if '訊號日' in frame.value)
    assert rejected.iloc[0]['原因'] == '訊號日趨勢資料未知'
    trades = next(frame.value for frame in rendered.dataframe if '買賣' in frame.value)
    assert trades.iloc[0]['買賣'] == '賣出' and trades.iloc[0]['原因'] == '轉弱要求退出'


def test_mixed_case_explains_separate_accounts_without_fake_trade_ledger(monkeypatch):
    report, calls = fixture(), []
    monkeypatch.setattr(ui, 'overview', lambda: report)
    def load(saved_report, selected):
        calls.append(selected['rule'])
        source = next(result for result in report['results'] if result['rule'] == 'entry_only'
                      and (result['basis'], result['scenario'], result['delay']) == ('official', 'stress', 0))
        benchmark = next(result for result in report['baselines'] if (result['basis'], result['scenario']) == ('official', 'stress'))
        return {**deepcopy(selected), 'available': True,
                'components': [{'sleeve': 'strategy', 'initial_weight': .5, 'case_file': source['case_file']},
                               {'sleeve': 'benchmark', 'initial_weight': .5, 'case_file': benchmark['case_file']}]}
    monkeypatch.setattr(ui, 'load_case', load)
    rendered = app()
    rendered.checkbox(key='regime_switch_details').set_value(True).run()
    rendered.selectbox(key='regime_switch_method').set_value('mix_entry').run()
    assert not rendered.exception and calls[-1] == 'mix_entry'
    assert any('沒有一份合併下單帳本' in info.value for info in rendered.info)
    components = next(frame.value for frame in rendered.dataframe if '獨立帳戶' in frame.value)
    assert list(components['起初資金比重']) == ['50.00%', '50.00%']
    assert set(components['獨立帳戶']) == {ui.NAMES['entry_only'], ui.NAMES['benchmark']}
    assert not any('事件' in frame.value or '買賣' in frame.value for frame in rendered.dataframe)


def test_changed_case_source_hides_details_with_specific_note(monkeypatch):
    monkeypatch.setattr(ui, 'overview', fixture)
    monkeypatch.setattr(ui, 'load_case', lambda *args: {'available': False, 'note': '逐筆來源已變更，請重新研究。'})
    rendered = app()
    rendered.checkbox(key='regime_switch_details').set_value(True).run()
    assert not rendered.exception
    assert any('逐筆來源已變更' in warning.value for warning in rendered.warning)
    assert not any('事件' in frame.value for frame in rendered.dataframe)
