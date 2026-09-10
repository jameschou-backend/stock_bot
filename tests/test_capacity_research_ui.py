"""Synthetic capacity UI comparisons: correct controls and lazy evidence loading."""
from copy import deepcopy

from streamlit.testing.v1 import AppTest

from app import capacity_research_ui as ui


def make_row(rule, basis, scenario, delay=0):
    returns = {'control3': .5, 'capacity6': .6, 'matched3': 1.2, 'residual3': 1.1, 'benchmark': .4}
    extra = (2 if basis == 'snapshot' else 0) + (.1 if scenario == 'base' else 0) + delay / 100
    reference = 'matched3' if rule == 'residual3' else 'control3'
    return {'rule': rule, 'name': ui.NAMES[rule], 'basis': basis, 'scenario': scenario, 'delay': delay,
            'reference_rule': reference, 'contrast_vs_reference': returns[rule] - returns[reference],
            'case_file': f'case-{rule}-{basis}-{scenario}-{delay}.json',
            'summary': {'total_return': returns[rule] + extra, 'cagr': (returns[rule] + extra) / 5,
                'max_drawdown': -.4 if rule == 'capacity6' else -.3,
                'mean_active_weight': .7 if rule == 'capacity6' else 0. if rule == 'benchmark' else .6,
                'mean_cash_weight': 0, 'entered_cohorts': 0 if rule == 'benchmark' else 4,
                'rejected_event_count': 0 if rule == 'benchmark' else 3, 'final_liquidation_complete': True,
                'total_cost': .03 if scenario == 'stress' else .02, 'turnover': 6., 'trade_count': 18,
                'annual_returns': {'2022': (returns[rule] + extra) / 2, '2023': .1}},
            'rejection_counts': {} if rule == 'benchmark' else {'slots_full': 2, 'overlapping_member': 1},
            'valuation_audit': {'finding_count': 0, 'findings': []}}


def fixture():
    results = [make_row(rule, basis, scenario, delay) for basis, scenario, delay in ui.SCENARIOS.values()
               for rule in ui.NAMES if rule != 'benchmark']
    return {'available': True, 'start': '2022-01-03', 'end': '2026-06-23', 'signal_end': '2025-12-31',
            'results': results,
            'baselines': [make_row('benchmark', basis, scenario) for basis in ('official', 'snapshot') for scenario in ('base', 'stress')],
            'charts': {rule: [{'date': '2022-01-03', 'nav': 1.}, {'date': '2026-06-23', 'nav': 1.5}] for rule in ui.NAMES},
            'signal_stats': {basis: {'original_count': 20, 'trend_count': 10, 'scoreable_trend_count': 8,
                'score_rejections': {'missing_prices': 2}, 'multiple_event_days': 3, 'reordered_days': 1,
                'reorder_examples': [{'date': '2022-01-04', 'original_order': ['a', 'b'], 'residual_order': ['b', 'a']}]}
                for basis in ('official', 'snapshot')},
            'cohort_comparisons': [{'basis': basis, 'scenario': scenario, 'delay': delay, 'rule': 'residual3',
                                   'reference': 'matched3', 'shared': 3, 'only_rule': ['b'], 'only_reference': ['a']}
                                  for basis, scenario, delay in ui.SCENARIOS.values()],
            'elapsed_seconds': 1.3, 'preparation_elapsed_seconds': .2, 'finmind_requests': 0,
            'limitations': ['全部歷史已使用，沒有未見測試。']}


def run_app():
    return AppTest.from_string('from app.capacity_research_ui import render\nrender()').run(timeout=15)


def results_table(app):
    return next(frame.value for frame in app.dataframe if '累積試算淨報酬' in frame.value)


def test_missing_report_has_no_details_or_metrics(monkeypatch):
    calls = []
    monkeypatch.setattr(ui, 'overview', lambda: {'available': False, 'note': '尚未完成部位比較。'})
    monkeypatch.setattr(ui, 'load_case', lambda *args: calls.append(args))
    app = run_app()
    assert not app.exception and not calls and not app.metric and not app.dataframe
    assert app.info[0].value == '尚未完成部位比較。'


def test_five_scenarios_show_correct_returns_costs_and_lazy_details(monkeypatch):
    report, calls = fixture(), []
    monkeypatch.setattr(ui, 'overview', lambda: report)
    monkeypatch.setattr(ui, 'load_case', lambda *args: calls.append(args))
    app = run_app()
    assert not app.exception and len(results_table(app)) == 5
    assert {'平均個股比重', '實際買入事件', '執行拒絕總數', '其中額滿拒絕', '最大跌幅'}.issubset(results_table(app))
    for choice, scenario in ui.SCENARIOS.items():
        app.selectbox(key='capacity_scenario').set_value(choice).run()
        assert not app.exception
        expected = next(row for row in report['results'] if row['rule'] == 'capacity6'
                        and (row['basis'], row['scenario'], row['delay']) == scenario)
        assert app.metric[1].value == ui.pct(expected['summary']['total_return'])
        displayed = results_table(app).set_index('方法').loc[ui.NAMES['capacity6']]
        assert displayed['累積試算淨報酬'] == ui.pct(expected['summary']['total_return'])
        assert displayed['最大跌幅'] == '40.00%'
        assert displayed['平均個股比重'] == '70.00%' and displayed['其中額滿拒絕'] == 2
        costs = next(frame.value for frame in app.dataframe if '累計成本／期初資金' in frame.value)
        assert costs.set_index('方法').loc[ui.NAMES['capacity6'], '累計成本／期初資金'] == ui.pct(expected['summary']['total_cost'])
        slip = '0.45%' if scenario[1] == 'stress' else '0.30%'
        assert any('每邊滑價 ' + slip in caption.value for caption in app.caption)
    assert any('第 2 個交易日買入' in caption.value for caption in app.caption)
    ordering = next(frame.value for frame in app.dataframe if '原處理順序' in frame.value)
    assert '原訊號日' not in ordering and ordering.iloc[0]['原定買入日'] == '2022-01-04'
    assert any('額外延遲情境仍顯示原定日期' in caption.value for caption in app.caption)
    assert not calls


def test_residual_compares_against_matched_candidates_not_full_pool(monkeypatch):
    monkeypatch.setattr(ui, 'overview', fixture)
    app = run_app()
    assert not app.exception
    # New ranking beats the full pool, but loses its correct matched control.
    assert app.metric[2].value == '110.00%' and app.metric[0].value == '50.00%'
    comparisons = next(frame.value for frame in app.dataframe if '淨報酬差（百分點）' in frame.value).set_index('比較')
    assert comparisons.loc['更換排序：同子集新排序 − 同子集原排序', '淨報酬差（百分點）'] == '-10.00'
    assert comparisons.loc['資料篩選：可評分子集原排序 − 全部原訊號', '淨報酬差（百分點）'] == '+70.00'
    assert comparisons.loc['增加部位：6 部位 − 原 3 部位', '最大跌幅差（百分點）'] == '+10.00'
    assert any('相同可評分事件中，新排序未提高' in info.value for info in app.info)
    assert any('是資料篩選效果' in caption.value for caption in app.caption)
    selections = next(frame.value for frame in app.dataframe if '共同買入事件數' in frame.value)
    assert selections.iloc[0]['對照'] == ui.NAMES['matched3'] and selections.iloc[0]['共同買入事件數'] == 3


def test_valuation_and_terminal_flags_remain_visible(monkeypatch):
    report = fixture()
    primary = next(row for row in report['results'] if row['rule'] == 'capacity6'
                   and (row['basis'], row['scenario'], row['delay']) == ('official', 'stress', 0))
    primary['summary']['final_liquidation_complete'] = False
    primary['valuation_audit']['finding_count'] = 2
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    assert not app.exception
    assert any('不能當已核實績效' in item.value for item in app.error)
    assert any('未平倉估值' in item.value for item in app.warning)
    primary_row = results_table(app).set_index('方法').loc[ui.NAMES['capacity6']]
    assert primary_row['估值疑點筆數'] == 2 and primary_row['期末'] == '含未平倉估值'


def test_no_reordering_is_reported_as_limited_power(monkeypatch):
    report = fixture()
    report['signal_stats']['official'].update(reordered_days=0, reorder_examples=[])
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    assert not app.exception
    assert any('檢驗力有限' in item.value for item in app.info)
    missing = next(frame.value for frame in app.dataframe if '無法評分原因' in frame.value)
    assert missing.iloc[0]['無法評分原因'] == '必要收盤價缺失'


def test_capacity_tradeoff_and_same_actual_events_are_explained_without_historical_constants(monkeypatch):
    report = fixture()
    for row in report['results']:
        if row['rule'] == 'capacity6':
            row['summary']['total_return'] -= .2
            row['contrast_vs_reference'] -= .2
            row['summary']['max_drawdown'] = -.15
    for comparison in report['cohort_comparisons']:
        comparison.update(shared=4, only_rule=[], only_reference=[])
    monkeypatch.setattr(ui, 'overview', lambda: report)
    app = run_app()
    assert not app.exception
    assert any('未提高這組累積試算報酬' in item.value and '最大跌幅減少 15.00 個百分點' in item.value for item in app.info)
    assert any('實際買入事件相同' in item.value and '無法判定完整殘差選股是否有效' in item.value for item in app.info)
    assert any('2 個趨勢事件無法評分' in item.value and '不能解讀為公司品質較差' in item.value for item in app.caption)
    app.selectbox(key='capacity_scenario').set_value('官方價格・買入再晚一天').run()
    assert not app.exception
    assert any('實際買入事件相同' in item.value for item in app.info)


def test_different_actual_events_do_not_receive_same_event_claim(monkeypatch):
    monkeypatch.setattr(ui, 'overview', fixture)
    app = run_app()
    assert not app.exception
    assert not any('實際買入事件相同' in item.value for item in app.info)


def test_details_use_exact_saved_case_and_explain_distinct_score_definitions(monkeypatch):
    report, calls = fixture(), []
    monkeypatch.setattr(ui, 'overview', lambda: report)
    def load(saved, row):
        assert saved is report
        calls.append((row['rule'], row['basis'], row['scenario'], row['delay']))
        return {**deepcopy(row), 'available': True,
                'cohorts': [{'event_id': 'a', 'members': ['2330'], 'entry_date': '2022-01-05',
                             'exit_date': '2022-04-06', 'exit_reason': 'scheduled_exit'}],
                'rejections': [{'event_id': 'b', 'entry_date': '2022-01-05', 'reason': 'slots_full'}],
                'score_rejections': [{'event_id': 'c', 'reason': 'missing_prices'}],
                'score_diagnostics': [{'event_id': 'a', 'signal_date': '2022-01-04',
                    'fit_start': '2021-06-14', 'fit_end': '2021-12-07',
                    'score_start': '2021-12-08', 'score_end': '2022-01-04',
                    'alpha': .001, 'beta': 1.2, 'original_priority': .2,
                    'residual_score': .1, 'reason': 'scored'}],
                'executions': [{'date': '2022-01-05', 'event_id': 'a', 'stock_id': '2330',
                                'side': 'buy', 'reason': 'event_entry'}]}
    monkeypatch.setattr(ui, 'load_case', load)
    app = run_app()
    assert not calls
    app.checkbox(key='capacity_details').set_value(True).run()
    assert calls == [('capacity6', 'official', 'stress', 0)]
    app.selectbox(key='capacity_method').set_value('residual3').run()
    app.selectbox(key='capacity_scenario').set_value('舊價格・基本成本').run()
    assert not app.exception and calls[-1] == ('residual3', 'snapshot', 'base', 0)
    assert any('不能直接比較大小' in caption.value for caption in app.caption)
    scores = next(frame.value for frame in app.dataframe if '原排序分數' in frame.value)
    assert scores.iloc[0]['原排序分數'] == .2 and scores.iloc[0]['大盤校正分數'] == .1
    assert scores.iloc[0]['估計窗口迄日'] == '2021-12-07' and scores.iloc[0]['評分結果'] == '可評分'
    trades = next(frame.value for frame in app.dataframe if '買賣' in frame.value)
    assert trades.iloc[0]['買賣'] == '買入' and trades.iloc[0]['原因'] == '事件買入'


def test_changed_detail_source_does_not_show_unverified_trades(monkeypatch):
    monkeypatch.setattr(ui, 'overview', fixture)
    monkeypatch.setattr(ui, 'load_case', lambda *args: {'available': False, 'note': '成交來源已變更。'})
    app = run_app()
    app.checkbox(key='capacity_details').set_value(True).run()
    assert not app.exception
    assert any('成交來源已變更' in item.value for item in app.warning)
    assert not any('買賣' in frame.value for frame in app.dataframe)
