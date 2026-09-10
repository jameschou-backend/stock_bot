"""Saved comparisons of event capacity and matched-candidate priority changes."""
import pandas as pd
import streamlit as st

from app.capacity_research import load_case, overview
from app.regime_switch_ui import REASONS, _ledger, pct


NAMES = {'control3': '原訊號・3 部位', 'capacity6': '同訊號・6 部位',
         'matched3': '可評分子集・原排序 3 部位',
         'residual3': '同子集・大盤校正排序 3 部位', 'benchmark': '0050 持有'}
SCENARIOS = {'官方價格・壓力成本': ('official', 'stress', 0),
             '官方價格・基本成本': ('official', 'base', 0),
             '舊價格・壓力成本': ('snapshot', 'stress', 0),
             '舊價格・基本成本': ('snapshot', 'base', 0),
             '官方價格・買入再晚一天': ('official', 'stress', 1)}
SCORE_REASONS = {'scored': '可評分', 'insufficient_history': '歷史窗口不足',
    'stock_not_in_prices': '缺少股票行情', 'missing_prices': '必要收盤價缺失',
    'missing_returns': '必要日報酬無效', 'price_anomaly': '窗口價格品質未通過',
    'benchmark_variance_too_small': '估計窗口大盤波動不足',
    'nonfinite_regression': '迴歸係數無效', 'nonfinite_score': '分數無效'}


def _name(row):
    return NAMES[row['rule']]


def _detail(report, row):
    case = load_case(report, row)
    if not case.get('available'):
        st.warning(case.get('note', '這份明細目前無法讀取。'))
        return
    if case.get('cohorts'):
        st.dataframe(pd.DataFrame([{'事件': item['event_id'], '持股': '、'.join(item['members']),
            '買入日': item['entry_date'], '退出日': item.get('exit_date') or '尚未退出',
            '退出原因': REASONS.get(item.get('exit_reason'), item.get('exit_reason') or '仍持有'),
            '退出受阻天數': item.get('blocked_exit_sessions', 0)} for item in case['cohorts']]),
            hide_index=True, use_container_width=True)
    for key, title in [('gate_rejections', '訊號日未通過趨勢條件'),
                       ('score_rejections', '無法評分的事件'), ('rejections', '執行時未買入的原因'),
                       ('executions', '逐筆成交')]:
        if case.get(key):
            frame = _ledger(case[key])
            if '原因' in frame:
                frame['原因'] = frame['原因'].map(lambda value: SCORE_REASONS.get(value, value))
            st.write(title)
            st.dataframe(frame, hide_index=True, use_container_width=True)
    if case.get('score_diagnostics'):
        st.write('每個分數使用哪段歷史？')
        st.caption('原分數是 20 日累計超額報酬；大盤校正分數是逐日扣除估計 α、β 影響後的報酬加總。兩欄定義不同，不能直接比較大小，只比較各自產生的順位。')
        scores = pd.DataFrame(case['score_diagnostics'])
        if 'reason' in scores:
            scores['reason'] = scores['reason'].map(lambda value: SCORE_REASONS.get(value, REASONS.get(value, value)))
        st.dataframe(scores.rename(columns={'event_id': '事件', 'stock_id': '代號', 'signal_date': '原訊號日',
            'fit_start': '估計窗口起日', 'fit_end': '估計窗口迄日',
            'score_start': '評分窗口起日', 'score_end': '評分窗口迄日',
            'fit_observations': '估計觀察數', 'score_observations': '評分觀察數',
            'alpha': '估計每日截距 α', 'beta': '估計市場敏感度 β',
            'original_priority': '原排序分數', 'residual_score': '大盤校正分數', 'reason': '評分結果',
            'missing_price_dates': '缺價日期', 'anomaly_dates': '價格疑點日期',
            'market_fit_variance': '估計窗口大盤變異數', 'available_prices': '可用歷史價數'}),
            hide_index=True, use_container_width=True)
    if case['valuation_audit']['findings']:
        st.write('保留對帳的持有估值疑點')
        st.dataframe(_ledger(case['valuation_audit']['findings']).rename(columns={'執行日': '估值日'}),
                     hide_index=True, use_container_width=True)
    if case['summary'].get('unliquidated_positions'):
        st.warning('這個方法期末含未平倉估值，並非全部可提領現金。')
        st.dataframe(_ledger(case['summary']['unliquidated_positions']), hide_index=True, use_container_width=True)


def render():
    st.subheader('多留幾個部位，或換個同日排序，能改善嗎？')
    report = overview()
    if not report.get('available'):
        st.info(report.get('note', '部位與排序研究尚未完成。'))
        return
    st.write('沿用同一批趨勢合格的領先訊號，分別測試 3 改 6 個部位，以及只調整同日事件的先後順位。')
    st.warning('全段歷史已經使用，這是探索試算，沒有未見測試，也尚未取得實盤資格。')
    st.caption(f"期間 {report['start']}～{report['end']}；新訊號截止 {report['signal_end']}。每筆最多使用當日資產的 1/3 或 1/6，持有 63 個交易日，其餘持有 0050；總起始資金相同、沒有槓桿。")
    selected = st.selectbox('部位與排序比較條件', list(SCENARIOS), key='capacity_scenario')
    basis, scenario, delay = SCENARIOS[selected]
    rows = [row for row in report['results'] if (row['basis'], row['scenario'], row['delay']) == SCENARIOS[selected]]
    baseline = next(row for row in report['baselines'] if (row['basis'], row['scenario']) == (basis, scenario))
    by_rule = {row['rule']: row for row in rows}
    all_rows = rows + [baseline]
    questionable = [row for row in all_rows if row['valuation_audit']['finding_count']]
    if questionable:
        st.error('下列方法仍有持有估值疑點，收益保留供對帳，不能當已核實績效：' + '、'.join(_name(row) for row in questionable))
    marked = [row for row in all_rows if not row['summary']['final_liquidation_complete']]
    if marked:
        st.warning('期末含未平倉估值，並非全部可提領現金：' + '、'.join(_name(row) for row in marked))
    for column, rule in zip(st.columns(3), ('control3', 'capacity6', 'residual3')):
        column.metric(NAMES[rule] + '：累積試算淨報酬', pct(by_rule[rule]['summary']['total_return']))
    table = []
    for row in all_rows:
        summary, rejected = row['summary'], row['rejection_counts']
        table.append({'方法': _name(row), '累積試算淨報酬': pct(summary['total_return']),
            '年化': pct(summary['cagr']), '最大跌幅': pct(abs(summary['max_drawdown'])),
            '平均個股比重': pct(summary['mean_active_weight']),
            '實際買入事件': summary['entered_cohorts'], '執行拒絕總數': summary['rejected_event_count'],
            '其中額滿拒絕': rejected.get('slots_full', 0), '估值疑點筆數': row['valuation_audit']['finding_count'],
            '期末': '已清算' if summary['final_liquidation_complete'] else '含未平倉估值'})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    st.caption(f"每邊滑價 {'0.45%' if scenario == 'stress' else '0.30%'}，另扣每邊 0.1425% 手續費、股票賣稅 0.3% 與 0050 賣稅 0.1%，包含換股前後所有交易。原訊號後第 {1 + delay} 個交易日買入，不使用買入日行情重排。")
    st.write('要看對照，才能分清楚改善來自哪裡。')
    comparisons = []
    labels = {'capacity6': '增加部位：6 部位 − 原 3 部位',
              'residual3': '更換排序：同子集新排序 − 同子集原排序',
              'matched3': '資料篩選：可評分子集原排序 − 全部原訊號'}
    for rule in ('capacity6', 'residual3', 'matched3'):
        row = by_rule[rule]
        reference = by_rule[row['reference_rule']]
        comparisons.append({'比較': labels[rule],
            '淨報酬差（百分點）': f"{row['contrast_vs_reference'] * 100:+.2f}",
            '最大跌幅差（百分點）': f"{(abs(row['summary']['max_drawdown']) - abs(reference['summary']['max_drawdown'])) * 100:+.2f}"})
    st.dataframe(pd.DataFrame(comparisons), hide_index=True, use_container_width=True)
    st.caption('淨報酬差越高越好；最大跌幅差為正，代表跌得更深。新排序必須與同一可評分子集的原排序比較；可評分子集相對全部訊號的差異，是資料篩選效果。')
    matched_comparison = next((item for item in report['cohort_comparisons']
        if (item['basis'], item['scenario'], item['delay'], item['rule'], item['reference'])
        == (basis, scenario, delay, 'residual3', 'matched3')), None)
    if matched_comparison and not matched_comparison['only_rule'] and not matched_comparison['only_reference']:
        st.info('實際買入事件相同，差異限於先後處理及資金分配；這輪無法判定完整殘差選股是否有效。')
    score_stats = report['signal_stats'][basis]
    if score_stats['scoreable_trend_count'] < score_stats['trend_count']:
        st.caption(f"本價格版本有 {score_stats['trend_count'] - score_stats['scoreable_trend_count']} 個趨勢事件無法評分，代表所需歷史窗口的資料條件未滿足，不能解讀為公司品質較差。")
    if not questionable:
        if by_rule['capacity6']['contrast_vs_reference'] <= 0:
            drawdown_difference = (abs(by_rule['capacity6']['summary']['max_drawdown'])
                                   - abs(by_rule['control3']['summary']['max_drawdown'])) * 100
            if drawdown_difference < 0:
                tradeoff = f'但最大跌幅減少 {abs(drawdown_difference):.2f} 個百分點，呈現報酬與跌幅的取捨。'
            elif drawdown_difference > 0:
                tradeoff = f'且最大跌幅增加 {drawdown_difference:.2f} 個百分點。'
            else:
                tradeoff = '最大跌幅相同。'
            st.info('增加到 6 部位未提高這組累積試算報酬；' + tradeoff + '分散部位也可能稀釋贏家的比重。')
        if by_rule['residual3']['contrast_vs_reference'] <= 0:
            st.info('在相同可評分事件中，新排序未提高這組累積試算報酬。')
    with st.expander('看曲線、年度收益、費用與拒絕原因'):
        if (basis, scenario, delay) == ('official', 'stress', 0):
            chart = pd.concat([pd.Series({point['date']: point['nav'] for point in report['charts'][rule]}, name=name)
                               for rule, name in NAMES.items()], axis=1)
            chart.index = pd.to_datetime(chart.index)
            st.line_chart(chart)
            st.caption('曲線為官方價格、壓力成本，起初資金設為 1。')
        else:
            st.caption('完整共同曲線保留在官方價格、壓力成本；下表是目前選定情境。')
        st.dataframe(pd.DataFrame([{'方法': _name(row), **{year: pct(value) for year, value in row['summary']['annual_returns'].items()}}
                                  for row in all_rows]), hide_index=True, use_container_width=True)
        st.caption('2026 年只到研究截止日；年度收益沿用完整持有路徑。')
        st.dataframe(pd.DataFrame([{'方法': _name(row), '累計成本／期初資金': pct(row['summary']['total_cost']),
            '雙向成交額／期初資金': f"{row['summary']['turnover']:.2f} 倍", '成交筆數': row['summary']['trade_count'],
            '重疊拒絕': row['rejection_counts'].get('overlapping_member', 0),
            '無法成交拒絕': row['rejection_counts'].get('entry_instruments_not_tradable', 0)} for row in all_rows]),
            hide_index=True, use_container_width=True)
    with st.expander('排序真的改了嗎？看候選池與實際持有差異'):
        stats = report['signal_stats'][basis]
        st.write(f"原始 {stats['original_count']} 個領先事件，趨勢條件通過 {stats['trend_count']} 個，其中 {stats['scoreable_trend_count']} 個可評分。同日多事件 {stats['multiple_event_days']} 天，排序改變 {stats['reordered_days']} 天。")
        st.caption('大盤校正排序只調整既有事件的同日處理順序，不是重新選股票或完整殘差動能選股。分數缺失明確排除，兩個子集方法同步使用相同候選池。')
        if not stats['reordered_days']:
            st.info('本次沒有改變同日事件處理順序，對新排序效應的檢驗力有限。')
        if stats['score_rejections']:
            st.dataframe(pd.DataFrame([{'無法評分原因': SCORE_REASONS.get(reason, REASONS.get(reason, reason)), '事件數': count}
                                      for reason, count in stats['score_rejections'].items()]), hide_index=True, use_container_width=True)
        if stats['reorder_examples']:
            st.caption('下表使用訊號準備時的原定買入日；額外延遲情境仍顯示原定日期，實際成交日期請查看成交明細。')
            st.dataframe(pd.DataFrame([{'原定買入日': example['date'], '原處理順序': ' → '.join(example['original_order']),
                '大盤校正後順序': ' → '.join(example['residual_order'])} for example in stats['reorder_examples']]),
                hide_index=True, use_container_width=True)
        changes = [item for item in report['cohort_comparisons'] if (item['basis'], item['scenario'], item['delay']) == (basis, scenario, delay)]
        if changes:
            st.dataframe(pd.DataFrame([{'方法': NAMES[item['rule']], '對照': NAMES[item['reference']],
                '共同買入事件數': item['shared'], '只有本方法買入': '、'.join(item['only_rule']) or '無',
                '只有對照買入': '、'.join(item['only_reference']) or '無'} for item in changes]),
                hide_index=True, use_container_width=True)
    if st.checkbox('查看這個方法的成交與分數', key='capacity_details'):
        rule = st.selectbox('檢查哪個部位或排序方法', [row['rule'] for row in all_rows], index=1,
                            format_func=lambda value: NAMES[value], key='capacity_method')
        _detail(report, next(row for row in all_rows if row['rule'] == rule))
    with st.expander('研究限制'):
        for note in report['limitations']:
            st.caption(note)
    st.caption(f"效能：分數準備 {report['preparation_elapsed_seconds']:.1f} 秒，20 組方法與 4 組基準含稽核 {report['elapsed_seconds']:.1f} 秒；研究使用 {report['finmind_requests']} 次 FinMind。選單只讀結果，明細勾選後才載入。")
