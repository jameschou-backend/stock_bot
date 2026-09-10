"""Read-only comparison of cash switching and independently funded mixtures."""
import math

import pandas as pd
import streamlit as st

from app.regime_switch_research import load_case, overview


NAMES = {'always': '所有領先訊號', 'entry_only': '趨勢只管進場',
         'idle_cash': '轉弱只收回閒置 0050', 'exit_cash': '轉弱連個股一起退出',
         'mix_always': '初始一半領先策略、一半 0050',
         'mix_entry': '初始一半趨勢進場、一半 0050', 'benchmark': '0050 持有'}
SCENARIOS = {'官方價格・壓力成本': ('official', 'stress', 0),
             '官方價格・基本成本': ('official', 'base', 0),
             '舊價格・壓力成本': ('snapshot', 'stress', 0),
             '舊價格・基本成本': ('snapshot', 'base', 0),
             '官方價格・狀態與買入再晚一天': ('official', 'stress', 1)}
STATES = {'ON': '均線上方', 'OFF': '均線下方', 'UNKNOWN': '資料未知', 'CASH': '初始現金'}
REASONS = {
    'forced_exit': '轉弱要求退出', 'scheduled_exit': '持有期滿',
    'terminal_liquidation': '研究期末清算', 'final_exit_blocked': '期末退出受阻',
    'initial_benchmark': '初始投入 0050', 'park_on': '轉強後現金投入 0050',
    'park_off': '轉弱後 0050 轉現金', 'fund_event_basket': '賣出 0050 支付新持股',
    'event_entry': '事件買入', 'basket_entry': '事件買入',
    'return_to_benchmark': '退出持股後轉回 0050',
    'scheduled_return_to_benchmark': '到期持股轉回 0050',
    'buy_benchmark': '買入 0050', 'sell_benchmark': '賣出 0050',
    'market_state_not_on': '執行指令非均線上方', 'trend_unknown': '訊號日趨勢資料未知',
    'trend_off': '訊號日位於均線下方', 'overlapping_member': '與現有持股重疊',
    'slots_full': '同時事件名額已滿', 'entry_instruments_not_tradable': '必要標的無法成交',
    'no_benchmark_funding': '可轉出的 0050 資金不足', 'terminal_session': '期末不再開倉',
    'outside_window': '超出研究期間', 'unknown_member': '標的行情不存在',
    'invalid_signal_date': '訊號日期無效', 'invalid_entry_date': '買入日期無效',
    'signal_date_outside_calendar': '訊號日期超出行情日曆',
    'signal_date_not_trading_session': '訊號日不是行情交易日',
    'entry_date_not_after_signal': '買入日未晚於訊號日',
    'entry_date_outside_calendar': '買入日期超出行情日曆',
    'entry_date_not_trading_session': '買入日不是行情交易日',
    'entry_date_not_next_session': '原買入日不是訊號後下一交易日',
    'entry_delay_outside_calendar': '延後買入超出行情日曆'}


def pct(value):
    return f'{value:.2%}' if value is not None and math.isfinite(value) else '未知'


def _name(row):
    return NAMES.get(row['rule'], row.get('name', row['rule']))


def _ledger(records):
    """Translate recorded facts without inventing combined-account trades."""
    frame = pd.DataFrame(records)
    for column in ('reason', 'exit_reason', 'action'):
        if column in frame:
            frame[column] = frame[column].map(lambda value: REASONS.get(value, value))
    for column in ('state', 'from_state', 'to_state', 'park_state', 'from_park_state', 'trend_state'):
        if column in frame:
            frame[column] = frame[column].map(lambda value: STATES.get(value, value))
    if 'side' in frame:
        frame['side'] = frame['side'].map({'buy': '買入', 'sell': '賣出'})
    if 'members' in frame:
        frame['members'] = frame['members'].map(lambda values: '、'.join(values) if isinstance(values, list) else values)
    return frame.rename(columns={'date': '執行日', 'event_id': '事件', 'stock_id': '代號',
        'signal_date': '訊號日', 'entry_date': '預定買入日', 'decision_date': '依據觀察日',
        'trend_decision_date': '趨勢觀察日', 'members': '持股', 'reason': '原因',
        'exit_reason': '退出原因', 'side': '買賣', 'price': '還原單位價格', 'units': '還原單位數量',
        'cost': '成本／期初資金', 'notional': '成交額／期初資金', 'action': '受阻動作',
        'from_state': '前指令', 'to_state': '本次指令', 'state': '指令',
        'trend_state': '訊號日狀態', 'park_state': '閒置資金目標',
        'from_park_state': '前閒置資金目標', 'park_decision_date': '配置依據觀察日',
        'observed_return_since_last_quote': '距前次有效報價漲跌',
        'price_basis_difference': '兩價格版本差異'})


def _case_details(report, row):
    detail = load_case(report, row)
    if not detail.get('available'):
        st.warning(detail.get('note', '逐筆紀錄目前無法讀取。'))
        return
    if detail.get('components'):
        st.info('這是起初各分 50% 資金的兩個獨立帳戶，之後不再平衡。收益與成本按初始資金加總，沒有一份合併下單帳本。')
        components = []
        all_rows = report['results'] + report['baselines']
        for component in detail['components']:
            source = next((item for item in all_rows if item['case_file'] == component['case_file']), None)
            components.append({'獨立帳戶': _name(source) if source else component['sleeve'],
                               '起初資金比重': pct(component['initial_weight'])})
        st.dataframe(pd.DataFrame(components), hide_index=True, use_container_width=True)
        st.caption('要檢查成交，請在上方方法選單選擇對應策略或「0050 持有」，分別查看各帳戶原始紀錄。')
        return
    cohorts = detail.get('cohorts', [])
    st.write(f"實際買入 {detail['summary'].get('entered_cohorts', 0)} 個事件，完成 {detail['summary'].get('completed_cohorts', 0)} 個。")
    if cohorts:
        st.dataframe(pd.DataFrame([{'事件': item['event_id'], '持股': '、'.join(item['members']),
            '買入日': item['entry_date'], '退出日': item.get('exit_date') or '尚未退出',
            '退出原因': REASONS.get(item.get('exit_reason'), item.get('exit_reason') or '仍持有'),
            '要求轉弱退出日': item.get('forced_exit_requested_on') or '無',
            '退出受阻天數': item.get('blocked_exit_sessions', 0)} for item in cohorts]),
            hide_index=True, use_container_width=True)
    for key, label in [('state_changes', '已延遲的配置指令'),
                       ('forced_exit_requests', '轉弱退出要求'), ('etf_blocked', '0050 配置受阻'),
                       ('gate_rejections', '訊號日未通過趨勢條件'), ('rejections', '執行時未買入的原因'),
                       ('executions', '逐筆成交')]:
        if detail.get(key):
            st.write(label)
            st.dataframe(_ledger(detail[key]), hide_index=True, use_container_width=True)
    findings = detail.get('valuation_audit', {}).get('findings', [])
    if findings:
        st.write('保留對帳的估值疑點')
        st.dataframe(_ledger(findings).rename(columns={'執行日': '估值日'}), hide_index=True, use_container_width=True)
    if detail['summary'].get('unliquidated_positions'):
        st.warning('這個方法期末仍有未平倉估值，並非全部可提領現金。')
        st.dataframe(_ledger(detail['summary']['unliquidated_positions']), hide_index=True, use_container_width=True)


def render():
    st.subheader('市場轉弱時，要不要切策略、留現金？')
    report = overview()
    if not report.get('available'):
        st.info(report.get('note', '情境切換研究尚未完成。'))
        return
    st.write('先看 0050 是否高於最近 120 次有效收盤均價，再比較只限制新買入、收回閒置資金，以及連個股一起退出。')
    st.warning('這段歷史已經用過，全部是探索結果，沒有未見測試，也尚未取得實盤資格。')
    st.caption(f"共同期間 {report['start']}～{report['end']}；新訊號截止 {report['signal_end']}。最多同時 3 個事件，原持有期 63 個交易日；現金利息假設 0。")
    selected = st.selectbox('情境切換比較條件', list(SCENARIOS), key='regime_switch_scenario')
    basis, scenario, delay = SCENARIOS[selected]
    rows = [row for row in report['results'] if (row['basis'], row['scenario'], row['delay']) == SCENARIOS[selected]]
    benchmark = next(row for row in report['baselines'] if (row['basis'], row['scenario']) == (basis, scenario))
    primary = next(row for row in rows if row['rule'] == 'exit_cash')
    entry = next(row for row in rows if row['rule'] == 'entry_only')
    fixed = next(row for row in rows if row['rule'] == 'mix_entry')
    all_rows = rows + [benchmark]
    unresolved = primary['valuation_audit']['finding_count']
    if unresolved:
        st.error(f'主要方法有 {unresolved} 筆持有估值疑點。下列數字保留供對帳，不能解讀成已核實績效。')
    elif any(row['valuation_audit']['finding_count'] for row in (entry, fixed, benchmark)):
        st.warning('比較方法仍有估值疑點，現階段無法據此確認哪個方法較好。')
    else:
        lower = [row for row in (entry, fixed, benchmark)
                 if primary['summary']['total_return'] <= row['summary']['total_return']]
        deeper = [row for row in (entry, fixed, benchmark)
                  if abs(primary['summary']['max_drawdown']) > abs(row['summary']['max_drawdown'])]
        if lower:
            st.info('轉弱連個股退出的累積試算報酬未勝過：' + '、'.join(_name(row) for row in lower) + '。目前不支持僅為提高報酬而採用這種切換。')
        if deeper:
            st.warning('轉弱連個股退出的最大跌幅仍大於：' + '、'.join(_name(row) for row in deeper) + '。切換並未保證較小虧損。')
        if not lower and not deeper:
            st.info('這組歷史試算的報酬與最大跌幅較有利，仍需未來資料驗證；不能把切換情境當成多次獨立成功。')
    marked = [row for row in all_rows if not row['summary']['final_liquidation_complete']]
    if marked:
        st.warning('以下方法期末含未平倉估值，並非全部可提領現金：' + '、'.join(_name(row) for row in marked))
    for column, row in zip(st.columns(3), (primary, entry, fixed)):
        column.metric(_name(row) + '：累積試算淨報酬', pct(row['summary']['total_return']))
    table = []
    for row in all_rows:
        summary = row['summary']
        table.append({'方法': _name(row), '累積試算淨報酬': pct(summary['total_return']),
            '年化': pct(summary['cagr']), '最大跌幅': pct(abs(summary['max_drawdown'])),
            '平均現金比重': pct(summary['mean_cash_weight']),
            '轉弱要求／已退出事件': f"{summary.get('forced_exit_requested_cohorts', 0)}／{summary.get('forced_exit_completed_cohorts', 0)}",
            '估值疑點筆數': row['valuation_audit']['finding_count'],
            '期末': '已清算' if summary['final_liquidation_complete'] else '含未平倉估值'})
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    st.caption(f"每邊滑價 {'0.45%' if scenario == 'stress' else '0.30%'}，另扣每邊 0.1425% 手續費、股票賣稅 0.3% 與 0050 賣稅 0.1%。每次切換都計入成本；狀態觀察後第 {1 + delay} 個交易日才執行，事件買入也延後相同交易日數。")
    st.caption('固定混合是起初各分 50% 資金，兩份資金獨立、之後不再平衡，比例會自然漂移。廣度方案先前因資料不足而未知，尚不能評價成敗。')
    with st.expander('看資產曲線、各年收益與切換成本'):
        if (basis, scenario, delay) == ('official', 'stress', 0):
            chart = pd.concat([pd.Series({point['date']: point['nav'] for point in report['charts'][rule]}, name=NAMES[rule])
                               for rule in NAMES], axis=1)
            chart.index = pd.to_datetime(chart.index)
            st.line_chart(chart)
            st.caption('曲線為目前所選的官方價格、壓力成本；起初資金設為 1。')
        else:
            st.caption('完整共同曲線保留於「官方價格・壓力成本」；下表顯示目前選定情境。')
        st.dataframe(pd.DataFrame([{'方法': _name(row), **{year: pct(value) for year, value in row['summary']['annual_returns'].items()}}
                                  for row in all_rows]), hide_index=True, use_container_width=True)
        st.caption('2026 年只到研究截止日。各年收益均沿用完整持有路徑，不能任意拼接當年的勝出方法。')
        st.dataframe(pd.DataFrame([{'方法': _name(row), '累計成本／期初資金': pct(row['summary']['total_cost']),
            '雙向成交額／期初資金': f"{row['summary']['turnover']:.2f} 倍", '成交筆數': row['summary']['trade_count']}
            for row in all_rows]), hide_index=True, use_container_width=True)
    with st.expander('看 0050 何時改變趨勢狀態'):
        stats = report['states']['stats'][basis]
        st.caption(f"這裡列觀察日，與執行日不同；正常於下一交易日執行，本情境延遲 {1 + delay} 個交易日。當天無報價或不足 120 次有效觀察為未知。未知不發出新退出要求；已發出的股票退出要求持續等待成交。")
        st.write('觀察日數：' + '、'.join(f'{STATES.get(state, state)} {count} 日' for state, count in stats['state_days'].items()))
        transitions = pd.DataFrame(stats['observed_transitions'])
        if not transitions.empty:
            transitions['state'] = transitions['state'].map(STATES)
            st.dataframe(transitions.rename(columns={'date': '觀察日', 'state': '狀態', 'close': '0050 收盤', 'ma120': '最近 120 次有效收盤均價'}),
                         hide_index=True, use_container_width=True)
    if st.checkbox('查看這個方法的成交與切換', key='regime_switch_details'):
        selected_rule = st.selectbox('檢查哪個方法', [row['rule'] for row in all_rows],
                                    index=next(i for i, row in enumerate(all_rows) if row['rule'] == 'exit_cash'),
                                    format_func=lambda key: NAMES[key], key='regime_switch_method')
        _case_details(report, next(row for row in all_rows if row['rule'] == selected_rule))
    with st.expander('研究限制'):
        for note in report['limitations']:
            st.caption(note)
    st.caption(f"效能：狀態準備 {report['states']['elapsed_seconds']:.1f} 秒，30 組方法與 4 組基準含稽核 {report['elapsed_seconds']:.1f} 秒。研究使用 {report['finmind_requests']} 次 FinMind；切換選單只讀結果，成交紀錄勾選後才載入。")
