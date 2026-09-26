"""Inspect exact research accounts and a matched benchmark without a new run."""
import json
from pathlib import Path

from app.backtest_tool_ui import verified_bytes
from skills.account_statistics import monthly_pairs
from skills.backtest_contract import validate_completed_account
from skills.exit_policy import REASON_LABELS
from scripts.replay_million import summarize

ROOT = Path(__file__).resolve().parents[1]


def load_comparison(publication, name, root=ROOT):
    row = publication['cases'][name]
    if row.get('completed') is not True:
        raise ValueError('未完成的帳戶不能顯示全期績效')
    case = json.loads(verified_bytes(row['result'], root, '.json'))
    if (case.get('completed') is not True or case.get('live_qualified') is not False
            or case['config'] != row['config'] or case['summary'] != row['summary']):
        raise ValueError('帳戶版本與研究摘要不同')
    original = json.loads(verified_bytes(publication['original_publication'], root, '.json'))
    key = 'benchmark_combined' if case['config']['factor_mask'] & 1 else 'benchmark_control'
    benchmark = json.loads(verified_bytes(original['cases'][key]['result'], root, '.json'))
    if benchmark.get('completed') is not True or benchmark['config'].get('benchmark') is not True:
        raise ValueError('缺少完整0050對照帳戶')
    account, comparison = case['account'], benchmark['account']
    dates = [r['date'] for r in comparison['daily']]
    for value, expected in ((account, row['summary']), (comparison, benchmark['summary'])):
        validate_completed_account(value, dates, publication['start'], publication['end'])
        if summarize(value) != expected or value['settings']['initial_cash'] != publication['initial_cash']:
            raise ValueError('帳本重算結果與發布摘要不同')
    monthly_pairs(account, comparison, dates)
    for key in ('commission', 'minimum_fee', 'slippage', 'participation', 'odd_participation'):
        if account['settings'][key] != comparison['settings'][key]:
            raise ValueError('帳戶與0050使用不同成本或成交容量')
    if benchmark['summary']['total_return'] != row['metrics']['benchmark_return']:
        raise ValueError('0050帳戶與顯示基準不同')
    return account, comparison


def scenario_label(mask):
    labels = [label for bit, label in ((1, '滑價加倍'), (2, '進場多晚一天'), (4, '出場多晚一天')) if mask & bit]
    return '、'.join(labels) if labels else '一般成交'


def render(publication, arms):
    import pandas as pd
    import streamlit as st
    if not st.checkbox('查看逐筆買賣與每日資產', key='research_account_detail_enabled'):
        return
    arm = st.selectbox('帳戶規則', list(arms), format_func=arms.get, key='research_detail_arm')
    mask = st.selectbox('成交情境', list(range(8)), format_func=scenario_label, key='research_detail_stress')
    name = f'{arm}_{mask}'
    if not publication['cases'][name]['completed']:
        st.warning('此情境因資料不足停止，不能顯示成完整績效。')
        return
    try:
        account, benchmark = load_comparison(publication, name)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error('帳戶明細驗證未通過：' + str(exc))
        return
    daily = pd.DataFrame(account['daily'])
    daily['benchmark_nav'] = [r['nav'] for r in benchmark['daily']]
    st.line_chart(daily.set_index('date')[['nav', 'benchmark_nav']].rename(
        columns={'nav': '策略資產', 'benchmark_nav': '0050資產'}))
    st.caption('100萬元起始本金、獲利複投。資產包含現金、持股及未交付權利；應收款不等於可買進現金。')
    labels = dict(date='成交日', signal_date='訊號日', stock_id='代號', name='名稱', side='買賣',
        qty='股數', reference_price='成交參考價', gross='成交價金', total_cost='費稅滑價',
        cash_after='成交後現金', reason='原因')
    trades = pd.DataFrame(account['trades'])
    shown = trades.reindex(columns=list(labels)).rename(columns=labels)
    if not shown.empty:
        shown['買賣'] = shown['買賣'].map({'buy': '買進', 'sell': '賣出'})
        reasons = dict(REASON_LABELS, leader_entry='符合當日候選及資金規則', support20='前日收盤跌破只上移的20日支撐',
                       pyramid_add='強勢突破後的本批唯一加碼')
        shown['原因'] = shown['原因'].map(lambda value: reasons.get(value, value))
    st.dataframe(shown, hide_index=True, use_container_width=True)
    st.caption('成交參考價另加帳本中的滑價與費稅；日資料成交估算未重建盤中排隊。所有明細來自所選封存帳戶。')
    for key, title, frame in [('trades', '全部買賣', trades), ('daily', '每日資產與0050', daily),
                              ('orders', '委託與未成交原因', pd.DataFrame(account['orders']))]:
        st.download_button('下載' + title, frame.to_csv(index=False).encode('utf-8-sig'),
            file_name=f'{name}-{key}.csv', mime='text/csv', key='research_detail_' + key)
    if publication['cases'][name]['config'].get('technical_mode') not in (None, 'control'):
        case = json.loads(verified_bytes(publication['cases'][name]['result'], ROOT, '.json'))
        with st.expander('當時的支撐與股數規劃'):
            plans = pd.json_normalize(case['technical_entries'])
            if case['config']['technical_mode'] == 'support20':
                plans['risk_budget'] = None
            columns = {'date': '預定買進日', 'signal_date': '原訊號日', 'stock_id': '代號',
                'context.support_raw': '原訊號支撐價', 'reference_price': '買進規劃參考價',
                'planned_stop_price': '計畫停損參考價', 'risk_budget': '計畫風險金額',
                'planned_loss': '上限股數的計畫損失', 'allowed_qty': '規劃股數上限', 'filled_qty': '實際成交股數'}
            st.dataframe(plans.reindex(columns=list(columns)).rename(columns=columns),
                         hide_index=True, use_container_width=True)
            st.caption('支撐及距離固定取原訊號日；買進規劃價只用執行前已知價格。計畫停損價用來算股數，並不是保證成交價或已掛出的停損單。只測支撐的組別不套用2%股數限制。')
            st.download_button('下載配置與支撐紀錄',
                json.dumps({'entries': case['technical_entries'], 'support': case['support_decisions']},
                           ensure_ascii=False, indent=2),
                file_name=f'{name}-technical-decisions.json', mime='application/json',
                key='research_detail_technical')
    if publication['cases'][name]['config'].get('pattern_filter') is True:
        case = json.loads(verified_bytes(publication['cases'][name]['result'], ROOT, '.json'))
        with st.expander('當時的突破條件'):
            entries = pd.json_normalize(case['pattern_entries'])
            columns = {'date': '預定買進日', 'signal_date': '原訊號日', 'stock_id': '代號',
                'context.breakout20': '突破20日高點', 'context.contraction10': '區間收斂',
                'context.volume_expansion': '成交量放大', 'allowed_qty': '篩選後股數',
                'filled_qty': '成交股數'}
            shown = entries.reindex(columns=list(columns)).rename(columns=columns)
            for title in ('突破20日高點', '區間收斂', '成交量放大'):
                shown[title] = shown[title].map({True: '符合', False: '不符'}).fillna('資料不足')
            st.dataframe(shown, hide_index=True, use_container_width=True)
            st.caption('三條件都用原訊號日已知資料。篩選後股數仍可能因資金、風險配置或行情限制減少；這張表不代表保證成交。未進到配置階段的候選可另查委託與未成交原因。')
            st.download_button('下載突破判斷紀錄',
                json.dumps(case['pattern_entries'], ensure_ascii=False, indent=2),
                file_name=f'{name}-pattern-decisions.json', mime='application/json',
                key='research_detail_pattern')
    if publication['cases'][name]['config'].get('pyramid_enabled') is True:
        case = json.loads(verified_bytes(publication['cases'][name]['result'], ROOT, '.json'))
        with st.expander('當時的加碼判斷'):
            decisions = pd.json_normalize(case['pyramid_decisions'])
            selected = (decisions[decisions.status.isin(['waiting_extra_entry_delay', 'filled', 'unfilled'])]
                        if not decisions.empty else decisions)
            columns = {'date': '判斷日', 'stock_id': '代號', 'status': '結果',
                'created_instruction.signal_date': '新指令訊號日',
                'pending_before.signal_date': '等待中原訊號日',
                'execution_capacity.qty': '執行股數上限', 'filled_qty': '成交股數'}
            shown = selected.reindex(columns=list(columns)).rename(columns=columns)
            shown['結果'] = shown['結果'].map({'filled': '已成交', 'unfilled': '未成交',
                                              'waiting_extra_entry_delay': '多等一個交易日'})
            st.dataframe(shown, hide_index=True, use_container_width=True)
            st.caption('這裡只列出已建立的加碼指令；所有不符條件、無預算或取消原因都保留在下載紀錄。加碼不重設原持股出場期限。')
            st.download_button('下載全部加碼判斷', json.dumps(case['pyramid_decisions'], ensure_ascii=False, indent=2),
                file_name=f'{name}-pyramid-decisions.json', mime='application/json', key='research_detail_pyramid')
