"""Readable stress attribution, with sealed sources checked before rendering."""
from copy import deepcopy
from pathlib import Path
import threading

from app.backtest_full_pass_ui import _file_signature
from app.backtest_tool_ui import verified_bytes
from scripts.research_exit_scenarios import read, sha, summarize
from skills.execution_factorial import LABELS
from scripts.research_execution_factorial import analyze

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/forward_simulation/execution_factorial_20260926.json'
_CACHE = {}
_LOCK = threading.RLock()
FAILURES = {'at_lower_limit': '跌停限制', 'at_upper_limit': '漲停限制',
    'board_only_below_one_lot': '數量不足一張', 'board_only_odd_remainder': '整張成交後的零股餘數',
    'overlapping_member': '已有同檔持股', 'partial_capacity_or_cash': '成交容量或現金僅足部分成交',
    'resource_cash_locked': '資金被其他計畫保留', 'resource_slots_locked': '持股名額被保留',
    'single_price_session': '當天僅有單一成交價', 'slots_full': '持股名額已滿',
    'official_trading_suspension': '官方停牌', 'not_general_board': '非一般板交易',
    'insufficient_cash': '可用現金不足'}


def validate(value, root=ROOT):
    import json
    if (value.get('schema') != 'execution_factorial_publication_v1'
            or any(value.get(k) is not False for k in ('live_qualified', 'strict_data_ready', 'unseen_validation'))
            or not value.get('all_completed') or value.get('candidate_count') != 454
            or value.get('labels') != list(LABELS)):
        raise ValueError('成交壓力報告範圍或資格不符')
    proof = json.loads(verified_bytes(value['offline_verification'], root, '.json'))
    manifest = value['run_manifest']
    if (proof.get('schema') != 'execution_factorial_offline_v1' or proof.get('passed') is not True
            or proof.get('all_completed') is not True or proof.get('compared_cases') != 12
            or proof['source_sha256'].get(manifest['path']) != manifest['sha256']):
        raise ValueError('缺少同一版本的十二帳戶離線重現證據')
    expected = {'factor_' + str(i) for i in range(8)} | {'noop_depth', 'noop_quote', 'benchmark_control', 'benchmark_combined'}
    if set(value['cases']) != expected:
        raise ValueError('缺少完整因素組合或對照帳戶')
    cases = {}
    for name, row in value['cases'].items():
        if proof['source_sha256'].get(row['result']['path']) != row['result']['sha256']:
            raise ValueError('帳戶不屬於同一份離線重播證據')
        account = json.loads(verified_bytes(row['result'], root, '.json'))
        if (account.get('completed') is not True or row.get('completed') is not True
                or row['summary'] != account['summary'] or row['summary'] != summarize(account['account'])):
            raise ValueError('顯示收益與完整帳戶不一致')
        if (row['summary']['start'], row['summary']['end'], row['summary']['initial_cash']) != (
                '2022-01-03', '2026-09-09', 1_000_000):
            raise ValueError('成交壓力帳戶期間或本金不一致')
        if name.startswith('factor_') and account['config']['factor_mask'] != int(name[7:]):
            raise ValueError('因素標籤與帳戶設定不一致')
        cases[name] = account
    if analyze(cases) != value['analysis']:
        raise ValueError('因素分攤或損益歸因與帳戶不一致')
    return value


def load(path=REPORT, root=ROOT):
    path, root = Path(path), Path(root).resolve()
    value = read(path)
    refs = dict(value['source_sha256'])
    refs[str(path.relative_to(root))] = path.with_suffix('.sha256').read_text().strip()
    refs[str(path.with_suffix('.sha256').relative_to(root))] = sha(path.with_suffix('.sha256'))
    signature = tuple((name, _file_signature(root / name, root)) for name in sorted(refs))
    key = (str(root), str(path))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == signature:
            return deepcopy(cached[1])
        for name, digest in refs.items():
            if sha(root / name) != digest:
                raise ValueError('來源版本已改變：' + name)
        validate(value, root)
        after = tuple((name, _file_signature(root / name, root)) for name in sorted(refs))
        if after != signature:
            raise ValueError('資料在核對期間變更')
        _CACHE[key] = (signature, deepcopy(value))
        return value


def render():
    import json
    import pandas as pd
    import streamlit as st
    st.title('成交壓力拆解')
    st.caption('2022/01/03–2026/09/09｜100 萬複利｜整張、5 個名額、閒錢現金｜固定同一批454個訊號')
    if not REPORT.exists():
        st.info('完整帳戶及離線比對完成後，才顯示本輪結果。')
        return
    try:
        value = load()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error('成交壓力結果目前不可採信：' + str(exc))
        return
    rows = []
    for mask, label in enumerate(value['labels']):
        s = value['cases']['factor_' + str(mask)]['summary']
        b = value['cases']['benchmark_combined' if mask & 1 else 'benchmark_control']['summary']
        rows.append({'情境': label, '累積淨報酬': f"{s['total_return']:.2%}",
                     '0050淨報酬': f"{b['total_return']:.2%}", '最大回撤': f"{s['max_drawdown']:.2%}",
                     '期末資產': round(s['final_nav']), '成交筆數': s['trade_count'],
                     '交易成本': round(s['costs']['total_cost'])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption('滑價加倍：每邊0.45%→0.90%；晚一天：比原規則多延後一個市場日。'
               '0050沒有策略進出訊號，僅對應滑價，沿用原買入及股利再投規則。')
    st.subheader('報酬差距如何形成')
    factor_names = {'slippage': '滑價加倍', 'entry_delay': '進場晚一天', 'exit_delay': '出場晚一天'}
    analysis = value['analysis']
    largest = min(analysis['single_factor_return_changes'], key=analysis['single_factor_return_changes'].get)
    st.info(f"本輪單一因素影響最大的是「{factor_names[largest]}」，"
            f"相對正常情境改變 {100 * analysis['single_factor_return_changes'][largest]:+.2f} 個百分點。")
    attribution = pd.DataFrame([{'因素': factor_names[key],
        '單獨施加影響（百分點）': round(100 * analysis['single_factor_return_changes'][key], 2),
        '含交互作用的平均分攤（百分點）': round(100 * contribution, 2)}
        for key, contribution in analysis['shapley_return_changes'].items()])
    st.dataframe(attribution, hide_index=True, use_container_width=True)
    st.caption('平均分攤採全部六種加入順序；能加總回總差距，但不是唯一因果解釋。'
               '零股報價、深度對整張版沒有作用，兩個對照帳戶已逐欄重現正常結果。')
    st.warning('延後成交會改變資金、名額及後續買進股票。這是歷史診斷，尚未通過未見資料或實盤驗證；出場延後包含停損故障壓力，不是延後停損建議。')
    st.subheader('哪些股票影響最大')
    gap = pd.DataFrame(analysis['stock_profit_changes'])
    st.dataframe(gap.head(10).rename(columns={'stock_id': '代號', 'name': '股票',
        'normal_profit': '正常情境損益', 'stress_profit': '全部壓力損益', 'difference': '壓力減正常'}),
        hide_index=True, use_container_width=True)
    st.caption('股票損益包含股利、期末未賣持股及應收，不全是已實現獲利；全部股票差額已核對回帳戶差額。')
    st.caption('整張版保留配股產生的零股餘額，這些餘額也占用持股名額；不假設它們已賣出或消失。')
    mask = st.selectbox('查看成交情境', list(range(8)), format_func=lambda m: value['labels'][m])
    chosen = value['cases']['factor_' + str(mask)]['summary']
    baseline = value['cases']['benchmark_combined' if mask & 1 else 'benchmark_control']['summary']
    st.subheader('逐年比較')
    st.dataframe(pd.DataFrame([{'年度': a['year'] + ('（至9/9）' if a['partial_year'] else ''),
        '策略淨報酬': f"{a['total_return']:.2%}", '0050淨報酬': f"{b['total_return']:.2%}",
        '期末資產': round(a['end_nav'])} for a, b in zip(chosen['annual'], baseline['annual'])]),
        hide_index=True, use_container_width=True)
    if st.checkbox('展開此情境的帳戶與未成交原因'):
        case = json.loads(verified_bytes(value['cases']['factor_' + str(mask)]['result'], ROOT, '.json'))
        account = case['account']
        daily = pd.DataFrame(account['daily'])
        st.line_chart(daily.set_index('date')[['nav']].rename(columns={'nav': '總資產'}))
        path = analysis['paths'][str(mask)]
        st.write(f"相較正常情境：少買 {len(path['only_base'])} 個進場事件，多買 {len(path['only_other'])} 個；"
                 f"共同買進事件中，{len(path['changed_quantity'])} 個數量不同。")
        st.dataframe(pd.DataFrame([{'未成交或部分成交原因': FAILURES.get(k, '其他：'+k), '委託紀錄數': v}
            for k, v in path['order_failure_rows'].items()]), hide_index=True, use_container_width=True)
        st.caption('上述為委託紀錄數，同一檔股票可能重複出現；整張拒絕的零股餘數也另記一列。')
        for key, label in (('daily', '每日資產'), ('trades', '全部買賣'), ('orders', '全部委託'), ('holdings', '每日持股')):
            st.download_button('下載' + label, pd.DataFrame(account[key]).to_csv(index=False).encode('utf-8-sig'),
                file_name=f'factor_{mask}_{key}.csv', mime='text/csv')
