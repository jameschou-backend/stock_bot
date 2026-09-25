"""Readable, source-bound results for the repaired historical selector."""
from copy import deepcopy
import json
from pathlib import Path
import threading

from app.backtest_full_pass_ui import _file_signature
from app.backtest_tool_ui import verified_bytes
from scripts.research_exit_scenarios import sha, summarize

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/forward_simulation/historical_selector_replay_20260925.json'
LABELS = {'original': '原股票池', 'identity': '修正身分與轉板歷史', 'omitted': '加入遺漏股票',
          'combined': '本次修正全部套用', 'benchmark': '0050 同條件基準'}
_CACHE = {}
_LOCK = threading.RLock()


def validate(value, root=ROOT):
    if (value.get('schema') != 'historical_selector_publication_v1'
            or any(value.get(k) is not False for k in ('live_qualified', 'strict_data_ready', 'unseen_validation'))
            or value.get('execution_policy') != 'board_only'
            or value.get('initial_cash') != 1_000_000
            or value.get('start') != '2022-01-03' or value.get('end') != '2026-09-09'):
        raise ValueError('新版回測範圍或研究資格不符')
    expected = {arm + '_' + stress for arm in LABELS for stress in ('control', 'combined')}
    if set(value['cases']) != expected:
        raise ValueError('新版回測必須完整保留十個對照案例')
    for key, schema in (('offline_verification', 'historical_selector_offline_v1'),
                        ('causality_verification', 'historical_selector_causality_v1')):
        descriptor = value[key]
        proof = json.loads(verified_bytes(descriptor, root, '.json'))
        manifest = value['run_manifest']
        if (proof.get('passed') is not True or proof.get('schema') != schema
                or proof.get('source_sha256', {}).get(manifest['path']) != manifest['sha256']):
            raise ValueError('回測尚未通過離線重現或未來資料隔離檢查')
        if key == 'causality_verification' and (proof.get('complete') is not True
                or proof.get('expected_checks') != 684
                or len(proof['cases']) != proof['expected_checks']
                or value['causality_checks'] != proof['expected_checks']
                or not all(r['passed'] is True for r in proof['cases'])):
            raise ValueError('未來資料隔離檢查尚未完成')
        if key == 'offline_verification' and (proof.get('compared_cases') != 10 or proof.get('compared_selectors') != 4):
            raise ValueError('離線帳戶對照數目不符')
    for name, row in value['cases'].items():
        case = json.loads(verified_bytes(row['result'], root, '.json'))
        if case.get('completed') is not row['completed']:
            raise ValueError('顯示完成狀態與帳戶不一致')
        config = case.get('config', {})
        if (config.get('board_only') is not True or config.get('stress') != name.rsplit('_', 1)[1]
                or config.get('benchmark') is not name.startswith('benchmark_')
                or case.get('live_qualified') is not False):
            raise ValueError('新版帳戶交易政策或資格不符')
        if row['completed']:
            if case['summary'] != row['summary'] or summarize(case['account']) != row['summary']:
                raise ValueError('顯示收益與每日帳戶不一致')
            if any(row['summary'][k] != value[k] for k in ('start', 'end', 'initial_cash')):
                raise ValueError('個別帳戶期間或本金與比較範圍不一致')
        elif row.get('summary') is not None or case.get('summary') or not row.get('reason'):
            raise ValueError('未完成案例不能顯示全期收益')
    return value


def load(path=REPORT, root=ROOT):
    """Hash on first use; unchanged file identities avoid re-reading GB-sized sources."""
    path, root = Path(path), Path(root).resolve()
    index = json.loads(path.read_text())
    refs = dict(index['source_sha256'])
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
                raise ValueError('新版回測來源已變更：' + name)
        value = validate(index, root)
        after = tuple((name, _file_signature(root / name, root)) for name in sorted(refs))
        if after != signature:
            raise ValueError('來源在核對期間變更，請重新驗證')
        _CACHE[key] = (signature, deepcopy(value))
        return value


def rows(value, stress):
    result = []
    for arm, label in LABELS.items():
        case = value['cases'][arm + '_' + stress]
        summary = case.get('summary')
        result.append({'資料版本': label, '狀態': '已完成日資料估算' if summary else '缺件阻擋',
            '候選訊號': value['signals'][arm]['candidates'] if arm != 'benchmark' else None,
            '累積淨報酬': f"{summary['total_return']:.2%}" if summary else '—',
            '最大回撤': f"{summary['max_drawdown']:.2%}" if summary else '—',
            '期末資產': f"{summary['final_nav']:,.0f} 元" if summary else '—',
            '限制': case.get('reason') or ''})
    return result


def render():
    import pandas as pd
    import streamlit as st
    st.subheader('股票池修正後，重新選股與回測')
    if not REPORT.exists():
        st.info('新版帳戶與驗證尚未封存；這裡不沿用舊報酬。')
        return
    try:
        value = load()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error('新版回測目前不可採信：' + str(exc))
        return
    st.caption('2022/01/03–2026/09/09｜本金 100 萬複利｜5 檔｜閒錢保留現金｜此輪只買賣整張')
    stress = st.radio('新版回測成交情境', ['control', 'combined'], horizontal=True,
                      format_func=lambda x: '正常成本' if x == 'control' else '壓力測試：延後成交與更高成本')
    st.dataframe(pd.DataFrame(rows(value, stress)), hide_index=True, use_container_width=True)
    left, right = (value['cases']['combined_' + s]['summary'] for s in ('control', 'combined'))
    bench = value['cases']['benchmark_combined']['summary']
    if left and right and bench and right['total_return'] <= bench['total_return']:
        st.warning('壓力測試仍落後 0050，尚未證明可以穩健跑贏基準，不啟用實盤。')
    st.caption('正常每邊滑價 0.45%；壓力每邊 0.90%，另沿用延後一個市場日等壓力規則。'
               '日成交量只提供容量上限，不能證明指定時間能成交；歷史零股逐筆與全市場完整性仍未核實。')
    st.caption(f"本次資料補抓 {value['preparation_calls']} 次；回測離線、無模型重訓。"
               f"完成 {value['causality_checks']} 次未來截斷／擾動檢查，及十個帳戶逐欄重現。")
    st.caption('補回 49 檔遺漏股票，另修復 9 檔轉板前的 6,133 筆行情。'
               '3717、5292 的上市日期差異仍待證實；這不是完整歷史全市場或未見資料驗證。')
    arm = st.selectbox('查看新版帳戶明細', list(LABELS), index=3, format_func=LABELS.get)
    selected = value['cases'][arm + '_' + stress]
    if st.checkbox('展開新版每日資產與全部交易', value=False):
        try:
            case = json.loads(verified_bytes(selected['result'], ROOT, '.json'))
        except (OSError, ValueError) as exc:
            st.error('帳戶明細驗證失敗：' + str(exc))
            return
        if not case['completed']:
            st.error(case['reason'])
            return
        daily = pd.DataFrame(case['account']['daily'])
        st.line_chart(daily.set_index('date')[['nav']], use_container_width=True)
        st.caption('淨值包含持股及應收，不等於可用現金；公司行動產生的零股餘數保留估值，不假造賣出。')
        for field, label in (('daily', '每日資產'), ('trades', '全部買賣'), ('orders', '委託與未成交'), ('holdings', '每日持股')):
            frame = pd.DataFrame(case['account'][field])
            st.download_button('下載新版' + label, frame.to_csv(index=False).encode('utf-8-sig'),
                file_name=arm + '_' + stress + '_' + field + '.csv', mime='text/csv')
        reasons = {'leader_entry': '領先股進場', 'loss12': '12% 收盤停損', 'time63': '63 日持有到期'}
        trades = pd.DataFrame(case['account']['trades'])
        if not trades.empty:
            trades['side'] = trades['side'].map({'buy': '買進', 'sell': '賣出'})
            trades['reason'] = trades['reason'].map(lambda x: reasons.get(x, x))
            columns = {'date': '成交日', 'stock_id': '代號', 'name': '股票', 'side': '買賣',
                       'qty': '股數', 'reference_price': '參考價', 'total_cost': '交易成本',
                       'cash_after': '成交後現金', 'reason': '依據'}
            st.caption('下表為回測估算成交；參考價另計滑價，交易成本已包含手續費、稅與滑價。'
                       '下載檔保留全部原始稽核欄位。')
            st.dataframe(trades[list(columns)].rename(columns=columns), hide_index=True, use_container_width=True)
