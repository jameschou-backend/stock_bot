"""Inspect sealed remnant-slot experiments without running or enabling orders."""
from copy import deepcopy
from pathlib import Path
import json
import threading

from app.backtest_full_pass_ui import _file_signature
from app.backtest_tool_ui import verified_bytes
from app.execution_factorial_ui import REPORT as ORIGINAL
from scripts.research_exit_scenarios import read, sha, summarize
from scripts.research_residual_slots import comparison
from skills.execution_factorial import LABELS

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/forward_simulation/residual_slots_20260926.json'
_CACHE = {}
_LOCK = threading.RLock()


def validate(value, root=ROOT):
    if (value.get('schema') != 'residual_slots_publication_v1'
            or any(value.get(k) is not False for k in ('live_qualified', 'strict_data_ready', 'unseen_validation'))
            or value.get('all_completed') is not True or value.get('candidate_count') != 454):
        raise ValueError('餘股研究範圍或資格不符')
    proof = json.loads(verified_bytes(value['offline_verification'], root, '.json'))
    manifest = value['run_manifest']
    if (proof.get('schema') != 'residual_slots_offline_v1' or proof.get('passed') is not True
            or proof.get('all_completed') is not True or proof.get('compared_cases') != 12
            or proof['source_sha256'].get(manifest['path']) != manifest['sha256']):
        raise ValueError('缺少同一版本的十二帳戶離線重現證據')
    expected = {'keep_0', 'keep_7', 'benchmark_control', 'benchmark_combined'} | {'release_'+str(i) for i in range(8)}
    if set(value['cases']) != expected:
        raise ValueError('缺少完整因素組合或對照帳戶')
    original = json.loads(verified_bytes(value['original_publication'], root, '.json'))
    if proof['source_sha256'].get(value['original_publication']['path']) != value['original_publication']['sha256']:
        raise ValueError('原規則不屬於同一份來源證據')
    cases = {}
    for name, row in value['cases'].items():
        if proof['source_sha256'].get(row['result']['path']) != row['result']['sha256']:
            raise ValueError('帳戶不屬於同一份離線重播證據')
        result = json.loads(verified_bytes(row['result'], root, '.json'))
        if (result.get('completed') is not True or row.get('completed') is not True
                or row['summary'] != result['summary'] or row['summary'] != summarize(result['account'])):
            raise ValueError('顯示收益與完整帳戶不一致')
        if (row['summary']['start'], row['summary']['end'], row['summary']['initial_cash']) != (
                '2022-01-03', '2026-09-09', 1_000_000):
            raise ValueError('餘股研究期間或本金不一致')
        if name.startswith(('keep_', 'release_')):
            policy, mask = name.split('_')
            if result['config']['residual_policy'] != policy or result['config']['factor_mask'] != int(mask):
                raise ValueError('因素標籤與帳戶設定不一致')
            if policy == 'release' and not all(result['audit'].get(k) is True for k in (
                    'residual_classification_rebuilt', 'residual_assets_retained',
                    'residual_risk_budget_rebuilt', 'active_slots_rebuilt')):
                raise ValueError('缺少餘股帳務與風險核對')
        if not name.startswith('release_'):
            old_name = 'factor_'+name[5:] if name.startswith('keep_') else name
            old = json.loads(verified_bytes(original['cases'][old_name]['result'], root, '.json'))
            if result['account'] != old['account']:
                raise ValueError('原規則／基準未逐欄重現')
        cases[name] = result
    if comparison(cases, original) != value['comparison']:
        raise ValueError('逐年、滾動或集中度比較與帳戶不一致')
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
    import pandas as pd
    import streamlit as st
    st.title('餘股與持股名額')
    st.caption('2022/01/03–2026/09/09｜100 萬複利｜整張成交｜5 個新部位名額｜閒錢現金')
    st.write('整張賣出後剩下的配股，繼續留在資產中。從次一交易日起，符合條件的餘股可釋放新部位名額；'
             '待交付配股也計入股數，合計達一張就仍占名額。')
    if not REPORT.exists():
        st.info('完整帳戶及第二次離線比對完成後，才顯示本輪結果。')
        return
    try:
        value = load()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error('餘股研究目前不可採信：' + str(exc))
        return
    wins = sum(r['excess_return'] > 0 for r in value['comparison'].values())
    st.info(f'八種固定情境中，有 {wins} 種累積報酬超過同條件0050；'
            '這個數字不是未來勝率，也不能單獨決定是否實戰。')
    rows = []
    for mask, label in enumerate(LABELS):
        r = value['comparison']['release_'+str(mask)]
        s = value['cases']['release_'+str(mask)]['summary']
        rows.append({'情境': label, '原規則淨報酬': f"{r['original_total_return']:.2%}",
            '釋放名額淨報酬': f"{s['total_return']:.2%}", '0050淨報酬': f"{r['benchmark_return']:.2%}",
            '最大回撤': f"{s['max_drawdown']:.2%}", '期末資產': round(s['final_nav']),
            '滾動一年勝率': f"{r['rolling252_win_rate']:.1%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption('淨報酬已扣模型中的交易成本。滑價加倍為每邊0.45%→0.90%；進出場延遲均多一個市場日。'
               '0050只對應滑價。滾動一年為252個市場日，各窗口重疊，不是獨立樣本成功率。')
    st.warning('這是使用過的歷史樣本；日線成交仍是估計，沒有補出歷史零股成交證據，也尚未驗證實際券商成交。策略尚未取得實戰資格。')
    st.info('餘股估值超過昨日淨值5%時，停止新增買進；每筆新部位預算先扣餘股曝險。'
            '不假設餘股已賣出，也不把同日賣出所得立即投入。')
    mask = st.selectbox('查看情境', list(range(8)), format_func=lambda m: LABELS[m])
    row = value['comparison']['release_'+str(mask)]
    selected = value['cases']['release_'+str(mask)]['summary']
    baseline = value['cases']['benchmark_combined' if mask&1 else 'benchmark_control']['summary']
    st.subheader('逐年比較')
    st.dataframe(pd.DataFrame([{'年度': a['year'] + ('（至9/9）' if a['partial_year'] else ''),
        '策略淨報酬': f"{a['total_return']:.2%}", '0050淨報酬': f"{b['total_return']:.2%}"}
        for a,b in zip(selected['annual'], baseline['annual'])]), hide_index=True, use_container_width=True)
    share = row['top_two_positive_profit_share']
    st.write(f"{row['rolling252_count']} 個滾動窗口中，最差超額報酬 {row['rolling252_worst_excess']:.2%}；"
             + (f"前兩大獲利股票占所有正獲利 {share:.1%}。" if share is not None else '沒有正獲利股票。'))
    st.caption(f"餘股最高占淨值 {row['max_residual_nav_ratio']:.2%}；"
               f"超過上限 {row['residual_cap_block_days']} 天；"
               f"持有股票最多 {row['max_total_holdings']} 檔（含餘股），開盤有效名額最多 {row['max_active_opening']} 個。")
    if st.checkbox('展開完整帳本與下載'):
        result = json.loads(verified_bytes(value['cases']['release_'+str(mask)]['result'], ROOT, '.json'))
        st.line_chart(pd.DataFrame(result['account']['daily']).set_index('date')[['nav']].rename(columns={'nav':'總資產'}))
        for key,label in (('daily','每日資產'),('trades','全部買賣'),('orders','委託及未成交'),('holdings','每日持股')):
            st.download_button('下載'+label, pd.DataFrame(result['account'][key]).to_csv(index=False).encode('utf-8-sig'),
                file_name=f'residual_{mask}_{key}.csv', mime='text/csv')
    st.subheader('大戶投既有紀錄核對')
    st.write('先使用你已有的紀錄：同一交易日的期初庫存、成交明細、期末庫存，以及可用買進額度。'
             '若當天有未成交或撤單，再附委託／撤單結果；若有配股，再附待交付股數。')
    st.caption('帳號及個資可遮蔽；保留日期、股票代號、買賣、股數、價格、手續費與稅。'
               '現有手動核帳入口在「策略驗證」。只有成交清單不足以證明盤中可用額度或撤單時序。')
