"""Read-only view of the latest three research priorities."""
from pathlib import Path
import hashlib
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/forward_simulation/priority_optimization_20260925.json'


def load(path=REPORT, root=ROOT):
    path, root = Path(path), Path(root).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('優先優化摘要已變動，請重新核對研究證據。')
    result = json.loads(raw)
    if (result['schema'] != 'priority_optimization_v1' or result['live_qualified'] is not False
            or result['unseen_validation'] is not False):
        raise ValueError('歷史研究不能直接提升實戰資格。')
    for name, digest in result['evidence_sha256'].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError('優先優化證據無法核對：' + name)
    return result


def comparison_rows(cases):
    rows = []
    for label, kind, policy in (
        ('五檔・可買零股', 'capacity', 'mixed'),
        ('五檔・只成交整股', 'capacity', 'board_only'),
        ('0050・可買零股', 'benchmark', 'mixed'),
        ('0050・只成交整股', 'benchmark', 'board_only'),
    ):
        row = {'方法': label}
        for stress, title in (('control', '原成本'), ('combined', '成交壓力')):
            case = cases[f'{kind}_{stress}_{policy}']
            row[title + '累積淨報酬'] = f"{case['summary']['total_return']:.2%}" if case['completed'] else '未完成'
            if stress == 'combined':
                row['壓力最大回撤'] = f"{case['summary']['max_drawdown']:.2%}" if case['completed'] else '未完成'
        rows.append(row)
    return rows


def missing_source_message(reason):
    if reason.startswith('Stock dividend data missing or invalid: '):
        event = reason.split(': ', 1)[1].split(';', 1)[0]
        return f'{event} 配股條件尚未完整核實，包含新股入帳日期。'
    if reason.startswith('Offline replay is missing price-limit evidence: '):
        return '缺少漲跌停來源：' + reason.rsplit(': ', 1)[1]
    if reason.startswith('Offline replay is missing price-limit date: '):
        return '缺少該日漲跌停資料：' + reason.rsplit(': ', 1)[1]
    return reason


def render():
    with st.expander('9/25 優先優化：防偷看、零股成交與0050比較', expanded=True):
        try:
            result = load()
        except (OSError, KeyError, ValueError) as exc:
            st.warning(str(exc))
            return
        audit = result['causality']
        if audit['passed'] and audit['complete']:
            st.write(f"① 防偷看：{audit['cutoff_count']}個日期、{audit['case_count']}組截斷／改動未來資料測試通過；{audit['accepted_candidates']}個歷史訊號完整重建一致。")
        else:
            st.error('① 防偷看：本輪因果稽核尚未全部通過。')
        st.caption('驗證的是封存至2026/9/9的價量資料及其歷史訊號；歷史名冊、當時公告與資料修訂版本仍未齊全。這不是最新每日選股的全面認證。')
        odd = result['oddlot']
        st.write(f"② 零股成交：三組壓力帳戶共有{odd['requested_stock_date_sides']}個股票／日期／方向需求，分布{odd['requested_dates']}日；封存來源中的合格撮合序列為{odd['sequence_tapes_declared']}。")
        st.caption('這是既有日資料帳戶的最低需求清單，含未成交委託；逐筆重播若改變持股路徑，還可能需要其他日期。日成交量與收盤對手量不能證明委託成交。')
        if result.get('oddlot_demand_csv'):
            st.download_button('下載零股補件清單（CSV）', (ROOT / result['oddlot_demand_csv']).read_bytes(),
                'odd-lot-data-demands.csv', 'text/csv', key='oddlot_scope_download')
        st.write('③ 減少零股依賴：只改整股成交限制，保留原選股、資金、名額與出場規則。')
        st.dataframe(pd.DataFrame(comparison_rows(result['board_cases'])), hide_index=True, use_container_width=True)
        st.caption('2022/1/3～2026/9/9，100萬元複利、閒錢現金；報酬已扣費且非年化。整股組以整股0050配對，另列原0050以免基準被換掉。全部仍是日資料成交假設。')
        for name, row in result['board_cases'].items():
            if not row['completed']:
                label = '原成本' if row['config']['stress'] == 'control' else '成交壓力'
                st.warning(f"整股限定／{label}未完成：{missing_source_message(row['reason'])} 不把中途帳戶當成全期收益。")
        st.write(result['next_direction'])
        st.caption('配股產生的殘股保留估值並占用名額，未假造賣出。排程仍暫停；這裡只讀封存結果，不下載行情或啟用策略。')
        st.download_button('下載三項研究結果', json.dumps(result, ensure_ascii=False, indent=2),
            'priority-optimization-20260925.json', 'application/json', key='priority_optimization_download')
