"""Concise paired comparison and explicit unresolved execution qualification."""
from pathlib import Path
import json
import hashlib
import pandas as pd
import streamlit as st

from app.intraday_limit_ui import load_case

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'artifacts/forward_simulation/reservation_bridge_delivery_20260914.json'


def load_report(path=REPORT, root=ROOT):
    report = json.loads(Path(path).read_text())
    if (not report['all_completed'] or len(report['cases']) != 8 or report['live_qualified'] is not False
            or report['strict_intraday_completed'] is not False or report['strict_intraday_return'] is not None
            or not report['offline']['identical'] or report['offline']['network_calls'] != 0):
        raise ValueError('配對研究尚未完成或錯誤宣稱逐筆驗證')
    for name, digest in report['code_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('配對研究規則已變動，需重新驗證')
    for item in report['cases'].values():
        account = load_case(item, root)
        actual = account['daily'][-1]['nav']/account['settings']['initial_cash']-1
        if abs(actual-item['summary']['total_return']) > 1e-10:
            raise ValueError('摘要與帳本報酬不一致')
    return report


def render(path=REPORT):
    with st.expander('原勝出策略重新核對：保留零股的配對比較', expanded=True):
        if not Path(path).exists():
            st.info('八組配對帳戶核對中，尚未發布結果。')
            return
        try:
            report = load_report(path)
        except (OSError, ValueError, KeyError) as exc:
            st.error(str(exc)); return
        st.warning(report['warning'])
        st.caption('2022/1/3–2026/9/9，本金100萬元複利；策略閒錢保留現金。每欄均扣交易成本。')
        rows = []
        for mode, label in [('original', '原日資料流程'), ('reserved', '開盤前預留現金與名額')]:
            row = {'流程': label}
            for stress, title in [('control', '一般'), ('combined', '合併壓力')]:
                for rank, name in [('capacity', '策略'), ('benchmark', '0050')]:
                    s = report['cases'][f'{rank}_{stress}_{mode}']['summary']
                    row[title+name] = f"{s['total_return']*100:+.2f}%"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.write('逐筆驗證：未完成。原策略缺事前限價、委託時間及完整零股成交序列，沒有可報告的逐筆策略報酬。')
        key = st.selectbox('查看原帳戶的當日資金／名額依賴', ['capacity_control_original', 'capacity_combined_original'],
                           format_func=lambda s:'一般條件' if 'control' in s else '合併壓力', key='bridge_dependencies')
        item = report['legacy_ledger_checks'][key]
        st.caption(f"需當日流入支應買單：{item['cash_dependent_days']} 天；需當日釋放名額：{item['slot_dependent_days']} 天。這些條件需要成交順序證據，並非僅憑日資料就能判定一定成交或一定不可能。")
        df = pd.DataFrame(item['dependencies'])
        st.dataframe(df.rename(columns={'date':'交易日', 'previous_cash':'前日現金',
            'buy_outflow':'買進支出', 'later_cash_required':'需當日流入',
            'needs_same_day_slot_release':'需當日釋放名額', 'buy_stocks':'買進股票', 'sell_stocks':'賣出股票'}),
            hide_index=True, use_container_width=True)
        st.download_button('下載資金與名額核對 CSV', df.to_csv(index=False).encode('utf-8-sig'),
                           key+'.csv', 'text/csv', key='bridge_download')
        st.caption(f"八組帳戶離線一致，{report['offline']['elapsed_seconds']:.1f} 秒、0 次請求。原有封存帳本未改寫。")
