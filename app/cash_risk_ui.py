"""Compact frozen research summary; no network calls or account mutations."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'artifacts/forward_simulation/cash_risk_delivery_20260913.json'


def render(path=REPORT):
    with st.expander('現金版補強：成交壓力、整戶減碼與資料缺口', expanded=True):
        if not path.is_file():
            st.info('補強研究尚未完成封存；不顯示暫存績效。')
            return
        try:
            report = json.loads(path.read_text())
            for name, expected in report['research_code_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                    raise ValueError('研究程式已改變，需要重新驗證：'+name)
            if not report['offline_identical']:
                raise ValueError('完整帳戶尚未通過離線重現')
        except (KeyError, ValueError, OSError) as exc:
            st.error('研究證據不可用：'+str(exc)); return
        st.warning('歷史壓力試驗，尚未取得實盤資格；全帳戶減碼仍是比較版本，沒有套用到目前帳本。')
        st.caption('2022/1/3–2026/9/9，本金100萬元複利、個股最多3檔、閒置現金。百分比皆為整段歷史扣成本報酬。')
        rows = report['comparison']
        names = list(rows)
        selected = st.selectbox('查看哪一組比較', names, key='cash_risk_comparison')
        st.dataframe(pd.DataFrame(rows[selected]), hide_index=True, use_container_width=True)
        st.write(report['conclusion'])
        st.write('仍待完成：')
        for item in report['remaining']:
            st.write('• '+item)
        st.caption(f"完整離線重播 {report['offline_seconds']:.1f} 秒；開啟本區塊不抓資料、不跑回測。")
        st.download_button('下載補強結果與來源指紋', path.read_text(),
                           'cash-risk-research.json', 'application/json', key='cash_risk_download')
