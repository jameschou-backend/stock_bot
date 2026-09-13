"""Read-only execution dependency inventory; never recompute account returns."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/contingent_audit_20260914.json'


def load_report(path=REPORT, root=ROOT):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('成交順序稽核檔案已變動，需重新驗證')
    report=json.loads(path.read_text())
    if (report.get('scope')!='legacy_path_dependency_inventory' or not report.get('audit_completed')
            or report.get('historical_replay_completed') is not False
            or report.get('live_qualified') is not False or report.get('total_return') is not None
            or report.get('network_calls')!=0):
        raise ValueError('依賴稽核不能當成完整逐筆回測報酬')
    for name,digest in (report['code_sha256'] | report['inputs_sha256']).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('成交順序來源或規則已變動：'+name)
    return report


def table(case):
    return pd.DataFrame([{'交易日':r['date'],'買進':', '.join(r['buy_stocks']),
        '賣出':', '.join(r['sell_stocks']),'需當日流入（元）':r['later_cash_required'],
        '需要釋放名額':'是' if r['needs_same_day_slot_release'] else '否',
        '清空持股最後一筆':', '.join(f"{s['stock_id']} {'零股' if s['channel']=='odd' else '整張'} {s['qty']}股" for s in r['closing_sales']) or '無清空持股',
        '整張逐筆覆蓋':str(sum(e['channel']=='board' and e['verified_ticks'] for e in r['evidence']))+'/'+str(sum(e['channel']=='board' for e in r['evidence'])),
        '核對狀態':'仍缺零股成交序列、事前委託及可用額度時間'} for r in case['rows']])


def render(path=REPORT):
    with st.expander('先賣再買：哪些交易還缺成交順序證據？',expanded=True):
        if not Path(path).exists():
            st.info('成交順序稽核尚未發布。');return
        try:report=load_report(path)
        except (OSError,KeyError,ValueError) as exc:
            st.error(str(exc));return
        st.write('已完成逐日缺件核對與成交回報流程測試；尚未完成歷史逐筆回測，這裡沒有新增策略報酬。')
        st.caption('固定隔日候選、股數與限價 → 等待賣出成交及可用額度 → 預留買單資金 → 只接受送單之後的成交。剩1股也仍占名額。')
        key=st.selectbox('查看換倉核對明細',['capacity_control_original','capacity_combined_original'],
            format_func=lambda s:'一般條件' if 'control' in s else '合併壓力',key='contingent_case')
        case=report['cases'][key]
        st.caption(f"共{case['dependency_days']}個依賴日，其中{case['slot_release_with_odd_final_sale_days']}天需要零股最後成交才能釋放名額；上輪稽核封存時的整張逐筆快取覆蓋{case['board_tick_cached']}/{case['board_tick_requests']}個股票日，後續補件進度見上方面板。")
        frame=table(case);st.dataframe(frame,hide_index=True,use_container_width=True)
        st.download_button('下載先賣再買核對明細',frame.to_csv(index=False).encode('utf-8-sig'),
            key+'-sequence.csv','text/csv',key='contingent_csv')
        st.caption('整張資料齊全也不能代替零股撮合；缺件不代表一定買不到。既有日資料報酬仍保留在上方配對表。')
