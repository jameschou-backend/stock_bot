"""Latest dated gap disposition, with bounded local integrity verification."""
from pathlib import Path
import hashlib
import json
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/completion_gaps_delivery_20260924_v3.json'


def load(path=REPORT,root=ROOT):
    path=Path(path);root=Path(root).resolve()
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('三項缺口摘要已變動')
    r=json.loads(raw)
    if r['schema']!='completion_gaps_v3' or r['live_qualified'] is not False:
        raise ValueError('缺口摘要不能提升實戰資格')
    for name,digest in r['evidence_sha256'].items():
        file=(root/name).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('缺口證據無法核對：'+name)
    return r


def render():
    with st.expander('9/24最新補件：歷史日期與零股核對',expanded=True):
        try:r=load()
        except (ValueError,KeyError,OSError) as exc:
            st.warning(str(exc));return
        st.write('已補歷史掛牌證據及零股檢查；尚未取得實戰資格。')
        cases=r['diversification']['cases']
        v3=cases['capacity_combined_3']['summary'];v5=cases['capacity_combined_5']['summary']
        bench=cases['benchmark_combined_0']['summary']
        st.write(f"① 策略：分散5檔壓力淨報酬{v5['total_return']:.2%}，高於3檔{v3['total_return']:.2%}，仍低於0050的{bench['total_return']:.2%}。尚未更換正式策略。")
        st.caption('2022/1/3～2026/9/9，100萬元複利、閒錢現金，扣費累積報酬，非年化。前日現金、買單預算及名額鎖至收盤；仍是日資料成交模型。')
        identity=r['listing_identity']
        st.write(f"② 歷史身分：原待查17檔再補{identity['newly_resolved']}檔掛牌日，剩{identity['remaining_unknown_starts']}檔。458個候選訊號及6組帳戶逐筆核對，已選股票的市場身分結果不變。")
        issues=identity['group_member_audit']['signal_date']['after']['issue_rows']
        if issues:
            st.warning(f'群組成員另有{issues}筆日期超出已核實掛牌區間，仍缺身分證據；不能直接判成上市前非法行情，也不能視為整個選股過程都通過。')
        corrected=len(identity['current_isin_date_corrections'])
        if corrected:
            st.caption(f'另修正{corrected}檔把現行證券表更新日當成首次掛牌日的核對錯誤；原始表與舊訊號保留。')
        if identity['unresolved_current_date_discrepancies']:
            st.caption(f"全市場另有{identity['unresolved_current_date_discrepancies']}檔現行證券表與公司基本資料日期不同，需釐清更名、換股或新代碼承接，尚未批次改寫。")
        st.write('③ 零股：已加入買賣雙邊配對檢查，避免成交量算兩次。官方免費樣本957筆、53,340股與日表吻合；金額仍差78元。2022年起始交易的逐筆撮合資料仍缺。')
        st.write('④ 大戶投：下方可中文逐筆填寫、附證據及下載核對紀錄；仍待真實券商回報。')
        st.caption('排程維持暫停。本輪身分核對使用本機封存資料，未重算或修改舊績效；完整逐日交易狀態與公告修訂仍未齊全。')
        st.download_button('下載本輪結果與待補證據',json.dumps(r,ensure_ascii=False,indent=2),
            'completion-gaps.json','application/json',key='completion_gaps_download')
