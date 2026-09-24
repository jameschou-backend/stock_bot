"""Latest dated gap disposition, with bounded local integrity verification."""
from pathlib import Path
import hashlib
import json
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/completion_gaps_delivery_20260924.json'


def load(path=REPORT,root=ROOT):
    path=Path(path);root=Path(root).resolve()
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('三項缺口摘要已變動')
    r=json.loads(raw)
    if r['schema']!='completion_gaps_v1' or r['live_qualified'] is not False:
        raise ValueError('缺口摘要不能提升實戰資格')
    for name,digest in r['evidence_sha256'].items():
        file=(root/name).resolve()
        if not file.is_relative_to(root) or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('缺口證據無法核對：'+name)
    return r


def render():
    with st.expander('9/24最新結果：三項實戰缺口',expanded=True):
        try:r=load()
        except (ValueError,KeyError,OSError) as exc:
            st.warning(str(exc));return
        st.write('執行工具已補；真實回報、穩健超額報酬及完整歷史成交證據尚未通過。')
        st.write('① 大戶投：已加入成交／撤單／額度時序核對，下方可上傳紀錄。尚無券商原始回報。')
        st.write('② 新改法未採用：保留候選至隔日，壓力淨報酬92.49%；原保守組115.73%，同規格0050為239.83%。')
        st.caption('2022/1/3～2026/9/9，100萬元複利；扣費累積報酬，非年化。正常2日組380.83%；仍是日資料模型，非已驗證實際成交。')
        st.write('③ 41檔回測起點均找到行情或停牌解釋，另補1檔原始上櫃日。仍缺40檔原始起日、完整公告修訂及零股撮合序列。')
        st.download_button('下載本輪結果與待補證據',json.dumps(r,ensure_ascii=False,indent=2),
            'completion-gaps.json','application/json',key='completion_gaps_download')
