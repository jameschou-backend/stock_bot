"""Latest dated gap disposition, with bounded local integrity verification."""
from pathlib import Path
import hashlib
import json
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/completion_gaps_delivery_20260924_v2.json'


def load(path=REPORT,root=ROOT):
    path=Path(path);root=Path(root).resolve()
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('三項缺口摘要已變動')
    r=json.loads(raw)
    if r['schema']!='completion_gaps_v2' or r['live_qualified'] is not False:
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
        st.write('已依序補研究、歷史日期及操作介面；三項實戰資格仍未全部通過。')
        cases=r['diversification']['cases']
        v3=cases['capacity_combined_3']['summary'];v5=cases['capacity_combined_5']['summary']
        bench=cases['benchmark_combined_0']['summary']
        st.write(f"① 策略：分散5檔壓力淨報酬{v5['total_return']:.2%}，高於3檔{v3['total_return']:.2%}，仍低於0050的{bench['total_return']:.2%}。尚未更換正式策略。")
        st.caption('2022/1/3～2026/9/9，100萬元複利、閒錢現金，扣費累積報酬，非年化。前日現金、買單預算及名額鎖至收盤；仍是日資料成交模型。')
        st.write(f"② 歷史資料：本輪另核實{r['listing_archive']['newly_resolved_since_boundary']}檔原始上櫃日，未明起日降至{r['listing_archive']['remaining_unknown_starts']}檔。完整公告修訂及零股撮合序列仍缺。")
        st.write('③ 大戶投：下方可中文逐筆填寫、附證據及下載核對紀錄。尚未取得真實券商回報，填寫成功不等於已認證。')
        st.caption('排程維持暫停。6組研究約52秒完成，0次新增資料請求，離線逐欄重現。')
        st.download_button('下載本輪結果與待補證據',json.dumps(r,ensure_ascii=False,indent=2),
            'completion-gaps.json','application/json',key='completion_gaps_download')
