"""Display a dated audit snapshot without querying vendors or changing strategy rules."""
import hashlib
import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT/'docs/adjustment_readiness_summary_20260914.json'


def load_summary(path=SUMMARY, root=ROOT):
    summary = json.loads(Path(path).read_text())
    report_path = (Path(root)/summary['report_path']).resolve()
    if not report_path.is_relative_to(Path(root).resolve()):
        raise ValueError('研究報告路徑超出專案')
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest()!=summary['report_sha256']:
        raise ValueError('研究報告已變更，請重新核對')
    report = json.loads(report_bytes)
    if summary['live_qualified'] or summary['applied_to_production'] or report['live_qualified'] or report['database_mutations']:
        raise ValueError('研究摘要不可宣告正式切換或實盤資格')
    return summary


def render():
    with st.expander('歷史資料修補進度',expanded=True):
        try:
            s=load_summary()
        except (OSError,ValueError,KeyError) as exc:
            st.warning('本機研究報告無法核對：'+str(exc));return
        st.caption('2026/9/14 研究快照，核對資料至 2026/9/9；不是即時行情。')
        st.write(f"官方事件來源已核對 {s['source_windows']} 個查詢區間；獨立因子草稿涵蓋 {s['shadow_stocks']:,} 檔、{s['shadow_rows']:,} 列。")
        comparable=s['candidate_count']-s['candidate_unresolved']
        st.write(f"原有 {s['candidate_count']} 筆候選：{comparable} 筆可比較，{s['candidate_unresolved']} 筆待補；20日動能變動超過0.1個百分點的有 {s['candidate_changed_above_0_1pp']} 筆。")
        st.write(f"已核對 {s['cross_market_transitions']} 筆跨市場轉板；另有 {s['unresolved_post_end_companies']} 檔終止掛牌後行情待核對。")
        st.warning('草稿尚未套用正式資料。仍缺完整歷史名冊及零股成交資料，不能據此宣稱可實戰或完整回測跑贏0050。')
        st.caption('候選動能相近，不代表全市場重新選股、出場路徑或最終報酬相同。')
        st.download_button('下載資料修補摘要',json.dumps(s,ensure_ascii=False,indent=2),
                           'adjustment-readiness.json','application/json',key='adjustment_readiness_download')
