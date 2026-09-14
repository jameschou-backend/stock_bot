"""Display a dated audit snapshot without querying vendors or changing strategy rules."""
import hashlib
import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT/'docs/adjustment_readiness_summary_20260914.json'
COMPLETION = ROOT/'docs/data_completion_summary_20260914.json'


def load_completion(path=COMPLETION,root=ROOT):
    summary=json.loads(Path(path).read_text())
    if summary['schema']!='historical_data_completion_v1' or summary['live_qualified']:
        raise ValueError('Invalid historical completion summary')
    result={}
    for key,item in summary['sources'].items():
        source=(Path(root)/item['path']).resolve()
        if not source.is_relative_to(Path(root).resolve()):raise ValueError('Source outside project')
        raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Repair evidence changed: '+key)
        result[key]=json.loads(raw)
    repair,derived=result['price_repair'],result['derived_rebuild']
    if (not repair['applied'] or not derived['completed']
            or repair['registry_sha256']!=summary['sources']['price_registry']['sha256']
            or derived['registry_sha256']!=repair['registry_sha256']):
        raise ValueError('Price repair and derived rebuild are not jointly verified')
    return result


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
        st.write(f"已核對 {s['cross_market_transitions']} 筆跨市場轉板。")
        try:
            done=load_completion()
        except (OSError,ValueError,KeyError) as exc:
            st.warning('最新修補證據無法核對：'+str(exc))
            done=None
        if done:
            r,d,m,o=(done[k] for k in ('price_repair','derived_rebuild','market_identity','odd_lot_sample'))
            st.success(f"行情修復完成：{r['reviewed_stocks']} 檔、{r['raw_rows']} 筆錯誤行情已隔離，移除 {r['invalidated_labels']} 筆受污染標籤，重算 {d['rows']:,} 筆特徵。")
            st.write(f"官方目前名冊：{m['current_rows']:,} 個四碼證券，其中 {m['current_ordinary_stocks']:,} 檔屬股票類；另有 {m['ended_episodes']} 筆歷史終止紀錄。")
            st.write(f"原有 {m['signal_rows']} 筆候選的市場／掛牌日期可對上；完整歷史名冊仍有 {m['missing_historical_starts']} 筆掛牌起日待核實，證券類別與公告時間封存也未齊全。")
            st.write(f"零股格式已能辨認實際成交與試算；官方免費範例 {o['record_count']} 筆全部是試算，不能用來證明歷史成交。")
        st.warning('還原因子仍是未套用的研究草稿。完整歷史名冊與零股成交仍未齊全，不能宣稱可實戰或完整回測跑贏0050。')
        st.caption('候選動能相近，不代表全市場重新選股、出場路徑或最終報酬相同。')
        st.download_button('下載資料修補摘要',json.dumps(dict(adjustment_snapshot=s,verified_repair=done),ensure_ascii=False,indent=2),
                           'adjustment-readiness.json','application/json',key='adjustment_readiness_download')
