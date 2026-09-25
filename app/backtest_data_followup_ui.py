"""Surface source corrections and conflicts without promoting historical performance."""
import json
from pathlib import Path

from app.backtest_tool_ui import verified_bytes
from skills.publication_versions import digest, verify_archive

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'artifacts/forward_simulation/backtest_data_followup_20260925.json'


def load(path=REPORT, root=ROOT):
    path, root = Path(path), Path(root).resolve()
    raw = path.read_bytes()
    if digest(raw) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('最新補件摘要指紋不一致')
    value = json.loads(raw)
    if (value.get('schema') != 'backtest_data_followup_v1' or value.get('live_qualified') is not False
            or value.get('performance_recomputed') is not False):
        raise ValueError('資料補件不能變更策略績效或實盤資格')
    sources = value['reports']
    if set(sources) != {'ordinary', 'identity', 'odd_lot', 'publication'}:
        raise ValueError('最新補件報告範圍不完整')
    reports = {k:json.loads(verified_bytes(v,root,'.json')) for k,v in sources.items()}
    from skills.board_tape_reconciliation import verify_report as verify_board
    from scripts.audit_historical_universe_followup import verify_report as verify_identity
    from scripts.prepare_odd_lot_evidence_request import verify_request
    odd_directory=(root/sources['odd_lot']['path']).parent
    odd_manifest=json.loads((odd_directory/'manifest.json').read_text())
    verified = dict(ordinary=verify_board(root/sources['ordinary']['path'],root),
        identity=verify_identity(root/sources['identity']['path']),
        odd_lot=verify_request(odd_directory),
        publication=verify_archive(root/sources['publication']['path']))
    if reports != verified or odd_manifest != json.loads((odd_directory/'manifest.json').read_text()):
        raise ValueError('補件資料在來源核對期間已變動')
    base = value['base_data_report']
    verified_bytes(base,root,'.json')
    for key in ('ordinary','odd_lot'):
        if reports[key]['input_sha256'].get(base['path']) != base['sha256']:
            raise ValueError('成交補件未對應原20案的相同資料範圍')
    return dict(value,odd_lot_exports=odd_manifest['files_sha256']), reports


def render():
    import streamlit as st
    if not REPORT.exists():
        return
    st.subheader('最新資料核對：哪些補好了，哪些仍有衝突')
    try:
        index, reports = load()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.error(f'最新資料補件無法核對：{exc}')
        return
    board, identities, odd, publication = (reports[k] for k in ('ordinary','identity','odd_lot','publication'))
    summary = board['summary']
    st.write(f"**普通盤日總量對帳：{summary['same_scope_aggregate_matched']} 股日一致、"
             f"{summary['aggregate_conflicts']} 股日衝突、"
             f"{summary['independent_daily_source_missing']} 股日待補獨立來源。**")
    if summary['aggregate_conflicts']:
        st.warning('有逐筆檔與官方日成交量／金額不一致；保留衝突，不把有檔案當作完整成交證據。')
    st.caption('日總量一致仍不證明每一筆時序完整或自己的委託能成交；上方為原20案已知路徑的資料核對。')
    st.write(f"上市日期未知：{identities['previous_unknown_starts']} → {identities['remaining_unknown_starts']}；"
             f"待核證券類別：{identities['unconfirmed_categories']}；"
             f"待釐清日期差異：{identities['unresolved_current_date_discrepancies']}。")
    st.write('已補1507停止交易區間：2022/4/14起不可當作仍正常交易；終止上市日不能代替停牌日。')
    st.write(f"**歷史零股仍缺 {odd['total_stock_days']:,} 股日。** 已備妥市場／月份清單與供應商詢問規格，尚未採購。")
    st.write(f"公告原文已保存 {len(publication['observations'])} 篇，分開記錄刊登日期與這個版本首次取得時間。"
             '這是華新科的局部資料，尚非全市場歷史公告或完整修訂史。')
    st.caption('本輪沒有重算報酬、沒有啟動排程；策略仍未取得實盤資格。')
    with st.expander('下載核對結果與零股需求清單'):
        for key,label in (('ordinary','普通盤對帳'),('identity','上市身分補件'),('publication','公告版本紀錄')):
            st.download_button('下載'+label,json.dumps(reports[key],ensure_ascii=False,indent=2),
                key+'-followup.json','application/json',key='data_followup_'+key)
        directory=(ROOT/index['reports']['odd_lot']['path']).parent
        csv_path=str((directory/'stock_days.csv').relative_to(ROOT))
        st.download_button('下載零股股日清單',verified_bytes(dict(path=csv_path,
            sha256=index['odd_lot_exports']['stock_days.csv']),ROOT,'.csv'),
            'historical-odd-lot-stock-days.csv','text/csv',key='data_followup_odd_csv')
