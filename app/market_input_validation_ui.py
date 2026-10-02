"""Read the dated primary-source coverage without running research or ingestion."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

from app.backtest_full_pass_ui import _file_signature
from skills.market_input_validation import CHECK_NAMES, require

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path('artifacts/forward_simulation/three_black_market_inputs_20261002.json')
_CACHE = {}
_LOCK = threading.RLock()


def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load(root=ROOT,path=REPORT):
    root = Path(root).resolve()
    path = root/path
    with _LOCK:
        cached = _CACHE.get((root,path))
        if cached:
            signatures,result = cached
            try:
                same = all(_file_signature(p,root) == sig for p,sig in signatures.items())
            except (OSError,ValueError):
                _CACHE.pop((root,path),None)
                raise
            if same:
                return deepcopy(result)
            _CACHE.pop((root,path),None)
        require(path.resolve().is_relative_to(root), '核對報告超出專案範圍')
        sidecar = path.with_suffix('.sha256')
        signatures = {p:_file_signature(p,root) for p in (path,sidecar)}
        require(_digest(path) == sidecar.read_text().strip(), '核對報告內容變動，須重新核對')
        value = json.loads(path.read_text())
        require(value.get('schema') in ('market_input_validation_v1', 'market_input_validation_v2'), '不支援此資料核對版本')
        require(all(value.get(k) is False for k in ('live_qualified','actual_fill_verified','unseen_validation','return_recomputed')),
                '資料核對不得提升實戰資格或冒稱重算報酬')
        require(set(value['checks']) == set(CHECK_NAMES) and all(type(v) is bool for v in value['checks'].values()),
                '核對項目不完整')
        require(value['complete_verified_data'] is all(value['checks'].values()), '資料完整狀態不一致')
        if value['schema'] == 'market_input_validation_v2':
            supplement = value.get('supplement', {})
            require(all(type(supplement.get(k)) is int and supplement[k] >= 0 for k in (
                'added_source_days', 'source_count', 'legacy_status_unknown'))
                and supplement['added_source_days'] <= supplement['source_count']
                and supplement['legacy_status_unknown'] <= supplement['source_count'], '補件數量不一致')
        for c in value['coverage'].values():
            if 'verified' in c:
                require(type(c['required']) is int and type(c['verified']) is int
                        and 0 <= c['verified'] <= c['required'] and c['missing'] == c['required']-c['verified']
                        and c['complete'] is (c['required'] > 0 and c['verified'] == c['required']),
                        '核對覆蓋數量不一致')
        refs = value['source_sha256']
        require(bool(refs), '缺少原始來源')
        for name,expected in refs.items():
            source = root/name
            require(source.resolve().is_relative_to(root), '核對來源超出專案範圍')
            signatures[source] = _file_signature(source,root)
            require(_digest(source) == expected, '原始來源已變動：'+name)
        require(all(_file_signature(p,root) == sig for p,sig in signatures.items()), '讀取期間來源變動')
        _CACHE[(root,path)] = signatures,deepcopy(value)
        return value


def overview(root=ROOT):
    try:
        value = load(root)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return dict(available=False,live_qualified=False,note='資料核對尚不可讀：'+str(exc))
    return dict(available=True,live_qualified=False,start=value['start'],end=value['end'],
        complete_verified_data=value['complete_verified_data'],counts=value['counts'],coverage=value['coverage'],
        source_days=value['source_days'],required_market_days=len(value['request_plan']),
        missing_market_days=value['requests_lower_bound'],checks=value['checks'],repair_summary=value['repair_summary'],
        candidate_count=value['identity']['candidate_count'],
        candidate_identity_issues=len(value['identity']['candidate_issues']),
        supplement=value.get('supplement') if value['schema'] == 'market_input_validation_v2' else None,
        observed_price_conflicts=len(value['price_conflicts']),report=str(REPORT))


def render(root=ROOT):
    import streamlit as st
    value = overview(root)
    st.subheader('回測行情與歷史股票名單核對')
    if not value['available']:
        st.warning(value['note'])
        return
    st.caption(f"三黑K策略來源：{value['start']} ～ {value['end']}；這是資料核對，不是新的績效回測。")
    if value['supplement']:
        supplement = value['supplement']
        st.info(f"本次新增核對 {supplement['added_source_days']:,} 張官方市場日表。")
        if supplement['legacy_status_unknown']:
            st.caption(f"其中 {supplement['legacy_status_unknown']} 張沿用早期官方快取，已核對原始檔與來源；舊收據未記錄 HTTP 狀態。")
    if not value['complete_verified_data']:
        st.warning('部分官方資料已核對，尚未完成全期間驗證。不能因此視為可實戰。')
    repairs = value['repair_summary']
    if repairs:
        st.info(f"已補回 {repairs['missing_quotes_repaired']} 筆缺漏行情，官方四個價格欄位相符；"
                f"重新計算後候選訊號 {repairs['original_candidate_count']:,} → {repairs['repaired_candidate_count']:,} 筆。")
        if repairs['full_account_replay_required']:
            st.warning('補件改變訊號或持股資料，必須重跑帳戶；舊報酬不能視為修復後結果。')
        else:
            st.caption('候選清單與順序相同，補件後期間沒有持有這些股票；未另外宣稱新的報酬。')
    c = value['coverage']['ordinary_fill_prices']
    a,b,d = st.columns(3)
    a.metric('普通盤成交股日價格',f"{c['verified']:,} / {c['required']:,}")
    b.metric('候選身分異常',f"{value['candidate_identity_issues']:,} / {value['candidate_count']:,}")
    d.metric('尚缺官方市場日表',f"{value['missing_market_days']:,}")
    st.caption('一個股日＝一檔股票的一個交易日；一張市場日表涵蓋該市場多檔股票。日成交總量、普通盤量及零股量分開核對。')
    with st.expander('查看各項覆蓋與限制'):
        labels = dict(ordinary_fill_prices='普通盤成交價',traded_stock_day_prices='成交股日的普通盤行情（不認證零股）',
                      holding_marks='持股收盤估值',candidate_signal_prices='全部候選訊號日價格')
        st.dataframe([dict(項目=label,已核對=value['coverage'][key]['verified'],
                           應核對=value['coverage'][key]['required']) for key,label in labels.items()],
                     hide_index=True,use_container_width=True)
        st.write('普通盤容量還需要同口徑的前20日資料；候選身分沒有異常，也不代表完整歷史股票名單已認證。')
        st.write('報告已記錄缺漏日期。資料不齊時，完整驗證模式會停止，不會刪除缺資料的股票繼續宣稱通過。')
