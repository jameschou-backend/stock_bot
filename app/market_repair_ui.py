"""Present the repaired-input replay separately from the prior data audit."""
from copy import deepcopy
import json
from pathlib import Path
import threading

from app.backtest_full_pass_ui import _file_signature
from skills.ordinary_volume_bundle import digest

REPORT = Path('artifacts/forward_simulation/market_input_repair_20261002.json')
_CACHE = {}
_LOCK = threading.RLock()


def load(root):
    root = Path(root).resolve()
    path = root/REPORT
    sidecar = path.with_suffix('.sha256')
    with _LOCK:
        cached = _CACHE.get(root)
        if cached:
            signatures, value = cached
            if all(_file_signature(p, root) == s for p, s in signatures.items()):
                return deepcopy(value)
            _CACHE.pop(root, None)
        signatures = {p: _file_signature(p, root) for p in (path, sidecar)}
        if digest(path) != sidecar.read_text().strip():
            raise ValueError('修復報告內容已變動')
        value = json.loads(path.read_text())
        if (value.get('schema') != 'market_input_repair_replay_v1'
                or any(value.get(k) is not False for k in ('live_qualified', 'actual_fill_verified', 'unseen_validation'))):
            raise ValueError('修復結果不得提高實戰或成交認證')
        if value.get('complete_verified_data') is not False:
            raise ValueError('此修復版本仍有資料認證限制')
        if not value.get('source_sha256'):
            raise ValueError('修復報告缺少封存來源')
        if (value['research']['completed'] is not value['research']['repeat_identical']
                or value['research']['volume_policy'] != 'legacy_total_research'):
            raise ValueError('研究回放缺少一致性驗證或成交假設')
        strict = value['strict']
        if set(strict['cases']) != {'three_black', 'benchmark'}:
            raise ValueError('嚴格診斷缺少策略或基準')
        complete = all(c['completed'] for c in strict['cases'].values())
        capacity = complete and strict['blocked_board_orders'] == 0 and all(
            c['ordinary_capacity_complete'] for c in strict['cases'].values())
        if strict['completed'] is not complete or strict['capacity_complete'] is not capacity:
            raise ValueError('嚴格診斷完成狀態與容量認證不一致')
        for name, expected in value['source_sha256'].items():
            source = (root/name).resolve()
            if not source.is_relative_to(root):
                raise ValueError('修復來源超出專案範圍')
            signatures[source] = _file_signature(source, root)
            if digest(source) != expected:
                raise ValueError('修復來源已變動：'+name)
        if any(_file_signature(p, root) != s for p, s in signatures.items()):
            raise ValueError('修復證據在讀取期間變動')
        _CACHE[root] = signatures, deepcopy(value)
        return value


def overview(root):
    try:
        result = load(root)
        return dict(available=True, **{k: v for k, v in result.items() if k != 'source_sha256'}, report=str(REPORT))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(available=False, live_qualified=False, note='修復回放尚不可讀：'+str(exc))


def render(root):
    import streamlit as st
    if not (Path(root)/REPORT).exists():
        return
    value = overview(root)
    st.subheader('修復後重新回測')
    if not value['available']:
        st.warning(value['note'])
        return
    repairs, research, strict = value['repairs'], value['research'], value['strict']
    st.caption(f"固定同一套策略：{value['start']} ～ {value['end']}，本金100萬元、3檔個股、閒置現金。")
    st.write(f"補入 {repairs['quotes_added']:,} 筆行情；候選 {repairs['original_candidates']:,} → {repairs['repaired_candidates']:,} 筆。")
    if research['completed']:
        rows = []
        for name, label in [('three_black', '三黑K出場策略'), ('benchmark', '0050基準')]:
            summary = research['cases'][name]['summary']
            rows.append(dict(帳戶=label, 累積報酬=f"{summary['total_return']:.2%}",
                             最大回撤=f"{summary['max_drawdown']:.2%}", 期末資產=f"{summary['final_nav']:,.0f} 元"))
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption('以上為修正行情與身分後的研究對照：沿用全日量估計普通盤容量，買賣價採各渠道日高低中點；兩次離線帳戶結果一致。')
    else:
        st.warning('修正後研究帳戶尚未完整通過，未顯示部分期間作為總報酬。')
    if not strict['completed']:
        st.warning('同口徑普通盤成交驗證尚未完成，研究報酬不能視為已驗證可成交。')
    elif strict['blocked_board_orders']:
        st.warning(f"嚴格資料診斷有 {strict['blocked_board_orders']:,} 筆普通盤委託因缺量證據被阻擋；這種跳過資料的結果不列為策略績效。")
    elif not strict['capacity_complete']:
        st.warning('回放已結束，但普通盤容量仍未完整認證；不能因此視為可實戰。')
    with st.expander('查看未解決項目與修復證據'):
        for item in value['limitations']:
            st.write('• '+item)
        st.json(strict)
    st.caption('下方保留修復前的市場資料核對，供追溯原先問題。')
