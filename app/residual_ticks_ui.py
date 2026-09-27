"""Show current five-slot execution evidence, including explicit blocked cases."""
from pathlib import Path
from copy import deepcopy
import json
import threading

from app.backtest_tool_ui import verified_bytes
from app.backtest_full_pass_ui import _file_signature
from scripts.research_exit_scenarios import sha, summarize

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/forward_simulation/residual_ticks_20260928.json'
LABELS={'strategy_normal':'五檔個股｜一般', 'benchmark_normal':'0050基準｜一般',
        'strategy_stress':'五檔個股｜較低量與較高成本', 'benchmark_stress':'0050基準｜較低量與較高成本'}
_CACHE={}
_LOCK=threading.RLock()


def _validate(path=REPORT,root=ROOT):
    path,root=Path(path),Path(root).resolve()
    if sha(path)!=path.with_suffix('.sha256').read_text().strip():
        raise ValueError('逐筆研究報告雜湊不符')
    value=json.loads(path.read_text())
    if (value.get('schema')!='residual_ticks_v1' or value.get('offline_identical') is not True
            or value.get('baseline_reproduced') is not True or value.get('compared_cases')!=4
            or value.get('live_qualified') is not False or value.get('unseen_validation') is not False
            or value.get('network_calls')!=0 or set(value['cases'])!=set(LABELS)):
        raise ValueError('缺少同版本完整重播證據')
    for name,digest in value['source_sha256'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root) or sha(target)!=digest:
            raise ValueError('逐筆研究來源已變更：'+name)
    for row in value['cases'].values():
        result=json.loads(verified_bytes(row['result'],root,'.json'))
        if row['completed']!=result['completed'] or row['summary']!=result['summary']:
            raise ValueError('研究狀態與完整結果不同')
        if row['completed']:
            if row['summary']!=summarize(result['account']) or not result['audit']['tick_fills_rebuilt']:
                raise ValueError('逐筆績效尚未核對')
        elif row['summary'] is not None or 'account' in result:
            raise ValueError('資料不足的帳戶不能顯示期間報酬')
    if value['all_completed'] != all(r['completed'] for r in value['cases'].values()):
        raise ValueError('整體狀態與各組結果不一致')
    return value


def load(path=REPORT,root=ROOT):
    path,root=Path(path).resolve(),Path(root).resolve()
    value=json.loads(path.read_text())
    refs=set(value['source_sha256']) | {str(path.relative_to(root)),str(path.with_suffix('.sha256').relative_to(root))}
    refs.update(row['result']['path'] for row in value['cases'].values())
    signature=tuple((name,_file_signature(root/name,root)) for name in sorted(refs))
    key=(str(path),str(root))
    with _LOCK:
        if key in _CACHE and _CACHE[key][0]==signature:
            return deepcopy(_CACHE[key][1])
        verified=_validate(path,root)
        if signature!=tuple((name,_file_signature(root/name,root)) for name in sorted(refs)):
            raise ValueError('核對期間來源已變更')
        _CACHE[key]=(signature,deepcopy(verified))
        return verified


def render():
    import pandas as pd
    import streamlit as st
    with st.expander('目前五檔策略：隔天限價能否成交？',expanded=True):
        st.caption('收盤訊號 → 次一交易日限價 → 穿價成交量檢查；未額外延後一天。')
        if not REPORT.exists():
            st.info('完整重播核對完成後，這裡會顯示各組結果與缺資料位置。')
            return
        try:
            value=load()
        except (OSError,ValueError,KeyError,TypeError) as exc:
            st.error('本輪逐筆結果目前不可採信：'+str(exc))
            return
        st.write('本金100萬元、5檔個股、整張、閒錢保留現金。買單隔天09:01起有效，'
                 '13:25截止；用前日調整後參考價掛單，同價不算成交，未成交買單當天取消。')
        st.warning('逐筆成交仍是估計；本輪不授予實戰資格。缺資料的組別不顯示報酬。')
        rows=[]
        for name,row in value['cases'].items():
            summary=row['summary']
            rows.append({'組別':LABELS[name],'狀態':'完整重播' if row['completed'] else '缺資料，中止',
                '總報酬':f"{summary['total_return']:+.2%}" if summary else '—',
                '最大回撤':f"{summary['max_drawdown']:.2%}" if summary else '—',
                '原因':row.get('reason') or ''})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        name=st.selectbox('查看本輪逐筆計畫',list(LABELS),format_func=LABELS.get,key='residual_tick_case')
        result=json.loads(verified_bytes(value['cases'][name]['result'],ROOT,'.json'))
        plans=(result['account']['tick_plans'] if result['completed'] else result['partial_diagnostics']['plans'])
        shown=pd.DataFrame(plans)
        if not shown.empty:
            fields={'date':'執行日','signal_date':'訊號日','stock_id':'代號','side':'買賣',
                    'limit_price':'限價','planned_qty':'計畫股數','reserved_cash':'預留現金','rejection':'略過原因'}
            shown=shown[list(fields)].rename(columns=fields)
            shown['買賣']=shown['買賣'].map({'buy':'買進','sell':'賣出'})
            st.dataframe(shown,hide_index=True,use_container_width=True)
        st.download_button('下載本輪計畫、成交與核對結果',json.dumps(result,ensure_ascii=False,indent=2),
                           name+'.json','application/json',key='residual_tick_download')
        st.caption('四組各重跑兩次、原五檔日線帳戶逐欄重現；離線重現包含相同的中止原因，並不表示資料已齊全。')
