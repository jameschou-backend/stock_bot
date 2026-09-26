"""Show the unchanged strategy's earlier-period failure beside its recent gains."""
from pathlib import Path
import json
from app.backtest_tool_ui import verified_bytes
from scripts.replay_million import summarize
from skills.backtest_contract import validate_completed_account

ROOT=Path(__file__).resolve().parents[1]
PUBLICATION=Path('artifacts/forward_simulation/index_earlier_20260927.json')
ARMS={'equal':'正二75%月再平衡對照','trend200':'200日趨勢主規則','trend180':'180日鄰近參數','trend220':'220日鄰近參數'}

def load(root=ROOT):
    root=Path(root);path=root/PUBLICATION
    value=json.loads(verified_bytes(dict(path=str(PUBLICATION),sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    if (value.get('schema')!='index_earlier_publication_v1'
            or any(value.get(k) is not False for k in ('adopted','live_qualified','unseen_validation','strict_data_ready'))
            or (value['start'],value['end'],value['initial_cash'])!=('2016-01-04','2021-12-30',1000000)):
        raise ValueError('早期驗證範圍或資格標記不符')
    proof=json.loads(verified_bytes(value['offline_verification'],root,'.json'))
    if (proof.get('passed') is not True or proof.get('compared_cases')!=34 or proof.get('newly_executed_cases')!=68
            or len(proof.get('runs',[]))!=2 or proof['runs'][0]['path']==proof['runs'][1]['path']
            or value['run_manifest'] not in proof['runs']):raise ValueError('缺少兩輪34個完整帳戶比對')
    for ref in proof['runs']:verified_bytes(ref,root,'.json')
    manifest=json.loads(verified_bytes(value['run_manifest'],root,'.json'))
    folder=Path(value['run_manifest']['path']).parent
    report=json.loads(verified_bytes(dict(path=str(folder/'report.json'),sha256=manifest['files_sha256']['report.json']),root,'.json'))
    for key in ('cases','benchmarks','validation','data_quality','all_completed'):
        if value[key]!=report[key]:raise ValueError('早期發布內容與封存帳戶不符')
    if set(value['cases'])!={f'{arm}_{m}' for arm in ARMS for m in range(8)} or set(value['benchmarks'])!={'control','combined'}:
        raise ValueError('早期組別不完整')
    for row in [*value['cases'].values(),*value['benchmarks'].values()]:
        case=Path(row['result']['path'])
        if (not case.is_relative_to(folder) or manifest['files_sha256'].get(str(case.relative_to(folder)))!=row['result']['sha256']
                or row['completed'] is not True):raise ValueError('早期帳戶未完成或未連結來源')
    return value

def overview(root=ROOT):
    try:
        value=load(root)
        return dict(available=True,start=value['start'],end=value['end'],initial_cash=value['initial_cash'],
            live_qualified=False,unseen_validation=False,validation=value['validation'],data_quality=value['data_quality'],
            arms={a:dict(label=label,normal_return=value['cases'][a+'_0']['summary']['total_return'],
                all_stresses_return=value['cases'][a+'_7']['summary']['total_return'],
                winning_stresses=value['validation']['arms'][a]['benchmark_winning_stresses']) for a,label in ARMS.items()},
            benchmark_returns={k:v['summary']['total_return'] for k,v in value['benchmarks'].items()},
            publication=dict(path=str(PUBLICATION),sha256=(Path(root)/PUBLICATION.with_suffix('.sha256')).read_text().strip()))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return dict(available=False,reason=str(exc),live_qualified=False)

def detail(value,name,root=ROOT):
    case=json.loads(verified_bytes(value['cases'][name]['result'],root,'.json'))
    key='combined' if case['config']['factor_mask']&1 else 'control'
    benchmark=json.loads(verified_bytes(value['benchmarks'][key]['result'],root,'.json'))
    dates=[r['date'] for r in benchmark['account']['daily']]
    for item,expected in ((case,value['cases'][name]),(benchmark,value['benchmarks'][key])):
        validate_completed_account(item['account'],dates,value['start'],value['end'])
        if item['summary']!=summarize(item['account']) or item['summary']!=expected['summary'] or item['config']!=expected['config']:
            raise ValueError('早期明細與摘要不同')
    for k in ('initial_cash','commission','minimum_fee','participation','odd_participation','slippage'):
        if case['account']['settings'][k]!=benchmark['account']['settings'][k]:raise ValueError('早期基準成本或本金不符')
    return case,benchmark

def render():
    import pandas as pd
    import streamlit as st
    if not (ROOT/PUBLICATION).exists():return
    with st.expander('同一規則換到2016–2021，結果如何？',expanded=True):
        try:
            value=load();rows=[]
            for arm,label in ARMS.items():
                a,b=value['cases'][arm+'_0'],value['cases'][arm+'_7'];gate=value['validation']['arms'][arm]
                rows.append({'規則':label,'一般淨報酬':f"{a['summary']['total_return']:.2%}",
                    '全部壓力淨報酬':f"{b['summary']['total_return']:.2%}",
                    '累積贏0050情境':str(gate['benchmark_winning_stresses'])+'/8','最差回撤':f"{gate['worst_drawdown']:.2%}"})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.write(f"早期同成本0050：一般 {value['benchmarks']['control']['summary']['total_return']:.2%}；滑價加倍 {value['benchmarks']['combined']['summary']['total_return']:.2%}。")
            st.warning('200日主規則在另一段歷史八種情境都輸給0050。不能只憑2022之後的好結果採用，也不會事後把較好的對照組改稱原本主策略。')
            st.caption('兩段各自從100萬元開始，不能把兩段累積報酬直接相加或相乘成同一帳戶。00631L原規則不變；較早行情仍屬回溯檢查，限價推算與股息付款來源限制保留。')
            if st.checkbox('查看早期交易與資產',key='index_earlier_detail'):
                arm=st.selectbox('早期規則',list(ARMS),format_func=ARMS.get,key='index_earlier_arm')
                from app.research_account_detail import scenario_label
                mask=st.selectbox('早期成交情境',list(range(8)),format_func=scenario_label,key='index_earlier_mask')
                name=f'{arm}_{mask}';case,benchmark=detail(value,name);account=case['account']
                daily=pd.DataFrame(account['daily']);daily['0050']=[r['nav'] for r in benchmark['account']['daily']]
                st.line_chart(daily.set_index('date')[['nav','0050']].rename(columns={'nav':'策略'}))
                trades=pd.DataFrame(account['trades']);cols=['date','signal_date','side','qty','reference_price','total_cost','cash_after']
                st.dataframe(trades[cols].rename(columns=dict(zip(cols,['成交日','訊號日','買賣','股數','參考價','費稅滑價','成交後現金']))),hide_index=True,use_container_width=True)
                st.download_button('下載早期完整帳戶與判斷',json.dumps(case,ensure_ascii=False),file_name=name+'-2016-2021.json',mime='application/json',key='index_earlier_download')
            st.download_button('下載跨期驗證摘要',json.dumps(value,ensure_ascii=False),file_name='index-earlier.json',mime='application/json',key='index_earlier_summary')
        except (OSError,ValueError,KeyError,TypeError) as exc:st.error('早期驗證讀取失敗：'+str(exc))
