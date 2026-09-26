"""Load the uninterrupted capital account and expose remaining data limitations."""
from pathlib import Path
import json
from app.backtest_tool_ui import verified_bytes
from scripts.replay_million import summarize
from skills.backtest_contract import validate_completed_account

ROOT=Path(__file__).resolve().parents[1]
PUBLICATION=Path('artifacts/forward_simulation/index_continuous_20260927.json')
ARMS={'equal':'正二75%月再平衡對照','trend200':'200日趨勢主規則','trend180':'180日鄰近參數','trend220':'220日鄰近參數'}

def load(root=ROOT):
    root=Path(root);path=root/PUBLICATION
    value=json.loads(verified_bytes(dict(path=str(PUBLICATION),sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    if (value.get('schema')!='index_continuous_publication_v1'
            or any(value.get(k) is not False for k in ('adopted','live_qualified','unseen_validation','strict_data_ready'))
            or (value['start'],value['end'],value['initial_cash'])!=('2016-01-04','2026-09-09',1000000)):
        raise ValueError('連續資金驗證範圍或資格標記不符')
    proof=json.loads(verified_bytes(value['offline_verification'],root,'.json'))
    if (proof.get('passed') is not True or proof.get('compared_cases')!=38 or proof.get('newly_executed_cases')!=76
            or len(proof.get('runs',[]))!=2 or proof['runs'][0]['path']==proof['runs'][1]['path']
            or value['run_manifest'] not in proof['runs']):raise ValueError('缺少兩輪38個完整帳戶比對')
    for ref in proof['runs']:verified_bytes(ref,root,'.json')
    manifest=json.loads(verified_bytes(value['run_manifest'],root,'.json'))
    folder=Path(value['run_manifest']['path']).parent
    report=json.loads(verified_bytes(dict(path=str(folder/'report.json'),sha256=manifest['files_sha256']['report.json']),root,'.json'))
    for key in ('cases','benchmarks','reference_controls','validation','data_quality','all_completed'):
        if value[key]!=report[key]:raise ValueError('連續資金發布內容與封存帳戶不符')
    if set(value['cases'])!={f'{arm}_{m}' for arm in ARMS for m in range(8)} or set(value['benchmarks'])!={'control','combined'}:
        raise ValueError('連續資金組別不完整')
    if set(value['reference_controls'])!={f'reference_{p}_{m}' for p in ('early','recent') for m in ('control','combined')}:
        raise ValueError('限價修正的四個比較帳戶不完整')
    for row in [*value['cases'].values(),*value['benchmarks'].values(),*value['reference_controls'].values()]:
        case=Path(row['result']['path'])
        if (not case.is_relative_to(folder) or manifest['files_sha256'].get(str(case.relative_to(folder)))!=row['result']['sha256']
                or row['completed'] is not True):raise ValueError('連續資金帳戶未完成或未連結來源')
    ref=value['limit_reconciliation']
    if ref['path']!=str(folder/'limit-audit.json') or ref['sha256']!=manifest['files_sha256']['limit-audit.json']:
        raise ValueError('限價核對未連結本次帳戶')
    verified_bytes(ref,root,'.json')
    impact=json.loads(verified_bytes(value['reference_impact'],root,'.json'))
    if set(impact['controls'])!=set(value['reference_controls']):raise ValueError('基準影響比對不完整')
    for name,row in impact['controls'].items():
        if row['new_result']!=value['reference_controls'][name]['result']:raise ValueError('基準影響連結不同帳戶')
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
            reference_controls={k:dict(account_identical=v['account_identical'],return_difference=v['return_difference']) for k,v in value['reference_controls'].items()},
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
            raise ValueError('連續資金明細與摘要不同')
    for k in ('initial_cash','commission','minimum_fee','participation','odd_participation','slippage'):
        if case['account']['settings'][k]!=benchmark['account']['settings'][k]:raise ValueError('連續資金基準成本或本金不符')
    return case,benchmark

def render():
    import pandas as pd
    import streamlit as st
    if not (ROOT/PUBLICATION).exists():return
    with st.expander('2016–2026連續投入：中間不重設100萬元',expanded=True):
        try:
            value=load();rows=[]
            for arm,label in ARMS.items():
                a,b=value['cases'][arm+'_0'],value['cases'][arm+'_7'];gate=value['validation']['arms'][arm]
                rows.append({'規則':label,'一般淨報酬':f"{a['summary']['total_return']:.2%}",
                    '全部壓力淨報酬':f"{b['summary']['total_return']:.2%}",
                    '累積贏0050情境':str(gate['benchmark_winning_stresses'])+'/8','最差回撤':f"{gate['worst_drawdown']:.2%}"})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.write(f"連續同成本0050：一般 {value['benchmarks']['control']['summary']['total_return']:.2%}；滑價加倍 {value['benchmarks']['combined']['summary']['total_return']:.2%}。")
            st.caption('2016/01/04–2026/09/09｜只在第一天投入100萬元｜2,604個市場日｜盈虧和持股直接延續｜整張交易｜其餘現金')
            st.warning('完整期間勝出也不能抹除2016–2021主規則落敗。00631L包含槓桿效果，0050及00631L上下限含明示推算；目前仍未取得實戰資格。')
            audit=json.loads(verified_bytes(value['limit_reconciliation'],ROOT,'.json'))
            controls=value['reference_controls'];impact=json.loads(verified_bytes(value['reference_impact'],ROOT,'.json'))
            identical=all(r['daily_assets_identical'] and r['trades_identical'] for r in impact['controls'].values())
            st.info(f"0050限價核對：{len(audit['provider_field_differences']):,}天的原始欄位與ETF規則推算不同，其中{len(audit['provider_ohlc_conflicts'])}天和行情衝突。四個分段基準重新執行後，每日資產及實際模擬成交{'均相同' if identical else '有差異，請查看下載報告'}。")
            if st.checkbox('顯示限價衝突與基準影響',key='index_continuous_limits'):
                st.json(dict(conflicts=audit['provider_ohlc_conflicts'],reference_controls={k:dict(account_identical=r['account_identical'],return_difference=r['return_difference'],changed_fields=impact['controls'][k]['changed_account_fields'],order_difference_count=len(impact['controls'][k]['order_differences'])) for k,r in controls.items()}))
                st.caption('近期除息日不再提前折減委託參考價，未達整張的申請股數可能不同；完整差異保留於下載摘要的 reference_impact 來源。')
                st.caption('保留資料源原始欄位；推算值不是逐日交易所原始上下限。')
            if st.checkbox('查看連續帳戶交易與資產',key='index_continuous_detail'):
                arm=st.selectbox('連續帳戶規則',list(ARMS),format_func=ARMS.get,key='index_continuous_arm')
                from app.research_account_detail import scenario_label
                mask=st.selectbox('連續帳戶成交情境',list(range(8)),format_func=scenario_label,key='index_continuous_mask')
                name=f'{arm}_{mask}';case,benchmark=detail(value,name);account=case['account']
                daily=pd.DataFrame(account['daily']);daily['0050']=[r['nav'] for r in benchmark['account']['daily']]
                st.line_chart(daily.set_index('date')[['nav','0050']].rename(columns={'nav':'策略'}))
                cols=['date','signal_date','side','qty','reference_price','total_cost','cash_after']
                st.dataframe(pd.DataFrame(account['trades'])[cols].rename(columns=dict(zip(cols,['成交日','訊號日','買賣','股數','參考價','費稅滑價','成交後現金']))),hide_index=True,use_container_width=True)
                st.download_button('下載連續完整帳戶與判斷',json.dumps(case,ensure_ascii=False),file_name=name+'-2016-2026.json',mime='application/json',key='index_continuous_download')
            st.download_button('下載連續資金驗證摘要',json.dumps(value,ensure_ascii=False),file_name='index-continuous.json',mime='application/json',key='index_continuous_summary')
        except (OSError,ValueError,KeyError,TypeError) as exc:st.error('連續資金驗證讀取失敗：'+str(exc))
