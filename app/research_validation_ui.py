"""Fast display of frozen research summaries bound to their reproduction proof."""
import json
import math
from pathlib import Path

from app.backtest_tool_ui import verified_bytes

ROOT=Path(__file__).resolve().parents[1]
FAMILIES={
    '出場機制':('exit_mechanisms',{'loss12':'原12%停損','fixed63':'固定63日',
        'trail20_12':'漲20%後移動停利','weak20':'個股趨勢轉弱','market_weak':'大盤與個股轉弱',
        'trend126':'強勢延長持有','adaptive':'綜合出場'}),
    '部位配置':('volatility_budget',{'equal':'原等額配置','vol30':'波動上限30%',
        'vol40':'波動上限40%','vol50':'波動上限50%'}),
    '候補有效期':('candidate_queue',{'valid1':'原當日有效','valid2':'多候補一天'}),
    '支撐與風險配置':('support_risk',{'control':'原12%停損＋等額配置',
        'support20':'加上20日支撐出場','risk2':'每筆計畫風險2%',
        'support_risk2':'支撐出場＋計畫風險2%'}),
    '收斂突破篩選':('pattern_cash',{'control':'原12%停損＋等額配置',
        'pattern':'原規則＋收斂突破','support_risk2':'支撐出場＋計畫風險2%',
        'support_risk2_pattern':'支撐與配置＋收斂突破'}),
}


def load_summary(path, family, arms, root=ROOT):
    root=Path(root).resolve();path=Path(path)
    descriptor=dict(path=str(path.relative_to(root)),sha256=path.with_suffix('.sha256').read_text().strip())
    value=json.loads(verified_bytes(descriptor,root,'.json'))
    if (value.get('schema')!=family+'_publication_v1' or value.get('live_qualified') is not False
            or value.get('unseen_validation') is not False or value.get('adopted') is not False
            or (value.get('start'),value.get('end'),value.get('initial_cash'))!=('2022-01-03','2026-09-09',1_000_000)):
        raise ValueError('研究範圍或資格標記不符')
    expected={f'{arm}_{m}' for arm in arms for m in range(8)}
    if set(value['cases'])!=expected:raise ValueError('缺少已登記的候選或壓力情境')
    proof=json.loads(verified_bytes(value['offline_verification'],root,'.json'))
    if (proof.get('passed') is not True or proof.get('compared_cases')!=len(expected)
            or proof.get('all_completed')!=value['all_completed']
            or len(proof.get('runs',[]))!=2 or proof['runs'][0]['path']==proof['runs'][1]['path']
            or value['run_manifest'] not in proof['runs']):
        raise ValueError('未連結兩輪完整案例比對')
    for run in proof['runs']:verified_bytes(run,root,'.json')
    manifest=json.loads(verified_bytes(value['run_manifest'],root,'.json'))
    folder=Path(value['run_manifest']['path']).parent
    frozen=json.loads(verified_bytes(dict(path=str(folder/'report.json'),
        sha256=manifest['files_sha256']['report.json']),root,'.json'))
    if value['cases']!=frozen['cases'] or value['all_completed']!=frozen['all_completed']:
        raise ValueError('顯示結果與重播報告不一致')
    if value.get('settlement_supplement') and any(value.get(k)!=frozen.get(k) for k in (
            'settlement_supplement','prior_completed_unchanged','repaired_cases')):
        raise ValueError('補件範圍或原帳戶一致性與重播報告不符')
    if value['all_completed']!=all(r['completed'] is True for r in value['cases'].values()):
        raise ValueError('完成狀態不一致')
    baseline=next(iter(arms))
    if any(value['cases'][f'{baseline}_{m}']['completed'] is not True for m in range(8)):
        raise ValueError('缺少完整對照帳戶')
    for name,row in value['cases'].items():
        case_path=Path(row['result']['path'])
        if (not case_path.is_relative_to(folder)
                or manifest['files_sha256'].get(str(case_path.relative_to(folder)))!=row['result']['sha256']):
            raise ValueError('結果未連結封存帳戶')
        if row['completed']:
            for metric in (row['summary']['total_return'],row['summary']['max_drawdown'],
                           row['metrics']['excess_return'],row['metrics']['benchmark_return']):
                if type(metric) not in (int,float) or not math.isfinite(metric):
                    raise ValueError('回測數字必須有限且有效')
            mask=int(name.rsplit('_',1)[1])
            if row['metrics']['benchmark_return']!=value['cases'][f'{baseline}_{mask}']['metrics']['benchmark_return']:
                raise ValueError('不同規則未使用同成本基準')
    return value


def comparison_rows(value,arms):
    rows=[]
    for arm,label in arms.items():
        cases=[value['cases'][f'{arm}_{m}'] for m in range(8)]
        available=[c for c in cases if c['completed']]
        pct=lambda c: f"{c['summary']['total_return']:.2%}" if c['completed'] else '資料阻擋'
        rows.append({'規則':label,'正常淨報酬':pct(cases[0]),'全部壓力淨報酬':pct(cases[7]),
            '跑贏0050的情境':f"{sum(c['metrics']['excess_return']>0 for c in available)}/8",
            '未完成情境':8-len(available),
            '已完成帳戶最差回撤':f"{min(c['summary']['max_drawdown'] for c in available):.2%}" if available else '—'})
    return rows


def load_uncertainty(path, research, root=ROOT):
    """Bind descriptive statistics to the exact displayed research manifest."""
    root=Path(root).resolve();path=Path(path)
    publication=json.loads(verified_bytes(dict(path=str(path.relative_to(root)),
        sha256=path.with_suffix('.sha256').read_text().strip()),root,'.json'))
    report=json.loads(verified_bytes(publication['report'],root,'.json'))
    if (publication.get('schema')!='account_statistics_publication_v1'
            or report.get('schema')!='account_statistics_v1'
            or any(report.get(k) is not False for k in ('live_qualified','unseen_validation',
                'selection_adjusted_statistics_verified','complete_trial_coverage_verified'))):
        raise ValueError('統計資格或版本標記不符')
    manifest=research['run_manifest']
    if report['source_sha256'].get(manifest['path'])!=manifest['sha256']:
        raise ValueError('統計與目前顯示的帳戶不是同一版本')
    study=report['studies'][str(Path(manifest['path']).parent)]
    if (set(study['cases'])!=set(research['cases']) or any(
            study['cases'][n]['available'] is not c['completed'] for n,c in research['cases'].items())):
        raise ValueError('統計案例不完整或完成狀態不符')
    for case in study['cases'].values():
        if not case['available']:continue
        b=case['bootstrap']
        if (case.get('scope')!='descriptive_current_account_not_selection_adjusted'
                or case.get('live_qualified') is not False or case['dsr'].get('available') is not False
                or any(type(b[k]) not in (float,int) or not math.isfinite(b[k])
                    for k in ('excess_sharpe_observed','excess_ci_low','excess_ci_high'))
                or b['excess_ci_low']>b['excess_ci_high']):
            raise ValueError('統計範圍或數值不符')
    return study


def uncertainty_rows(study,arms):
    rows=[]
    for arm,label in arms.items():
        for mask,scenario in ((0,'正常'),(7,'全部壓力')):
            c=study['cases'][f'{arm}_{mask}']
            if not c['available']:continue
            b=c['bootstrap']
            rows.append({'規則':label,'情境':scenario,
                '超額報酬／追蹤誤差':f"{b['excess_sharpe_observed']:.2f}",
                '95%區間':f"{b['excess_ci_low']:.2f} ～ {b['excess_ci_high']:.2f}"})
    return rows


def render():
    import pandas as pd
    import streamlit as st
    st.subheader('最新策略驗證')
    st.warning('尚未取得實戰資格。正常回測贏過0050，不代表延遲成交後仍有優勢。')
    st.caption('2022/01/03–2026/09/09｜100萬元複利｜5個部位｜整張成交｜閒錢現金｜已計交易成本')
    selected=st.selectbox('查看哪一組實驗',list(FAMILIES),key='research_validation_family')
    family,arms=FAMILIES[selected]
    version='completed_20260927' if family in ('exit_mechanisms','volatility_budget') else '20260927'
    path=ROOT/'artifacts/forward_simulation'/f'{family}_{version}.json'
    if not path.exists():
        st.info('這組實驗尚未發布兩輪比對完成的報告。');return
    try:value=load_summary(path,family,arms)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.error('報告驗證未通過：'+str(exc));return
    st.dataframe(pd.DataFrame(comparison_rows(value,arms)),hide_index=True,use_container_width=True)
    if family == 'support_risk':
        st.caption('初始支撐固定取原訊號日，持有後只上移；跌破後下一交易日提出賣出。計畫風險含來回費稅與本情境滑價，並受整張、現金與名額限制。')
        st.caption('2%是事前配置上限，不保證跳空或未成交後的損失也小於2%。原12%出場基準仍是入場日還原收盤，和事前配置的參考價不同。')
    if family == 'pattern_cash':
        st.caption('只在原訊號日同時突破前20日高點、前10日區間縮至再前10日的75%以內、量達前20日均量1.5倍時，才通過新增篩選。延後成交不重算訊號。')
        st.caption('對照帳戶核對後重用，新增規則各離線執行兩次。未通過篩選的股票不占買進名額；通過仍須符合現金與成交條件。')
    if value.get('settlement_supplement'):
        st.caption(f"配股資料補件後，本組{value['repaired_cases']}個中止案例已完成；"
                   f"原先{value['prior_completed_unchanged']}個完整帳戶逐欄相同，兩輪重播一致。")
    baseline=next(iter(arms))
    normal=value['cases'][f'{baseline}_0']['metrics']['benchmark_return']
    stress=value['cases'][f'{baseline}_7']['metrics']['benchmark_return']
    st.write(f'同成本0050：正常 **{normal:.2%}**；滑價加倍 **{stress:.2%}**。全部壓力包含滑價加倍、進場多晚一天、出場多晚一天。')
    st.caption('各情境是同一段歷史的壓力比較，不能把「幾組跑贏」當作未來勝率。回撤是從資產高點回落，與本金虧損不同。')
    blocked=[{'情境':n,'原因':c['reason']} for n,c in value['cases'].items() if not c['completed']]
    if blocked:
        with st.expander(f'{len(blocked)}個情境因資料不足停止，沒有補假設報酬'):
            st.dataframe(pd.DataFrame(blocked),hide_index=True,use_container_width=True)
    statistics_path=ROOT/'artifacts/forward_simulation/account_statistics_completed_20260927.json'
    if family == 'support_risk':
        statistics_path=ROOT/'artifacts/forward_simulation/account_statistics_support_20260927.json'
    if family == 'pattern_cash':
        statistics_path=ROOT/'artifacts/forward_simulation/account_statistics_pattern_cash_20260927.json'
    if family in ('exit_mechanisms','volatility_budget','support_risk','pattern_cash'):
        with st.expander('優勢有多不確定？查看月報酬統計'):
            try:
                study=load_uncertainty(statistics_path,value)
                st.dataframe(pd.DataFrame(uncertainty_rows(study,arms)),hide_index=True,use_container_width=True)
                st.caption('2022/02–2026/08，共55個月；首尾月份排除。以策略減0050的月報酬，除以差額波動並年化。區間跨過0，表示這段資料仍不足以排除沒有優勢。')
                st.caption('每6個月成組、配對重抽2,000次。這是既有帳戶的描述統計，未校正所有歷史試驗與挑選偏差；不是未來獲利機率，DSR尚不可確認。')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                st.error('統計報告驗證未通過：'+str(exc))
    from app.research_account_detail import render as render_account_detail
    render_account_detail(value, arms)
    st.download_button('下載完整結果摘要',json.dumps(value,ensure_ascii=False,indent=2),
        file_name=path.name,mime='application/json',key='download_validation_summary')
    st.caption('直接讀取封存且已重跑比對的報告，不會重新抓資料、重跑回測或下單。畫面不重新驗證全部歷史來源；新回測仍會完整核對來源。')
