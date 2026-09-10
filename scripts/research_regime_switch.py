#!/usr/bin/env python
"""Fixed causal cash-switch and independently funded mixed-portfolio research."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy

from app.file_lock import file_lock
from scripts import research_diffusion as source
from skills.diffusion_portfolio import simulate_baskets
from skills.regime_state import build_trend, execution_controls, gate_events
from skills.regime_portfolio import simulate_regime
from skills.regime_mix import mix_portfolios

CACHE=ROOT/'.cache/regime-switch-research'
PROBE=ROOT/'.cache/regime-probe-20260910'
SPEC=ROOT/'docs/prereg_regime_switch_20260910.md'
CODE=('scripts/research_regime_switch.py','skills/regime_state.py','skills/regime_portfolio.py','skills/regime_mix.py')
RULES={'always':'所有領先訊號','entry_only':'趨勢只管進場',
       'idle_cash':'轉弱只收回閒置0050','exit_cash':'轉弱連個股一起退出',
       'mix_always':'初始一半領先策略、一半0050','mix_entry':'初始一半趨勢進場、一半0050'}
START,END=source.START,source.END


def code_hashes():
    return {name:source.sha(ROOT/name) for name in CODE}


def versions():
    return {'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}


def prior_probe():
    if not (PROBE/'report.json').exists():
        raise ValueError('Missing frozen prior probe: .cache/regime-probe-20260910/report.json; restore its original sealed artifacts before comparing.')
    report=json.loads((PROBE/'report.json').read_text())
    checks={'protocol.md':report['protocol_sha256'],'probe.py':report['script_sha256'],**report['state_files_sha256']}
    if (set(checks)!={'protocol.md','probe.py','states-official.parquet','states-snapshot.parquet'}
            or report['diffusion_signal_manifest_sha256']!=source.sha(source.CACHE/'signals.json')):
        raise ValueError('Prior probe does not refer to the same sealed inputs')
    for name,digest in checks.items():
        if source.sha(PROBE/name)!=digest:
            raise ValueError('Prior probe source changed: '+name)
    return report,{'.cache/regime-probe-20260910/'+name:source.sha(PROBE/name) for name in (*checks,'report.json')}


def prepare_states():
    CACHE.mkdir(parents=True,exist_ok=True)
    source.verify_signals()
    _,probe_hashes=prior_probe()
    started=time.perf_counter();code=code_hashes();spec=source.sha(SPEC)
    files={};stats={}
    for basis in ('official','snapshot'):
        close=source.read_matrix(f'close-{basis}.parquet')['0050']
        trend=build_trend(close)
        for length in (400,900):
            pd.testing.assert_frame_equal(build_trend(close.iloc[:length]),trend.iloc[:length])
        path=CACHE/f'states-{basis}.parquet'
        source.save_matrix(path,trend)
        files[path.name]=source.sha(path)
        sliced=trend.loc[START:END]
        known=sliced.state.isin(['ON','OFF'])
        changes=sliced.loc[known & sliced.state.ne(sliced.state.where(known).ffill().shift(1))]
        stats[basis]={'state_days':sliced.state.value_counts().to_dict(),
            'observed_transitions':[{'date':str(day.date()),'state':row.state,'close':row.close,'ma120':row.ma120}
                                    for day,row in changes.iterrows()]}
    if code!=code_hashes() or spec!=source.sha(SPEC):
        raise ValueError('Research implementation changed during preparation')
    source.write_json(CACHE/'states.json',{'schema':1,'code_sha256':code,'protocol_sha256':spec,
        'diffusion_signal_manifest_sha256':source.sha(source.CACHE/'signals.json'),
        'prior_probe_files_sha256':probe_hashes,'files_sha256':files,'stats':stats,'versions':versions(),
        'elapsed_seconds':round(time.perf_counter()-started,3),'prefix_invariance_passed':True,'finmind_requests':0})
    print('已凍結逐日狀態，沒有計算新方法報酬。',flush=True)


def verify_states():
    source.verify_signals()
    _,probe_hashes=prior_probe()
    manifest=json.loads((CACHE/'states.json').read_text())
    if (manifest['schema']!=1 or manifest['code_sha256']!=code_hashes()
            or manifest['protocol_sha256']!=source.sha(SPEC) or manifest['versions']!=versions()
            or manifest['diffusion_signal_manifest_sha256']!=source.sha(source.CACHE/'signals.json')
            or manifest['prior_probe_files_sha256']!=probe_hashes
            or set(manifest['files_sha256'])!={'states-official.parquet','states-snapshot.parquet'}):
        raise ValueError('State provenance changed; explicitly run make prepare-regime-switch')
    for name,digest in manifest['files_sha256'].items():
        if source.sha(CACHE/name)!=digest:
            raise ValueError('Frozen state changed: '+name)
    return manifest


def case_name(rule,basis,scenario,delay):
    return f'case-{rule}-{basis}-{scenario}-{delay}.json'


def delay_events(events,index,delay):
    if isinstance(delay,bool) or not isinstance(delay,(int,np.integer)) or delay not in (0,1):
        raise ValueError('Event delay must be exactly 0 or 1 extra market session')
    delayed=[]
    for event in events:
        entry=pd.Timestamp(event['entry_date'])
        if pd.isna(entry) or entry.tzinfo is not None or entry!=entry.normalize() or entry not in index:
            raise ValueError('Frozen event entry is not an exact market session')
        position=int(index.get_loc(entry))+delay
        if position>=len(index):
            raise ValueError('Delayed frozen event has no execution session')
        delayed.append({**event,'entry_date':str(index[position].date())})
    return delayed


def combine_audits(*audits):
    findings={(row['date'],row['stock_id']):row for audit in audits for row in audit['findings']}
    return {'finding_count':len(findings),'unresolved_valuation_days':len({k[0] for k in findings}),
            'findings':[findings[k] for k in sorted(findings)]}


def finish_case(sim,rule,basis,scenario,delay,audit,files,gate_rejections=None,accepted=None):
    sim['summary']['annual_returns']=source.annual_returns(sim['curve'])
    sim['summary'].setdefault('mean_cash_weight',float(np.mean([r['cash']/r['nav'] for r in sim['curve']])))
    payload={'rule':rule,'basis':basis,'scenario':scenario,'delay':delay,
             'gate_rejections':gate_rejections or [],'valuation_audit':audit,**sim}
    name=case_name(rule,basis,scenario,delay)
    source.write_json(CACHE/name,payload);files[name]=source.sha(CACHE/name)
    return {'rule':rule,'name':RULES.get(rule,'0050持有'),'basis':basis,'scenario':scenario,'delay':delay,
            'case_file':name,'summary':sim['summary'],'valuation_audit':audit,
            'state_admitted':accepted,'state_rejected_count':len(gate_rejections or [])}


def run():
    started=time.perf_counter();manifest=verify_states()
    old,_=prior_probe()
    signal={b:json.loads((source.CACHE/f'signals-{b}.json').read_text())['entries']['leader_now'] for b in ('official','snapshot')}
    price={b:source.read_matrix(f'close-{b}.parquet') for b in signal}
    flags=source.read_matrix('trade-flags.parquet')
    states={b:pd.read_parquet(CACHE/f'states-{b}.parquet').set_index('date') for b in signal}
    for frame in states.values():frame.index=pd.to_datetime(frame.index)
    results=[];baselines=[];files={};charts={}
    for basis,close in price.items():
        anomalies=source.price_anomalies(close,price['snapshot' if basis=='official' else 'official'])
        for scenario,slippage in (('base',.003),('stress',.0045)):
            bm=simulate_baskets(close,flags,[],start=START,end=END,slippage=slippage,mode='benchmark')
            bm_audit=source.valuation_audit(bm,anomalies)
            baselines.append(finish_case(bm,'benchmark',basis,scenario,0,bm_audit,files))
            for delay in ((0,1) if (basis,scenario)==('official','stress') else (0,)):
                admitted,rejected=gate_events(signal[basis],states[basis],extra_entry_delay=delay)
                control=execution_controls(states[basis],delay=1+delay)
                sims={};audits={}
                for rule in ('always','entry_only','idle_cash','exit_cash'):
                    entries=delay_events(signal[basis],close.index,delay) if rule=='always' else admitted
                    if rule in ('always','entry_only'):
                        sim=simulate_baskets(close,flags,entries,start=START,end=END,slippage=slippage)
                        sim['summary']['annual_returns']=source.annual_returns(sim['curve'])
                        reference=next(r for r in old['results'] if (r['rule'],r['basis'],r['scenario'],r['delay'])
                                       ==('always' if rule=='always' else 'trend',basis,scenario,delay))
                        if (sim['summary']!=reference['summary'] or sim['cohorts']!=reference['cohorts']
                                or sim['rejections']!=reference['execution_rejections']):
                            raise ValueError('Prior control did not reproduce: '+rule)
                    else:
                        sim=simulate_regime(close,flags,entries,control,policy=rule,start=START,end=END,slippage=slippage)
                    sims[rule]=sim;audits[rule]=source.valuation_audit(sim,anomalies)
                    row=finish_case(sim,rule,basis,scenario,delay,audits[rule],files,
                                    [] if rule=='always' else rejected,len(entries))
                    results.append(row)
                    print(rule,basis,scenario,delay,'完成',flush=True)
                for rule,component in (('mix_always','always'),('mix_entry','entry_only')):
                    sim=mix_portfolios(sims[component],bm,weight=.5)
                    sim['components'][0]['case_file']=case_name(component,basis,scenario,delay)
                    sim['components'][1]['case_file']=case_name('benchmark',basis,scenario,0)
                    sims[rule]=sim
                    audit=combine_audits(audits[component],bm_audit)
                    results.append(finish_case(sim,rule,basis,scenario,delay,audit,files))
                if (basis,scenario,delay)==('official','stress',0):
                    for rule,sim in {**sims,'benchmark':bm}.items():
                        charts[rule]=[{'date':r['date'],'nav':r['nav']} for r in sim['curve']]
    for row in results:
        key=(row['basis'],row['scenario'],row['delay'])
        bm=next(r for r in baselines if (r['basis'],r['scenario'])==key[:2])
        row['excess_vs_0050']=row['summary']['total_return']-bm['summary']['total_return']
        for comparison in ('entry_only','mix_entry'):
            other=next(r for r in results if r['rule']==comparison and (r['basis'],r['scenario'],r['delay'])==key)
            row['excess_vs_'+comparison]=row['summary']['total_return']-other['summary']['total_return']
    default=[r for r in results if (r['basis'],r['scenario'],r['delay'])==('official','stress',0)]
    default_bm=next(r for r in baselines if (r['basis'],r['scenario'])==('official','stress'))
    leave_one_year=[{'omitted_year':year,'diagnostic_only':True,
        'compounded_remaining_years':{r['rule']:math.prod(1+v for y,v in r['summary']['annual_returns'].items() if y!=year)-1
                                    for r in [*default,default_bm]}}
                    for year in ('2022','2023','2024','2025')]
    verify_states()
    names=pd.read_parquet(source.CACHE/'companies.parquet').set_index('stock_id')['name'].to_dict()
    report={'schema':1,'experiment':'regime_switch_20260910','research_only':True,'live_qualified':False,
        'valid_strategy_evidence':False,'start':START,'end':END,'signal_end':'2025-12-31',
        'created_at':datetime.now(timezone.utc).isoformat(),'code_sha256':code_hashes(),
        'protocol_sha256':source.sha(SPEC),'state_manifest_sha256':source.sha(CACHE/'states.json'),
        'states':manifest,'case_files_sha256':files,'results':results,'baselines':baselines,'charts':charts,
        'names':names,'leave_one_year_diagnostic':leave_one_year,
        'control_reproduction_passed':True,'elapsed_seconds':round(time.perf_counter()-started,3),
        'finmind_requests':0,'prediction_model_training_runs':0,
        'limitations':[
            '本輪沒有未見測試：同一歷史已使用，價格／成本／延遲只屬敏感度檢查。',
            '固定混合只在起初各分一半，兩份資金独立、不再平衡；份額會隨收益漂移。',
            '每日收盤後判斷，下一交易日才執行；轉弱不能保證立即成交或限制最大虧損。',
            'UNKNOWN不當多頭，股票已發出的退出不取消；閒置配置沿用上次已知指令，初始未知留現金。',
            '股價是合成還原單位；未建完整漲跌停、最低手續費或零股深度模型，現金利息為零。',
            '當前名冊有存活者及市場移轉日期偏差；可疑持有估值保留，不能當已核實收益。',
            '刪除單一年度的複利只是集中度診斷，不是可實現投組或樣本外測試。',
            '廣度方案仍未評價：官方觀測價沒有有效收盤的日期不能由舊快照前值補成新觀测。']}
    source.write_json(CACHE/'report.summary.json',report)
    source.write_json(ROOT/'docs/research_regime_switch_20260910.json',report)
    lines=['# 情境切換、現金與固定混合：研究結果','',
        f'共同期間{START}～{END}，訊號截止2025-12-31；各自閒置資金按固定規格處理。',
        '官方參考價格；每邊滑價0.45%，另扣每邊0.1425%手續費及賣出稅。均為歷史探索試算。','',
        '| 方法 | 累積試算淨報酬 | 年化 | 最大回撤 | 平均現金 | 期末清算 | 估值疑點 |',
        '|---|---:|---:|---:|---:|---|---:|']
    for row in [*default,default_bm]:
        s=row['summary'];flag='完成' if s['final_liquidation_complete'] else '含未平倉估值'
        lines.append(f"| {row['name']} | {s['total_return']:.2%} | {s['cagr']:.2%} | {s['max_drawdown']:.2%} | {s['mean_cash_weight']:.2%} | {flag} | {row['valuation_audit']['finding_count']} |")
    lines+=['','## 限制','',*['- '+x for x in report['limitations']],
        '',f"狀態準備{manifest['elapsed_seconds']}秒；30組方法＋4組基準含稽核{report['elapsed_seconds']}秒；FinMind0次。",
        '', '`make prepare-regime-switch` 凍結狀態，`make research-regime-switch` 重現比較。逐筆成交與拒絕紀錄見 .cache/regime-switch-research/case-*.json；summary只保留小型彙總，頁面不觸發回測。']
    (ROOT/'docs/research_regime_switch_20260910.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'comparisons':len(results),'seconds':report['elapsed_seconds'],'controls_reproduced':True}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-states',action='store_true')
    args=parser.parse_args();CACHE.mkdir(parents=True,exist_ok=True)
    with file_lock(CACHE/'research.lock',timeout=0):
        if args.prepare_states:prepare_states()
        else:run()
