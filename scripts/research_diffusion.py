#!/usr/bin/env python
"""Frozen monthly price groups and causal leader/follower portfolio comparisons."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import duckdb
import numpy as np
import pandas as pd
import scipy

from app.file_lock import file_lock

CACHE = ROOT/'.cache/diffusion-research'
SPEC = ROOT/'docs/prereg_diffusion_20260910.md'
CODE = ('scripts/research_diffusion.py','skills/diffusion_signals.py','skills/diffusion_portfolio.py')
RULES = {'leader_now':'領先出現就買', 'leader_after':'擴散後買原領先股',
         'follower_after':'擴散後買接力股', 'basket_after':'擴散後同群整籃持有'}
START, END = '2022-01-03', '2026-06-23'
INPUT_FILES = {'close-official.parquet','close-snapshot.parquet','trade-flags.parquet',
               'volume.parquet','turnover.parquet','companies.parquet'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,default=str,indent=2))
    temp.replace(path)


def save_matrix(path, frame):
    tmp = path.with_suffix('.tmp')
    frame.rename_axis('date').reset_index().to_parquet(tmp,index=False)
    tmp.replace(path)


def read_matrix(name):
    frame=pd.read_parquet(CACHE/name).set_index('date')
    frame.index=pd.to_datetime(frame.index)
    return frame


def code_hashes():
    return {name:sha(ROOT/name) for name in CODE}


def input_transform_hash():
    source=inspect.getsource(prepare_inputs)+inspect.getsource(save_matrix)
    return hashlib.sha256(source.encode()).hexdigest()


def prepare_inputs():
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'inputs.json').exists():
        verify_inputs()
        print('重用凍結價量，不下載或重查DB。',flush=True)
        return
    started=time.perf_counter()
    source=ROOT/'.cache/event-group-research'
    meta=json.loads((source/'signal-inputs.json').read_text())
    original={}
    for name in ('close-official.parquet','close-snapshot.parquet','trade-flags.parquet'):
        digest=sha(source/name)
        if digest!=meta['files_sha256'][name]:
            raise ValueError('Frozen price source changed: '+name)
        original[str((source/name).relative_to(ROOT))]=digest
        shutil.copyfile(source/name,CACHE/name)
    official=read_matrix('close-official.parquet')
    flow=ROOT/'.cache/growth-flow-research'
    flow_meta=json.loads((flow/'inputs.json').read_text())
    for name in ('quotes.parquet','companies.parquet'):
        digest=sha(flow/name)
        if digest!=flow_meta['files_sha256'][name]:
            raise ValueError('Frozen flow source changed: '+name)
        original[str((flow/name).relative_to(ROOT))]=digest
    companies=pd.read_parquet(flow/'companies.parquet')
    companies.to_parquet(CACHE/'companies.parquet',index=False)
    dates=companies.set_index('stock_id').listed_date.reindex(official.columns)
    if dates.drop(index='0050').isna().any():
        raise ValueError('Unknown listing dates')
    dates.loc['0050']=pd.Timestamp.min
    listed=pd.DataFrame(official.index.to_numpy()[:,None]>=dates.to_numpy()[None,:],index=official.index,columns=official.columns)
    with duckdb.connect() as con:
        raw=con.execute("SELECT stock_id,trading_date,raw_close,raw_volume FROM read_parquet(?) WHERE trading_date>=DATE '2021-01-01' ORDER BY trading_date,stock_id",[str(flow/'quotes.parquet')]).df()
    if raw.duplicated(['stock_id','trading_date']).any():
        raise ValueError('Duplicate frozen quotes')
    volume=raw.pivot(index='trading_date',columns='stock_id',values='raw_volume').reindex(index=official.index,columns=official.columns)
    raw_close=raw.pivot(index='trading_date',columns='stock_id',values='raw_close').reindex_like(volume)
    prelisting=int((raw_close.notna() & ~listed).sum().sum())
    valid=listed & raw_close.gt(0) & volume.gt(0) & np.isfinite(raw_close) & np.isfinite(volume)
    save_matrix(CACHE/'volume.parquet',volume.where(valid))
    save_matrix(CACHE/'turnover.parquet',(raw_close*volume).where(valid))
    flags=read_matrix('trade-flags.parquet').fillna(False)&listed&valid
    save_matrix(CACHE/'trade-flags.parquet',flags)
    snapshots=read_matrix('close-snapshot.parquet')
    if not official.index.equals(snapshots.index) or not official.columns.equals(snapshots.columns):
        raise ValueError('Price bases are not aligned')
    if (official.notna()&~listed).any().any() or (snapshots.notna()&~listed).any().any():
        raise ValueError('Parent adjusted prices include pre-listing quotes')
    observed=pd.concat([official['0050'],snapshots['0050']],axis=1,keys=['official','snapshot']).dropna()
    diff=((observed.official/observed.official.shift(1))/(observed.snapshot/observed.snapshot.shift(1))-1).abs().dropna()
    if diff.gt(.005).any():
        raise ValueError('Reconcile 0050 observed returns before testing stock selection')
    write_json(CACHE/'inputs.json',{'schema':1,'prepared_at':datetime.now(timezone.utc).isoformat(),
        'input_transform_sha256':input_transform_hash(),
        'transform_versions':{'numpy':np.__version__,'pandas':pd.__version__,'duckdb':duckdb.__version__},
        'files_sha256':{name:sha(CACHE/name) for name in sorted(INPUT_FILES)},'parent_files_sha256':original,
        'parent_price_manifest_sha256':sha(source/'signal-inputs.json'),
        'benchmark_split':meta['action_check']['benchmark_split'],
        'price_rows':len(official),'assets':len(official.columns),'first_date':str(official.index[0].date()),
        'last_date':str(official.index[-1].date()),'prelisting_raw_cells_masked':prelisting,
        'benchmark_max_observed_return_difference_bp':float(diff.max()*10000),
        'universe':'current_official_ordinary_stock_cohort_with_past_listing_mask',
        'turnover_definition':'raw close times volume estimate, not actual trading money or net inflow',
        'elapsed_seconds':round(time.perf_counter()-started,3),'finmind_requests':0})
    print('價量已凍結；上市前遮蔽格數 '+str(prelisting),flush=True)


def verify_inputs():
    info=json.loads((CACHE/'inputs.json').read_text())
    if info['schema']!=1 or set(info['files_sha256'])!=INPUT_FILES:
        raise ValueError('Incomplete input manifest; prepare inputs explicitly')
    if (info.get('input_transform_sha256')!=input_transform_hash()
            or info.get('transform_versions')!={'numpy':np.__version__,'pandas':pd.__version__,'duckdb':duckdb.__version__}):
        raise ValueError('Input transformation changed; archive this task cache inputs.json and explicitly rebuild --prepare-inputs')
    for name,digest in info['files_sha256'].items():
        if sha(CACHE/name)!=digest:
            raise ValueError('Frozen input changed: '+name)
    return info


def prepare_signals():
    from skills.diffusion_signals import build_diffusion
    started=time.perf_counter()
    verify_inputs()
    provenance=code_hashes(); spec_sha=sha(SPEC)
    prices={basis:read_matrix(f'close-{basis}.parquet') for basis in ('official','snapshot')}
    volume=read_matrix('volume.parquet'); turnover=read_matrix('turnover.parquet')
    companies=pd.read_parquet(CACHE/'companies.parquet')
    files={};stats={}
    for basis,close in prices.items():
        result=build_diffusion(close,prices['snapshot' if basis=='official' else 'official'],volume,turnover,companies,start=START,signal_end='2025-12-31')
        if set(result['entries'])!=set(RULES):
            raise ValueError('Missing fixed signal arm')
        path=CACHE/f'signals-{basis}.json'
        write_json(path,result);files[path.name]=sha(path);stats[basis]=result['stats']
        print(basis, json.dumps(result['stats'],ensure_ascii=False,default=str),flush=True)
    if provenance!=code_hashes() or spec_sha!=sha(SPEC):
        raise ValueError('Protocol/code changed while constructing signals')
    verify_inputs()
    write_json(CACHE/'signals.json',{'schema':1,'code_sha256':provenance,'preregistration_sha256':spec_sha,
        'input_manifest_sha256':sha(CACHE/'inputs.json'),'files_sha256':files,'stats':stats,
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
        'elapsed_seconds':round(time.perf_counter()-started,3),'finmind_requests':0,'predictive_model_training_runs':0})


def verify_signals():
    info=json.loads((CACHE/'signals.json').read_text())
    if (info['schema']!=1 or info['code_sha256']!=code_hashes() or info['preregistration_sha256']!=sha(SPEC)
            or info['input_manifest_sha256']!=sha(CACHE/'inputs.json')
            or set(info['files_sha256'])!={'signals-official.json','signals-snapshot.json'}):
        raise ValueError('Signal provenance changed; run --prepare-signals explicitly')
    if info['versions']!={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__}:
        raise ValueError('Runtime versions changed; rebuild signals explicitly')
    for name,digest in info['files_sha256'].items():
        if sha(CACHE/name)!=digest:
            raise ValueError('Frozen signal changed: '+name)
    verify_inputs()
    return info


def annual_returns(curve):
    previous=1.; by_year={}
    for row in curve:
        year=row['date'][:4]
        by_year[year]=by_year.get(year,1.)*row['nav']/previous
        previous=row['nav']
    return {year:value-1 for year,value in by_year.items()}


def price_anomalies(close,other):
    # For valuation only: compare a resumed quote with the last observed mark.
    # Signal formation never fills missing daily returns this way.
    changes=close/close.ffill().shift(1)-1
    paired=((1+changes)/(other/other.ffill().shift(1))-1).abs()
    i,j=np.where((changes.abs().gt(.2)|paired.gt(.005)).to_numpy())
    lookup={}
    for day,column in zip(i,j):
        move=changes.iat[day,column];difference=paired.iat[day,column]
        lookup.setdefault(str(close.index[day].date()),[]).append({'stock_id':close.columns[column],
            'observed_return_since_last_quote':float(move) if pd.notna(move) else None,
            'price_basis_difference':float(difference) if pd.notna(difference) else None})
    return lookup


def valuation_audit(sim,anomalies):
    """Keep suspect outcomes; never erase a losing cohort after seeing its prices."""
    fills={}
    for fill in sim['executions']:
        fills.setdefault(fill['date'],[]).append(fill)
    units={}; findings=[]
    for row in sim['curve']:
        exposed={sid for sid,u in units.items() if u>1e-10}
        exposed.update(f['stock_id'] for f in fills.get(row['date'],[]) if f['side']=='buy')
        for item in anomalies.get(row['date'],[]):
            if item['stock_id'] in exposed:
                findings.append({'date':row['date'],**item})
        for fill in fills.get(row['date'],[]):
            sid=fill['stock_id']; units[sid]=units.get(sid,0)+(1 if fill['side']=='buy' else -1)*fill['units']
    return {'unresolved_valuation_days':len({row['date'] for row in findings}),
            'finding_count':len(findings),'findings':findings}


def run():
    from skills.diffusion_portfolio import simulate_baskets
    started=time.perf_counter(); info=verify_signals()
    signal={b:json.loads((CACHE/f'signals-{b}.json').read_text()) for b in ('official','snapshot')}
    prices={b:read_matrix(f'close-{b}.parquet') for b in ('official','snapshot')}
    anomalies={b:price_anomalies(c,prices['snapshot' if b=='official' else 'official']) for b,c in prices.items()}
    flags=read_matrix('trade-flags.parquet')
    results=[]; baselines=[]; charts={}
    for basis,close in prices.items():
        for scenario,slippage in (('base',.003),('stress',.0045)):
            bm=simulate_baskets(close,flags,[],start=START,end=END,mode='benchmark',slippage=slippage)
            bm['summary']['annual_returns']=annual_returns(bm['curve'])
            baselines.append({'basis':basis,'scenario':scenario,**bm})
            settings=[(basis,0)]
            if (basis,scenario)==('official','stress'): settings.append((basis,1))
            if (basis,scenario)==('snapshot','stress'): settings.append(('official',0))
            for signal_basis,delay in settings:
                for rule in RULES:
                    entries=[]
                    for event in signal[signal_basis]['entries'][rule]:
                        ix=int(close.index.searchsorted(event['entry_date']))+delay
                        if ix<len(close.index):
                            entries.append({**event,'entry_date':str(close.index[ix].date())})
                    sim=simulate_baskets(close,flags,entries,start=START,end=END,slippage=slippage)
                    sim['summary']['annual_returns']=annual_returns(sim['curve'])
                    audit=valuation_audit(sim,anomalies[basis])
                    row={'rule':rule,'name':RULES[rule],'basis':basis,'signal_basis':signal_basis,'scenario':scenario,'delay':delay,
                         'signal_count':len(entries),'excess_vs_0050':sim['summary']['total_return']-bm['summary']['total_return'],
                         'valuation_audit':audit,**sim}
                    results.append(row)
                    print(rule,basis,signal_basis,scenario,delay,'完成',flush=True)
                    if (basis,signal_basis,scenario,delay)==('official','official','stress',0):
                        charts[rule]=[{'date':r['date'],'nav':r['nav']} for r in sim['curve']]
            if (basis,scenario)==('official','stress'):
                charts['0050']=[{'date':r['date'],'nav':r['nav']} for r in bm['curve']]
    for row in results:
        key=tuple(row[k] for k in ('basis','signal_basis','scenario','delay'))
        basket=next(r for r in results if r['rule']=='basket_after' and tuple(r[k] for k in ('basis','signal_basis','scenario','delay'))==key)
        row['basket_comparison_paired']=row['rule']!='leader_now'
        row['excess_vs_basket']=(row['summary']['total_return']-basket['summary']['total_return']
                                 if row['basket_comparison_paired'] else None)
    verify_signals()
    limits=[
        '當前普通股名冊有存活者偏差，動態分群只解決使用今天產業分類回推的問題。',
        '價量相近不代表有共同訂單或已確認題材；本輪不分析新聞、營收或法說。',
        '歷史已被過往研究使用，不是未見測試；沒有任何一組自動取得實盤資格。',
        '成交占比使用收盤×成交量估值，分母是名冊內當時已上市股票，並非淨流入或完整歷史全市場。',
        '合成還原單位不含最低手續費、零股簿深及完整處置／漲跌停模型。',
        '兩版本一致不保證都正確；持有期間的可疑還原價保留並標記未解決估值，不能當已核實績效。',
        '確認後兩種選股與同群整籃採相同預定事件，但重疊／成交受阻造成實際持倉不同；領先出現就買包含未擴散事件，不能當成同群配對比較。',
        '本輪只完成族群先後擴散；月營收預期差、財務品質、融資修復尚未執行。']
    report={'schema':1,'experiment':'diffusion_20260910','research_only':True,'live_qualified':False,'valid_strategy_evidence':False,
        'created_at':datetime.now(timezone.utc).isoformat(),'start':START,'end':END,'signal_end':'2025-12-31',
        'code_sha256':info['code_sha256'],'preregistration_sha256':info['preregistration_sha256'],
        'signal_manifest_sha256':sha(CACHE/'signals.json'),'input_manifest_sha256':sha(CACHE/'inputs.json'),
        'inputs':verify_inputs(),'signal_info':info,'events':{b:s['events'] for b,s in signal.items()},
        'groups':{b:s['groups'] for b,s in signal.items()},'results':results,'baselines':baselines,'charts':charts,
        'elapsed_seconds':round(time.perf_counter()-started,3),'finmind_requests':0,'predictive_model_training_runs':0,'limitations':limits}
    write_json(CACHE/'report.full.json',report)
    # The full rejection identity lists remain in the sealed signal artifacts.
    report['groups']={b:[{**{k:v for k,v in g.items() if k not in ('exclusions','selected_ids','discarded_clusters')},
        'exclusion_counts':{k:len(v) for k,v in g.get('exclusions',{}).items()},
        'selected_count':len(g.get('selected_ids',[])), 'discarded_cluster_count':len(g.get('discarded_clusters',[]))}
        for g in rows] for b,rows in report['groups'].items()}
    report['results']=[{k:v for k,v in r.items() if k not in ('curve','executions')} for r in results]
    report['baselines']=[{k:v for k,v in r.items() if k not in ('curve','executions')} for r in baselines]
    write_json(CACHE/'report.summary.json',report)
    write_json(ROOT/'docs/research_diffusion_20260910.json',report)
    lines=['# 領先股之後的擴散：研究結果','','固定規格見 prereg_diffusion_20260910.md；本輪只完成第一方向。',
        '',f'共同期間 {START}～{END}。新領先訊號截止2025-12-31，之後只完成有限確認及退出；閒置持有0050。',
        '', '## 官方參考價格、壓力滑價每邊0.45%（另扣稅費）','',
        '| 方法 | 累積淨報酬 | 年化 | 最大回撤 | 相對0050（百分點） | 相對同群（百分點） | 期末清算 | 可疑持有估值筆數 |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in results:
        if (r['basis'],r['signal_basis'],r['scenario'],r['delay'])==('official','official','stress',0):
            s=r['summary'];excess=f"{r['excess_vs_basket']*100:.2f}" if r['excess_vs_basket'] is not None else '不適用'
            liquidated='完成' if s['final_liquidation_complete'] else '含未平倉估值'
            lines.append(f"| {r['name']} | {s['total_return']:.2%} | {s['cagr']:.2%} | {s['max_drawdown']:.2%} | {r['excess_vs_0050']*100:.2f} | {excess} | {liquidated} | {r['valuation_audit']['finding_count']} |")
    bm=next(r for r in baselines if (r['basis'],r['scenario'])==('official','stress'))['summary']
    liquidated='完成' if bm['final_liquidation_complete'] else '含未平倉估值'
    lines.append(f"| 0050持有 | {bm['total_return']:.2%} | {bm['cagr']:.2%} | {bm['max_drawdown']:.2%} | — | — | {liquidated} | — |")
    lines+=['', '以上為待核實的歷史試算；有可疑持有估值的列，不能當成已驗證收益。']
    lines+=['','## 限制','',*['- '+x for x in limits], '', '## 效能與重現','',
        f"價量準備 {report['inputs']['elapsed_seconds']} 秒；兩套分群與訊號 {info['elapsed_seconds']} 秒；24組投組比較含品質稽核 {report['elapsed_seconds']} 秒。本輪研究0次FinMind請求、0次預測模型訓練。",'',
        '`make prepare-diffusion` 建立或重用價量後生成訊號；`make research-diffusion` 離線重現24組。頁面切換只讀已存結果，不執行回測。', '',
        '完整收益、拒絕事件與來源雜湊見同名JSON；所有逐日成交與淨值保存在 .cache/diffusion-research/report.full.json。']
    (ROOT/'docs/research_diffusion_20260910.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'comparisons':len(results),'seconds':report['elapsed_seconds'],'stats':info['stats']},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-inputs',action='store_true')
    parser.add_argument('--prepare-signals',action='store_true')
    args=parser.parse_args();CACHE.mkdir(parents=True,exist_ok=True)
    with file_lock(CACHE/'research.lock',timeout=0):
        if args.prepare_inputs: prepare_inputs()
        elif args.prepare_signals: prepare_signals()
        else: run()
