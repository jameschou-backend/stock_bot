#!/usr/bin/env python
"""Run the fixed, preregistered non-ML comparison against frozen local data."""
from __future__ import annotations
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import duckdb
import numpy as np
import pandas as pd
from sqlalchemy import select
from app.db import get_session
from app.models import Stock
from app.file_lock import file_lock
from skills.rule_research import RULE_NAMES,scores_for,executable,simulate,metrics

INPUT_DIR=ROOT/'.cache/rule-research'


def prepare_inputs():
    """Explicit, idempotent repair of the three gaps recorded in the preregistration.

    All calls use the project's shared quota/cache. Never mutate the original
    snapshots or the production database. No automatic full-market backfill.
    """
    from app.config import load_config
    from app.finmind import fetch_dataset
    from skills.ingest_prices import _normalize_prices
    cfg=load_config()
    INPUT_DIR.mkdir(parents=True,exist_ok=True)
    requests=[('0050-raw.parquet',date(2016,2,15),date(2026,6,23),'0050'),
              ('raw-20260521.parquet',date(2026,5,21),date(2026,5,21),None),
              ('raw-20260522.parquet',date(2026,5,22),date(2026,5,22),None)]
    for name,start,end,sid in requests:
        path=INPUT_DIR/name
        if path.exists():
            print(f'已存在，跳過：{name}',flush=True)
            continue
        print(f'補研究快照：{name}（1 次查詢，重試另計）',flush=True)
        frame=_normalize_prices(fetch_dataset('TaiwanStockPrice',start,end,
            token=cfg.finmind_token,data_id=sid,requests_per_hour=cfg.finmind_requests_per_hour))
        if frame.empty or (sid is None and frame.stock_id.nunique()<1500):
            raise ValueError(f'{name} 行情不足，停止；已完成檔案保留')
        tmp=path.with_suffix('.tmp')
        frame.to_parquet(tmp,index=False)
        tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load_inputs():
    adjusted=ROOT/'artifacts/adj_prices/adj_prices_10y.parquet'
    raw=ROOT/'artifacts/cache/prices.parquet'
    benchmark_raw=ROOT/'.cache/rule-research/0050-raw.parquet'
    repairs=[INPUT_DIR/'raw-20260521.parquet',INPUT_DIR/'raw-20260522.parquet']
    paths=(adjusted,raw,benchmark_raw,*repairs)
    for path in paths:
        if not path.exists():
            raise ValueError(f'缺少指定研究快照：{path}；先執行 python scripts/research_rules.py --prepare-inputs；不自動替換資料源')
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    with get_session() as s:
        stocks=pd.read_sql(select(Stock.stock_id,Stock.name,Stock.market,Stock.security_type,
                                   Stock.is_listed,Stock.listed_date,Stock.delisted_date),s.get_bind())
    ordinary=stocks[(stocks.security_type=='stock') & stocks.market.isin(['TWSE','TPEX'])
                    & stocks.stock_id.str.fullmatch(r'[0-9]{4}')]
    allowed=ordinary[['stock_id']].copy()
    allowed=pd.concat([allowed,pd.DataFrame({'stock_id':['0050']})]).drop_duplicates()
    con=duckdb.connect()
    try:
        con.register('allowed',allowed)
        frame=con.execute('''SELECT a.stock_id,CAST(a.trading_date AS DATE) AS trading_date,
            a.close AS adj_close,r.close AS raw_close,r.volume AS raw_volume,
            r.high AS raw_high,r.low AS raw_low
            FROM read_parquet(?) a JOIN allowed u USING(stock_id)
            LEFT JOIN (SELECT * FROM read_parquet(?) WHERE stock_id!='0050'
                         AND trading_date NOT IN (DATE '2026-05-21',DATE '2026-05-22')
                       UNION ALL SELECT * FROM read_parquet(?)
                       UNION ALL SELECT * FROM read_parquet(?) WHERE stock_id!='0050') r
              ON a.stock_id=r.stock_id AND CAST(a.trading_date AS DATE)=r.trading_date
            WHERE a.trading_date >= '2016-02-15'
        ''',[str(adjusted),str(raw),str(benchmark_raw),[str(p) for p in repairs]]).df()
    finally: con.close()
    if frame.duplicated(['stock_id','trading_date']).any(): raise ValueError('快照有重複股票日期，停止研究')
    if frame.empty: raise ValueError('研究資料為空')
    frame['trading_date']=pd.to_datetime(frame.trading_date)
    frame.loc[frame.adj_close<=0,'adj_close']=np.nan
    fields={c:frame.pivot(index='trading_date',columns='stock_id',values=c).sort_index()
            for c in ('adj_close','raw_close','raw_volume','raw_high','raw_low')}
    close=fields['adj_close']
    liquid=fields['raw_close']*fields['raw_volume']
    flags=executable(fields['raw_close'],fields['raw_volume'],fields['raw_high'],fields['raw_low'])
    coverage=fields['raw_close'].notna().sum(axis=1)/close.notna().sum(axis=1)
    if (coverage<.9).any():
        raise ValueError('原始行情有整日覆蓋缺漏，不能把缺資料當成空手：'+
                         ', '.join(str(d.date()) for d in coverage.index[coverage<.9]))
    if '0050' not in close or close['0050'].dropna().empty: raise ValueError('缺少 0050 基準資料')
    source={'files_sha256':before,'last_date':str(close.index[-1].date()),'rows':len(frame),
            'stocks_including_benchmark':len(close.columns),
            'ordinary_metadata_not_currently_listed':int((~ordinary.is_listed).sum()),
            'metadata_missing_listing_date':int(ordinary.listed_date.isna().sum()),
            'minimum_daily_raw_coverage':float(coverage.min()),
            'stock_metadata_sha256':hashlib.sha256(stocks.sort_values('stock_id').to_json(orient='records',date_format='iso').encode()).hexdigest(),
            'large_adjusted_moves_over_50pct':int((close.pct_change(fill_method=None).abs()>.5).sum().sum()),
            'missing_raw_rows':int(frame.raw_close.isna().sum()),
            'read_seconds':None}
    # Ensure other writers did not change the input during preparation.
    if before!={str(p.relative_to(ROOT)):digest(p) for p in paths}:
        raise ValueError('研究期間快照已改變，結果不可比較')
    return close,liquid,flags,source


def main(output):
    started=time.perf_counter()
    print('[TIMER] load_prices start',flush=True)
    close,turnover,flags,source=load_inputs()
    source['read_seconds']=round(time.perf_counter()-started,3)
    print('[TIMER] load_prices done',flush=True)
    scores=scores_for(close,turnover)
    for frame in scores.values(): frame.loc[:,'0050']=np.nan
    segments={'2018_2022':('2018-01-01','2022-12-31'),
              '2023_2025':('2023-01-01','2025-12-31'),
              '2026_partial':('2026-01-01',source['last_date'])}
    results=[]
    for scenario,slip in [('base',.003),('stress',.0045)]:
        bm=simulate(close,flags,None,benchmark='0050',slippage=slip)
        for rule,values in scores.items():
            print(f'[TIMER] {rule}_{scenario} start',flush=True)
            run=simulate(close,flags,values,slippage=slip)
            comparison={name:{'strategy':metrics(run.curve,*dates),
                              'benchmark':metrics(bm.curve,*dates)} for name,dates in segments.items()}
            for stats in comparison.values():
                stats['excess_return']=stats['strategy']['total_return']-stats['benchmark']['total_return']
            results.append({'rule':rule,'name':RULE_NAMES[rule],'scenario':scenario,
                            'summary':run.summary,'benchmark_summary':bm.summary,'segments':comparison,
                            'equity_curve':run.curve.to_dict('records'),
                            'benchmark_curve':bm.curve.to_dict('records'),
                            'trades':run.trades,'decisions':run.decisions})
            print(f'[TIMER] {rule}_{scenario} done',flush=True)
    report={'schema':1,'research_only':True,'live_qualified':False,'offline':True,
            'preregistration':'docs/prereg_rules_20260909.md',
            'preregistration_sha256':digest(ROOT/'docs/prereg_rules_20260909.md'),
            'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
                           (ROOT/'skills/rule_research.py',ROOT/'scripts/research_rules.py')},
            'source':source,'elapsed_seconds':round(time.perf_counter()-started,3),
            'limitations':['還原價尚有對帳差異，不作實盤資格認證',
                '股票主檔未對齊歷史上市日期，可能混入當時興櫃及主檔分類為 stock 的存託憑證；有殘餘存活者偏差',
                '現有歷史區間已被過去研究使用，不是全新樣本外資料',
                '合成還原單位，不含最低手續費、交易單位及個人資金容量',
                '無法成交以原始價量近似；缺價持倉採最後可得價估值並另外統計'],
            'results':results}
    def clean(value):
        if isinstance(value,(pd.Timestamp,)): return value.isoformat()
        if isinstance(value,float) and not np.isfinite(value): return None
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list): return [clean(v) for v in value]
        return value
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.tmp')
    temp.write_text(json.dumps(clean(report),ensure_ascii=False,allow_nan=False))
    temp.replace(output)
    print(json.dumps({'report':str(output),'elapsed_seconds':report['elapsed_seconds'],
                     'source':source},ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/rule-research/report.json')
    parser.add_argument('--prepare-inputs',action='store_true',help='只補已記錄的研究行情缺口；最多 3 次初始查詢，重試另計')
    args=parser.parse_args()
    with file_lock(ROOT/'.cache/research-or-update.lock',timeout=0):
        if args.prepare_inputs: prepare_inputs()
        else: main(args.output)
