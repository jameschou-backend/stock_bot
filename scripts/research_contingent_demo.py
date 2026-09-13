#!/usr/bin/env python3
"""Freeze a preregistered one-day board example and an explicit odd-data block."""
from pathlib import Path
from dataclasses import asdict
from decimal import Decimal
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_five_axis as parent
from scripts.prepare_contingent_ticks import OUTPUT as SOURCES
from scripts.research_exit_scenarios import read,write,sha,encoded
from scripts.replay_contingent_day import run as replay,CODE as REPLAY_CODE
from skills.contingent_replay import cumulative_bill
from skills.contingent_execution import Plan
from app.contingent_execution_ui import load_report

OUTPUT=ROOT/'.cache/contingent-replay-20260914-verified'
TARGET=ROOT/'artifacts/forward_simulation/contingent_replay_delivery_20260914.json'
CODE=REPLAY_CODE+['scripts/prepare_contingent_ticks.py','scripts/research_contingent_demo.py']


def run(verify=False):
    started=time.monotonic()
    old=load_report();source=read(SOURCES/'summary.json')
    if not source['all_completed'] or len(source['rows'])!=43:
        raise ValueError('Complete 43-stock-day board inventory required')
    inputs={}
    for row in source['rows']:
        path=ROOT/row['path'];meta=path.with_suffix('.json')
        if sha(path)!=row['sha256'] or sha(meta)!=row['metadata_sha256']:
            raise ValueError('Prepared board evidence changed')
        inputs[row['path']]=sha(path);inputs[str(meta.relative_to(ROOT))]=sha(meta)
    context=parent.identity(parent.OUTPUT)
    _,data,_=parent.load_data(parent.OUTPUT)
    day='2022-01-10';decision='2022-01-07'
    prices={}
    for sid in ('8261','2884'):
        row=data.quotes.loc[(data.quotes.stock_id==sid)&(data.quotes.date.astype(str)==decision)]
        if len(row)!=1:raise ValueError('Missing unique prior close')
        amount=Decimal(str(row.iloc[0]['close']))*100
        if amount!=amount.to_integral_value():raise ValueError('Unrepresentable prior price')
        prices[sid]=int(amount)
    dep=old['cases']['capacity_control_original']['rows'][0]
    if dep['date']!=day:raise ValueError('Prerecorded example date changed')
    adv={o['stock_id']:int(o['prior_avg_volume20']) for o in dep['orders'] if o['channel']=='board'}
    gross=6000*prices['2884'];budget=gross+cumulative_bill(gross,'buy','2884',45)['total']
    spec=dict(date=day,decision_date=decision,prior_data_date=decision,
        calendar=[str(d.date()) for d in data.days if decision<=str(d.date())<=day],
        holdings={'8261':2000},available_cents=1260,slots=1,slippage_bps=45,
        prior_avg_volume_shares=adv,plans=[asdict(Plan('sell-8261','8261','sell','board',2000,prices['8261'],0,decision)),
        asdict(Plan('buy-2884','2884','buy','board',6000,prices['2884'],budget,decision))])
    sources=[dict(format='finmind_board',path=str(ROOT/r['path']),sha256=r['sha256'],metadata_sha256=r['metadata_sha256'],
        stock_id=r['stock_id'],date=r['date'],market=r['market']) for r in source['rows']
        if r['date']==day and r['stock_id'] in ('8261','2884')]
    cases={}
    for key,delay in [('no_reuse',None),('delay_0',0),('delay_1s',1_000_000),('missing_odd',0)]:
        case=dict(spec,credit_delay_us=delay)
        if key=='missing_odd':
            case=dict(case,holdings={'8261':2693},plans=[spec['plans'][0],
                asdict(Plan('sell-8261-odd','8261','sell','odd',693,prices['8261'],0,decision)),spec['plans'][1]])
        plan_path=OUTPUT/'plans'/f'{key}.json';out=OUTPUT/'cases'/f'{key}.json'
        document=dict(spec=case,sources=sources,prior_data_identity=context,
            notice='事後選日的單日執行示範；不等於原零股策略或完整歷史績效。')
        if plan_path.exists() and encoded(read(plan_path))!=encoded(document):
            raise ValueError('Frozen single-day plan changed')
        if verify and not plan_path.exists():raise ValueError('Missing frozen plan')
        if not verify:write(plan_path,document)
        result=replay(plan_path,out,verify)
        cases[key]=dict(path=str(out.relative_to(ROOT)),sha256=sha(out),plan_path=str(plan_path.relative_to(ROOT)),
            plan_sha256=sha(plan_path),completed=result['completed'],
            fills=[{k:f[k] for k in ('side','stock_id','qty','at','sent_at','price_cents')} for f in result['fills']],
            missing=result.get('missing',[]),unfilled=result.get('unfilled',[]))
    if not all(cases[k]['completed'] for k in ('no_reuse','delay_0','delay_1s')) or cases['missing_odd']['completed']:
        raise ValueError('Unexpected demonstration completeness')
    report=dict(scope='contingent_adapter_demonstration',historical_replay_completed=False,total_return=None,
        live_qualified=False,board_stock_days=43,finmind_preparation_requests=source['requests_lifetime'],
        missing_historical_odd=True,cases=cases,code_sha256={p:sha(ROOT/p) for p in CODE},inputs_sha256=inputs,
        prior_data_identity=context,network_calls=0,
        warning='單日整張串接示範；回用延遲是研究假設，尚非原零股策略的完整歷史回測。')
    if verify:
        if encoded(read(TARGET))!=encoded(report):raise ValueError('Published demo differs')
        if TARGET.with_suffix('.sha256').read_text().strip()!=sha(TARGET):raise ValueError('Demo digest changed')
    else:
        if TARGET.exists() and read(TARGET)!=report:raise ValueError('Publication is immutable')
        write(TARGET,report);TARGET.with_suffix('.sha256').write_text(sha(TARGET)+'\n')
    print(dict(verified=verify,seconds=round(time.monotonic()-started,3),network_calls=0,
        fills={k:len(c['fills']) for k,c in cases.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    run(parser.parse_args().verify)
