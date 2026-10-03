#!/usr/bin/env python3
"""Render a completed, prefix-verified October POC account without replaying it."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import export_poc_signal_explorer as prior
from scripts.export_signal_explorer import digest,number,pack_prices,json_for_script,render_html
from scripts.research_poc_latest_account import load_candidate_bundle

BASE=ROOT/'.cache/poc-latest-20261003'
BUNDLE=BASE/'inputs-v1'
END='2026-10-02'
BOUNDARY='2026-09-09'
ARMS=('original','benchmark','poc_base','poc_red')


def report_entries(bundle,report,end=END):
    active,pending=load_candidate_bundle(bundle,end)
    if (report.get('pending_terminal_signals')!=pending
            or report.get('source_signal_count')!=len(active)+len(pending)):
        raise ValueError('Published report differs from sealed terminal signals or source count')
    return active+pending


def load_latest(report_path,root=ROOT):
    path=Path(report_path).resolve();path.relative_to(root/'.cache/poc-latest-20261003')
    report=prior.read(path)
    sidecar=path.with_suffix('.sha256')
    if not sidecar.is_file() or sidecar.read_text().strip()!=digest(path):
        raise ValueError('Latest report needs its sealed SHA sidecar')
    if (report.get('all_completed') is not True or set(report.get('cases',{}))!=set(ARMS)
            or report.get('end')!=END or report.get('start')!='2024-01-02'
            or report.get('input_bundle')!=str(BUNDLE.relative_to(ROOT))
            or report.get('live_qualified') is not False):
        raise ValueError('Require all four complete fixed-scope latest accounts')
    refs=dict(report['source_sha256']);refs[str(path.relative_to(root))]=digest(path)
    refs[str(sidecar.relative_to(root))]=digest(sidecar)
    profiles_meta=report['profile_data']
    if profiles_meta.get('schema')!='poc_latest_profiles_v1':raise ValueError('Unexpected profile snapshot schema')
    for arm in ARMS:
        item=report['cases'][arm]
        if (item.get('completed') is not True or not item.get('prefix_parity',{}).get('all_exact')
                or item['prefix_parity'].get('through')!=BOUNDARY):
            raise ValueError('Case lacks completed prefix equality: '+arm)
        prior.merge_refs(refs,{item['path']:item['sha256']})
        profile=profiles_meta['arms'][arm]
        prior.merge_refs(refs,{profile['path']:profile['sha256']})
    manifest_path=root/'.cache/poc-latest-20261003/inputs-v1/manifest.json'
    if str(manifest_path.relative_to(root)) not in refs:raise ValueError('Latest inputs are not bound by account report')
    manifest=prior.read(manifest_path)
    if manifest.get('events_extension_complete') is not True or manifest.get('end')!=END:
        raise ValueError('Incomplete latest input scope')
    prior.merge_refs(refs,{str((manifest_path.parent/name).relative_to(root)):h
                          for name,h in manifest['files_sha256'].items()})
    prior.verify_refs(refs,root)
    report_entries(manifest_path.parent,report)
    cases,profiles={},{}
    for arm in ARMS:
        item=report['cases'][arm];value=prior.read(root/item['path'])
        if (value.get('completed') is not True or value['summary']!=item['summary']
                or value.get('prefix_parity')!=item.get('prefix_parity')):
            raise ValueError('Latest case differs from its published parent')
        if value['summary']['end']!=END or value['summary']['start']!='2024-01-02':
            raise ValueError('Case account period changed')
        cases[arm]=value
        profiles[arm]=prior.read(root/profiles_meta['arms'][arm]['path'])
    return cases,profiles,refs


def terminal_signals(entries,adjusted,quotes,companies,end=END):
    """The final close is a signal, although its next market session is unknown."""
    rows=[r for r in entries if r.get('entry_date') is None]
    if any(r['signal_date']!=end for r in rows):raise ValueError('Unresolved non-terminal entry date')
    if len({r['event_id'] for r in rows})!=len(rows):raise ValueError('Duplicate terminal candidate')
    raw=quotes.loc[quotes.date.eq(pd.Timestamp(end))].set_index('stock_id')
    if not raw.index.is_unique:raise ValueError('Duplicate terminal raw quote')
    names=companies.set_index('stock_id');i=adjusted.index.get_loc(pd.Timestamp(end))
    result=[]
    for rank,row in enumerate(sorted(rows,key=lambda r:(-r['priority'],r['event_id'])),1):
        sid=row['members'][0];q=raw.loc[sid];opened,close=number(q['open']),number(q['close'])
        if opened is None or close is None or min(opened,close)<=0:raise ValueError('Unknown terminal candle')
        ev=row['leader_evidence'];own,bench=ev['leader_return20'],ev['benchmark_return20']
        if not math.isclose(own-bench,row['priority'],abs_tol=1e-12,rel_tol=0):raise ValueError('Terminal score differs')
        window=adjusted[sid].iloc[max(0,i-60):i]
        previous=number(window.max()) if len(window)==60 and window.notna().all() else None
        adj=number(adjusted.at[pd.Timestamp(end),sid])
        result.append(dict(signal_id=row['event_id'],stock_id=sid,name=names.at[sid,'name'],
            market=names.at[sid,'market'],signal_date=end,entry_date=None,daily_rank=rank,candidate_count=len(rows),
            priority=row['priority'],stock_return20=own,benchmark_return20=bench,volume_ratio=ev['leader_volume_ratio'],
            signal_open_raw=opened,signal_close_raw=close,signal_close_adjusted=adj,
            previous60_high_adjusted=previous,close_to_prior60_high_pct=adj/previous-1 if adj and previous else None,
            turnover_mean20=None,turnover_median20=None,trend_state=None,
            candle='red' if close>opened else 'black' if close<opened else 'doji',
            available_at=end+' 收盤資料完成後',source_scope='finmind_extension_unverified',pending_entry=True))
    return result


def pending_decision(signal,red_required):
    passed=signal['candle']=='red'
    return dict(signal_id=signal['signal_id'],red_gate_required=red_required,
        red_gate=passed if red_required else None,
        red_gate_status=signal['candle'] if red_required else 'not_required',
        poc_status='not_evaluated',poc_reason='pending_next_session',poc_reconstructed=False,
        selection_status='red_gate_rejected' if red_required and not passed else 'pending_next_session',
        selection_reason='no_observed_next_market_session',candidate_rank_after_red=None,
        fallback=False,fallback_reason=None,entry_date=None,selection_available_at=None,
        selected_for_planning=False,simulated_buy_qty=0,simulated_buy_dates=[],order_count=0,
        order_failures=[],pending_entry=True)


def build_payload(entries,cases,profiles,adjusted,quality,quotes,eligibility,companies,*,end=END):
    if set(cases)!=set(ARMS):raise ValueError('All four accounts required')
    active=[r for r in entries if r.get('entry_date') is not None]
    stock_arms={a:cases[a] for a in ('poc_red','poc_base','original')}
    payload=prior.build_payload(active,stock_arms,profiles,adjusted,quality,quotes,eligibility,companies,end=end)
    pending=terminal_signals(entries,adjusted,quotes,companies,end)
    for signal in pending:
        for arm,strategy in payload['strategies'].items():
            if any(t['event_id']==signal['signal_id'] for t in strategy['trades']):
                raise ValueError('Pending terminal signal already has a modeled fill')
            if any(q['event_id']==signal['signal_id'] for q in cases[arm]['profile_queries']):
                raise ValueError('Pending terminal candidate has a future POC planning query')
            strategy['decisions'][signal['signal_id']]=pending_decision(signal,arm=='poc_red')
        sid=signal['stock_id']
        if sid not in payload['stocks']:
            dates=adjusted.index[adjusted.index>=payload['metadata']['price_start']]
            prices,issues=pack_prices(dates,quotes.loc[quotes.stock_id.eq(sid)],adjusted[sid],quality[sid],eligibility[sid])
            payload['stocks'][sid]=dict(stock_id=sid,name=signal['name'],market=signal['market'],prices=prices,
                signal_ids=[],invalid_bar_count=sum(issues.values()),quality_issue_counts=issues)
        stock=payload['stocks'][sid];stock['signal_ids'].append(signal['signal_id'])
        if stock['prices'][-1][7]!=signal['signal_close_adjusted']:raise ValueError('Terminal candle marker differs')
    payload['signals'].extend(pending)
    terminal=payload['days'][-1]
    terminal.update(signal_ids=[s['signal_id'] for s in pending],signal_count=len(pending),
                    signal_status='pending_next_session' if pending else 'complete')
    for signal in payload['signals']:
        if signal['signal_date']>BOUNDARY:signal['source_scope']='finmind_extension_unverified'
    totals=Counter()
    for stock in payload['stocks'].values():totals.update(stock['quality_issue_counts'])
    meta=payload['metadata']
    meta.update(data_as_of=end,date_end=end,signal_last_date=end,historical_end=BOUNDARY,
        source_label='封存原策略＋固定規則日期延伸；回測與訊號截止 '+end,
        signal_count=len(payload['signals']),stock_count=len(payload['stocks']),
        pending_signal_count=len(pending),invalid_bar_count=sum(totals.values()),quality_issue_counts=dict(totals),
        zero_signal_days=sum(d['signal_count']==0 for d in payload['days']),
        archive_url='signal_explorer_20260909.html',
        source_scope_labels={'frozen_repaired_history':'9/9以前修復封存歷史',
            'finmind_extension_unverified':'9/10起 FinMind 延伸與新增官方事件/盤別核對；歷史身分與還原價未完整官方認證'})
    meta['limitations']=[
        '帳戶自2024/01/02以100萬元開始，2026視圖延續原有資產與持股。',
        '截至'+end+'收盤；最後一日訊號等待下一個可觀察交易日，不推測未來買入價格或成交。',
        '9/9以前經濟帳本、資產及POC查詢與封存版本逐項一致；延伸期沿同規則，未重新調參。',
        'POC只顯示本版本真實查詢；未查不等於未上移。永久品質未知時當日合格候選回原排序。',
        'POC為原始價，K線為同日還原因子調整OHLC；不可當成同一座標。',
        'HL2為日高低中點研究成交假設，普通盤可成交量未完整認證，非券商實際成交。',
        '2026/10/2仍有21筆TWSE全日成交量與輸入不同，原因未釐清；原始OHLC價格已比對，不能因此宣稱成交量完整認證。',
        '全期摘要從2024至'+end+'；歷史模式應隱藏所選日期後的成交、走勢和預約結果。',
        '延伸期身分及還原價未完整歷史官方認證；本研究不代表全市場股票池或實戰認證。']
    benchmark=cases['benchmark']
    payload['benchmark']=dict(label='0050研究基準',summary=deepcopy(benchmark['summary']),
        account_days={r['date']:deepcopy(r) for r in benchmark['account']['daily']
                      if meta['date_start']<=r['date']<=end})
    return payload


def run(args):
    cases,profiles,refs=load_latest(args.report)
    entries=report_entries(BUNDLE,prior.read(args.report))
    payload=build_payload(entries,cases,profiles,
        pd.read_parquet(BUNDLE/'close-official.parquet').set_index('date'),
        pd.read_parquet(BUNDLE/'close-quality.parquet').set_index('date'),
        pd.read_parquet(BUNDLE/'quotes-unmasked.parquet'),
        pd.read_parquet(BUNDLE/'eligibility.parquet').set_index('date'),pd.read_parquet(BUNDLE/'companies.parquet'))
    report=Path(args.report).resolve()
    payload['metadata']['parent_report_sha256']={str(report.relative_to(ROOT)):digest(report)}
    payload['metadata']['verified_source_count']=len(refs)
    dest=Path(args.payload).resolve();dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json_for_script(payload),encoding='utf-8')
    outputs={str(dest.relative_to(ROOT)):digest(dest)}
    for name in ('scripts/export_poc_latest_explorer.py','tests/test_poc_latest_explorer.py',
                 'scripts/export_poc_signal_explorer.py','scripts/export_signal_explorer.py'):
        refs[name]=digest(ROOT/name)
    if args.output:
        template=Path(args.template).resolve();refs[str(template.relative_to(ROOT))]=digest(template)
        path=Path(args.output).resolve();path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(render_html(template.read_text(),payload),encoding='utf-8')
        outputs[str(path.relative_to(ROOT))]=digest(path)
    receipt=dict(schema='poc_latest_explorer_receipt_v1',created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs,output_sha256=outputs,metadata=payload['metadata'],no_network=True,no_strategy_execution=True)
    path=Path(args.receipt).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(signal_count=len(payload['signals']),pending_signal_count=payload['metadata']['pending_signal_count'],
                         output_sha256=outputs),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',help='Omit to generate payload and receipt only')
    parser.add_argument('--template',default='ui/poc_latest_signal_explorer.html')
    parser.add_argument('--payload',default='.cache/poc-latest-20261003/explorer/payload.json')
    parser.add_argument('--receipt',default='.cache/poc-latest-20261003/explorer/receipt.json')
    run(parser.parse_args())
