#!/usr/bin/env python3
"""Append newly scanned days, preserving published historical results exactly.

Both scans must already exist. No market fetch, strategy evaluation or account
replay takes place here; receipts bind the old and new results independently.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.scan_market_strategies import digest, read_json, entry_events
from skills.strategy_scanner.presentation import render_html


def read_scan(folder):
    folder=Path(folder).resolve()
    folder.relative_to(ROOT)
    receipt=read_json(folder/'receipt.json')
    for name,expected in receipt['files_sha256'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder) or digest(path)!=expected:
            raise ValueError('Published scan source changed')
    return read_json(folder/'scan.json'),dict(path=str((folder/'receipt.json').relative_to(ROOT)),
        sha256=digest(folder/'receipt.json'))


def combine(previous, addition, previous_ref, addition_ref):
    for scan in (previous,addition):
        if (scan.get('schema')!='multi_strategy_scan_v1' or scan.get('live_qualified') is not False
                or scan.get('account_independent') is not True or not scan.get('days')):
            raise ValueError('Require independent nonempty sealed scans')
        dates=[d['date'] for d in scan['days']]
        if dates!=sorted(set(dates)) or (dates[0],dates[-1])!=(scan['start'],scan['end']):
            raise ValueError('Scan dates differ from declared period')
    if previous['end']>=addition['start']:
        raise ValueError('Cannot rerun or replace a published date')
    for key in ('strategies','evaluated_strategy_ids','signal_timing','execution_model','portfolio_model','returns_inherited'):
        if previous[key]!=addition[key]:raise ValueError('Daily strategy contract changed: '+key)
    old_ids=[s['stock_id'] for s in previous['days'][-1]['stocks']]
    if any([s['stock_id'] for s in d['stocks']]!=old_ids for d in addition['days']):
        raise ValueError('Frozen scan universe changed')
    return dict(addition,start=previous['start'],days=previous['days']+addition['days'],
        provenance=dict(addition['provenance'],daily_append=dict(previous=previous_ref,addition=addition_ref,
            historical_days_reused=len(previous['days']),historical_days_recomputed=0,
            appended_dates=[d['date'] for d in addition['days']],historical_rows_exact=True)))


def run(previous,addition,output):
    output=Path(output).resolve();output.relative_to(ROOT)
    if output.exists():raise ValueError('Choose a new immutable output directory')
    old,old_ref=read_scan(previous);new,new_ref=read_scan(addition)
    result=combine(old,new,old_ref,new_ref)
    entry_ids={s['id'] for s in result['strategies'] if s['kind']=='entry'}
    entries={sid:entry_events(result,sid) for sid in result['evaluated_strategy_ids'] if sid in entry_ids}
    counts=Counter({sid:0 for sid in result['evaluated_strategy_ids']})
    for day in result['days']:
        for stock in day['stocks']:counts.update(k for k,v in stock['results'].items() if v['status']=='matched')
    summary=dict(schema=result['schema'],start=result['start'],end=result['end'],source_end=result['source_end'],
        trading_days=len(result['days']),stocks_per_day={d['date']:len(d['stocks']) for d in result['days']},
        active_strategies=len(result['evaluated_strategy_ids']),catalog_modules=len(result['strategies']),
        signals_per_strategy=dict(counts),entry_export_strategy_count=len(entries),matches_include_non_entry_conditions=True,
        outcomes=dict(sum((Counter(d['counts']) for d in result['days']),Counter())),
        external_data_requests=0,account_independent=True,live_qualified=False,backtest_run=False,
        historical_days_reused=len(old['days']),historical_days_recomputed=0,
        appended_dates=[d['date'] for d in new['days']])
    # Reject source changes between reading and publishing, even if output seems usable.
    for ref in (old_ref,new_ref):
        if digest(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Source receipt changed during append')
    output.mkdir(parents=True)
    for name,data in [('scan.json',result),('entry-signals.json',entries),('summary.json',summary)]:
        (output/name).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    (output/'index.html').write_text(render_html(result))
    receipt=dict(created_at=datetime.now(timezone.utc).isoformat(),
        files_sha256={p.name:digest(p) for p in sorted(output.iterdir())},provenance=result['provenance'],
        append_code_sha256=digest(Path(__file__)))
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    for name in ('previous','addition','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(**vars(parser.parse_args()))
