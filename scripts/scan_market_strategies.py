#!/usr/bin/env python3
"""Run a bounded offline multi-strategy scan, without schedules or orders."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from skills.strategy_scanner.engine import scan_market
from skills.strategy_scanner.presentation import render_html

DEFAULT_BUNDLE=ROOT/'.cache/poc-latest-20261003/inputs-v1'
DEFAULT_POC=ROOT/'.cache/poc-daily-opportunities-20261004/snapshots/20261004T063227825688Z/0063/report.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''): h.update(part)
    return h.hexdigest()


def read_json(path):
    def invalid(value): raise ValueError('Invalid non-finite JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def load_poc(report_path, *, root=ROOT, bundle=None, manifest_hash=None):
    if report_path is None:
        return [],dict(status='not_supplied',scope='no_POC_inferred_from_daily_OHLCV')
    root=Path(root).resolve();path=Path(report_path).resolve()
    path.relative_to(root)
    sidecar=path.with_suffix('.sha256')
    if digest(path)!=sidecar.read_text().strip(): raise ValueError('POC report hash mismatch')
    report=read_json(path)
    if report.get('schema')!='poc_daily_profiles_v1' or report.get('all_signals_materialized') is not True:
        raise ValueError('Require account-independent daily POC report')
    if bundle is not None:
        key=str((Path(bundle).resolve()/'manifest.json').relative_to(root))
        if not manifest_hash or report.get('source_sha256',{}).get(key)!=manifest_hash:
            raise ValueError('POC report is not bound to this scan input bundle')
    item=report['profiles']; profiles=(root/item['path']).resolve(); profiles.relative_to(root)
    if digest(profiles)!=item['sha256']: raise ValueError('Daily POC profile hash mismatch')
    rows=read_json(profiles)
    if not isinstance(rows,list) or any(r.get('account_independent') is not True for r in rows):
        raise ValueError('POC records must be account-independent')
    return rows,dict(status='loaded',report=str(path.relative_to(root)),report_sha256=digest(path),
        profiles_sha256=digest(profiles),end=report['end'],row_count=len(rows),
        scope='original_candidate_stockdays_only_not_every_stockday',
        direct_artifacts_verified=True,ancestor_raw_files_reverified=False,
        reconstructed=True,known_at_signal_realtime_receipts_verified=False)


def entry_events(payload, strategy_id, *, first_only=False):
    """Portfolio-agnostic adapter: preserve every matching setup, no top-N cut."""
    if strategy_id not in payload['evaluated_strategy_ids']:
        raise ValueError('Strategy was not evaluated')
    specs={s['id']:s for s in payload['strategies']}
    events=[]
    for day in payload['days']:
        for stock in day['stocks']:
            result=stock['results'][strategy_id]
            if result['status']!='matched' or first_only and result['first_signal'] is not True:
                continue
            events.append(dict(event_id=f"{strategy_id}:{specs[strategy_id]['version']}:{day['date']}:{stock['stock_id']}",
                strategy_id=strategy_id,strategy_version=specs[strategy_id]['version'],
                signal_date=day['date'],stock_id=stock['stock_id'],members=[stock['stock_id']],
                earliest_execution='next_market_session',first_signal=result['first_signal'],
                reasons=result['reasons'],metrics=result['metrics'],regime=stock['regime'],
                market_regime=day['market_regime'],regime_fit=result['regime_fit'],
                source='multi_strategy_scan_v1',account_independent=True,live_qualified=False))
    return events


def run(args):
    from skills.strategy_scanner.data import load_bundle
    started=time.perf_counter()
    bundle=Path(args.bundle).resolve()
    if not args.end:
        # Metadata is used only to choose a date; the adapter verifies its SHA.
        end=read_json(bundle/'manifest.json')['end']
    else: end=args.end
    start=args.start or end
    data=load_bundle(bundle,start=start,end=end)
    profiles,poc_info=load_poc(args.poc_report,bundle=bundle,
        manifest_hash=data['provenance']['source_hashes']['manifest.json'])
    provenance=dict(data['provenance'],poc=poc_info,
        source_code_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            sorted((ROOT/'skills/strategy_scanner').glob('*.py'))+[Path(__file__).resolve()]},
        external_data_requests=0,scheduler_enabled=False,
        research_scope='frozen_local_universe_not_certified_all_historical_listings',
        optional_data_policy='missing_is_unknown_no_synthetic_chips_or_news',
        original_returns_not_inherited=True)
    result=scan_market(data['bars'],data['calendar'],start=start,end=end,names=data['names'],
        original_signals=data['original_signals'],poc=profiles,provenance=provenance)
    outputs={sid:entry_events(result,sid) for sid in result['evaluated_strategy_ids']}
    summary=dict(schema=result['schema'],start=result['start'],end=result['end'],
        source_end=result['source_end'],trading_days=len(result['days']),
        stocks_per_day={d['date']:len(d['stocks']) for d in result['days']},
        active_strategies=len(result['evaluated_strategy_ids']),
        catalog_modules=len(result['strategies']),signals_per_strategy={k:len(v) for k,v in outputs.items()},
        outcomes=dict(sum((Counter(d['counts']) for d in result['days']),Counter())),
        elapsed_seconds=round(time.perf_counter()-started,3),external_data_requests=0,
        account_independent=True,live_qualified=False,backtest_run=False)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise ValueError('Choose an empty output directory; previous scans are immutable')
    for name,value in [('scan.json',result),('entry-signals.json',outputs),('summary.json',summary)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    (output/'index.html').write_text(render_html(result))
    receipt=dict(created_at=datetime.now(timezone.utc).isoformat(),
        files_sha256={p.name:digest(p) for p in sorted(output.iterdir())},provenance=provenance)
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    print(json.dumps(dict(summary,output=str(output)),ensure_ascii=False))
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    p.add_argument('--start',help='First signal date; default is end')
    p.add_argument('--end',help='Last signal date; default is frozen bundle end')
    p.add_argument('--poc-report',type=Path,default=DEFAULT_POC,
        help='Hash-bound independent POC report; does not fetch missing data')
    p.add_argument('--without-poc',action='store_true',help='Explicitly mark POC as unavailable')
    p.add_argument('--output',type=Path,required=True,help='New empty output directory')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.without_poc: args.poc_report=None
    run(args)
