#!/usr/bin/env python3
"""Audit dated issuer evidence; missing news is never a negative event label."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts.research_early_signal_losses import digest
from scripts.export_signal_explorer import verify_evidence

SPEC = ROOT/'docs/research_sequential_prereg_20261003.md'
SPEC_SHA = '7a8247ef6304e684e085d6360d9e2bf2a4649c1b1e84da48f7afebac0576efa1'


def audit_event(event, root=ROOT):
    """Do not substitute meeting dates or a permissive exploratory eligibility flag."""
    result = {k:event.get(k) for k in ('event_id','stock_id','source_date','source_url')}
    issues = []
    relative = event.get('source_local_path')
    path = (root/relative).resolve() if isinstance(relative,str) else None
    document_ok = bool(path and path.is_relative_to(root.resolve()) and path.is_file()
                       and digest(path) == event.get('source_sha256'))
    if not document_ok: issues.append('missing_or_changed_original_document')
    if event.get('original_document_retrieved') is not True: issues.append('original_not_retrieved')
    if event.get('first_publication_verified') is not True: issues.append('first_publication_unverified')
    stamp = None
    try:
        stamp = pd.Timestamp(event['first_publication_at'])
        if pd.isna(stamp) or stamp.tzinfo is None: raise ValueError('Need timezone')
        stamp = stamp.tz_convert('Asia/Taipei')
    except (KeyError, ValueError, TypeError):
        issues.append('missing_verified_publication_timestamp')
    proof = event.get('publication_evidence_local_path')
    proof_path = (root/proof).resolve() if isinstance(proof,str) else None
    if not (proof_path and proof_path.is_relative_to(root.resolve()) and proof_path.is_file()
            and digest(proof_path) == event.get('publication_evidence_sha256')):
        issues.append('missing_publication_evidence')
    early, realized = event.get('order_or_production'), event.get('realized_growth')
    stage = ('early' if early is True and realized is False else
             'realized' if realized is True and early is False else None)
    if stage is None: issues.append('stage_unknown_or_conflicting')
    sid = event.get('stock_id')
    if not isinstance(sid,str) or len(sid)!=4 or not sid.isascii() or not sid.isdigit():
        issues.append('invalid_ordinary_stock_identity')
    result.update(document_verified=document_ok, stage=stage,
        known_at=stamp.isoformat() if stamp is not None and not pd.isna(stamp) else None,
        eligible=not issues, issues=issues)
    return result


def available_event(event, signal_date):
    if event['eligible'] is not True: return False
    cutoff = pd.Timestamp(signal_date,tz='Asia/Taipei') + pd.Timedelta(days=1)
    return pd.Timestamp(event['known_at']) < cutoff


def run(events_path, output):
    if output.exists(): raise ValueError('Use a new output directory')
    if digest(SPEC)!=SPEC_SHA: raise ValueError('Prereg changed')
    refs,_ = verify_evidence(ROOT/'.cache/all-signals-2019-20261002/inputs',
                            ROOT/'.cache/signal-rank-20261003/rank-v2')
    events=json.loads(events_path.read_text())
    if len({e['event_id'] for e in events})!=len(events): raise ValueError('Duplicate event identity')
    audited=[audit_event(e) for e in events]
    signals=pd.read_parquet(ROOT/'.cache/signal-rank-20261003/rank-v2/signal-ranks.parquet',
                            columns=['signal_id','stock_id','signal_date'])
    matches=[]
    for r in signals.to_dict('records'):
        eligible=[e['event_id'] for e in audited if e['stock_id']==r['stock_id'] and available_event(e,r['signal_date'])]
        matches.append(dict(**r,eligible_events=eligible,event_coverage='verified_positive_only' if eligible else 'unknown'))
    for p in (SPEC,Path(__file__),events_path): refs[str(p.relative_to(ROOT))]=digest(p)
    for e in events:
        for key in ('source_local_path','publication_evidence_local_path'):
            relative=e.get(key)
            if isinstance(relative,str):
                p=(ROOT/relative).resolve()
                if p.is_relative_to(ROOT) and p.is_file(): refs[str(p.relative_to(ROOT))]=digest(p)
    output.mkdir(parents=True)
    for name,value in [('events.json',audited),('signal-coverage.json',matches)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    report=dict(schema='event_stage_time_audit_v1',created_at=datetime.now(timezone.utc).isoformat(),
        events=len(events),stocks=len({e['stock_id'] for e in events}),
        original_documents_verified=sum(e['document_verified'] for e in audited),
        eligible_events=sum(e['eligible'] for e in audited),
        issue_counts=dict(Counter(i for e in audited for i in e['issues'])),
        signals=len(matches),signals_with_verified_positive=sum(bool(r['eligible_events']) for r in matches),
        complete_news_coverage=False,negative_control_identifiable=False,
        performance_comparison='not_identifiable',trial_count=0,live_qualified=False,
        source_sha256=refs,output_sha256={str(p.relative_to(ROOT)):digest(p) for p in output.iterdir()})
    p=output/'report.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'report.sha256').write_text(digest(p)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','output_sha256')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--events',type=Path,default=ROOT/'.cache/theme-catalyst-20260927-v4/events.json')
    args=parser.parse_args();run(args.events.resolve(),args.output.resolve())
