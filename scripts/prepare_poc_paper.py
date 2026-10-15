#!/usr/bin/env python3
"""Prepare the selected original POC strategy without replaying any account."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import poc_paper_preparation as p, forward_journal as j
from skills.all_entry_accounts import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('protocol','capture','status'))
    parser.add_argument('--journal',type=Path,default=p.PATH)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--audit-sha')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='protocol':
        payload=p.protocol_payload();write(p.PROTOCOL,payload)
        print(json.dumps(dict(protocol=str(p.PROTOCOL),sha256=p.sha(p.PROTOCOL),paper_enabled=False)))
        return
    protocol,ref=p.load_protocol()
    if args.command=='status':
        print(json.dumps(p.observation_summary(args.journal),ensure_ascii=False));return
    if args.output is None: parser.error('capture requires a new --output report path')
    if args.output.exists(): parser.error('report is immutable; use status or choose a new output')
    if bool(args.audit)!=bool(args.audit_sha): parser.error('--audit and --audit-sha are required together')
    from app.research_terminal_service import ResearchTerminal
    from app.workbench_service import data_status
    from app.db import get_session
    from app.models import TradingCalendar
    from sqlalchemy import select
    kwargs={} if args.audit is None else dict(daily_audit=str(args.audit.resolve().relative_to(ROOT)),daily_sha=args.audit_sha)
    snapshot=p.scanner_snapshot(ResearchTerminal(**kwargs))
    status=data_status()
    with get_session() as db:
        sessions=[str(d) for d in db.scalars(select(TradingCalendar.trading_date).where(
            TradingCalendar.is_open.is_(True)).order_by(TradingCalendar.trading_date))]
    with j.connection(args.journal) as con:
        observation=p.observe(con,protocol,ref,snapshot,status,sessions)
    report=p.validate(p.readiness(protocol,ref,snapshot,observation,status))
    report['observation_summary']=p.observation_summary(args.journal)
    value=write(args.output,report)
    args.output.with_suffix('.sha256').write_text(value+'\n')
    print(json.dumps(dict(path=str(args.output),sha256=value,signal_date=snapshot['signal_date'],
        observation=observation['kind'],reused_accounts=8,historical_backtests_replayed=0,paper_enabled=False),ensure_ascii=False))


if __name__=='__main__': main()
