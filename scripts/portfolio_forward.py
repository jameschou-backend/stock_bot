#!/usr/bin/env python3
"""Separate v2 paper account. Never sends brokerage orders."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import forward_portfolio as p, forward_portfolio_service as service, forward_journal as j


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('status','close','plan','release','record','export'))
    parser.add_argument('--journal',type=Path,default=p.PATH)
    parser.add_argument('--signals-journal',type=Path,default=j.DEFAULT_PATH)
    parser.add_argument('--actions-reviewed',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.command=='close': result=service.capture_close(args.journal,args.actions_reviewed)
    elif args.command=='plan': result=service.save_proposals(args.journal,args.signals_journal)
    elif args.command=='release': result=service.release_funded_intents(args.journal)
    elif args.command=='record':
        if args.report is None: parser.error('record requires --report JSON')
        payload=json.loads(args.report.read_text())
        if payload['kind'] not in ('fill','cancel','entitlement','delivery'):
            parser.error('record accepts fill/cancel/entitlement/delivery only')
        with j.connection(args.journal) as con: result=p.submit(con,payload)
    else:
        result=p.summary(args.journal)
        if args.command=='status': result.pop('rows')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
