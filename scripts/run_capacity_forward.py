#!/usr/bin/env python3
"""Independent capacity/control/0050 simulation. Never submits brokerage orders."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import capacity_forward as policy, capacity_forward_runner as runner
from app.file_lock import file_lock

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','run','status','refresh','pause','resume'])
    parser.add_argument('--root',type=Path,default=policy.ROOT)
    parser.add_argument('--reason',default='')
    parser.add_argument('--role',choices=['strategy','control'],default='strategy')
    args=parser.parse_args()
    if args.command=='init': result=dict(path=str(policy.initialize(args.root)))
    elif args.command=='run': result=runner.run(args.root)
    elif args.command=='status': result=runner.status(args.root)
    else:
        with file_lock(args.root/'.run.lock',timeout=0):
            if args.command=='refresh':result=runner.refresh(runner.books(args.root))
            else:result=policy.pause(args.root/(args.role+'.sqlite3'),args.command=='pause',args.reason)
    if args.command!='status': runner.export(args.root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
