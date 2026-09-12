#!/usr/bin/env python3
"""Run one bounded observation/settlement cycle on sealed CASH-ALLOCATION simulation books only."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app import forward_cash_policy as sim, forward_cash_automation as auto


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','run','status','export'])
    parser.add_argument('--observe-seconds',type=int,default=20)
    args=parser.parse_args()
    if args.command=='init':result=dict(path=str(sim.initialize()))
    elif args.command=='run':result=auto.run(observe_seconds=args.observe_seconds)
    else:
        result=auto.export()
        if args.command=='status':
            result['books']={k:{x:v[x] for x in ['cash','nav','price_date','fill_count']} for k,v in result['books'].items()}
            result['recent_runs']=[dict(stage=x['body']['stage'],status=x['body']['status'],recorded_at=x['recorded_at']) for x in result['recent_runs'][-5:]]
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
