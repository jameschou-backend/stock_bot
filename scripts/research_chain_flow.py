#!/usr/bin/env python
"""Update at most 32 Sponsor snapshots, or analyze local cache without network."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import chain_flow_research as research
from app.file_lock import file_lock
from app.finmind import FinMindQuotaError
from app.news_research import atomic_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config=None
    if args.fetch:
        from app.config import load_config
        config=load_config()
    try:
        with file_lock(research.CACHE/'run.lock',timeout=0):
            report=research.run(fetch=args.fetch,config=config)
            if args.output: atomic_json(args.output,report)
    except FinMindQuotaError as exc:
        if args.output: atomic_json(args.output,{'error':'FinMind quota paused','retry_after_seconds':exc.retry_after_seconds})
        print(str(exc),file=sys.stderr)
        return 75
    print({k:report[k] for k in ['as_of','summary']})
    print({k:v for k,v in report['collection'].items() if k!='inputs'})

if __name__=='__main__': sys.exit(main())
