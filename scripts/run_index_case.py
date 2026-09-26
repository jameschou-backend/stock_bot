#!/usr/bin/env python3
"""Revalidate sources and replay one frozen continuous ETF case offline."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from skills.index_case_job import run,ARMS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--mask',type=int,choices=range(8),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fresh',action='store_true');a=p.parse_args()
    result=run(a.arm,a.mask,a.output,fresh=a.fresh)
    print('completed',result['metrics'],flush=True)
