#!/usr/bin/env python3
"""Read-only closing preview; never marks a corporate review as approved."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.capacity_close_preview import inspect
from app.capacity_forward import ROOT

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=inspect(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(observed_at=result['observed_at'],source_requests=0,
        books={role:dict(cash=b['cash'],checks=b['checks']) for role,b in result['books'].items()}),ensure_ascii=False,indent=2))
