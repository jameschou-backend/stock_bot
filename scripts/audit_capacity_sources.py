#!/usr/bin/env python3
"""Reconstruct each recorded fill's source ages without network or ledger edits."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.capacity_source_review import review
from app.capacity_forward import ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = review(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(observed_at=result['observed_at'], guard=result['guard'],
        books={role: dict(fills=len(b['fills']), fresh_at_fill=sum(f['source_freshness_passed'] for f in b['fills']),
                          fresh_now=b['current']['source_freshness_passed']) for role,b in result['books'].items()}),
        ensure_ascii=False, indent=2))
