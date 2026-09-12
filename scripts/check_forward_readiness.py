#!/usr/bin/env python3
"""Inspect sealed cash books, optionally refresh corporate evidence; no fills or approval."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import forward_readiness as readiness

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    result = readiness.refresh() if args.refresh else readiness.inspect()
    print(json.dumps(result, ensure_ascii=False, indent=2))
