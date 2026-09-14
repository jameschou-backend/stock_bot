#!/usr/bin/env python3
"""Explicit activation and execution of the separately sealed source guard."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import capacity_forward as policy, capacity_forward_runner as legacy
from app import capacity_source_guard as guard, capacity_guard_runner as runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['activate', 'run', 'status'])
    parser.add_argument('--root', type=Path, default=policy.ROOT)
    args = parser.parse_args()
    if args.command == 'activate':
        result = guard.activate(args.root)
    elif args.command == 'run':
        result = runner.run(args.root)
    else:
        guard.verify(args.root)
        result = legacy.status(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
