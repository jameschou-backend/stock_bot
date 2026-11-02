#!/usr/bin/env python3
"""Capture or verify Taiwan files without changing data, models or backtests."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.tw_baseline_guard import capture, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("capture", "verify"))
    p.add_argument("--baseline", type=Path, required=True)
    args = p.parse_args()
    result = capture(ROOT, args.baseline) if args.action == "capture" else verify(ROOT, args.baseline)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("unchanged", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
