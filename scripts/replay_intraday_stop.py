#!/usr/bin/env python3
"""Replay a known position and timestamp groups; never submit broker orders."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.intraday_peak_exit import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = replay(json.loads(raw))
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    # Exclusive creation keeps an older audited result intact.
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'remaining_qty', 'net_sell_proceeds')}))


if __name__ == '__main__':
    main()
