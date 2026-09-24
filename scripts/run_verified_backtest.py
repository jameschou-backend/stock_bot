#!/usr/bin/env python3
"""Run or preflight a versioned offline account comparison, with per-case resume."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.verified_backtest_tool import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('daily', 'strict'), default='daily')
    parser.add_argument('--policy', choices=('mixed', 'board_only', 'all'), default='all')
    parser.add_argument('--stress', choices=('control', 'combined', 'all'), default='all')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--fresh', action='store_true', help='Recompute selected cases and compare with sealed references/cache')
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/backtest-tool/latest.json')
    args = parser.parse_args()
    try:
        report = run(output=args.output, mode=args.mode, policy=args.policy, stress=args.stress,
                     preflight_only=args.preflight_only, fresh=args.fresh)
    except (ValueError, FileNotFoundError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(report['status'], report['metrics'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
