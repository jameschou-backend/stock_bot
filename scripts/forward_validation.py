#!/usr/bin/env python3
"""Freeze rules/signals and retain observed quotes. No brokerage connection."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import forward_journal as journal, forward_service as service

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('status', 'prepare', 'freeze', 'plan', 'quotes'))
    p.add_argument('--signals', type=Path, default=service.DEFAULT_SIGNALS)
    p.add_argument('--stocks', nargs='+', default=['0050'])
    p.add_argument('--journal', type=Path, default=journal.DEFAULT_PATH)
    args = p.parse_args()
    if args.command == 'prepare':
        from scripts.prepare_forward_signals import prepare
        result = {'signals':str(prepare())}
    elif args.command == 'plan': result = service.plan_frozen_candidates(args.journal)
    elif args.command == 'freeze': result = service.freeze_today(args.journal, args.signals)
    elif args.command == 'quotes': result = service.capture_quotes(args.stocks, args.journal)
    else:
        result = journal.summary(args.journal)
        result.pop('rows')
    print(json.dumps(result, ensure_ascii=False, indent=2))
