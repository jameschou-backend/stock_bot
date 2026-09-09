#!/usr/bin/env python
"""Bounded automatic headline scan or offline historical review."""
from datetime import date, datetime
from pathlib import Path
import argparse
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import news_research as news
from app.file_lock import file_lock
from app.finmind import FinMindQuotaError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['scan', 'review'])
    parser.add_argument('--end', type=date.fromisoformat, default=datetime.now(ZoneInfo('Asia/Taipei')).date())
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--stock-id')
    parser.add_argument('--fetch', action='store_true', help='近期掃描才可用，最多 14 天，每天最多一次網路請求')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = None
    if args.fetch:
        from app.config import load_config
        config = load_config()
    try:
        with file_lock(news.CACHE/'run.lock', timeout=0):
            report = news.run(args.mode, args.end, args.days, stock_id=args.stock_id, fetch=args.fetch, config=config)
            if args.output: news.atomic_json(args.output, report)
    except FinMindQuotaError as exc:
        if args.output:
            news.atomic_json(args.output, {'error': 'FinMind quota paused', 'retry_after_seconds': exc.retry_after_seconds})
        print(str(exc), file=sys.stderr)
        return 75
    print({'summary': report['summary'], 'collection': report['collection'],
           'topics': [(t['name'], t['articles'], t['operating_clues']) for t in report['themes']]})


if __name__ == '__main__':
    sys.exit(main())
