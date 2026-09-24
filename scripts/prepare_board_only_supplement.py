#!/usr/bin/env python3
"""Bounded preparation in a new cache; never mutate sealed replay evidence."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read, write, sha
from skills.replay_market_feeds import ReplayMarketFeeds

SOURCE = ROOT / '.cache/board-only-20260925'
OUTPUT = ROOT / '.cache/board-only-source-supplement-20260925'
MAX_FETCHES = 10


def prepare(stock_ids):
    for sid in stock_ids:
        if len(sid) != 4 or not sid.isascii() or not sid.isdigit():
            raise ValueError('Only four-digit Taiwan stock IDs are permitted')
    output = OUTPUT / 'inputs'
    ledger_path = OUTPUT / 'source-ledger.json'
    if not output.exists():
        for name, digest in read(SOURCE / 'manifest.json')['files_sha256'].items():
            if sha(SOURCE / name) != digest:
                raise ValueError('Sealed parent changed: ' + name)
        shutil.copytree(SOURCE / 'inputs', output)
        write(ledger_path, dict(parent_manifest_sha256=sha(SOURCE / 'manifest.json'),
            preparation_code_sha256=sha(Path(__file__)), max_finmind_fetch_attempts=MAX_FETCHES,
            finmind_fetch_attempts=0, requests=[]))
    ledger = read(ledger_path)
    if (ledger['parent_manifest_sha256'] != sha(SOURCE / 'manifest.json')
            or ledger['preparation_code_sha256'] != sha(Path(__file__))):
        raise ValueError('Preparation source/code changed; choose another version')
    feeds = ReplayMarketFeeds(output / 'execution-feeds', offline=False)
    for sid in stock_ids:
        before = read(output / 'execution-feeds/index.json')
        cached = 'limits:' + sid in before['entries']
        if not cached:
            if ledger['finmind_fetch_attempts'] >= MAX_FETCHES:
                raise ValueError('Supplement request budget exhausted')
            ledger['finmind_fetch_attempts'] += 1
            write(ledger_path, ledger)
        status = 'failed'
        try:
            rows = feeds.get_limits(sid)
            status = 'success'
        finally:
            after = read(output / 'execution-feeds/index.json')
            old, new = before['request_counters'], after['request_counters']
            ledger['requests'].append(dict(stock_id=sid, status=status, preexisting_cache=cached,
                counter_delta={key: new[key]-old[key] for key in new},
                index_sha256=sha(output / 'execution-feeds/index.json')))
            write(ledger_path, ledger)
        print(json.dumps(dict(stock_id=sid, rows=len(rows), first_date=min(rows), last_date=max(rows),
            finmind_fetch_attempts=ledger['finmind_fetch_attempts'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limits', nargs='+', required=True)
    args = parser.parse_args()
    with file_lock(ROOT / '.cache/board-only-supplement-prepare.lock', timeout=0):
        prepare(args.limits)
