#!/usr/bin/env python3
"""Bounded source preparation, separate from the strictly offline account replay."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from app.finmind import fetch_dataset
from scripts.research_exit_scenarios import read, write, sha, TrackedCorporateActions
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable

PARENT = ROOT / '.cache/board-only-source-supplement-20260925/inputs'
OUTPUT = ROOT / '.cache/sector-account-sources-20260925'
MAXIMUM = {'finmind': 200, 'official': 100}


class RequestBudget:
    """Persist attempts before transport; interrupted attempts are never free retries."""
    def __init__(self, path, *, maximum=None):
        self.path = Path(path)
        self.maximum = dict(MAXIMUM if maximum is None else maximum)
        if self.path.exists():
            self.state = read(self.path)
            if self.state.get('maximum') != self.maximum:
                raise ValueError('Preparation budget changed; use a new explicit source version')
        else:
            self.state = dict(maximum=self.maximum, attempts={k: 0 for k in self.maximum}, requests=[])
            write(self.path, self.state)

    def call(self, provider, identity, function, *args, **kwargs):
        if self.state['attempts'][provider] >= self.maximum[provider]:
            raise ReplayDataUnavailable('Preparation request budget exhausted: ' + provider)
        # A failed request is kept for review, rather than retried by every case.
        previous = [r for r in self.state['requests'] if r['provider'] == provider and r['identity'] == identity]
        if previous:
            raise ReplayDataUnavailable('Preparation request already attempted; review evidence before retry: ' + identity)
        self.state['attempts'][provider] += 1
        row = dict(provider=provider, identity=identity, status='started')
        self.state['requests'].append(row)
        write(self.path, self.state)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__)
            write(self.path, self.state)
            raise
        row.update(status='success')
        if provider == 'finmind':
            row['shared_cache_hit'] = bool(result.attrs.get('cache_hit', False))
        write(self.path, self.state)
        return result

    def finmind(self, dataset, start, end, **kwargs):
        if dataset not in ('TaiwanStockPriceLimit', 'TaiwanStockDividend'):
            raise ValueError('Dataset is outside the bounded execution preparation')
        kwargs.update(max_retries=0, requests_per_hour=6000, timeout=30)
        identity = f'{dataset}:{kwargs.get("data_id")}:{start}:{end}'
        return self.call('finmind', identity, fetch_dataset, dataset, start, end, **kwargs)

    def official(self, url, **kwargs):
        import requests
        from skills.replay_market_feeds import URLS
        if url not in URLS.values():
            raise ValueError('Only official daily odd-lot preparation endpoints are permitted')
        identity = url + '?' + json.dumps(kwargs.get('params', {}), sort_keys=True)
        response = self.call('official', identity, requests.get, url, **kwargs)
        return response


def initialize(output=OUTPUT, parent=PARENT):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if (not output.is_relative_to(ROOT / '.cache') or output.is_relative_to(parent)
            or parent.is_relative_to(output)):
        raise ValueError('Preparation must use a separate directory inside the project cache')
    feeds = ReplayMarketFeeds(parent / 'execution-feeds', offline=True).manifest()
    references = {str(p.relative_to(parent)): sha(p) for p in parent.rglob('*')
                  if p.is_file() and p.suffix != '.lock'}
    provenance = dict(parent=str(parent.relative_to(ROOT)), parent_files_sha256=references,
                      parent_feed_index_sha256=feeds['manifest_sha256'])
    if output.exists():
        if read(output / 'parent.json') != provenance:
            raise ValueError('Preparation parent evidence changed')
        for name, digest in references.items():
            if name == 'execution-feeds/index.json':
                continue  # The new index grows; original raw/normalized evidence must remain identical.
            if sha(output / 'inputs' / name) != digest:
                raise ValueError('Copied parent source changed: ' + name)
    else:
        shutil.copytree(parent, output / 'inputs')
        write(output / 'parent.json', provenance)
    return output / 'inputs'


def prepare(output=OUTPUT, parent=PARENT, additions=()):
    # Imported after argument parsing so --help never starts a feature calculation.
    from scripts.research_sector_accounts import load_inputs, corporate_overrides
    from skills.sector_account_replay import run_case, configurations
    from app.config import load_config
    inputs = initialize(output, parent)
    budget = RequestBudget(Path(output) / 'budget.json')
    overrides = corporate_overrides([ROOT / 'docs/backtest_corporate_completion_20260925.json', *additions])
    data, _ = load_inputs()
    token = load_config().finmind_token
    rows = {}
    for name, config in configurations():
        print('preparing', name, flush=True)
        feeds = ReplayMarketFeeds(inputs / 'execution-feeds', offline=False,
                                  token=token, finmind_fetch=budget.finmind, http_get=budget.official)
        corporate = TrackedCorporateActions(data.events, inputs / 'dividends', token,
                                            offline=False, overrides=overrides)
        with patch('skills.replay_corporate_actions.fetch_dataset', budget.finmind):
            result = run_case(data, config, inputs, overrides, feeds=feeds, corporate=corporate)
        rows[name] = dict(path_reached_end=bool(result['completed']), reason=result.get('reason'))
        # No performance from an online preparation run is published as a backtest.
        write(Path(output) / 'preparation.json', dict(cases=rows, attempts=budget.state['attempts'],
            performance_report=False, live_qualified=False, preparation_code_sha256=sha(__file__)))
        print(name, rows[name], flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--parent', type=Path, default=PARENT,
                        help='Verified prior inputs to copy into a new bounded preparation version')
    parser.add_argument('--corporate-additions', action='append', type=Path, default=[])
    parser.add_argument('--fetch', action='store_true', help='Explicitly allow bounded source requests')
    args = parser.parse_args()
    if not args.fetch:
        parser.error('Use --fetch for source preparation; account replay never fetches')
    with file_lock(ROOT / '.cache/sector-account-prepare.lock', timeout=0):
        prepare(args.output, args.parent, args.corporate_additions)
