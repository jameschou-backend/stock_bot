#!/usr/bin/env python3
"""Bounded execution-source preparation for the dated catalyst experiment."""
from pathlib import Path
import argparse
import json
import sys
from urllib.parse import urlparse
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import load_config
from app.file_lock import file_lock
from scripts.prepare_sector_account_sources import initialize, RequestBudget
from scripts.research_theme_catalyst import inputs, ledger, plans, ADDITIONS
from scripts.research_sector_accounts import corporate_overrides
from scripts.research_exit_scenarios import read, write, sha, TrackedCorporateActions
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.sector_account_replay import run_case

OUTPUT = ROOT/'.cache/theme-catalyst-sources-20260927'
PARENT = ROOT/'.cache/sector-account-sources-20260925/inputs'


class Budget(RequestBudget):
    def official(self, url, **kwargs):
        hold = ROOT/'.cache/official-origin-holds'/(urlparse(url).hostname+'.json')
        if hold.exists() and read(hold).get('status') == 'blocked':
            raise ReplayDataUnavailable('Existing official-origin hold: '+urlparse(url).hostname)
        return super().official(url, **kwargs)


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    with file_lock(ROOT/'.cache/theme-catalyst.lock', timeout=0):
        cache = initialize(output, PARENT)
        budget = Budget(output/'budget.json', maximum={'finmind':60, 'official':80})
        data, frames, expected = inputs()
        jobs, signals, observed = plans(data, frames, ledger())
        overrides = corporate_overrides(ADDITIONS)
        token = load_config().finmind_token
        rows = {}
        for name, selected, config in jobs:
            feeds = ReplayMarketFeeds(cache/'execution-feeds', offline=False, token=token,
                finmind_fetch=budget.finmind, http_get=budget.official)
            corp = TrackedCorporateActions(data.events, cache/'dividends', token, offline=False, overrides=overrides)
            with patch('skills.replay_corporate_actions.fetch_dataset', budget.finmind):
                result = run_case(selected, config, cache, overrides, feeds=feeds, corporate=corp)
            rows[name] = dict(path_reached_end=result['completed'], reason=result.get('reason'))
            # Online path discovery is not published as performance.
            write(output/'preparation.json', dict(cases=rows, attempts=budget.state['attempts'],
                preparation_code_sha256=sha(Path(__file__)), online_returns_published=False,
                signal_code_sha256=sha(ROOT/'skills/theme_catalyst.py'),
                source_documents_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                    (ROOT/'docs/leo_catalyst_sources_20260927.json', ROOT/'docs/leo_peer_sources_20260927.json')}))
            print(name, rows[name], flush=True)
        return dict(cases=rows, attempts=budget.state['attempts'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUTPUT)
    print(json.dumps(prepare(p.parse_args().output), ensure_ascii=False, indent=2))
