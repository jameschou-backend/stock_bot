#!/usr/bin/env python3
"""Prepare and gate the frozen holder-flow accounts before offline backtesting."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.file_lock import file_lock
from app.finmind import FinMindQuotaError
from scripts.prepare_theme_catalyst import Budget
from scripts.prepare_sector_account_sources import RequestBudget
from scripts.research_first_bar import inputs as load_inputs
from scripts.research_holder_flow import PUBLICATION
from scripts.research_surge_anatomy import clean
from scripts.research_theme_catalyst import source_closure, CACHE
from scripts.research_sector_accounts import corporate_overrides
from scripts.research_exit_scenarios import TrackedCorporateActions
from scripts.audit_current_causality_20260925 import verify_hashes
from skills.holder_flow import ACCOUNT_ARMS
from skills.sector_account_replay import run_case
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only
from skills.replay_market_feeds import ReplayDataUnavailable, parse_odd, URLS
from skills.account_source_preflight import (digest, read, write, merge_sources,
    inventory, source_identity, gate, CachedPreparationFeeds, prepared_case)

DEFAULT = ROOT / '.cache/holder-flow-prepared-20260927'
CODE = [Path(__file__), ROOT/'skills/account_source_preflight.py', ROOT/'skills/execution_factorial.py']


class ExecutionBudget(Budget):
    """A reviewed normal response can recover only its exact endpoint.

    Other held origins/endpoints stay held. A new security response immediately
    stops this endpoint too; no alternate identities, challenge solving or retry.
    """
    def official(self, url, **kwargs):
        recovery = self.path.parent/'official-recovery.json'
        held = self.path.parent/'official-recovery-stopped.json'
        if url != URLS['twse'] or not recovery.exists():
            return super().official(url, **kwargs)
        if held.exists():
            raise ReplayDataUnavailable('Recovered official endpoint stopped; review '+str(held))
        proof = read(recovery)
        raw = ROOT/proof['path']
        if digest(raw) != proof['sha256']:
            raise ReplayDataUnavailable('Official endpoint recovery proof changed')
        record = read(raw)
        if record.get('url') != url:
            raise ReplayDataUnavailable('Official recovery proof covers another endpoint')
        parse_odd(record, 'twse', record['day'])
        observed = datetime.fromisoformat(record['retrieved_at'])
        age = (datetime.now(timezone.utc)-observed).total_seconds()
        if not 0 <= age <= 86400:
            raise ReplayDataUnavailable('Official endpoint recovery requires a fresh reviewed response')
        response = RequestBudget.official(self, url, **kwargs)
        if response.status_code in (403, 428):
            write(held, dict(url=url, status=response.status_code, automatic_retry=False,
                             observed_at=datetime.now(timezone.utc).isoformat()))
            raise ReplayDataUnavailable('Official endpoint security response; preparation stopped')
        return response


def study(cache=None):
    """Use the already sealed signals, without another parameter search."""
    if digest(PUBLICATION) != PUBLICATION.with_suffix('.sha256').read_text().strip():
        raise ValueError('Holder-flow publication changed')
    pub = read(PUBLICATION)
    report_path = ROOT / pub['report']['path']
    if digest(report_path) != pub['report']['sha256'] or pub['reproducibility']['passed'] is not True:
        raise ValueError('Holder-flow report is not verified')
    report = read(report_path)
    manifest = read(report_path.parent / 'manifest.json')
    bound = [r for r in pub['reproducibility']['runs'] if ROOT/r['path'] == report_path.parent/'manifest.json']
    if len(bound) != 1 or digest(report_path.parent/'manifest.json') != bound[0]['sha256']:
        raise ValueError('Signal manifest is not bound to publication')
    expected = {ROOT/p: h for p, h in report['source_sha256'].items()}
    expected.update({report_path.parent/p: h for p, h in manifest['files_sha256'].items()})
    expected.update({p: digest(p) for p in [PUBLICATION, PUBLICATION.with_suffix('.sha256'), report_path.parent/'manifest.json', *CODE]})
    verify_hashes(expected)
    entries = read(report_path.parent / 'entries.json')
    _, _, _, data, sources = load_inputs()
    expected.update(sources)
    closure, overrides = source_closure(CACHE)
    expected.update(closure)
    extra = cache/'corporate-overrides.json' if cache else None
    if extra is not None and extra.exists():
        overrides = corporate_overrides([ROOT/'docs/backtest_corporate_completion_20260925.json', extra])
        expected[extra] = digest(extra)
        expected.update({ROOT/p:h for p,h in read(extra).get('evidence_sha256', {}).items()})
    if cache and (cache/'official-recovery.json').exists():
        proof = read(cache/'official-recovery.json')
        expected[ROOT/proof['path']] = proof['sha256']
    verify_hashes(expected)
    return data, entries, overrides, expected


def jobs(data, entries):
    for stress in ('control', 'combined'):
        for arm in ('benchmark', *ACCOUNT_ARMS):
            config = dict(arm='benchmark' if arm == 'benchmark' else 'relative_strength', stress=stress,
                          benchmark=arm == 'benchmark', board_only=False, position_count=0 if arm == 'benchmark' else 5)
            yield arm+'_'+stress, replace(data, entries_by_arm={'relative_strength': entries.get(arm, [])}), config


def identity(cache, expected):
    return dict(prepared_sources=source_identity(cache/'inputs'),
                preparation_audit={name: digest(cache/name) for name in
                    ('reuse.json', 'budget.json', 'official-recovery.json', 'official-recovery-stopped.json') if (cache/name).exists()},
                research_sources={str(p.relative_to(ROOT)): h for p,h in expected.items()})


def prepare(cache, fetch):
    # Preserve previous blocked/completed receipts before a mutable preparation
    # workspace is extended. Historical results never disappear on a retry.
    for name in ('preparation.json', 'preflight.json', 'progress.json'):
        if (cache/name).exists():
            write(cache/'history'/(digest(cache/name)+'-'+name), read(cache/name))
    data, entries, overrides, expected = study(cache)
    ids = {e['stock_id'] for es in entries.values() for e in es} | {'0050'}
    if not (cache/'reuse.json').exists():
        donors = [CACHE, *sorted(p.parent.parent for p in (ROOT/'.cache').glob('*/inputs/execution-feeds/index.json')
                                if p.parent.parent != CACHE and p.parent.parent != cache/'inputs')]
        receipt = merge_sources(cache/'inputs', donors, ids)
    else:
        receipt = read(cache/'reuse.json')
    write(cache/'inventory.json', inventory(cache/'inputs', ids))
    if receipt['conflicts']:
        raise ValueError('Donor conflicts require review; see reuse.json')
    if not fetch:
        print('Local reuse complete; inventory recorded. Use --fetch for bounded preparation.', flush=True)
        return
    budget = ExecutionBudget(cache/'budget.json', maximum={'finmind': 2000, 'official': 600})
    token = load_config().finmind_token
    feeds = CachedPreparationFeeds(cache/'inputs/execution-feeds', offline=False, token=token,
                                    finmind_fetch=budget.finmind, http_get=budget.official)
    cases = {}
    for name, selected, config in jobs(data, entries):
        corporate = TrackedCorporateActions(data.events, cache/'inputs/dividends', token, offline=False, overrides=overrides)
        print('preparing', name, flush=True)
        try:
            with patch('skills.replay_corporate_actions.fetch_dataset', budget.finmind):
                result = prepared_case(selected, config, cache/'inputs', overrides, feeds=feeds, corporate=corporate)
        except FinMindQuotaError:
            write(cache/'progress.json', dict(cases=cases, quota_paused=True, attempts=budget.state['attempts']))
            raise
        cases[name] = dict(completed=result['completed'], reason=result.get('reason'),
                          last_daily_date=(result.get('account', result.get('partial_account', {})).get('daily') or [{}])[-1].get('date'))
        write(cache/'progress.json', dict(cases=cases, attempts=budget.state['attempts'], performance_report=False))
        print(name, cases[name], 'requests', budget.state['attempts'], flush=True)
    verify_hashes(expected)
    write(cache/'preparation.json', dict(identity=identity(cache, expected), cases=cases,
        attempts=budget.state['attempts'], performance_report=False, live_qualified=False))
    write(cache/'inventory.json', inventory(cache/'inputs', ids))
    check(cache, data, entries, expected)


def check(cache, data, entries, expected):
    receipt = read(cache/'preparation.json') if (cache/'preparation.json').exists() else {}
    result = gate(receipt, identity(cache, expected), [name for name, _, _ in jobs(data, entries)])
    write(cache/'preflight.json', result)
    print('preflight', result, flush=True)
    return result


def replay(cache, output):
    if output.exists():
        raise ValueError('Use a new immutable replay output')
    with offline_only():
        data, entries, overrides, expected = study(cache)
        result = check(cache, data, entries, expected)
        if not result['ready']:
            raise ValueError('Preflight blocked; no performance backtest started. See preflight.json')
        before = identity(cache, expected)
        output.mkdir(parents=True)
        summaries = {}
        feeds = CachedPreparationFeeds(cache/'inputs/execution-feeds', offline=True)
        for name, selected, config in jobs(data, entries):
            append_trial_registry(dict(source='holder_flow_prepared_account', case=name, status='started',
                output=str(output.relative_to(ROOT)), timestamp=datetime.now(timezone.utc).isoformat()))
            result = prepared_case(selected, config, cache/'inputs', overrides, feeds=feeds)
            write(output/(name+'.json'), clean(result))
            append_trial_registry(dict(source='holder_flow_prepared_account', case=name,
                status='completed' if result['completed'] else 'incomplete', reason=result.get('reason'),
                output=str(output.relative_to(ROOT)), timestamp=datetime.now(timezone.utc).isoformat()))
            if not result['completed']:
                raise ValueError('Prepared account changed behavior: '+name+' '+result.get('reason', ''))
            summaries[name] = result['summary']
            print('replayed', name, flush=True)
        verify_hashes(expected)
        if before != identity(cache, expected):
            raise ValueError('Prepared sources changed during replay')
        write(output/'manifest.json', dict(identity=before, all_accounts_completed=True,
            live_qualified=False, summaries=summaries,
            files_sha256={p.name:digest(p) for p in sorted(output.glob('*.json'))}))


def compare(cache, folders):
    if folders[0] == folders[1]:
        raise ValueError('Two independent replay directories required')
    with offline_only():
        data, entries, _, expected = study(cache)
        if not check(cache, data, entries, expected)['ready']:
            raise ValueError('Current preflight is not ready')
        manifests = []
        names = {name+'.json' for name, _, _ in jobs(data, entries)}
        for folder in folders:
            manifest = read(folder/'manifest.json')
            if (manifest['identity'] != identity(cache, expected)
                    or manifest.get('all_accounts_completed') is not True
                    or manifest.get('live_qualified') is not False
                    or set(manifest['files_sha256']) != names):
                raise ValueError('Replay manifest is incomplete or changed')
            for name, checksum in manifest['files_sha256'].items():
                if digest(folder/name) != checksum:
                    raise ValueError('Replay output changed: '+name)
            manifests.append(manifest)
        if manifests[0] != manifests[1]:
            raise ValueError('Independent replays differ')
        write(cache/'reproducibility.json', dict(passed=True, live_qualified=False,
            runs=[dict(path=str(f.relative_to(ROOT)), manifest_sha256=digest(f/'manifest.json')) for f in folders],
            scope='Fourteen complete accounts, identical daily-model results. Not live fill qualification.'))
        print('All fourteen accounts reproduce exactly in two offline runs.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=DEFAULT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--check', action='store_true')
    group.add_argument('--run', type=Path)
    group.add_argument('--compare', nargs=2, type=Path)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    cache = args.cache.resolve()
    if not cache.is_relative_to(ROOT/'.cache') or (args.run and not args.run.resolve().is_relative_to(ROOT/'.cache')):
        parser.error('Use project .cache paths')
    if args.compare and any(not p.resolve().is_relative_to(ROOT/'.cache') for p in args.compare):
        parser.error('Use project .cache paths')
    if args.fetch and not args.prepare:
        parser.error('--fetch requires --prepare')
    with file_lock(ROOT/'.cache/holder-flow-source-preflight.lock', timeout=0):
        if args.prepare:
            prepare(cache, args.fetch)
        elif args.check:
            with offline_only():
                data, entries, _, expected = study(cache)
                result = check(cache, data, entries, expected)
            sys.exit(0 if result['ready'] else 2)
        elif args.compare:
            compare(cache, [p.resolve() for p in args.compare])
        else:
            replay(cache, args.run.resolve())
