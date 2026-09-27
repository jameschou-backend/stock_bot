#!/usr/bin/env python3
"""Frozen official-document strategy using the existing integer-share account."""
from dataclasses import replace
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from scripts.audit_current_causality_20260925 import BASE, ORIGINAL, QUARANTINE, provenance, verify_hashes
from scripts.research_exit_scenarios import read, write, sha
from scripts.research_sector_accounts import corporate_overrides
from scripts.research_cash_allocation import INPUT
from skills.theme_catalyst import START, END, COHORT, ARMS, build_signals, validate_ledger
from skills.sector_account_replay import SectorAccountInputs, run_case
from skills.scenario_exit_replay import ExitSignals
from skills.verified_backtest_tool import offline_only, source_context
from skills.replay_market_feeds import ReplayMarketFeeds
from skills.trial_registry import append_trial_registry
from skills.execution_factorial import stock_pnl
from scripts.research_surge_anatomy import clean

SOURCE_DOCS = [ROOT/'docs/leo_catalyst_sources_20260927.json', ROOT/'docs/leo_peer_sources_20260927.json']
SPEC = ROOT/'docs/prereg_theme_catalyst_20260927.md'
CACHE = ROOT/'.cache/theme-catalyst-sources-20260927/inputs'
ADDITIONS = [ROOT/'docs/backtest_corporate_completion_20260925.json']


def source_closure(cache):
    """Verify our preparation receipt, inherited bytes, and the actual engine."""
    identity, _ = source_context()
    expected = {ROOT/k:v for k,v in identity['source_sha256'].items()}
    ReplayMarketFeeds(cache/'execution-feeds', offline=True).manifest()
    receipt = read(cache.parent/'preparation.json')
    if receipt['preparation_code_sha256'] != sha(ROOT/'scripts/prepare_theme_catalyst.py'):
        raise ValueError('Preparation code changed; new preparation required')
    if (receipt['signal_code_sha256'] != sha(ROOT/'skills/theme_catalyst.py') or
            receipt['source_documents_sha256'] != {str(p.relative_to(ROOT)):sha(p) for p in SOURCE_DOCS}):
        raise ValueError('Preparation used a different signal recipe or document ledger')
    parent = read(cache.parent/'parent.json')
    for name, digest in parent['parent_files_sha256'].items():
        if sha(ROOT/parent['parent']/name) != digest:
            raise ValueError('Preparation parent changed: '+name)
        if name != 'execution-feeds/index.json' and sha(cache/name) != digest:
            raise ValueError('Inherited source bytes changed: '+name)
    budget = read(cache.parent/'budget.json')
    if budget['maximum'] != {'finmind':60, 'official':80} or any(
            type(budget['attempts'][k]) is not int or not 0 <= budget['attempts'][k] <= maximum
            for k,maximum in budget['maximum'].items()):
        raise ValueError('Preparation exceeded registered request budget')
    for p in cache.rglob('*'):
        if p.is_file() and p.suffix != '.lock':
            expected[p] = sha(p)
    extras = [cache.parent/n for n in ('parent.json','budget.json','preparation.json')]
    extras += [ROOT/p for p in ('skills/sector_account_replay.py','skills/conservative_diversification.py',
        'skills/slot_reuse_replay.py','skills/pending_share_entitlements.py','skills/trial_registry.py',
        'scripts/research_sector_accounts.py','scripts/prepare_sector_account_sources.py')]
    # Resolve overrides using the same collision checks as the parent, then seal
    # every override document and the cited official evidence.
    overrides = corporate_overrides(ADDITIONS)
    from scripts.research_cash_allocation import OVERRIDES
    from scripts.research_chip import ADDITIONS as CHIP
    from scripts.research_board_only_supplement import SOURCES
    for p in [OVERRIDES, CHIP, ROOT/'docs/intraday_corporate_additions_20260914.json',
              SOURCES/'overrides.json', *ADDITIONS]:
        extras.append(p)
        extras.extend(ROOT/name for name in read(p).get('evidence_sha256',{}))
    expected.update({p:sha(p) for p in extras})
    return expected, overrides


def ledger():
    rows = []
    for path in SOURCE_DOCS:
        for incoming in read(path)['events']:
            e = dict(incoming)
            if 'observed_at' not in e:
                e['observed_at'] = e['retrieved_date']
            if len(e['observed_at']) != 10:
                stamp = pd.Timestamp(e['observed_at'])
                if stamp.tzinfo is None:
                    raise ValueError('Collection timestamp requires a time zone')
                e['observed_timestamp'] = e['observed_at']
                e['observed_at'] = str(stamp.tz_convert('Asia/Taipei').date())
            # Pre-April documents remain in source journals, not this experiment.
            if '2025-04-15' <= e['source_date'] <= '2026-01-31':
                rows.append(e)
    validate_ledger(rows)
    return sorted(rows, key=lambda e:(e['source_date'], e['stock_id'], e['event_id']))


def source_documents(events):
    expected = {}
    for e in events:
        relative = e.get('source_local_path')
        if relative is None:
            continue
        path = (ROOT/relative).resolve()
        if not path.is_relative_to(ROOT/'.cache') or sha(path) != e['source_sha256']:
            raise ValueError('Issuer document bytes or path changed: '+e['event_id'])
        expected[path] = e['source_sha256']
    return expected


def inputs():
    expected = provenance()
    pool = ['0050', *COHORT, '3491']
    frames = []
    for name in ('close-official', 'raw-close', 'raw-volume'):
        frame = pd.read_parquet(BASE/(name+'.parquet'), columns=['date', *pool]).set_index('date')
        frame.index = pd.to_datetime(frame.index)
        frames.append(frame)
    refs = read(INPUT/'manifest.json')['references']
    for row in refs.values():
        p = ROOT/row['path']
        if sha(p) != row['sha256']:
            raise ValueError('Changed account source: '+str(p))
        expected[p] = row['sha256']
    quotes = pd.read_parquet(ROOT/refs['quotes']['path'], filters=[('stock_id','in',pool)])
    quotes['date'] = pd.to_datetime(quotes.date)
    for row in read(QUARANTINE)['quarantine']:
        quotes = quotes[~(quotes.stock_id.eq(row['stock_id']) & quotes.date.eq(pd.Timestamp(row['date'])))]
    companies = pd.read_parquet(ROOT/refs['companies']['path'])
    if not set(COHORT+('3491',)).issubset(set(companies.stock_id)):
        raise ValueError('Ordinary company identities missing')
    cal = pd.read_parquet(ROOT/refs['calendar']['path'])
    days = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open,'date']))
    if not days.equals(frames[0].index):
        raise ValueError('Price and account calendars differ')
    events = pd.read_parquet(ROOT/refs['events']['path'])
    data = SectorAccountInputs(quotes, companies, days, {}, events,
        ExitSignals(frames[0], days), '2026-09-27', start=START, end=END)
    return data, frames, expected


def plans(data, frames, events):
    signals = {}
    for augmented in (False, True):
        for delay in (0, 5):
            key = f'{"augmented" if augmented else "primary"}_delay{delay}'
            signals[key] = build_signals(*frames, events, augmented=augmented, delay=delay)
    observed = build_signals(*frames, events, augmented=True, mode='observed')
    if any(observed['entries_by_arm'].values()):
        raise ValueError('Newly observed evidence generated historical signals')
    result = []
    for stress in ('control', 'combined'):
        result.append((f'benchmark_{stress}', data, dict(arm='benchmark', stress=stress,
            benchmark=True, board_only=False, position_count=0)))
    for key, signal in signals.items():
        for arm in ARMS:
            # Parent config selects the established execution engine, not the
            # new strategy identity (which is carried separately in the case key).
            selected = replace(data, entries_by_arm={'relative_strength':signal['entries_by_arm'][arm]})
            for stress in ('control', 'combined'):
                result.append((f'{key}_{arm}_{stress}', selected,
                    dict(arm='relative_strength', stress=stress, benchmark=False, board_only=False, position_count=5)))
    return result, signals, observed


def causality(frames, events):
    checks = []
    base = build_signals(*frames, events, augmented=True)
    for day in ('2025-06-30','2025-09-30','2025-12-31','2026-03-31'):
        cutoff = frames[0].index[frames[0].index <= pd.Timestamp(day)][-1]
        truncated = build_signals(*(f.loc[:cutoff] for f in frames), events, augmented=True)
        mutated = [f.copy() for f in frames]
        for f in mutated:
            f.loc[f.index > cutoff] *= 1.7
        changed = build_signals(*mutated, events, augmented=True)
        past = lambda r:[{k:v for k,v in d.items() if k!='scheduled'} for d in r['decisions'] if d['signal_date'] <= str(cutoff.date())]
        if clean(past(base)) != clean(past(truncated)) or clean(past(base)) != clean(past(changed)):
            raise ValueError('Future prices changed past catalyst decisions')
        future_removed = build_signals(*frames, [e for e in events if e['source_date'] <= str(cutoff.date())], augmented=True)
        if clean(past(base)) != clean(past(future_removed)):
            raise ValueError('Future issuer documents changed past catalyst decisions')
        checks.append(dict(cutoff=str(cutoff.date()), truncation=True, future_mutation=True, future_document_removal=True))
    return checks


def annotate_annual_periods(summary):
    """A first-year subset must not be presented as a whole calendar year."""
    for annual in summary['annual']:
        year = str(annual['year'])
        annual['period_start'] = max(summary['start'], year+'-01-01')
        annual['period_end'] = min(summary['end'], year+'-12-31')
        annual['partial_year'] = (annual['period_start'] != year+'-01-01' or
            annual['period_end'] != year+'-12-31')


def run(output, cache=CACHE):
    output, cache = Path(output).resolve(), Path(cache).resolve()
    if not output.is_relative_to(ROOT/'.cache') or output.exists():
        raise ValueError('A new immutable project cache output is required')
    started = time.perf_counter()
    with file_lock(ROOT/'.cache/theme-catalyst.lock', timeout=0), offline_only():
        data, frames, expected = inputs()
        source_expected, overrides = source_closure(cache)
        expected.update(source_expected)
        own = [*SOURCE_DOCS, SPEC, Path(__file__), ROOT/'skills/theme_catalyst.py',
            ROOT/'tests/test_theme_catalyst.py', ROOT/'scripts/prepare_theme_catalyst.py']
        expected.update({p:sha(p) for p in own})
        events = ledger()
        expected.update(source_documents(events))
        jobs, signals, observed = plans(data, frames, events)
        checks = causality(frames, events)
        output.mkdir(parents=True)
        write(output/'signals.json', clean(signals))
        write(output/'events.json', events)
        cases = {}
        for name, selected, config in jobs:
            append_trial_registry(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                source='theme_catalyst', case=name, output=str(output.relative_to(ROOT)), status='started',
                preregistration_sha256=sha(SPEC)))
            result = run_case(selected, config, cache, overrides)
            result.update(strategy_case=name, historical_first_publication_verified=False)
            if result['completed']:
                annotate_annual_periods(result['summary'])
                account = result['account']
                result['average_cash_fraction'] = sum(r['cash']/r['nav'] for r in account['daily'])/len(account['daily'])
                marks = {r['stock_id']:dict(price=r['price']) for r in account['holdings']}
                result['stock_pnl'] = stock_pnl(account, marks)
                # Whole-account exports retain actual integer board/odd shares.
                for table in ('trades','daily','cash_ledger','holdings'):
                    pd.DataFrame(account[table]).to_csv(output/(name+'-'+table+'.csv'), index=False, encoding='utf-8-sig')
            write(output/(name+'.json'), result)
            row = {k:v for k,v in result.items() if k in ('completed','summary','audit','reason','candidate_count','execution',
                'average_cash_fraction','stock_pnl')}
            if result['completed']:
                row['final_nav'] = result['account']['daily'][-1]['nav']
                row['trade_count'] = len(result['account']['trades'])
            row['result'] = dict(path=str((output/(name+'.json')).relative_to(ROOT)), sha256=sha(output/(name+'.json')))
            cases[name] = row
            print(name, 'completed' if result['completed'] else result.get('reason'), flush=True)
        for name, row in cases.items():
            if name.startswith('benchmark'):
                continue
            bm = cases['benchmark_'+name.rsplit('_',1)[1]]
            row['benchmark_return'] = bm['summary']['total_return'] if bm['completed'] else None
            row['excess_return'] = row['summary']['total_return'] - row['benchmark_return'] if row['completed'] and bm['completed'] else None
        verify_hashes(expected)
        report = clean(dict(schema='theme_catalyst_v1', completed=True, cases=cases,
            all_accounts_completed=all(r['completed'] for r in cases.values()),
            live_qualified=False, adopted=False, unseen_validation=False,
            historical_first_publication_verified=False, valid_unbiased_strategy_evidence=False,
            start=START, end=END, initial_cash=1_000_000, cohort=list(COHORT), augmented_stock='3491',
            events=events, causality_checks=checks, strict_observed_historical_signals=0,
            source_sha256={str(p.relative_to(ROOT)):v for p,v in expected.items()},
            signal_counts={key:{arm:len(rows) for arm,rows in s['entries_by_arm'].items()} for key,s in signals.items()},
            finmind_requests=0, network_calls=0, elapsed_seconds=round(time.perf_counter()-started,3),
            limitations=['文件日是假設可取得日期，尚無首次發布版本；updated文件排除。',
                '固定七檔是公開文章列舉，非完整產業母體；昇達科是已知贏家的另加組。',
                '營運披露未知不等於没有衛星業務；嚴格門檻可能漏掉受惠公司。',
                '日資料與零股日量是成交模型，不保證委託撮合；缺件帳戶不報全期績效。',
                '所有期間已見；沒有校正多次研究、存活者與歷史版本偏差。']))
        write(output/'report.json', report)
        write(output/'manifest.json', dict(schema='theme_catalyst_manifest_v1', source_sha256=report['source_sha256'],
            files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=CACHE)
    args = p.parse_args()
    report = run(args.output, args.cache)
    print(json.dumps(dict(cases=len(report['cases']), completed=sum(r['completed'] for r in report['cases'].values()),
        seconds=report['elapsed_seconds']), ensure_ascii=False))
