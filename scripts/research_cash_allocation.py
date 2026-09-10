#!/usr/bin/env python3
"""Compare three frozen cash policies using the sealed loss12 stock strategy."""
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import argparse
import math
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.config import load_config
from app.file_lock import file_lock
from scripts import research_exit_scenarios as parent
from scripts.prepare_cash_allocation_inputs import prepare as prepare_inputs, verify as verify_inputs
from scripts.research_exit_scenarios import (encoded, sha, read, write, _safe, fingerprint,
    RunInputs, TrackedCorporateActions, audit, summarize, exit_statistics, provider_snapshot)
from skills.cash_allocation_replay import CashAllocationReplay
from skills.million_replay import Replay
from skills.replay_market_feeds import ReplayMarketFeeds
from skills.scenario_exit_replay import ExitSignals

INPUT = ROOT / '.cache/cash-allocation-inputs'
OUTPUT = ROOT / '.cache/cash-allocation'
PARENT = ROOT / '.cache/exit-research'
SPEC = ROOT / 'docs/prereg_cash_allocation_20260910.md'
OVERRIDES = ROOT / 'docs/exit_corporate_overrides_20260910.json'
MODES = ('always_0050', 'cash', 'trend_0050')
MODE_LABELS = {'always_0050': '剩餘資金買0050', 'cash': '剩餘資金保留現金', 'trend_0050': '大盤向上才買0050'}
CODE = (*parent.CODE, 'scripts/research_cash_allocation.py', 'scripts/prepare_cash_allocation_inputs.py',
        'skills/cash_allocation_replay.py', 'app/exit_research.py')
LIMITATIONS = [line for line in parent.LIMITATIONS if not line.startswith(('沿用完整458', '所得閒置'))] + [
    '固定loss12個股規則及完整458個候選；配置改變淨值、交易數量、容量及後續部位，不能當成相同成交清單的純0050報酬歸因。',
    '現金不計息。趨勢配置只看前一市場日0050的120個有效還原收盤均線；OFF才嘗試退出，UNKNOWN暫停新增配置交易。',
    '資產比重是每日收盤比重的算術平均。ETF淨損益是元的帳務歸因，包含成本與股利；不是個股選擇超額報酬。',
]


def verify_sources():
    seed = verify_inputs(INPUT)
    sources = dict(seed['parent_files_sha256'])
    for path in (INPUT / 'manifest.json', SPEC, OVERRIDES):
        sources[str(path.relative_to(ROOT))] = sha(path)
    return dict(schema=1, mode_order=list(MODES), stock_exit_mode='loss12', source_files_sha256=sources,
        code_sha256={name: sha(ROOT / name) for name in CODE},
        runtime_versions={'python': platform.python_version(), **{name: version(name)
            for name in ('numpy', 'pandas', 'pyarrow', 'requests', 'PyYAML', 'python-dotenv')}},
        child_cache_directory=str(INPUT.resolve()))


def _stamps(context):
    result = {}
    for name in set(context['source_files_sha256']) | set(context['code_sha256']):
        s = _safe(ROOT, name).stat()
        result[name] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    return result


def _stable(context, stamps):
    if _stamps(context) != stamps:
        raise ValueError('Cash-allocation source or code changed during execution')


def load_inputs():
    refs = read(INPUT / 'manifest.json')['references']
    path = lambda name: _safe(ROOT, refs[name]['path'])
    entries = read(path('signals'))['entries']
    if (len(entries) != 458 or len({row['event_id'] for row in entries}) != 458
            or any(len(row['members']) != 1 for row in entries)):
        raise ValueError('The complete 458 single-stock candidate entries are required')
    pool = sorted({'0050'} | {row['members'][0] for row in entries})
    quotes = pd.read_parquet(path('quotes'), filters=[('stock_id', 'in', pool)])
    companies = pd.read_parquet(path('companies'))
    calendar = pd.read_parquet(path('calendar'))
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, 'date']))
    close = pd.read_parquet(path('close_official'), columns=['date', *pool]).set_index('date')
    close.index = pd.to_datetime(close.index)
    previous = {'strategy': read(PARENT / 'cases/loss12.json')['account'],
                'benchmark': read(PARENT / 'cases/benchmark.json')['account']}
    return RunInputs(quotes, companies, days, entries, pd.read_parquet(path('events')),
                     ExitSignals(close, days), previous)


def provider_pair(data, *, offline):
    token = None if offline else load_config().finmind_token
    return (ReplayMarketFeeds(INPUT / 'execution-feeds', offline=offline, token=token),
        TrackedCorporateActions(data.events, INPUT / 'dividends', token, offline=offline,
                                overrides=read(OVERRIDES)['overrides']))


def check_provider_snapshot(snapshot, *, exact=False):
    current = ReplayMarketFeeds(INPUT / 'execution-feeds', offline=True).manifest()
    old = snapshot['execution_feeds']
    if exact and current != old:
        raise ValueError('Sealed execution sources changed')
    if any(current.get(key) != old.get(key) for key in ('schema', 'parser_sha256', 'limitations', 'cache_directory')):
        raise ValueError('Execution provider identity changed')
    for key in ('entries', 'files_sha256'):
        if any(current.get(key, {}).get(name) != value for name, value in old.get(key, {}).items()):
            raise ValueError('Completed-case execution evidence changed')
    for key, value in old.get('request_counters', {}).items():
        count = current.get('request_counters', {}).get(key)
        if type(count) is not int or count < value:
            raise ValueError('Execution request counter reset')
    for name, digest in snapshot['corporate_sources']['files_sha256'].items():
        if sha(_safe(INPUT / 'dividends', name)) != digest:
            raise ValueError('Completed-case dividend source changed')
    if snapshot['corporate_sources']['overrides'] != read(OVERRIDES)['overrides']:
        raise ValueError('Completed-case corporate facts changed')


def allocation_statistics(account):
    """Reconcile daily exposure and terminal dollar P&L; no alpha attribution."""
    holdings = defaultdict(lambda: {'etf': 0., 'stock': 0.})
    final_day = account['daily'][-1]['date']
    final_prices = {}
    for row in account['holdings']:
        holdings[row['date']]['etf' if row['stock_id'] == '0050' else 'stock'] += row['market_value']
        if row['date'] == final_day:
            final_prices[row['stock_id']] = row['price']
    daily = []
    for row in account['daily']:
        values = dict(cash=row['cash'], etf=holdings[row['date']]['etf'],
                      stock=holdings[row['date']]['stock'], receivable=row['receivable'])
        if not math.isclose(sum(values.values()), row['nav'], abs_tol=.06, rel_tol=0):
            raise ValueError('Exposure amounts do not reconcile to NAV')
        daily.append(dict(date=row['date'], **{key + '_weight': value / row['nav'] for key, value in values.items()}))
    means = {'mean_' + key + '_weight': sum(row[key + '_weight'] for row in daily) / len(daily)
             for key in ('cash', 'etf', 'stock', 'receivable')}
    trades = [row for row in account['trades'] if row['stock_id'] == '0050']
    trade_flow = sum(row['cash_change'] for row in trades)
    # Entitlements and their payments are distinct journals. Count only paid
    # cash here, and only still-outstanding rights below; never both.
    paid = sum(row['cash_change'] for row in account['cash_ledger']
               if row.get('stock_id') == '0050' and row['kind'] in ('dividend_payment', 'fractional_share_payment'))
    rights = 0.
    for row in account['receivables']:
        if row['stock_id'] != '0050':
            continue
        if row['kind'] == 'cash':
            rights += row['amount']
        else:
            if row.get('qty', 0) and '0050' not in final_prices:
                raise ValueError('ETF shares receivable has no final valuation price')
            rights += row.get('qty', 0) * final_prices.get('0050', 0)
            rate = row.get('fractional_cash_per_share')
            if row.get('fraction', 0) and rate is None:
                raise ValueError('ETF fractional receivable lacks settlement terms')
            rights += row.get('fraction', 0) * (rate or 0)
    final_etf = holdings[final_day]['etf']
    # All initial capital is cash. Net ETF cash flows + terminal ETF assets
    # form a self-contained subledger even when funding stocks mid-period.
    etf_pnl = trade_flow + paid + final_etf + rights
    stock_buys = [row for row in account['trades'] if row['stock_id'] != '0050' and row['side'] == 'buy']
    return dict(**means, daily_exposure=daily, etf_buy_count=sum(row['side'] == 'buy' for row in trades),
        etf_sell_count=sum(row['side'] == 'sell' for row in trades), etf_trade_count=len(trades),
        etf_total_cost=sum(row['total_cost'] for row in trades), etf_net_trade_cashflow=trade_flow,
        etf_cash_distributions_paid=paid, etf_final_market_value=final_etf,
        etf_outstanding_receivable_value=rights, etf_net_pnl=etf_pnl,
        stock_and_other_net_pnl=account['daily'][-1]['nav']-account['settings']['initial_cash']-etf_pnl,
        attribution_unit='NTD', attribution_is_selection_alpha=False,
        minimum_stock_entry_prior_avg_amount=min((row['prior_avg_amount20'] for row in stock_buys), default=None))


def run_case(mode, data, feeds, corp):
    args = (data.quotes, data.companies, data.days, data.entries, feeds, corp)
    kwargs = dict(start=data.start, end=data.end)
    if mode == 'benchmark':
        engine = Replay(*args, benchmark=True, **kwargs)
    else:
        class ProgressReplay(CashAllocationReplay):
            progress_month = None

            def corporate_day(self, day):
                month = str(pd.Timestamp(day).date())[:7]
                if not feeds.offline and month != self.progress_month:
                    print(mode + ': preparing ' + month, flush=True)
                    self.progress_month = month
                return super().corporate_day(day)
        engine = ProgressReplay(*args, exit_signals=data.features, allocation_mode=mode, **kwargs)
    account = engine.run()
    checked = audit(account)
    if mode in ('always_0050', 'benchmark'):
        if encoded(account) != encoded(data.parent['strategy' if mode == 'always_0050' else 'benchmark']):
            raise ValueError(mode + ' does not exactly reproduce the sealed parent account')
    states, decisions = getattr(engine, 'exit_states', {}), getattr(engine, 'exit_decisions', [])
    summary = {**summarize(account), **exit_statistics(account, states, data.days),
               'allocation': allocation_statistics(account)}
    return dict(schema=1, mode=mode, stock_exit_mode='loss12', exit_mode='loss12',
        label='0050持有對照' if mode == 'benchmark' else MODE_LABELS[mode], account=account,
        exit_states=states, exit_decisions=decisions, allocation_decisions=getattr(engine, 'allocation_decisions', []),
        summary=summary, audit=checked)


def _case_paths(output, mode):
    if mode not in (*MODES, 'benchmark'):
        raise ValueError('Unknown checkpoint mode')
    return output / 'cases' / f'{mode}.json', output / 'cases' / f'{mode}.manifest.json'


def save_case(output, case, context, sources):
    path, meta = _case_paths(output, case['mode'])
    write(path, case)
    write(meta, dict(schema=1, mode=case['mode'], context_fingerprint=fingerprint(context),
        case_file=str(path.relative_to(output)), case_sha256=sha(path), sources=sources))


def load_case(output, mode, context):
    path, meta = _case_paths(output, mode)
    if not meta.exists():
        return None
    manifest = read(meta)
    if (manifest.get('schema') != 1 or manifest.get('mode') != mode
            or manifest.get('context_fingerprint') != fingerprint(context)
            or manifest.get('case_file') != str(path.relative_to(output)) or sha(path) != manifest.get('case_sha256')):
        raise ValueError('Completed case changed or has incompatible code/spec/source: ' + mode)
    check_provider_snapshot(manifest['sources'])
    case = read(path)
    if case.get('mode') != mode or case.get('stock_exit_mode') != 'loss12':
        raise ValueError('Checkpoint identity mismatch')
    audit(case['account'])
    return case


def build_report(cases, benchmark, context, timings):
    control, base = cases['always_0050']['summary'], benchmark['summary']
    comparisons, annual = [], []
    for mode in MODES:
        row = cases[mode]['summary']
        allocation = row['allocation']
        comparisons.append(dict(mode=mode, label=MODE_LABELS[mode],
            **{key: row[key] for key in ('final_nav', 'total_return', 'cagr', 'max_drawdown', 'trade_count', 'stock_cohorts', 'reason_counts')},
            total_cost=row['costs']['total_cost'],
            excess_total_return_vs_always_0050=row['total_return']-control['total_return'],
            excess_total_return_vs_0050=row['total_return']-base['total_return'],
            drawdown_difference_vs_always_0050=row['max_drawdown']-control['max_drawdown'],
            depth_flag=row['depth_audit']['flag'],
            **{key: value for key, value in allocation.items() if key != 'daily_exposure'}))
        annual.extend(dict(mode=mode, label=MODE_LABELS[mode], **item) for item in row['annual'])
    annual.extend(dict(mode='benchmark', label='0050持有對照', **item) for item in base['annual'])
    common = dict(schema=1, mode_order=list(MODES), stock_exit_mode='loss12', exit_mode='loss12',
        performance=timings, limitations=LIMITATIONS, candidate_count=458, unseen_validation=False,
        live_qualified=False, auto_promote=False, execution_signal_lag_market_sessions=1)
    summary = dict(**common, labels=MODE_LABELS, comparisons=comparisons, annual=annual, benchmark=base,
        case_files={mode: f'cases/{mode}.json' for mode in MODES}, benchmark_case_file='cases/benchmark.json',
        complete_exit_date_definition='含後續配股的部位全部結束日期；不是第一次原持股清零日期。')
    return dict(**common, cases=cases, benchmark=benchmark, summary=summary, context_fingerprint=fingerprint(context))


def output_names():
    return ['report.json', 'summary.json', 'run.json', 'preparation_sessions.json',
        *[f'cases/{mode}{suffix}' for mode in (*MODES, 'benchmark') for suffix in ('.json', '.manifest.json')]]


def verification_files(context, sources, output, hashes):
    result = {}
    def add(path, digest):
        name = str(Path(path).relative_to(ROOT))
        if name in result and result[name] != digest:
            raise ValueError('Conflicting source hashes: ' + name)
        result[name] = digest
    for mapping in (context['source_files_sha256'], context['code_sha256']):
        for name, digest in mapping.items():
            add(ROOT / name, digest)
    for name, digest in read(INPUT / 'manifest.json')['seed_files_sha256'].items():
        add(INPUT / name, digest)
    for name, digest in sources['execution_feeds']['files_sha256'].items():
        add(INPUT / 'execution-feeds' / name, digest)
    add(INPUT / 'execution-feeds/index.json', sources['execution_feeds']['manifest_sha256'])
    for path in sorted((INPUT / 'dividends').glob('*.parquet')):
        add(path, sha(_safe(INPUT / 'dividends', path.name)))
    for name, digest in hashes.items():
        add(output / name, digest)
    return result


def preparation_usage(output, sources):
    seed = read(INPUT / 'manifest.json')
    counts = sources['execution_feeds']['request_counters']
    delta = {key: value-seed['seed_execution_index']['request_counters'].get(key, 0) for key, value in counts.items()}
    sessions = read(output / 'preparation_sessions.json')['sessions']
    corporate = [row['corporate_requests'] for row in sessions.values()]
    if any(type(value) is not int or value < 0 for value in [*delta.values(), *corporate]):
        raise ValueError('Invalid or reset preparation request counter')
    return dict(online_preparation_corporate_requests=sum(corporate), execution_feed_requests_since_seed=delta,
        online_preparation_corporate_requests_scope='Recorded successful driver fetches, including resumed sessions; excludes failed calls with unknown charge, external preflight and other processes.',
        offline_api_requests=0, seed_dividend_count=len(seed['seed_dividend_hashes']),
        total_child_dividend_count=len(list((INPUT / 'dividends').glob('*.parquet'))),
        elapsed_time_scope='This invocation only; input cloning, earlier attempts and external source research are excluded.')


def verify_report(output=OUTPUT):
    output = Path(output).resolve()
    meta = read(output / 'manifest.json')
    if (meta.get('schema') != 1 or meta.get('stock_exit_mode') != 'loss12' or meta.get('offline_identical') is not True
            or any(meta.get(key) is not False for key in ('live_qualified', 'unseen_validation', 'auto_promote'))
            or set(meta.get('files_sha256', {})) != set(output_names())
            or meta.get('verification_directories') != [str((INPUT / 'dividends').relative_to(ROOT))]):
        raise ValueError('Incomplete sealed cash-allocation manifest')
    context = verify_sources()
    if context != meta['context'] or read(output / 'run.json')['context'] != context:
        raise ValueError('Cash-allocation code, runtime, spec or input changed')
    for name, digest in meta['files_sha256'].items():
        if sha(_safe(output, name)) != digest:
            raise ValueError('Cash-allocation output hash changed: ' + name)
    check_provider_snapshot(meta['sources'], exact=True)
    if verification_files(context, meta['sources'], output, meta['files_sha256']) != meta.get('verification_files_sha256'):
        raise ValueError('Cash-allocation verification closure changed')
    return meta


def _offline_all(data, expected):
    feeds, corp = provider_pair(data, offline=True)
    before = feeds.manifest()
    for mode in ('benchmark', *MODES):
        actual = run_case(mode, data, feeds, corp)
        previous = expected['benchmark'] if mode == 'benchmark' else expected['cases'][mode]
        if encoded(actual) != encoded(previous):
            raise ValueError('Offline account/decision/state reproduction differs: ' + mode)
        print(mode + ': identical offline reproduction', flush=True)
    sources = provider_snapshot(feeds, corp)
    if sources['execution_feeds'] != before or sources['corporate_sources']['requests'] != 0:
        raise ValueError('Offline replay changed execution cache or used the API')
    return sources


def research(output=OUTPUT, *, offline=False):
    output = Path(output).absolute()
    if '..' in output.parts or not output.is_relative_to(ROOT):
        raise ValueError('Research output must be inside the repository')
    for protected in (INPUT, PARENT, ROOT / '.cache/exit-research-inputs', ROOT / '.cache/million-replay',
                      ROOT / '.cache/million-replay-inputs', ROOT / '.cache/million-replay-signals'):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Research output overlaps immutable or input evidence')
    if any(path.is_symlink() for path in (output, *output.parents) if path == ROOT or ROOT in path.parents):
        raise ValueError('Research output must not use symlinks')
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / '.research.lock', timeout=10):
        if offline:
            manifest = verify_report(output)
            report = read(output / 'report.json')
            _offline_all(load_inputs(), report)
            verify_report(output)
            return report
        if (output / 'manifest.json').exists():
            raise ValueError('Sealed cash-allocation research exists; use --offline-replay or a new --output')
        prepare_inputs(destination=INPUT)
        context = verify_sources()
        stamps = _stamps(context)
        if (output / 'run.json').exists():
            if read(output / 'run.json').get('context') != context:
                raise ValueError('Checkpoint code/spec/source changed; use a new explicit --output')
        else:
            write(output / 'run.json', dict(schema=1, context=context, created_at=datetime.now(timezone.utc).isoformat()))
        session_path = output / 'preparation_sessions.json'
        if not session_path.exists():
            write(session_path, dict(schema=1, sessions={}))
        sessions = read(session_path)
        data, cases = load_inputs(), {}
        started = time.perf_counter()
        local = provider_pair(data, offline=True)
        online = None
        for mode in ('benchmark', *MODES):
            _stable(context, stamps)
            case = load_case(output, mode, context)
            if case is not None:
                if mode in ('benchmark', 'always_0050'):
                    expected = data.parent['benchmark' if mode == 'benchmark' else 'strategy']
                    if encoded(case['account']) != encoded(expected):
                        raise ValueError('Resumed control differs from parent')
                print(mode + ': resumed verified completed case', flush=True)
            else:
                if mode in ('benchmark', 'always_0050'):
                    feeds, corp = local
                else:
                    if online is None:
                        online = provider_pair(data, offline=False)
                        session_id = uuid.uuid4().hex
                        def record(count):
                            sessions['sessions'][session_id] = {'corporate_requests': count}
                            write(session_path, sessions)
                        online[1].request_observer = record
                        record(online[1].requests)
                    feeds, corp = online
                print(mode + ': preparing account and execution evidence', flush=True)
                case = run_case(mode, data, feeds, corp)
                _stable(context, stamps)
                save_case(output, case, context, provider_snapshot(feeds, corp))
            if mode == 'benchmark':
                benchmark = case
            else:
                cases[mode] = case
        elapsed = time.perf_counter() - started
        if verify_sources() != context:
            raise ValueError('Sources changed before offline verification')
        before = time.perf_counter()
        sources = _offline_all(data, {'benchmark': benchmark, 'cases': cases})
        report = build_report(cases, benchmark, context, dict(preparation_seconds=elapsed,
            offline_seconds=time.perf_counter()-before, **preparation_usage(output, sources)))
        if verify_sources() != context:
            raise ValueError('Sources changed during offline verification')
        write(output / 'report.json', report)
        write(output / 'summary.json', report['summary'])
        hashes = {name: sha(output / name) for name in output_names()}
        write(output / 'manifest.json', dict(schema=1, stock_exit_mode='loss12',
            created_at=datetime.now(timezone.utc).isoformat(), context=context, sources=sources,
            files_sha256=hashes, verification_files_sha256=verification_files(context, sources, output, hashes),
            verification_directories=[str((INPUT / 'dividends').relative_to(ROOT))],
            offline_identical=True, live_qualified=False, unseen_validation=False, auto_promote=False))
        verify_report(output)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-and-replay', action='store_true')
    group.add_argument('--offline-replay', action='store_true')
    group.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.verify:
        with file_lock(args.output / '.research.lock', timeout=10):
            verify_report(args.output)
        print('Cash-allocation sources, checkpoints and sealed output verified')
    else:
        print(encoded(research(args.output, offline=args.offline_replay)['summary']['comparisons']))


if __name__ == '__main__':
    main()
