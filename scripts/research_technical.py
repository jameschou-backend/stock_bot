#!/usr/bin/env python3
"""Compare preregistered support exits, risk sizing, adding and pattern filtering."""
from collections import Counter
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
from scripts import research_cash_allocation as parent
from scripts.prepare_technical_inputs import prepare as prepare_inputs, verify as verify_inputs
from scripts.research_exit_scenarios import (encoded, sha, read, write, _safe, fingerprint,
    RunInputs, TrackedCorporateActions, audit, summarize, exit_statistics, provider_snapshot)
from skills.technical_replay import TechnicalReplay
from skills.million_replay import Replay
from skills.replay_market_feeds import ReplayMarketFeeds
from skills.technical_signals import TechnicalSignals

INPUT = ROOT / '.cache/technical-inputs'
OUTPUT = ROOT / '.cache/technical-research'
PARENT = ROOT / '.cache/cash-allocation'
SPEC = ROOT / 'docs/prereg_technical_20260910.md'
OVERRIDES = ROOT / 'docs/technical_corporate_overrides_20260910.json'
SOURCE_DOCS = (ROOT / 'docs/technical_corporate_sources_2834_20260910.md',)
MODES = ('control', 'support20', 'risk2', 'support_risk2', 'support_risk2_add', 'support_risk2_pattern')
MODE_LABELS = {'control': '原策略：12%停損、等額配置', 'support20': '加入支撐失效出場',
    'risk2': '改用2%風險配置', 'support_risk2': '支撐出場＋2%風險配置',
    'support_risk2_add': '支撐與風險配置＋強勢加碼', 'support_risk2_pattern': '支撐與風險配置＋突破篩選'}
CODE = (*parent.CODE, 'scripts/research_technical.py', 'scripts/prepare_technical_inputs.py',
        'skills/technical_replay.py', 'skills/technical_signals.py')
LIMITATIONS = [line for line in parent.parent.LIMITATIONS if not line.startswith(('沿用完整458', '所得閒置'))] + [
    '沿用完整458個候選與剩餘資金買0050；三個個股名額與原63日／12%停損規則保留，依預先規格逐項比較支撐、風險配置、加碼與突破篩選。',
    '支撐是落後20個市場日的還原最低價的低點代理；突破篩選是明確量價規則，沒有將主觀頭肩底、W底或三角型態事後貼標籤。',
    '2%是下單當時模型估計的風險預算，不保證實際虧損不超過2%；跳空、跌停、容量與未成交出場都可能造成超額損失。',
    '2%是單檔部位的下單風險預算，不是全帳戶風險上限；剩餘資金0050仍承擔大盤風險。',
    '六組統一修正小額賣出：若出售所得不足原費用，但現金足以支付差額，允許原規則範圍內成交並扣現金；不改稅費、滑價、漲跌停、容量或另賣0050籌費。原策略與0050仍須逐欄重現父研究。',
    '2834的2022配股畸零現金按已核實的取整元規則計算，NT$10面額僅為明示毛額估值假設；若產生正金額，付款日未知而保留應收，不視為已付現金或已確認費後淨額。',
    '加碼與型態篩選各自對照固定支撐＋風險配置，沒有把兩者一起加入或在同期間搜尋最佳參數。',
    '配置與篩選改變淨值、交易數量、容量及後續部位，不能當成相同成交清單的純訊號報酬歸因。',
    '資產比重是每日收盤比重的算術平均；ETF淨損益為含成本股利的帳務歸因，不是個股選擇超額報酬。',
]


def verify_sources():
    seed = verify_inputs(INPUT)
    sources = dict(seed['parent_files_sha256'])
    for path in (INPUT / 'manifest.json', SPEC, OVERRIDES, *SOURCE_DOCS):
        sources[str(path.relative_to(ROOT))] = sha(path)
    return dict(schema=1, mode_order=list(MODES), research_kind='technical', control_exit_mode='loss12', source_files_sha256=sources,
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
        raise ValueError('Technical research source or code changed during execution')


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
    previous = {'strategy': read(PARENT / 'cases/always_0050.json')['account'],
                'benchmark': read(PARENT / 'cases/benchmark.json')['account']}
    return RunInputs(quotes, companies, days, entries, pd.read_parquet(path('events')),
                     TechnicalSignals(close, quotes, days), previous)



def candidate_feature_audit(data):
    """Audit the complete frozen candidate pool before account eligibility gates."""
    rows = []
    for entry in data.entries:
        if len(entry['members']) != 1:
            raise ValueError('Candidate feature audit requires one stock per original event')
        index = data.days.get_indexer([pd.Timestamp(entry['entry_date'])])[0]
        if index < 0:
            raise ValueError('Candidate execution date is outside the audited market calendar')
        context = data.features.technical_context(int(index), entry['members'][0])
        if context['signal_date'] != entry['signal_date']:
            raise ValueError('Candidate technical feature does not use its original prior signal date')
        rows.append(dict(event_id=entry['event_id'], stock_id=entry['members'][0],
            entry_date=entry['entry_date'], original_signal_date=entry['signal_date'], **context))
    fields = ('support_available', 'risk_available', 'breakout20', 'contraction10',
              'volume_expansion', 'pattern_available', 'pattern_pass')
    summary = dict(candidate_count=len(rows), scope='complete_original_candidates_before_portfolio_eligibility',
        diagnostics_counts=dict(Counter(issue for row in rows for issue in row['diagnostics'])))
    for field in fields:
        for value, suffix in ((True, 'true'), (False, 'false'), (None, 'unknown')):
            summary[field + '_' + suffix + '_count'] = sum(row[field] is value for row in rows)
    return dict(schema=1, summary=summary, rows=rows,
        units={'adjusted_close': 'same-row adjusted close price',
               'support20': 'minimum adjusted low from the 20 market sessions before the signal day',
               'resistance20': 'maximum adjusted high from the 20 market sessions before the signal day',
               'raw_close': 'NTD/share', 'support_raw': 'NTD/share at the signal-day adjustment ratio',
               'risk_fraction': 'support price-distance fraction; excludes costs and is not risk2 planned loss',
               'raw_volume': 'shares', 'volume20': 'shares/session'},
        counts_are_orders_or_fills=False)

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
    return parent.allocation_statistics(account)


def technical_statistics(engine, account=None):
    """Count checks, requested orders and successful fills separately."""
    result = {}
    for name in ('exit', 'sizing', 'add', 'pattern'):
        rows = getattr(engine, name + '_decisions', [])
        result[name + '_decision_count'] = len(rows)
        for field in ('action', 'reason', 'phase', 'failure'):
            result[name + '_' + field + '_counts'] = dict(Counter(
                str(row[field]) for row in rows if isinstance(row, dict) and row.get(field) is not None))
    sizing = getattr(engine, 'sizing_decisions', [])
    adds = getattr(engine, 'add_decisions', [])
    patterns = getattr(engine, 'pattern_decisions', [])
    result.update(sizing_filled_count=sum(row.get('filled_qty', 0) > 0 for row in sizing),
        sizing_attempt_count=sum(row.get('requested_qty', 0) > 0 for row in sizing),
        sizing_requested_qty=sum(row.get('requested_qty', 0) for row in sizing),
        sizing_fill_qty=sum(row.get('filled_qty', 0) for row in sizing),
        add_attempt_count=sum(row.get('requested_qty', 0) > 0 for row in adds),
        add_successful_cohorts=len({row['event_id'] for row in adds if row.get('filled_qty', 0) > 0}),
        add_requested_qty=sum(row.get('requested_qty', 0) for row in adds),
        add_fill_qty=sum(row.get('filled_qty', 0) for row in adds),
        pattern_evaluated_count=len(patterns),
        pattern_pass_count=sum(row.get('pattern_pass') is True for row in patterns),
        pattern_fail_count=sum(row.get('pattern_pass') is False for row in patterns),
        pattern_unknown_count=sum(row.get('pattern_pass') is None for row in patterns))
    if account is not None:
        add_trades = [row for row in account['trades'] if row['reason'] == 'pyramid_add']
        funding = [row for row in account['trades'] if row['reason'] == 'fund_stock_add']
        fee_funded = [row for row in account['trades'] if row['side'] == 'sell' and row['cash_change'] < 0]
        zero_proceeds = [row for row in account['trades'] if row['side'] == 'sell' and row['cash_change'] == 0]
        result.update(add_filled_trade_count=len(add_trades),
            add_stock_trade_cost=sum(row['total_cost'] for row in add_trades),
            add_funding_etf_trade_cost=sum(row['total_cost'] for row in funding),
            fee_funded_exit_trade_count=len(fee_funded),
            fee_funded_exit_cash_paid=-sum(row['cash_change'] for row in fee_funded),
            zero_proceeds_exit_trade_count=len(zero_proceeds),
            nonpositive_proceeds_exit_trade_count=len(fee_funded)+len(zero_proceeds))
    return result


def audit_technical(engine, account, days, mode):
    """Reject causal, risk-budget, support-ratchet and fill-journal regressions."""
    positions = {str(pd.Timestamp(day).date()): i for i, day in enumerate(days)}
    rows_by_name = {name: getattr(engine, name + '_decisions', [])
                   for name in ('exit', 'sizing', 'add', 'pattern')}
    for name, rows in rows_by_name.items():
        for row in rows:
            day, signal = row.get('date'), row.get('signal_date')
            if day not in positions or signal not in positions or positions[day] != positions[signal] + 1:
                raise ValueError('Technical ' + name + ' decision must use exactly the prior market session')
    risk_modes = ('risk2', 'support_risk2', 'support_risk2_add', 'support_risk2_pattern')
    risk_ratios = []
    for name, reason in (('sizing', 'leader_entry'), ('add', 'pyramid_add')):
        for row in rows_by_name[name]:
            requested, filled = row.get('requested_qty'), row.get('filled_qty')
            if any(type(value) is not int or value < 0 for value in (requested, filled)) or filled > requested:
                raise ValueError('Technical decision has invalid requested or filled shares')
            matched = [trade for trade in account['trades'] if trade['date'] == row['date']
                and trade['event_id'] == row['event_id'] and trade['stock_id'] == row['stock_id']
                and trade['side'] == 'buy' and trade['reason'] == reason]
            if sum(trade['qty'] for trade in matched) != filled:
                raise ValueError('Technical decision filled quantity does not match actual trades')
            if requested and (mode in risk_modes or name == 'add'):
                planned, cap, nav = row.get('planned_risk'), row.get('risk_cap'), row.get('prior_nav')
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in (planned, cap, nav)):
                    raise ValueError('Technical risk plan is missing finite risk, cap or prior NAV')
                if nav <= 0 or cap < 0 or planned < 0 or not math.isclose(cap, nav*.02, abs_tol=1e-6, rel_tol=0) or planned > cap + 1e-6:
                    raise ValueError('Technical requested risk exceeds the preregistered 2% prior-NAV cap')
                risk_ratios.append(planned / nav)
    successful = [row['event_id'] for row in rows_by_name['add'] if row['filled_qty'] > 0]
    if len(successful) != len(set(successful)):
        raise ValueError('A stock cohort has more than one successful add')
    if mode != 'support_risk2_add' and rows_by_name['add']:
        raise ValueError('Adding decisions appeared outside the preregistered adding case')
    support = {}
    for row in rows_by_name['exit']:
        if 'support_floor' not in row:
            continue
        identity, floor = row['event_id'], row['support_floor']
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0:
            raise ValueError('Technical support floor must remain finite and positive')
        if identity in support and floor < support[identity]:
            raise ValueError('Technical support floor moved down within a cohort')
        support[identity] = floor
    for identity, floor in support.items():
        if getattr(engine, 'exit_states', {}).get(identity, {}).get('support_floor') != floor:
            raise ValueError('Final technical support state disagrees with decision history')
    return dict(checked_decisions=sum(map(len, rows_by_name.values())),
        risk_checked_positive_requests=len(risk_ratios),
        maximum_planned_risk_fraction=max(risk_ratios, default=None),
        support_ratchet_cohorts=len(support), successful_add_cohorts=len(successful),
        actual_loss_is_capped=False)


def run_case(mode, data, feeds, corp):
    args = (data.quotes, data.companies, data.days, data.entries, feeds, corp)
    kwargs = dict(start=data.start, end=data.end)
    if mode == 'benchmark':
        engine = Replay(*args, benchmark=True, **kwargs)
    else:
        class ProgressReplay(TechnicalReplay):
            progress_month = None

            def corporate_day(self, day):
                month = str(pd.Timestamp(day).date())[:7]
                if not feeds.offline and month != self.progress_month:
                    print(mode + ': preparing ' + month, flush=True)
                    self.progress_month = month
                return super().corporate_day(day)
        engine = ProgressReplay(*args, technical_signals=data.features, mode=mode, **kwargs)
    account = engine.run()
    checked = dict(account=audit(account), technical=audit_technical(engine, account, data.days, mode))
    if mode in ('control', 'benchmark'):
        if encoded(account) != encoded(data.parent['strategy' if mode == 'control' else 'benchmark']):
            raise ValueError(mode + ' does not exactly reproduce the sealed parent account')
    states, decisions = getattr(engine, 'exit_states', {}), getattr(engine, 'exit_decisions', [])
    summary = {**summarize(account), **exit_statistics(account, states, data.days),
               'allocation': allocation_statistics(account), 'technical': technical_statistics(engine, account)}
    return dict(schema=1, mode=mode, research_kind='technical', control_exit_mode='loss12',
        label='0050持有對照' if mode == 'benchmark' else MODE_LABELS[mode], account=account,
        exit_states=states, exit_decisions=decisions,
        sizing_decisions=getattr(engine, 'sizing_decisions', []),
        add_decisions=getattr(engine, 'add_decisions', []),
        pattern_decisions=getattr(engine, 'pattern_decisions', []),
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
    if case.get('mode') != mode or case.get('research_kind') != 'technical' or case.get('control_exit_mode') != 'loss12':
        raise ValueError('Checkpoint identity mismatch')
    audit(case['account'])
    return case


def build_report(cases, benchmark, context, timings, candidates):
    control, base = cases['control']['summary'], benchmark['summary']
    comparisons, annual = [], []
    for mode in MODES:
        row = cases[mode]['summary']
        comparison_mode = 'support_risk2' if mode in ('support_risk2_add', 'support_risk2_pattern') else 'control'
        comparison = cases[comparison_mode]['summary']
        allocation = row['allocation']
        comparisons.append(dict(mode=mode, label=MODE_LABELS[mode], comparison_base_mode=comparison_mode,
            **{key: row[key] for key in ('final_nav', 'total_return', 'cagr', 'max_drawdown', 'trade_count', 'stock_cohorts', 'reason_counts')},
            total_cost=row['costs']['total_cost'],
            excess_total_return_vs_base=row['total_return']-comparison['total_return'],
            excess_total_return_vs_control=row['total_return']-control['total_return'],
            excess_total_return_vs_0050=row['total_return']-base['total_return'],
            drawdown_difference_vs_base=row['max_drawdown']-comparison['max_drawdown'],
            drawdown_difference_vs_control=row['max_drawdown']-control['max_drawdown'],
            depth_flag=row['depth_audit']['flag'], **row['technical'],
            **{key: value for key, value in allocation.items() if key != 'daily_exposure'}))
        annual.extend(dict(mode=mode, label=MODE_LABELS[mode], **item) for item in row['annual'])
    annual.extend(dict(mode='benchmark', label='0050持有對照', **item) for item in base['annual'])
    common = dict(schema=1, mode_order=list(MODES), research_kind='technical', control_exit_mode='loss12',
        performance=timings, limitations=LIMITATIONS, candidate_count=candidates['summary']['candidate_count'], unseen_validation=False,
        live_qualified=False, auto_promote=False, execution_signal_lag_market_sessions=1)
    summary = dict(**common, labels=MODE_LABELS, comparisons=comparisons, annual=annual, benchmark=base,
        candidate_feature_summary=candidates['summary'], candidate_feature_file='candidate_features.json',
        comparison_groups={'support_and_sizing': ['control', 'support20', 'risk2', 'support_risk2'],
                           'adding': ['support_risk2', 'support_risk2_add'],
                           'pattern': ['support_risk2', 'support_risk2_pattern']},
        case_files={mode: f'cases/{mode}.json' for mode in MODES}, benchmark_case_file='cases/benchmark.json',
        complete_exit_date_definition='含後續配股的部位全部結束日期；不是第一次原持股清零日期。')
    return dict(**common, cases=cases, benchmark=benchmark, summary=summary,
                candidate_features=candidates, context_fingerprint=fingerprint(context))


def output_names():
    return ['report.json', 'summary.json', 'run.json', 'preparation_sessions.json', 'candidate_features.json',
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
        online_preparation_corporate_requests_scope='Recorded successful driver fetches, resumed sessions and explicitly registered source-preflight sessions; excludes unregistered external fetches, failed calls with unknown charge and other processes.',
        offline_api_requests=0, seed_dividend_count=len(seed['seed_dividend_hashes']),
        total_child_dividend_count=len(list((INPUT / 'dividends').glob('*.parquet'))),
        elapsed_time_scope='This invocation only; input cloning, earlier attempts and external source research are excluded.')


def verify_report(output=OUTPUT):
    output = Path(output).resolve()
    meta = read(output / 'manifest.json')
    if (meta.get('schema') != 1 or meta.get('research_kind') != 'technical' or meta.get('control_exit_mode') != 'loss12' or meta.get('offline_identical') is not True
            or any(meta.get(key) is not False for key in ('live_qualified', 'unseen_validation', 'auto_promote'))
            or set(meta.get('files_sha256', {})) != set(output_names())
            or meta.get('verification_directories') != [str((INPUT / 'dividends').relative_to(ROOT))]):
        raise ValueError('Incomplete sealed technical research manifest')
    context = verify_sources()
    if context != meta['context'] or read(output / 'run.json')['context'] != context:
        raise ValueError('Technical research code, runtime, spec or input changed')
    for name, digest in meta['files_sha256'].items():
        if sha(_safe(output, name)) != digest:
            raise ValueError('Technical research output hash changed: ' + name)
    check_provider_snapshot(meta['sources'], exact=True)
    if verification_files(context, meta['sources'], output, meta['files_sha256']) != meta.get('verification_files_sha256'):
        raise ValueError('Technical research verification closure changed')
    return meta


def _offline_all(data, expected):
    if encoded(candidate_feature_audit(data)) != encoded(expected['candidate_features']):
        raise ValueError('Offline candidate feature audit reproduction differs')
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
    for protected in (INPUT, PARENT, ROOT / '.cache/cash-allocation-inputs', ROOT / '.cache/exit-research', ROOT / '.cache/exit-research-inputs', ROOT / '.cache/million-replay',
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
            raise ValueError('Sealed technical research exists; use --offline-replay or a new --output')
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
        candidates = candidate_feature_audit(data)
        candidate_path = output / 'candidate_features.json'
        if candidate_path.exists():
            if encoded(read(candidate_path)) != encoded(candidates):
                raise ValueError('Completed candidate feature audit changed')
        else:
            _stable(context, stamps)
            write(candidate_path, candidates)
        started = time.perf_counter()
        local = provider_pair(data, offline=True)
        online = None
        for mode in ('benchmark', *MODES):
            _stable(context, stamps)
            case = load_case(output, mode, context)
            if case is not None:
                if mode in ('benchmark', 'control'):
                    expected = data.parent['benchmark' if mode == 'benchmark' else 'strategy']
                    if encoded(case['account']) != encoded(expected):
                        raise ValueError('Resumed control differs from parent')
                print(mode + ': resumed verified completed case', flush=True)
            else:
                if mode in ('benchmark', 'control'):
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
        sources = _offline_all(data, {'benchmark': benchmark, 'cases': cases, 'candidate_features': candidates})
        report = build_report(cases, benchmark, context, dict(preparation_seconds=elapsed,
            offline_seconds=time.perf_counter()-before, **preparation_usage(output, sources)), candidates)
        if verify_sources() != context:
            raise ValueError('Sources changed during offline verification')
        if encoded(read(candidate_path)) != encoded(candidates):
            raise ValueError('Candidate feature audit changed during execution')
        write(output / 'report.json', report)
        write(output / 'summary.json', report['summary'])
        hashes = {name: sha(output / name) for name in output_names()}
        write(output / 'manifest.json', dict(schema=1, research_kind='technical', control_exit_mode='loss12',
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
        print('Technical research sources, checkpoints and sealed output verified')
    else:
        print(encoded(research(args.output, offline=args.offline_replay)['summary']['comparisons']))


if __name__ == '__main__':
    main()
