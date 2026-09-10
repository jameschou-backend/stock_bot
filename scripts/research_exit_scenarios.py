#!/usr/bin/env python3
"""Compare preregistered exits, checkpoint each case, then reproduce offline."""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import argparse
import hashlib
from importlib.metadata import version
import json
import math
from numbers import Real
from pathlib import Path
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from app.config import load_config
from app.file_lock import file_lock
from scripts.prepare_exit_inputs import verify as verify_exit_inputs
from scripts.replay_million import audit, summarize, verify_report as verify_parent_report
from skills.exit_policy import MODES, MODE_LABELS
from skills.million_replay import Replay, UnresolvedAction
from skills.replay_corporate_actions import CorporateActions
from skills.replay_market_feeds import ReplayMarketFeeds
from skills.scenario_exit_replay import ExitSignals, ScenarioExitReplay


INPUT = ROOT / '.cache/exit-research-inputs'
OUTPUT = ROOT / '.cache/exit-research'
PARENT = ROOT / '.cache/million-replay'
SPEC = ROOT / 'docs/prereg_exit_scenarios_20260910.md'
OVERRIDES = ROOT / 'docs/exit_corporate_overrides_20260910.json'
CODE = ('scripts/research_exit_scenarios.py', 'scripts/prepare_exit_inputs.py',
    'scripts/replay_million.py', 'skills/scenario_exit_replay.py', 'skills/exit_policy.py',
    'skills/regime_state.py', 'skills/million_replay.py', 'skills/replay_market_feeds.py',
    'skills/replay_corporate_actions.py', 'app/finmind.py', 'app/finmind_cache.py',
    'app/rate_limiter.py', 'app/file_lock.py', 'app/config.py')
LIMITATIONS = [
    '既有歷史已反覆研究；不是未見資料或實盤證據，不自動替換正式策略。',
    '沿用完整458個候選；提早退出改變後續名額與預算，並非相同45部位的純出場比較。',
    '所得閒置資金仍投入0050；大盤轉弱退出個股不代表整戶空手。',
    '停損及停利門檻是還原收盤訊號；最早次一市場日按實際日資料與成本模擬，不能保證門檻價成交。',
    '普通盤與零股日資料缺完整委託簿；最後對手報價數量不足另列，日量限制不保證可成交。',
    '使用固定當前公司名冊，仍有存活者偏誤；末日持股及應收保留，未假設全部清倉。',
    '現金股利採毛額；未計個人所得稅、補充保費及郵匯費；不參與現金增資認購。',
    '2881及2736畸零股按公告面額折現至元；2880按面額作估值假設。未核實淨額及付款日，分列毛額現金應收，沒有在整股交付日提前入帳。',
    '2880於2024-08-30發放權利證書；本研究不模擬證書交易，等2024-09-30換普通股才可售，期間以普通股價代理估值且持續占用名額。這可能改變後續買入與績效，偏差方向不確定。',
    '2880現金股利先取整元，未模擬全體股東尾數排序分配；單戶可能少計1元，不是已核實實收。',
]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Non-finite research JSON: '+str(path))
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('Duplicate research JSON key: '+str(path))
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=invalid, object_pairs_hook=unique)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(encoded(value)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe(base, name):
    name = Path(name)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('Unsafe research evidence path')
    path = base/name
    if any(part.is_symlink() for part in (path,*path.parents) if part == base or base in part.parents) or not path.is_file():
        raise ValueError('Research evidence missing or symlinked: '+str(path))
    return path


def verify_sources():
    """Full source checks occur at run boundaries, never per market day."""
    inputs = verify_exit_inputs(INPUT)
    verify_parent_report(PARENT)
    sources = dict(inputs['parent_files_sha256'])
    for path in (INPUT/'manifest.json', PARENT/'manifest.json', SPEC, OVERRIDES):
        sources[str(path.relative_to(ROOT))] = sha(path)
    return dict(schema=1, mode_order=list(MODES), source_files_sha256=sources,
        code_sha256={name:sha(ROOT/name) for name in CODE},
        runtime_versions={'python':platform.python_version(), **{name:version(name)
                           for name in ('numpy','pandas','pyarrow','requests','PyYAML','python-dotenv')}},
        child_cache_directory=str(INPUT.resolve()))


def fingerprint(context):
    return hashlib.sha256(encoded(context).encode()).hexdigest()


def _stamps(context):
    paths = set(context['source_files_sha256']) | set(context['code_sha256'])
    result = {}
    for name in paths:
        path = _safe(ROOT, name)
        info = path.stat()
        result[name] = (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
    return result


def _stable(context, stamps):
    if _stamps(context) != stamps:
        raise ValueError('Research source or code changed during execution')


@dataclass
class RunInputs:
    quotes: pd.DataFrame
    companies: pd.DataFrame
    days: pd.DatetimeIndex
    entries: list
    events: pd.DataFrame
    features: ExitSignals
    parent: dict
    start: str = '2022-01-03'
    end: str = '2026-09-09'


def load_inputs():
    refs = read(INPUT/'manifest.json')['references']
    path = lambda name: _safe(ROOT,refs[name]['path'])
    entries = read(path('signals'))['entries']
    if len(entries) != 458 or len({row['event_id'] for row in entries}) != 458:
        raise ValueError('The preregistered complete 458-entry pool is required')
    if any(len(row['members']) != 1 for row in entries):
        raise ValueError('Single-stock entry events required')
    pool = sorted({'0050'}|{row['members'][0] for row in entries})
    # Read the large quote source once, restricted to the full candidate pool.
    quotes = pd.read_parquet(path('quotes'), filters=[('stock_id','in',pool)])
    companies = pd.read_parquet(path('companies'))
    calendar = pd.read_parquet(path('calendar'))
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open,'date']))
    adjusted = pd.read_parquet(path('close_official'), columns=['date',*pool]).set_index('date')
    adjusted.index = pd.to_datetime(adjusted.index)
    features = ExitSignals(adjusted,days)
    events = pd.read_parquet(path('events'))
    return RunInputs(quotes,companies,days,entries,events,features,read(PARENT/'report.json'))


class TrackedCorporateActions(CorporateActions):
    request_observer = None

    def prepare(self, sid):
        before = self.requests
        try:
            return super().prepare(sid)
        finally:
            if self.request_observer is not None and self.requests != before:
                self.request_observer(self.requests)

    def on_date(self, sid, day):
        rows = super().on_date(sid,day)
        for row in rows:
            if row.get('kind') != 'stock_dividend':
                continue
            invalid = []
            for field in ('shares_per_share','fractional_cash_per_share'):
                value = row.get(field)
                # The original engine needs a fractional-cash term only when
                # the actual entitled quantity has a fractional remainder.
                if field=='fractional_cash_per_share' and value is None:
                    continue
                try:
                    valid = (not isinstance(value,bool) and isinstance(value,(Real,Decimal))
                             and math.isfinite(value) and value>=0)
                except (OverflowError,ValueError,TypeError):
                    valid = False
                if not valid:
                    invalid.append(field)
            if not row.get('pay_date'):
                invalid.append('pay_date')
            if invalid:
                raise UnresolvedAction(f'Stock dividend data missing or invalid: {sid} {day}; '+', '.join(invalid))
        return rows


def provider_pair(data, *, offline):
    token = None if offline else load_config().finmind_token
    return (ReplayMarketFeeds(INPUT/'execution-feeds',offline=offline,token=token),
            TrackedCorporateActions(data.events,INPUT/'dividends',token,offline=offline,
                             overrides=read(OVERRIDES)['overrides']))


def provider_snapshot(feeds, corp):
    return {'execution_feeds':feeds.manifest(), 'corporate_sources':corp.manifest()}


def check_provider_snapshot(snapshot, *, exact=False):
    """Append-only cache growth is allowed for a completed case's resumption."""
    current = ReplayMarketFeeds(INPUT/'execution-feeds',offline=True).manifest()
    old = snapshot['execution_feeds']
    if exact and current != old:
        raise ValueError('Sealed execution sources changed')
    for key in ('schema','parser_sha256','limitations','cache_directory'):
        if current.get(key) != old.get(key):
            raise ValueError('Execution provider identity changed')
    for key in ('entries','files_sha256'):
        if any(current.get(key,{}).get(name) != value for name,value in old.get(key,{}).items()):
            raise ValueError('Completed-case execution evidence changed')
    for key,value in old.get('request_counters',{}).items():
        present = current.get('request_counters',{}).get(key)
        if type(present) is not int or present < value:
            raise ValueError('Execution request counter reset')
    for name,digest in snapshot['corporate_sources']['files_sha256'].items():
        if sha(_safe(INPUT/'dividends',name)) != digest:
            raise ValueError('Completed-case dividend source changed')
    if snapshot['corporate_sources']['overrides'] != read(OVERRIDES)['overrides']:
        raise ValueError('Completed-case corporate facts changed')


def exit_statistics(account, states, days):
    positions = {str(pd.Timestamp(day).date()):i for i,day in enumerate(days)}
    cohorts = {row['event_id']:row for row in account['cohorts']}
    waits = []
    for identity,state in states.items():
        if not state.get('trigger_reason'):
            continue
        target,signal = state['target_date'],state['signal_date']
        if target not in positions or signal not in positions or positions[target] != positions[signal]+1:
            raise ValueError('Exit trigger must use the immediately previous market session')
        sales = [row for row in account['trades'] if row['side']=='sell'
                 and row['stock_id']==state['stock_id'] and row['stock_id']!='0050'
                 and row['event_id']==identity]
        if any(row['date'] < target for row in sales):
            raise ValueError('Sale predates its exit instruction')
        first = sales[0]['date'] if sales else None
        complete = cohorts[identity].get('exit_date')
        waits.append(dict(event_id=identity, stock_id=state['stock_id'], reason=state['trigger_reason'],
            signal_date=signal, target_date=target, first_fill_date=first, complete_exit_date=complete,
            signal_to_first_fill_sessions=positions[first]-positions[signal] if first else None,
            target_to_first_fill_sessions=positions[first]-positions[target] if first else None,
            target_to_complete_sessions=positions[complete]-positions[target] if complete else None,
            pending_at_end=complete is None))
    filled = [row['target_to_first_fill_sessions'] for row in waits if row['first_fill_date']]
    odd = [row for row in account['trades'] if row['channel']=='odd']
    missing,exceeds = [],[]
    for row in odd:
        depth = row.get('odd_ask_qty' if row['side']=='buy' else 'odd_bid_qty')
        if depth is None:
            missing.append(row['sequence'])
        elif row['qty'] > depth:
            exceeds.append(row['sequence'])
    fractional_pending = [r for r in account['receivables'] if r['kind']=='cash'
        and r['action_id'].endswith('-fractional-cash') and r['pay_date'] is None]
    restricted = [r for r in account['corporate_actions'] if r.get('certificate_restriction')]
    return dict(reason_counts=dict(Counter(row['reason'] for row in waits)),triggered_positions=len(waits),
        restricted_certificate_actions=restricted,
        unverified_fractional_cash_receivables=fractional_pending,
        unverified_fractional_cash_amount=sum(r['amount'] for r in fractional_pending),
        pending_exit_positions=sum(row['pending_at_end'] for row in waits),
        mean_target_to_first_fill_sessions=sum(filled)/len(filled) if filled else None,
        max_target_to_first_fill_sessions=max(filled) if filled else None, exit_waits=waits,
        signal_lag_market_sessions=1,
        depth_audit=dict(odd_trade_count=len(odd),missing_opposing_depth_count=len(missing),
            exceeds_last_opposing_depth_count=len(exceeds),missing_depth_trade_sequences=missing,
            exceeds_depth_trade_sequences=exceeds,flag=bool(missing or exceeds),execution_guaranteed=False))


def run_case(mode, data, feeds, corp):
    args = (data.quotes,data.companies,data.days,data.entries,feeds,corp)
    kwargs = dict(start=data.start,end=data.end)
    if mode == 'benchmark':
        engine = Replay(*args,benchmark=True,**kwargs)
    else:
        class ProgressReplay(ScenarioExitReplay):
            progress_month = None

            def corporate_day(self, day):
                month = str(pd.Timestamp(day).date())[:7]
                if not feeds.offline and month != self.progress_month:
                    print(mode+': preparing '+month,flush=True)
                    self.progress_month = month
                return super().corporate_day(day)
        engine = ProgressReplay(*args,exit_signals=data.features,mode=mode,**kwargs)
    account = engine.run()
    decisions = getattr(engine,'exit_decisions',[])
    states = getattr(engine,'exit_states',{})
    checked = audit(account)
    if mode in ('fixed63','benchmark'):
        old = data.parent['strategy' if mode=='fixed63' else 'benchmark']
        if encoded(account) != encoded(old):
            raise ValueError(mode+' does not exactly reproduce the sealed parent account')
    summary = {**summarize(account), **exit_statistics(account,states,data.days)}
    return dict(schema=1,mode=mode,label='0050持有對照' if mode=='benchmark' else MODE_LABELS[mode],
        account=account,exit_decisions=decisions,exit_states=states,summary=summary,audit=checked)


def _case_paths(output, mode):
    if mode not in (*MODES,'benchmark'):
        raise ValueError('Unknown checkpoint mode')
    return output/'cases'/f'{mode}.json',output/'cases'/f'{mode}.manifest.json'


def save_case(output, case, context, sources):
    path,manifest_path = _case_paths(output,case['mode'])
    write(path,case)
    # The manifest is the commit marker. An interrupted unsealed .json is not a
    # completed checkpoint and can be deterministically regenerated.
    write(manifest_path,dict(schema=1,mode=case['mode'],context_fingerprint=fingerprint(context),
        case_file=str(path.relative_to(output)),case_sha256=sha(path),sources=sources))


def load_case(output, mode, context):
    path,manifest_path = _case_paths(output,mode)
    if not manifest_path.exists():
        return None
    meta = read(manifest_path)
    if (meta.get('schema') != 1 or meta.get('mode') != mode
            or meta.get('context_fingerprint') != fingerprint(context)
            or meta.get('case_file') != str(path.relative_to(output)) or sha(path) != meta.get('case_sha256')):
        raise ValueError('Completed case changed or has incompatible code/spec/source: '+mode)
    check_provider_snapshot(meta['sources'])
    case = read(path)
    if case.get('mode') != mode:
        raise ValueError('Checkpoint case identity mismatch')
    audit(case['account'])
    return case


def build_report(cases, benchmark, context, timings):
    fixed = cases['fixed63']['summary']
    base = benchmark['summary']
    comparisons,annual = [],[]
    for mode in MODES:
        summary = cases[mode]['summary']
        comparisons.append(dict(mode=mode,label=MODE_LABELS[mode],final_nav=summary['final_nav'],
            total_return=summary['total_return'],cagr=summary['cagr'],max_drawdown=summary['max_drawdown'],
            total_cost=summary['costs']['total_cost'],trade_count=summary['trade_count'],
            stock_cohorts=summary['stock_cohorts'],reason_counts=summary['reason_counts'],
            excess_total_return_vs_fixed63=summary['total_return']-fixed['total_return'],
            excess_total_return_vs_0050=summary['total_return']-base['total_return'],
            drawdown_difference_vs_fixed63=summary['max_drawdown']-fixed['max_drawdown'],
            pending_exit_positions=summary['pending_exit_positions'],depth_flag=summary['depth_audit']['flag'],
            mean_target_to_first_fill_sessions=summary['mean_target_to_first_fill_sessions'],
            signal_lag_market_sessions=1))
        annual.extend(dict(mode=mode,label=MODE_LABELS[mode],**row) for row in summary['annual'])
    annual.extend(dict(mode='benchmark',label='0050持有對照',**row) for row in base['annual'])
    return dict(schema=1,mode_order=list(MODES),cases=cases,benchmark=benchmark,
        summary=dict(schema=1,mode_order=list(MODES),labels=MODE_LABELS,
            comparisons=comparisons,annual=annual,benchmark=base,
            case_files={mode:f'cases/{mode}.json' for mode in MODES},
            benchmark_case_file='cases/benchmark.json',limitations=LIMITATIONS,
            unseen_validation=False,live_qualified=False,auto_promote=False,
            candidate_count=458,remaining_cash_asset='0050',execution_signal_lag_market_sessions=1,
            performance=timings,
            complete_exit_date_definition='含後續配股的部位全部結束日期；不是第一次原持股清零日期。'),
        context_fingerprint=fingerprint(context),performance=timings,limitations=LIMITATIONS,
        candidate_count=458,unseen_validation=False,live_qualified=False,auto_promote=False,
        execution_signal_lag_market_sessions=1,remaining_cash_asset='0050')


def output_names():
    return ['report.json','summary.json','run.json','preparation_sessions.json',
        *[f'cases/{mode}{suffix}' for mode in (*MODES,'benchmark') for suffix in ('.json','.manifest.json')]]


def verification_files(context, sources, output, output_hashes):
    """Complete repository-relative closure, excluding the manifest itself."""
    result = {}
    def add(path, digest):
        name = str(Path(path).relative_to(ROOT))
        if name in result and result[name] != digest:
            raise ValueError('Conflicting source hashes: '+name)
        result[name] = digest
    for mapping in (context['source_files_sha256'],context['code_sha256']):
        for name,digest in mapping.items():
            add(ROOT/name,digest)
    seed = read(INPUT/'manifest.json')
    for name,digest in seed['seed_files_sha256'].items():
        add(INPUT/name,digest)
    feeds = sources['execution_feeds']
    for name,digest in feeds['files_sha256'].items():
        add(INPUT/'execution-feeds'/name,digest)
    add(INPUT/'execution-feeds/index.json',feeds['manifest_sha256'])
    for path in sorted((INPUT/'dividends').glob('*.parquet')):
        add(path,sha(_safe(INPUT/'dividends',path.name)))
    for name,digest in output_hashes.items():
        add(output/name,digest)
    return result


def preparation_usage(output, sources):
    seed = read(INPUT/'manifest.json')
    current = sources['execution_feeds']['request_counters']
    inherited = seed['seed_execution_index']['request_counters']
    delta = {key:value-inherited.get(key,0) for key,value in current.items()}
    if any(value < 0 for value in delta.values()):
        raise ValueError('Execution request counter fell below the inherited baseline')
    sessions = read(output/'preparation_sessions.json')['sessions']
    counts = [row['corporate_requests'] for row in sessions.values()]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError('Invalid corporate request record')
    return dict(online_preparation_corporate_requests=sum(counts),
        online_preparation_corporate_requests_scope='Recorded successful FinMind fetches in driver provider sessions, including resumed runs; excludes failed calls with unknown charge, external preflight and other processes.',
        execution_feed_requests_since_seed=delta,offline_api_requests=0,
        seed_dividend_count=len(seed['seed_dividend_hashes']),
        total_child_dividend_count=len(list((INPUT/'dividends').glob('*.parquet'))),
        elapsed_time_scope='This invocation only; earlier preparation and external preflight are excluded.')


def verify_report(output=OUTPUT):
    output = Path(output).resolve()
    manifest = read(output/'manifest.json')
    if (manifest.get('schema')!=1 or manifest.get('offline_identical') is not True
            or manifest.get('live_qualified') is not False
            or manifest.get('unseen_validation') is not False or manifest.get('auto_promote') is not False
            or set(manifest.get('files_sha256',{})) != set(output_names())
            or manifest.get('verification_directories') != [str((INPUT/'dividends').relative_to(ROOT))]):
        raise ValueError('Incomplete sealed exit-research manifest')
    context = verify_sources()
    if context != manifest['context'] or read(output/'run.json')['context'] != context:
        raise ValueError('Exit research code, runtime, spec or input changed')
    for name,digest in manifest['files_sha256'].items():
        if sha(_safe(output,name)) != digest:
            raise ValueError('Exit research output hash changed: '+name)
    check_provider_snapshot(manifest['sources'],exact=True)
    closure = verification_files(context,manifest['sources'],output,manifest['files_sha256'])
    if closure != manifest.get('verification_files_sha256'):
        raise ValueError('Exit research verification file closure changed')
    # The source verifier, provider verifier, dividend scan and output loop
    # above already verify every part of this closure. Do not rehash the large
    # shared quote matrix once more merely to expose the UI's inventory.
    return manifest


def _offline_all(output, data, context, expected):
    feeds,corp = provider_pair(data,offline=True)
    before = feeds.manifest()
    for mode in ('benchmark',*MODES):
        current = run_case(mode,data,feeds,corp)
        previous = expected['benchmark'] if mode=='benchmark' else expected['cases'][mode]
        if encoded(current) != encoded(previous):
            raise ValueError('Offline account/decision/state reproduction differs: '+mode)
        print(mode+': identical offline reproduction',flush=True)
    sources = provider_snapshot(feeds,corp)
    if sources['execution_feeds'] != before or sources['corporate_sources']['requests'] != 0:
        raise ValueError('Offline replay changed execution cache or used the API')
    return sources


def research(output=OUTPUT, *, offline=False):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Research output must be inside the repository for relative evidence paths')
    output.mkdir(parents=True,exist_ok=True)
    with file_lock(output/'.research.lock',timeout=10):
        if offline:
            manifest = verify_report(output)
            data = load_inputs()
            report = read(output/'report.json')
            _offline_all(output,data,manifest['context'],report)
            verify_report(output)
            return report
        if (output/'manifest.json').exists():
            raise ValueError('Sealed exit research exists; use --offline-replay or a new --output')
        context = verify_sources()
        stamps = _stamps(context)
        run_path = output/'run.json'
        if run_path.exists():
            if read(run_path).get('context') != context:
                raise ValueError('Checkpoint code/spec/source changed; use a new explicit --output')
        else:
            write(run_path,dict(schema=1,context=context,created_at=datetime.now(timezone.utc).isoformat()))
        session_path = output/'preparation_sessions.json'
        if not session_path.exists():
            write(session_path,dict(schema=1,sessions={}))
        sessions = read(session_path)
        data = load_inputs()
        started = time.perf_counter()
        offline_feeds,offline_corp = provider_pair(data,offline=True)
        benchmark = load_case(output,'benchmark',context)
        if benchmark is None:
            benchmark = run_case('benchmark',data,offline_feeds,offline_corp)
            save_case(output,benchmark,context,provider_snapshot(offline_feeds,offline_corp))
        elif encoded(benchmark['account']) != encoded(data.parent['benchmark']):
            raise ValueError('Resumed benchmark differs from parent')
        cases,online_pair = {},None
        for mode in MODES:
            _stable(context,stamps)
            case = load_case(output,mode,context)
            if case is not None:
                if mode=='fixed63' and encoded(case['account']) != encoded(data.parent['strategy']):
                    raise ValueError('Resumed fixed63 differs from parent')
                print(mode+': resumed verified completed case',flush=True)
            else:
                if mode=='fixed63':
                    feeds,corp = offline_feeds,offline_corp
                else:
                    if online_pair is None:
                        online_pair = provider_pair(data,offline=False)
                        session_id = uuid.uuid4().hex
                        def record_requests(count):
                            sessions['sessions'][session_id] = {'corporate_requests':count}
                            write(session_path,sessions)
                        online_pair[1].request_observer = record_requests
                        record_requests(online_pair[1].requests)
                    feeds,corp = online_pair
                print(mode+': preparing account and execution evidence',flush=True)
                case = run_case(mode,data,feeds,corp)
                _stable(context,stamps)
                save_case(output,case,context,provider_snapshot(feeds,corp))
            cases[mode] = case
        elapsed = time.perf_counter()-started
        if verify_sources() != context:
            raise ValueError('Sources changed before offline verification')
        before = time.perf_counter()
        sources = _offline_all(output,data,context,{'benchmark':benchmark,'cases':cases})
        report = build_report(cases,benchmark,context,dict(preparation_seconds=elapsed,
            offline_seconds=time.perf_counter()-before,**preparation_usage(output,sources)))
        if verify_sources() != context:
            raise ValueError('Sources changed during offline verification')
        write(output/'report.json',report)
        write(output/'summary.json',report['summary'])
        hashes = {name:sha(output/name) for name in output_names()}
        write(output/'manifest.json',dict(schema=1,created_at=datetime.now(timezone.utc).isoformat(),
            context=context,sources=sources,files_sha256=hashes,
            verification_files_sha256=verification_files(context,sources,output,hashes),
            verification_directories=[str((INPUT/'dividends').relative_to(ROOT))],
            offline_identical=True,live_qualified=False,unseen_validation=False,auto_promote=False))
        verify_report(output)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-and-replay',action='store_true',help='Resume compatible cases, then reproduce every case offline')
    group.add_argument('--offline-replay',action='store_true')
    group.add_argument('--verify',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    if args.verify:
        with file_lock(args.output/'.research.lock',timeout=10):
            verify_report(args.output)
        print('Exit research sources, checkpoints and sealed output verified')
    else:
        report = research(args.output,offline=args.offline_replay)
        print(encoded(report['summary']['comparisons']))


if __name__ == '__main__':
    main()
