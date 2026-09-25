#!/usr/bin/env python3
"""Offline full-account comparison of fixed relative-strength/peer-turnover rules."""
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import argparse
import hashlib
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np

from app.file_lock import file_lock
from scripts.research_exit_scenarios import read, write, sha, encoded
from scripts import research_board_only_supplement as sealed
from scripts.research_cash_allocation import INPUT as CASH_INPUT, OVERRIDES
from scripts.research_chip import ADDITIONS
from scripts.audit_current_causality_20260925 import BASE, ORIGINAL, QUARANTINE
from skills.verified_backtest_tool import offline_only, source_context, export_csv
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_comparison
from skills.replay_market_feeds import ReplayMarketFeeds
from skills.scenario_exit_replay import ExitSignals
from skills.sector_account_replay import (SectorAccountInputs, ARMS, START, END,
    configurations, build_signals, run_case)

OUTPUT = ROOT / '.cache/sector-accounts-20260925'
DEFAULT_CACHE = sealed.SOURCES / 'inputs'
SPEC = ROOT / 'docs/prereg_sector_account_20260925.md'
MEMBERS = ROOT / '.cache/chain-flow-research/members.parquet'
CODE = ['skills/sector_account_replay.py', 'scripts/research_sector_accounts.py',
        'skills/pending_share_entitlements.py', 'tests/test_pending_share_entitlements.py',
        'tests/test_sector_account_replay.py', 'skills/surge_anatomy.py', 'skills/surge_sector.py',
        'docs/prereg_sector_account_20260925.md']
LARGE = {'account','partial_account','resource_plans','slot_decisions','board_decisions','exit_decisions','exit_states'}
LABELS = {'relative_strength':'相對強勢', 'strength_with_turnover':'相對強勢＋同行成交升溫', 'benchmark':'0050持有'}
LIMITATIONS = [
    '目前產業分類回推歷史；membership_point_in_time=false，取得日不能倒填為訊號日已知名冊。',
    '固定目前公司名冊仍有存活者偏差；公告當時可得時間與修訂版本未完全核實。',
    '前輪事件標籤是20日價格結果，本輪沿用12%停損及63日上限；不是相同持有期。',
    '已見历史的探索性帳戶比較；無未見驗證、無實盤資格、不自動替換正式策略。',
    '日量、收盤及最後報價不是逐筆委託證據；日級模型完成不代表實際一定成交。',
    '現金不計息；淨值含持股及應收，非全部可動用現金。只整張殘股仍估值、占名額。',
    '候選共同母體要求同行成交條件可判定；結果不代表所有未分類或未知同行股票。',
]


def _inside(path):
    path = Path(path).absolute()
    if '..' in path.parts or not path.is_relative_to(ROOT):
        raise ValueError('All experiment paths must stay inside the project')
    if any(p.is_symlink() for p in (path,*path.parents) if p==ROOT or ROOT in p.parents):
        raise ValueError('Experiment evidence cannot use symlinks')
    return path


def sources(cache, additions):
    """Reuse the reviewed parent closure, then add all new inputs and code."""
    identity, _ = source_context()
    refs = dict(identity['source_sha256'])
    own = [ROOT/p for p in CODE] + [MEMBERS, MEMBERS.with_suffix('.meta.json'), QUARANTINE]
    own += [BASE/name for name in ('close-official.parquet','raw-close.parquet','raw-volume.parquet')]
    own += [ORIGINAL/'companies.parquet', CASH_INPUT/'manifest.json']
    reference = read(CASH_INPUT/'manifest.json')['references']
    own += [ROOT/reference[key]['path'] for key in ('quotes','companies','calendar','events')]
    overrides = corporate_overrides(additions)
    paths = [OVERRIDES, ADDITIONS, ROOT/'docs/intraday_corporate_additions_20260914.json', sealed.SOURCES/'overrides.json']
    for path in [*paths, *additions]:
        payload = read(path)
        if path==ROOT/'docs/backtest_corporate_completion_20260925.json':
            own.append(ROOT/'skills/backtest_corporate_completion.py')
        own.append(path)
        for name,digest in payload.get('evidence_sha256',{}).items():
            source = _inside(ROOT/name)
            if sha(source)!=digest:
                raise ValueError('Corporate source evidence changed: '+name)
            own.append(source)
    if not (cache/'execution-feeds/index.json').is_file() or not (cache/'dividends').is_dir():
        raise ValueError('Prepared execution-feeds/index.json and dividends directory required')
    feeds = ReplayMarketFeeds(cache/'execution-feeds', offline=True).manifest()
    own += [cache/'execution-feeds/index.json']
    own += [cache/'execution-feeds'/name for name in feeds['files_sha256']]
    own += sorted((cache/'dividends').glob('*.parquet'))
    # Prepared child evidence keeps its attempt budget and parent relationship,
    # not just the final raw files. Never silently detach a fetched source from
    # the preparation receipt that authorized and recorded it.
    if (cache.parent/'parent.json').exists():
        preparation = read(cache.parent/'parent.json')
        parent = _inside(ROOT/preparation['parent'])
        for name,digest in preparation['parent_files_sha256'].items():
            if sha(_inside(parent/name))!=digest:
                raise ValueError('Prepared source parent changed: '+name)
            if name!='execution-feeds/index.json' and sha(_inside(cache/name))!=digest:
                raise ValueError('Prepared source replaced inherited bytes: '+name)
        own.append(cache.parent/'parent.json')
        for name in ('budget.json','preparation.json'):
            if (cache.parent/name).exists():
                own.append(cache.parent/name)
        if (cache.parent/'budget.json').exists():
            budget=read(cache.parent/'budget.json')
            if any(type(budget['attempts'][key]) is not int or not 0<=budget['attempts'][key]<=value
                   for key,value in budget['maximum'].items()):
                raise ValueError('Preparation request budget was exceeded')
        preparer=ROOT/'scripts/prepare_sector_account_sources.py'
        own.append(preparer)
        if (cache.parent/'preparation.json').exists():
            recorded=read(cache.parent/'preparation.json').get('preparation_code_sha256')
            if recorded!=sha(preparer):
                raise ValueError('Preparation receipt code changed')
    for path in own:
        path = _inside(path)
        digest = sha(path)
        name = str(path.relative_to(ROOT))
        if name in refs and refs[name]!=digest:
            raise ValueError('New source conflicts with sealed parent: '+name)
        refs[name] = digest
    return dict(source_sha256=refs, runtime=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__),
        initial_cash=1_000_000, start=START, end=END, membership_point_in_time=False,
        cache_directory=str(cache.relative_to(ROOT)), corporate_additions=[str(p.relative_to(ROOT)) for p in additions]), overrides


def corporate_overrides(additions=()):
    if isinstance(additions,(str,Path)):
        additions = [Path(additions)]
    result = {}
    paths = [OVERRIDES,ADDITIONS,ROOT/'docs/intraday_corporate_additions_20260914.json',sealed.SOURCES/'overrides.json']
    for path in [*paths,*additions]:
        path = _inside(path)
        payload = read(path)
        if path==ROOT/'docs/backtest_corporate_completion_20260925.json':
            from skills.backtest_corporate_completion import load_corporate_completion
            incoming = load_corporate_completion(ROOT)
        else:
            incoming = payload['overrides']
        for name,digest in payload.get('evidence_sha256',{}).items():
            if sha(_inside(ROOT/name))!=digest:
                raise ValueError('Corporate source evidence changed: '+name)
        for key,value in incoming.items():
            if key in result and result[key]!=value:
                raise ValueError('Corporate additions conflict with established terms: '+key)
            result[key] = value
    return result


def load_inputs():
    tick = time.monotonic()
    print('[TIMER] sector signal inputs start',flush=True)
    paths = [BASE/(key+'.parquet') for key in ('close-official','raw-close','raw-volume')]
    paths += [ORIGINAL/'companies.parquet',MEMBERS,MEMBERS.with_suffix('.meta.json'),SPEC,Path(__file__),
              ROOT/'skills/sector_account_replay.py',ROOT/'skills/surge_anatomy.py',ROOT/'skills/surge_sector.py']
    feature_identity = dict(files=file_identities(paths,ROOT),start=START,end=END,
                           python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
                           pyarrow=version('pyarrow'),schema=1)
    digest = hashlib.sha256(encoded(feature_identity).encode()).hexdigest()
    signal_path = ROOT/'.cache/sector-account-signal-cache'/f'{digest}.json'
    frames = {'close-official':pd.read_parquet(BASE/'close-official.parquet').set_index('date')}
    frames['close-official'].index = pd.to_datetime(frames['close-official'].index)
    companies = pd.read_parquet(ORIGINAL/'companies.parquet')
    members = pd.read_parquet(MEMBERS)
    metadata = read(MEMBERS.with_suffix('.meta.json'))
    retrieved = metadata['retrieved_at']
    if isinstance(retrieved, bool) or not isinstance(retrieved, (int,float)) or not np.isfinite(retrieved):
        raise ValueError('Actual membership retrieval timestamp is missing')
    snapshot = datetime.fromtimestamp(retrieved, timezone.utc).date().isoformat()
    if signal_path.exists():
        cached = read(signal_path)
        signals = cached['signals']
        if (cached['identity']!=feature_identity
                or cached['signals_sha256']!=hashlib.sha256(encoded(signals).encode()).hexdigest()):
            raise ValueError('Cached causal sector signals changed')
        print('[TIMER] sector signals cache hit',flush=True)
    else:
        print('[TIMER] sector features compute',flush=True)
        for key in ('raw-close','raw-volume'):
            frames[key] = pd.read_parquet(BASE/(key+'.parquet')).set_index('date')
            frames[key].index = pd.to_datetime(frames[key].index)
        signals = build_signals(frames['close-official'], frames['raw-close'], frames['raw-volume'], companies, members,
                                membership_snapshot_date=snapshot)
        if file_identities(paths,ROOT)!=feature_identity['files']:
            raise ValueError('Feature sources changed during signal construction')
        write(signal_path,dict(identity=feature_identity,signals=signals,
            signals_sha256=hashlib.sha256(encoded(signals).encode()).hexdigest()))
        print('[TIMER] sector causal signals cached',flush=True)
    signals['membership_metadata'] = metadata
    signals['membership_timestamp_timezone'] = 'UTC'
    signals['excluded_members_outside_fixed_cohort'] = sorted(set(members.stock_id)-set(companies.stock_id))
    pool = sorted({'0050'} | {e['stock_id'] for arm in ARMS for e in signals['entries_by_arm'][arm]})
    refs = read(CASH_INPUT/'manifest.json')['references']
    quotes = pd.read_parquet(ROOT/refs['quotes']['path'], filters=[('stock_id','in',pool)])
    quotes['date'] = pd.to_datetime(quotes.date)
    for row in read(QUARANTINE)['quarantine']:
        quotes = quotes[~(quotes.stock_id.eq(row['stock_id']) & quotes.date.eq(pd.Timestamp(row['date'])))]
    companies = pd.read_parquet(ROOT/refs['companies']['path'])
    calendar = pd.read_parquet(ROOT/refs['calendar']['path'])
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open,'date']))
    if not days.equals(frames['close-official'].index):
        raise ValueError('Account and feature market calendars differ')
    events = pd.read_parquet(ROOT/refs['events']['path'])
    data = SectorAccountInputs(quotes, companies, days, signals['entries_by_arm'], events,
                              ExitSignals(frames['close-official'][pool],days),snapshot)
    print('[TIMER] sector inputs ready '+str(round(time.monotonic()-tick,3))+'s '+str({arm:len(signals['entries_by_arm'][arm]) for arm in ARMS}),flush=True)
    return data, signals


def requirements(data, signals, cache):
    index = read(cache/'execution-feeds/index.json')
    pool = sorted({'0050'} | {e['stock_id'] for arm in ARMS for e in signals['entries_by_arm'][arm]})
    position = {str(day.date()):i for i,day in enumerate(data.days)}
    requests = []
    # The augmented arm is a subset; retain rule memberships without duplicating
    # candidate source requirements. Holding/exit dates depend on actual paths.
    augmented = {e['event_id'] for e in signals['entries_by_arm']['strength_with_turnover']}
    for event in signals['entries_by_arm']['relative_strength']:
        i = position[event['entry_date']]
        requests.append(dict(stock_id=event['stock_id'], event_id=event['event_id'],
            signal_date=event['signal_date'], control_entry_date=event['entry_date'],
            combined_entry_date=str(data.days[i+1].date()) if i+1<len(data.days) else None,
            also_turnover_arm=event['event_id'] in augmented))
    quotes = data.quotes.set_index(['date','stock_id'])
    missing_quote = []
    for row in requests:
        for key in ('control_entry_date','combined_entry_date'):
            if row[key] and row[key]<=data.end:
                point = (pd.Timestamp(row[key]),row['stock_id'])
                if point not in quotes.index or quotes.loc[point,['open','high','low','close','volume']].isna().any():
                    missing_quote.append(dict(stock_id=row['stock_id'],date=row[key],kind=key))
    return dict(scope='Potential candidate coverage; actual holdings and exits determine the required execution path.',
        stock_ids=pool, candidate_entry_dates=requests,
        missing_limits_stocks=[sid for sid in pool if 'limits:'+sid not in index['entries']],
        missing_dividend_stocks=[sid for sid in pool if not (cache/'dividends'/f'{sid}.parquet').is_file()],
        missing_candidate_quotes=missing_quote, available_execution_entries=len(index['entries']),
        available_dividend_stocks=len(list((cache/'dividends').glob('*.parquet'))),
        execution_quote_range=[str(data.days[0].date()),data.end],
        adjusted_price_range=[str(data.days[0].date()),data.end],
        corporate_event_rows=int(data.events.stock_id.isin(pool).sum()),
        raw_quote_rows=len(data.quotes), automatic_source_fetch=False)


def _label(config):
    return LABELS[config['arm']]+' · '+('只整張' if config['board_only'] else '整張＋零股')+' · '+(
        '一般成本' if config['stress']=='control' else '合併壓力')


def run(output=OUTPUT, cache=DEFAULT_CACHE, additions=(), *, preflight_only=False,
        offline_replay=False, strict_pit=False):
    output, cache = _inside(output), _inside(cache)
    additions = [_inside(path) for path in additions]
    if not output.is_relative_to(ROOT/'.cache') or output==cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError('Use a distinct output inside .cache, separate from source inputs')
    if output.exists() and not offline_replay:
        raise ValueError('Output exists; preserve it and use a new output or --offline-replay')
    tick = time.monotonic()
    with file_lock(ROOT/'.cache/sector-accounts.lock',timeout=0), offline_only():
        identity, overrides = sources(cache,additions)
        if offline_replay:
            manifest = read(output/'manifest.json')
            if read(output/'identity.json')!=identity:
                raise ValueError('Frozen sector account sources/code changed')
            for name,digest in manifest['files_sha256'].items():
                if sha(output/name)!=digest:
                    raise ValueError('Frozen sector account output changed: '+name)
        data, signals = load_inputs()
        needs = requirements(data,signals,cache)
        rows, cases, comparisons = [], {}, []
        for name,config in configurations():
            if strict_pit:
                result = dict(completed=False, config=config,
                    reason='strict PIT blocked: current membership snapshot is not historical membership evidence',
                    membership_point_in_time=False,membership_snapshot_date=data.membership_snapshot_date,
                    live_qualified=False,unseen_validation=False)
            elif preflight_only:
                result = dict(completed=False, config=config, reason='Preflight only; account not executed',
                    membership_point_in_time=False,membership_snapshot_date=data.membership_snapshot_date,
                    live_qualified=False,unseen_validation=False)
            else:
                print('running '+name,flush=True)
                result = run_case(data,config,cache,overrides)
            path = output/'cases'/f'{name}.json'
            if offline_replay:
                if encoded(result)!=encoded(read(path)):
                    raise ValueError('Offline full case differs: '+name)
            else:
                write(path,result)
            cases[name] = result
            row = dict(name=name,label=_label(config),status='completed_daily' if result['completed'] else 'blocked',
                config=config,reason=result.get('reason',''),result_path=str(path.relative_to(ROOT)),result_sha256=sha(path),
                total_return=None,max_drawdown=None,artifact_paths={})
            if result['completed']:
                row.update({key:result['summary'][key] for key in ('total_return','final_nav','max_drawdown','annual','costs')})
                for key in ('trades','daily'):
                    row['artifact_paths'][key] = export_csv(output/'exports'/f'{name}-{key}.csv',result['account'][key])
            rows.append(row)
            print(name, 'COMPLETE' if result['completed'] else 'BLOCKED '+result['reason'],flush=True)
        for name,config in configurations():
            case = cases[name]
            if config['benchmark'] or not case['completed']:
                continue
            base = f"benchmark_{config['stress']}_{'board_only' if config['board_only'] else 'mixed'}"
            if cases[base]['completed']:
                comparisons.append(dict(strategy=name,benchmark=base,
                    audit=validate_comparison(case,cases[base]),
                    excess_return=case['summary']['total_return']-cases[base]['summary']['total_return']))
        if file_identities([ROOT/p for p in identity['source_sha256']],ROOT)!=identity['source_sha256']:
            raise ValueError('Sources changed during sector account replay')
        # Directory additions are source changes too, even if no old file changed.
        if sources(cache,additions)[0]!=identity:
            raise ValueError('Source inventory changed during sector account replay')
        report = dict(format='sector_account_v1',status='exploratory' if all(r['status']=='completed_daily' for r in rows) else 'blocked',
            completed=all(c['completed'] for c in cases.values()),start=data.start,end=data.end,initial_cash=1_000_000,
            case_rows=rows,comparisons=comparisons,candidate_counts={arm:len(signals['entries_by_arm'][arm]) for arm in ARMS},
            membership_point_in_time=False,membership_snapshot_date=data.membership_snapshot_date,
            membership_metadata=signals['membership_metadata'],live_qualified=False,unseen_validation=False,
            strict_pit=strict_pit,preflight_only=preflight_only,future_labels_used=False,
            schedules_restarted=False,broker_orders_sent=False,limitations=LIMITATIONS,
            source_sha256=identity['source_sha256'],metrics=dict(elapsed_seconds=round(time.monotonic()-tick,3),
                network_calls=0,finmind_requests=0,database_writes=0,executed_cases=0 if strict_pit or preflight_only else 12))
        if offline_replay:
            previous = read(output/'report.json')
            actual = dict(report); previous = dict(previous)
            actual.pop('metrics');previous.pop('metrics')
            if actual!=previous:
                raise ValueError('Offline report differs')
            write(output/'offline.json',dict(all_cases_identical=True,manifest_sha256=sha(output/'manifest.json'),
                network_calls=0,elapsed_seconds=report['metrics']['elapsed_seconds']))
        else:
            write(output/'identity.json',identity)
            write(output/'signals.json',signals)
            write(output/'requirements.json',needs)
            write(output/'report.json',report)
            files = [p for p in output.rglob('*') if p.is_file() and p.name not in ('manifest.json','offline.json')]
            write(output/'manifest.json',dict(files_sha256={str(p.relative_to(output)):sha(p) for p in files},
                membership_point_in_time=False,live_qualified=False,unseen_validation=False))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--cache','--inputs',dest='cache',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--corporate-additions',action='append',type=Path,default=[])
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--offline-replay',action='store_true')
    parser.add_argument('--strict-pit',action='store_true')
    args=parser.parse_args()
    result=run(args.output,args.cache,args.corporate_additions,preflight_only=args.preflight_only,
               offline_replay=args.offline_replay,strict_pit=args.strict_pit)
    print(json.dumps(dict(status=result['status'],candidate_counts=result['candidate_counts'],metrics=result['metrics']),ensure_ascii=False))
