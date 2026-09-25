#!/usr/bin/env python3
"""Actually rebuild the corrected universe, then run matched, offline accounts."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from scripts import research_board_only_supplement as sealed
from scripts.research_priority import rolling_comparison
from scripts.prepare_million_signals import official_adjusted
from scripts.prepare_historical_cohort_supplement import verify as verify_raw, DIRECTORY
from scripts.prepare_historical_selector_quality import verify as verify_quality, OUTPUT as QUALITY
from scripts.audit_historical_universe_completion import verify_report as verify_identity, TARGET as IDENTITY
from skills.backtest_case_cache import file_identities
from skills.backtest_corporate_completion import DOCUMENT, load_corporate_completion
from skills.verified_backtest_tool import source_context, offline_only
from skills.historical_selector_replay import build_signals, eligibility_matrix, HistoricalBoardReplay
from skills.board_only_verified_replay import BoardOnlyVerifiedReplay, BoardOnlyVerifiedBenchmark
from skills.scenario_exit_replay import ExitSignals
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.backtest_contract import validate_completed_account
from scripts.prepare_historical_listing_prefix import verify as verify_prefix, OUTPUT as PREFIX
from skills.historical_listing_prefix import restore as restore_prefix

BASE = ROOT / '.cache/five-axis-20260913/rebuild'
PARENT = ROOT / '.cache/backtest-corporate-completion-20260925/probe-v2'
OUTPUT = ROOT / '.cache/historical-selector-replay-20260925'
ARMS = ('original', 'identity', 'omitted', 'combined')
CODE = ('scripts/research_historical_selector_replay.py', 'scripts/prepare_historical_selector_quality.py',
        'scripts/prepare_historical_selector_execution.py',
        'scripts/prepare_historical_listing_prefix.py', 'skills/historical_listing_prefix.py',
        'skills/historical_selector_replay.py', 'skills/historical_diffusion_signals.py',
        'skills/historical_universe_completion.py', 'skills/backtest_corporate_completion.py',
        'docs/prereg_historical_selector_replay_20260925.md')


def inventory():
    identity, _ = source_context()
    report = verify_identity()
    raw, quality = verify_raw(), verify_quality()
    prefix = verify_prefix()
    load_corporate_completion(ROOT)
    refs = dict(identity['source_sha256'])
    refs.update(report['source_sha256'])
    refs.update(read(ROOT / DOCUMENT)['evidence_sha256'])
    for folder in (PARENT, BASE):
        for name, digest in read(folder / 'manifest.json')['files_sha256'].items():
            if sha(folder / name) != digest:
                raise ValueError('Parent input changed: ' + str(folder / name))
            refs[str((folder / name).relative_to(ROOT))] = digest
    paths = [IDENTITY, IDENTITY.with_suffix('.sha256'), DIRECTORY / 'manifest.json', DIRECTORY / 'quotes.parquet',
             QUALITY / 'manifest.json', QUALITY / 'manifest.sha256', ROOT / DOCUMENT]
    paths += [QUALITY / p for p in quality['files_sha256']]
    paths += [PREFIX / 'manifest.json', PREFIX / 'manifest.sha256']
    paths += [PREFIX / p for p in prefix['files_sha256']]
    paths += [ROOT / p for p in CODE]
    refs.update(file_identities(paths, ROOT))
    return refs, report, raw, quality


def load_frames():
    frames = {name: pd.read_parquet(BASE / (name + '.parquet')).set_index('date')
              for name in ('close-official', 'close-quality', 'raw-close', 'raw-volume')}
    for frame in frames.values():
        frame.index = pd.to_datetime(frame.index)
    return frames


def supplemental_companies(report, raw):
    rows = []
    for sid in raw['plan']['stock_ids']:
        episodes = [e for e in report['episodes'] if e['stock_id'] == sid and e['category'] == '股票']
        if not episodes or any(e['start'] is None for e in episodes):
            raise ValueError('Supplement has no verified ordinary listing: ' + sid)
        earliest, latest = min(episodes, key=lambda e: e['start']), max(episodes, key=lambda e: e['start'])
        rows.append(dict(stock_id=sid, name=earliest['name'], listed_date=pd.Timestamp(earliest['start']),
                         industry=None, market=latest['market'].upper()))
    return pd.DataFrame(rows)


def augment(frames, raw, events):
    quotes = pd.read_parquet(DIRECTORY / 'quotes.parquet')
    quotes['date'] = pd.to_datetime(quotes.date)
    bad = {(r['stock_id'], pd.Timestamp(r['date'])) for r in raw['summary']['quarantine']}
    quotes = quotes.loc[[(sid, day) not in bad for sid, day in zip(quotes.stock_id, quotes.date)]]
    new_ids = raw['plan']['stock_ids']
    if set(new_ids) & set(frames['raw-close']):
        raise ValueError('Supplement overlaps original quote columns')
    raw_close = quotes.pivot(index='date', columns='stock_id', values='close').reindex(
        index=frames['raw-close'].index, columns=new_ids)
    volume = quotes.pivot(index='date', columns='stock_id', values='volume').reindex_like(raw_close)
    official = official_adjusted(raw_close, events)
    source_frames = []
    for sid in new_ids:
        frame = pd.read_parquet(QUALITY / (sid + '.parquet'))
        if not frame.empty:
            source_frames.append(frame[['date', 'stock_id', 'close']])
    quality = pd.concat(source_frames, ignore_index=True)
    quality['date'] = pd.to_datetime(quality.date)
    quality = quality.pivot(index='date', columns='stock_id', values='close').reindex_like(raw_close)
    quality = quality.where(raw_close.gt(0) & volume.gt(0))
    extensions = {'raw-close': raw_close, 'raw-volume': volume,
                  'close-official': official, 'close-quality': quality}
    return {name: pd.concat([frame, extensions[name]], axis=1) for name, frame in frames.items()}, quotes


def prepare_arm(arm, frames, companies, report, supplement_ids):
    frames = {k: v.copy() for k, v in frames.items()}
    if arm == 'original':
        mask = None
    elif arm == 'omitted':
        # Single-variable control: preserve original membership assumptions and
        # only give the new cohort its evidence-bound dates/exclusions.
        mask = pd.DataFrame({r.stock_id: frames['raw-close'].index >= pd.Timestamp(r.listed_date)
                             for r in companies.itertuples()}, index=frames['raw-close'].index)
        mask['0050'] = True
        exact = eligibility_matrix(report, companies[companies.stock_id.isin(supplement_ids)], mask.index)
        mask.loc[:, supplement_ids] = exact[supplement_ids]
    else:
        mask = eligibility_matrix(report, companies, frames['raw-close'].index)
    if mask is not None:
        for name in frames:
            frames[name] = frames[name].where(mask.reindex(columns=frames[name].columns))
    return frames, mask


def case(data, config, inputs, additions, identity_report=None, identity_stock_ids=None):
    feeds = ReplayMarketFeeds(inputs / 'execution-feeds', offline=True)
    overrides = (read(sealed.parent.OVERRIDES)['overrides'] | read(sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT / 'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp = TrackedCorporateActions(data.events, inputs / 'dividends', None, offline=True, overrides=overrides)
    args = (data.quotes, data.companies, data.days, data.entries, feeds, corp)
    kwargs = dict(start=data.start, end=data.end, stress_mode=config['stress'])
    if config['benchmark']:
        engine = BoardOnlyVerifiedBenchmark(*args, **kwargs)
    else:
        engine_type = HistoricalBoardReplay if identity_report else BoardOnlyVerifiedReplay
        if identity_report:
            kwargs['identity_report'] = identity_report
            kwargs['identity_stock_ids'] = identity_stock_ids
        engine = engine_type(*args, **kwargs, exit_signals=data.features,
                            action_dates=list(zip(data.events.stock_id, data.events.event_date)))
    try:
        account = engine.run()
        validate_completed_account(account, [str(d.date()) for d in data.days], data.start, data.end)
        audit = sealed.audit_account(account, engine.resource_plans, getattr(engine, 'slot_decisions', []),
                                     engine.board_decisions, config['benchmark'])
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return sealed.parent.blocked(config, str(exc), engine)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return sealed.parent.blocked(config, str(exc), engine)
    return dict(completed=True, config=config, account=account, summary=summarize(account), audit=audit,
        resource_plans=engine.resource_plans, slot_decisions=getattr(engine, 'slot_decisions', []),
        board_decisions=engine.board_decisions, identity_decisions=getattr(engine, 'identity_decisions', []),
        live_qualified=False, unseen_validation=False)


def run(output, signals_only=False, execution_inputs=None):
    started = time.monotonic()
    output = Path(output).resolve()
    if not output.is_relative_to(OUTPUT.resolve()) or output == OUTPUT.resolve():
        raise ValueError('Choose a new directory under the historical selector replay cache')
    execution_inputs = Path(execution_inputs).resolve() if execution_inputs is not None else None
    if output.exists():
        raise ValueError('Choose a new output directory; keep every prior result')
    print('verifying source identities', flush=True)
    refs, report, raw, _ = inventory()
    write(output / 'identity.json', refs)
    if execution_inputs is None:
        inputs = PARENT / 'inputs'
    else:
        # Caller-prepared additions are separately sealed, never implicit downloads.
        manifest = read(execution_inputs / 'manifest.json')
        for name, digest in manifest['files_sha256'].items():
            if sha(execution_inputs / name) != digest:
                raise ValueError('Prepared execution input changed: ' + name)
        refs.update(file_identities([execution_inputs / 'manifest.json'] +
            [execution_inputs / name for name in manifest['files_sha256']], ROOT))
        write(output / 'identity.json', refs)
        inputs = execution_inputs / 'inputs'
    write(output / 'execution-source.json', dict(path=str(inputs.relative_to(ROOT)), offline=True))
    original, _ = sealed.parent.source.inputs()
    frames = load_frames()
    companies = pd.read_parquet(ROOT / '.cache/million-replay-signals/companies.parquet')
    prefix_report = verify_prefix()
    prefix_quotes = pd.read_parquet(PREFIX / 'quotes.parquet')
    prefix_quotes['date'] = pd.to_datetime(prefix_quotes.date)
    bad_prefix = {(r['stock_id'], pd.Timestamp(r['date'])) for r in prefix_report['summary']['quarantine']}
    prefix_quotes = prefix_quotes.loc[[(s, d) not in bad_prefix for s, d in zip(prefix_quotes.stock_id, prefix_quotes.date)]]
    prefix_quality = {}
    for row in prefix_report['plan']['rows']:
        q = pd.read_parquet(PREFIX / (row['stock_id'] + '.parquet'))
        q['date'] = pd.to_datetime(q.date)
        prefix_quality[row['stock_id']] = q.set_index('date')['close']
    corrected, corrected_companies, prefix_anchors = restore_prefix(frames, companies, original.events,
        prefix_quotes, prefix_quality, prefix_report['plan'])
    write(output / 'listing-prefix.json', dict(anchors=prefix_anchors,
        unconfirmed_discrepancies=prefix_report['plan']['unconfirmed_discrepancies']))
    expanded_companies = pd.concat([companies, supplemental_companies(report, raw)], ignore_index=True)
    corrected_expanded_companies = pd.concat([corrected_companies, supplemental_companies(report, raw)], ignore_index=True)
    expanded, supplement_quotes = augment(frames, raw, original.events)
    corrected_expanded, _ = augment(corrected, raw, original.events)
    all_quotes = pd.read_parquet(ROOT / '.cache/million-replay-inputs/quotes.parquet')
    all_quotes['date'] = pd.to_datetime(all_quotes.date)
    bad = {(r['stock_id'], pd.Timestamp(r['date'])) for r in read(sealed.parent.source.five.AUDIT)['quarantine']}
    all_quotes = all_quotes.loc[[(sid, day) not in bad for sid, day in zip(all_quotes.stock_id, all_quotes.date)]]
    all_quotes = pd.concat([all_quotes, supplement_quotes], ignore_index=True)
    corrected_quotes = pd.concat([all_quotes, prefix_quotes], ignore_index=True)
    if corrected_quotes.duplicated(['stock_id', 'date']).any():
        raise ValueError('Restored prefix overlaps the old quote snapshot')
    additions = load_corporate_completion(ROOT)
    results, signal_summaries, baseline_entries = {}, {}, None
    with offline_only():
        for arm in ARMS:
            has_new = arm in ('omitted', 'combined')
            arm_companies = {'original': companies, 'identity': corrected_companies,
                             'omitted': expanded_companies, 'combined': corrected_expanded_companies}[arm]
            arm_base = {'original': frames, 'identity': corrected, 'omitted': expanded,
                        'combined': corrected_expanded}[arm]
            arm_frames, mask = prepare_arm(arm, arm_base, arm_companies,
                                           report, raw['plan']['stock_ids'])
            tick = time.monotonic()
            signals = build_signals(arm_frames, arm_companies, mask)
            entries = signals['entries']
            if arm == 'original':
                baseline_entries = read(BASE / 'signals.json')['entries']
                if encoded(entries) != encoded(baseline_entries):
                    raise ValueError('Neutral selector did not reproduce original entries')
            write(output / arm / 'signals.json', signals)
            for name, frame in arm_frames.items():
                frame.rename_axis('date').reset_index().to_parquet(output / arm / (name + '.parquet'), index=False)
            arm_companies.to_parquet(output / arm / 'companies.parquet', index=False)
            if mask is not None:
                mask.rename_axis('date').reset_index().to_parquet(output / arm / 'eligibility.parquet', index=False)
            old_ids, new_ids = {e['event_id'] for e in baseline_entries}, {e['event_id'] for e in entries}
            signal_summaries[arm] = dict(candidates=len(entries), elapsed_seconds=round(time.monotonic()-tick, 3),
                added=sorted(new_ids-old_ids), removed=sorted(old_ids-new_ids),
                supplement_candidates=sum(e['members'][0] in raw['plan']['stock_ids'] for e in entries),
                cohort_stocks=len(arm_companies), full_historical_market=False)
            print('signals', arm, len(entries), 'supplement', signal_summaries[arm]['supplement_candidates'], flush=True)
            if signals_only:
                continue
            pool = sorted({'0050'} | {e['members'][0] for e in entries})
            quote_source = corrected_quotes if arm in ('identity', 'combined') else all_quotes
            quotes = quote_source[quote_source.stock_id.isin(pool)].copy()
            if mask is not None:
                valid = [(day in mask.index and bool(mask.at[day, sid])) for sid, day in zip(quotes.stock_id, quotes.date)]
                quotes = quotes.loc[valid]
            data = replace(original, quotes=quotes, entries=entries, companies=arm_companies,
                           features=ExitSignals(arm_frames['close-official'][pool], original.days))
            for stress in ('control', 'combined'):
                configs = [dict(stress=stress, benchmark=False, board_only=True, position_count=5)]
                if arm == 'original':
                    configs += [dict(stress=stress, benchmark=True, board_only=True, position_count=0)]
                for config in configs:
                    name = ('benchmark' if config['benchmark'] else arm) + '_' + stress
                    print('account', name, flush=True)
                    identity_guard = report if arm in ('identity', 'combined') else None
                    # New stocks always use dated identity/settlement guards in the omitted-only arm.
                    if arm == 'omitted':
                        identity_guard = report
                    result = case(data, config, inputs, additions, identity_guard,
                                  raw['plan']['stock_ids'] if arm == 'omitted' else None)
                    if arm == 'original' and result['completed']:
                        parent_name = ('benchmark' if config['benchmark'] else 'capacity') + '_' + stress + '_board_only'
                        if encoded(result['account']) != encoded(read(PARENT / 'cases' / (parent_name + '.json'))['account']):
                            raise ValueError('Original full account did not reproduce: ' + name)
                    target = output / 'cases' / (name + '.json')
                    write(target, result)
                    results[name] = dict(completed=result['completed'], summary=result.get('summary'),
                        reason=result.get('reason'), path=str(target.relative_to(ROOT)), sha256=sha(target))
                    print(name, result.get('summary', {}).get('total_return', result.get('reason')), flush=True)
        for name, row in results.items():
            if name.startswith('benchmark') or not row['completed']:
                continue
            benchmark = results['benchmark_' + ('combined' if name.endswith('combined') else 'control')]
            if benchmark['completed']:
                row['benchmark'] = benchmark['summary']
                row['rolling252'] = rolling_comparison(read(ROOT / row['path'])['account'],
                                                       read(ROOT / benchmark['path'])['account'])
    if file_identities([ROOT / name for name in refs], ROOT) != refs:
        raise ValueError('Inputs or code changed during replay; result cannot be published')
    summary = dict(schema='historical_selector_replay_v1', signals=signal_summaries, cases=results,
        network_calls=0, database_writes=0, elapsed_seconds=round(time.monotonic()-started, 3),
        performance_recomputed=bool(results), all_completed=bool(results) and all(r['completed'] for r in results.values()),
        live_qualified=False, strict_data_ready=False, unseen_validation=False,
        limitations=['Known reconstructed cohort is not the complete historical market',
            'Daily board execution is estimated, not verified order/auction fills',
            'Revised price and legal-identity archives do not prove historical publication availability',
            'Missing settlement or execution sources block accounts; partial accounts have no full-period returns'])
    write(output / 'report.json', summary)
    write(output / 'manifest.json', dict(files_sha256={str(p.relative_to(output)): sha(p)
        for p in output.rglob('*') if p.is_file() and p.suffix != '.lock'}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--signals-only', action='store_true')
    parser.add_argument('--execution-inputs', type=Path)
    args = parser.parse_args()
    with file_lock(OUTPUT / '.run.lock', timeout=0):
        result = run(args.output, args.signals_only, args.execution_inputs)
    print(json.dumps(dict(all_completed=result['all_completed'], elapsed_seconds=result['elapsed_seconds'])))
