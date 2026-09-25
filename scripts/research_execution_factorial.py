#!/usr/bin/env python3
"""A fixed eight-cell experiment plus no-op and benchmark reproduction checks."""
from dataclasses import replace
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from app.historical_selector_ui import load as load_publication, REPORT as PUBLICATION
from scripts import research_historical_selector_replay as parent
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.execution_factorial import FactorialReplay, flags, LABELS, shapley, stock_pnl, path_comparison, fixed_trade_slippage_delta, load_capital_terms, CORPORATE_DOCUMENT
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.scenario_exit_replay import ExitSignals
from skills.verified_backtest_tool import offline_only

OUTPUT = ROOT / '.cache/execution-factorial-20260926'
SPEC = ROOT / 'docs/prereg_execution_factorial_20260926.md'
CODE = [Path(__file__), ROOT / 'skills/execution_factorial.py', SPEC, ROOT / CORPORATE_DOCUMENT]


def load_data(publication):
    folder = (ROOT / publication['run_manifest']['path']).parent
    arm = folder / 'combined'
    original, _ = parent.sealed.parent.source.inputs()
    entries = read(arm / 'signals.json')['entries']
    companies = pd.read_parquet(arm / 'companies.parquet')
    pool = sorted({'0050'} | {e['members'][0] for e in entries})
    quotes = pd.read_parquet(ROOT / '.cache/million-replay-inputs/quotes.parquet')
    quotes['date'] = pd.to_datetime(quotes.date)
    audit = read(parent.sealed.parent.source.five.AUDIT)
    bad = {(r['stock_id'], pd.Timestamp(r['date'])) for r in audit['quarantine']}
    quotes = quotes.loc[[(s, d) not in bad for s, d in zip(quotes.stock_id, quotes.date)]]
    for directory in (parent.DIRECTORY, parent.PREFIX):
        extra = pd.read_parquet(directory / 'quotes.parquet')
        extra['date'] = pd.to_datetime(extra.date)
        report = read(directory / 'manifest.json')
        bad = {(r['stock_id'], pd.Timestamp(r['date'])) for r in report['summary']['quarantine']}
        extra = extra.loc[[(s, d) not in bad for s, d in zip(extra.stock_id, extra.date)]]
        quotes = pd.concat([quotes, extra], ignore_index=True)
    quotes = quotes.loc[quotes.stock_id.isin(pool)]
    if quotes.duplicated(['stock_id', 'date']).any():
        raise ValueError('Duplicate quote in fixed factorial input')
    mask = pd.read_parquet(arm / 'eligibility.parquet').set_index('date')
    mask.index = pd.to_datetime(mask.index)
    quotes = quotes.loc[[(d in mask.index and bool(mask.at[d, s])) for s, d in zip(quotes.stock_id, quotes.date)]]
    close = pd.read_parquet(arm / 'close-official.parquet', columns=['date', *pool]).set_index('date')
    close.index = pd.to_datetime(close.index)
    data = replace(original, quotes=quotes, companies=companies, entries=entries,
                   features=ExitSignals(close, original.days))
    inputs = ROOT / read(folder / 'execution-source.json')['path']
    return data, inputs, read(parent.IDENTITY)


def run_factor(data, inputs, identity, additions, mask):
    feeds = ReplayMarketFeeds(inputs / 'execution-feeds', offline=True)
    overrides = (read(parent.sealed.parent.OVERRIDES)['overrides'] | read(parent.sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT / 'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp = TrackedCorporateActions(data.events, inputs / 'dividends', None, offline=True, overrides=overrides)
    engine = FactorialReplay(data.quotes, data.companies, data.days, data.entries, feeds, corp,
        start=data.start, end=data.end, identity_report=identity, factor_mask=mask,
        exit_signals=data.features, action_dates=list(zip(data.events.stock_id, data.events.event_date)))
    config = dict(factor_mask=mask, factors=flags(mask), benchmark=False, board_only=True, position_count=5)
    try:
        account = engine.run()
        validate_completed_account(account, [str(d.date()) for d in data.days], data.start, data.end)
        audit = parent.sealed.audit_account(account, engine.resource_plans, engine.slot_decisions,
                                            engine.board_decisions, False)
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return parent.sealed.parent.blocked(config, str(exc), engine)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return parent.sealed.parent.blocked(config, str(exc), engine)
    return dict(completed=True, config=config, account=account, summary=summarize(account), audit=audit,
        resource_plans=engine.resource_plans, slot_decisions=engine.slot_decisions,
        board_decisions=engine.board_decisions, identity_decisions=engine.identity_decisions,
        stock_pnl=stock_pnl(account, engine.marks), live_qualified=False, unseen_validation=False)


def analyze(cases):
    active = {m: cases['factor_' + str(m)] for m in range(8)}
    if not all(r['completed'] for r in cases.values()):
        return dict(complete=False, reason='Missing evidence blocks full-factorial attribution')
    outcomes = {m: row['summary']['total_return'] for m, row in active.items()}
    base, worst = active[0], active[7]
    names = {t['stock_id']: t['name'] for row in active.values() for t in row['account']['trades']}
    pnl_gap = []
    for sid in sorted(set(base['stock_pnl']) | set(worst['stock_pnl'])):
        normal = base['stock_pnl'].get(sid, {}).get('profit', 0.)
        stress = worst['stock_pnl'].get(sid, {}).get('profit', 0.)
        pnl_gap.append(dict(stock_id=sid, name=names.get(sid, sid), normal_profit=normal,
                            stress_profit=stress, difference=stress-normal))
    gap = worst['summary']['final_nav'] - base['summary']['final_nav']
    if abs(sum(r['difference'] for r in pnl_gap) - gap) > .03:
        raise ValueError('Stock profit differences do not explain the final asset difference')
    singles = {factor: outcomes[1 << i] - outcomes[0]
               for i, factor in enumerate(('slippage', 'entry_delay', 'exit_delay'))}
    return dict(complete=True, total_return_change=outcomes[7]-outcomes[0], final_asset_change=gap,
        single_factor_return_changes=singles, shapley_return_changes=shapley(outcomes),
        nonadditive_return_change=outcomes[7]-outcomes[0]-sum(singles.values()),
        paths={str(m): path_comparison(base['account'], row['account']) for m, row in active.items()},
        stock_profit_changes=sorted(pnl_gap, key=lambda row: row['difference']),
        fixed_original_fills_extra_slippage=fixed_trade_slippage_delta(base['account']),
        fixed_fills_note='Arithmetic only; same historical fills cannot be assumed after cash and slot paths change')


def run(output):
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT) or output == OUTPUT:
        raise ValueError('Keep sealed results; choose a new directory under the factorial cache')
    publication = load_publication()
    capital_terms, capital_evidence = load_capital_terms(ROOT)
    refs = dict(publication['source_sha256'])
    refs.update(capital_evidence)
    refs.update(file_identities([PUBLICATION, PUBLICATION.with_suffix('.sha256'), *CODE], ROOT))
    write(output / 'identity.json', refs)
    cases = {}
    with offline_only():
        data, inputs, identity = load_data(publication)
        if len(data.entries) != 454:
            raise ValueError('Frozen signal count changed')
        additions = parent.load_corporate_completion(ROOT) | capital_terms
        # Verify both endpoints before interpreting the six new combinations.
        for mask in (0, 7, 1, 2, 4, 3, 5, 6):
            print('running factor', mask, flush=True)
            value = run_factor(data, inputs, identity, additions, mask)
            if mask in (0, 7):
                old = read(ROOT / publication['cases']['combined_' + ('control' if mask == 0 else 'combined')]['result']['path'])
                if not value['completed'] or encoded(value['account']) != encoded(old['account']):
                    raise ValueError('Factor endpoint does not reproduce the sealed account: ' + str(mask))
            cases['factor_' + str(mask)] = value
            write(output / 'cases' / ('factor_' + str(mask) + '.json'), value)
            print(mask, value.get('summary', {}).get('total_return', value.get('reason')), flush=True)
        for mode in ('depth', 'quote', 'control', 'combined'):
            benchmark = mode in ('control', 'combined')
            name = 'benchmark_' + mode if benchmark else 'noop_' + mode
            value = parent.case(data, dict(stress=mode, benchmark=benchmark, board_only=True,
                position_count=0 if benchmark else 5), inputs, additions, None if benchmark else identity)
            expected = (read(ROOT / publication['cases'][name]['result']['path'])['account']
                        if benchmark else cases['factor_0']['account'])
            if not value['completed'] or encoded(value['account']) != encoded(expected):
                raise ValueError('Benchmark/no-op control did not reproduce: ' + name)
            cases[name] = value
            write(output / 'cases' / (name + '.json'), value)
            print(name, 'identical', flush=True)
    analysis = analyze(cases)
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Source or code changed during factorial replay')
    rows = {name: dict(completed=c['completed'], summary=c.get('summary'), reason=c.get('reason'),
                      config=c['config'], result=dict(path=str((output / 'cases' / (name + '.json')).relative_to(ROOT)),
                      sha256=sha(output / 'cases' / (name + '.json')))) for name, c in cases.items()}
    report = dict(schema='execution_factorial_v1', cases=rows, analysis=analysis,
        start='2022-01-03', end='2026-09-09', initial_cash=1_000_000, candidate_count=454,
        labels=list(LABELS), all_completed=all(c['completed'] for c in cases.values()),
        network_calls=0, database_writes=0, elapsed_seconds=round(time.monotonic()-started, 3),
        live_qualified=False, strict_data_ready=False, unseen_validation=False,
        limitations=['Daily board fills remain estimates; odd-lot evidence is not added',
            'Shapley shares depend on the three fixed factors and average their interactions',
            '0050 has no strategy entry/exit signals; only slippage changes for this benchmark',
            'Used historical data is not an unseen validation sample'])
    write(output / 'report.json', report)
    write(output / 'manifest.json', dict(files_sha256={str(p.relative_to(output)): sha(p)
        for p in output.rglob('*.json') if p.name != 'manifest.json'}))
    return report


def verify(left, right, output):
    left, right, output = (Path(p).resolve() for p in (left, right, output))
    if left == right or output.exists():
        raise ValueError('Compare separate runs and preserve previous evidence')
    refs = {}
    reports = []
    for folder in (left, right):
        report = read(folder / 'report.json')
        if report.get('all_completed') is not True or report.get('analysis', {}).get('complete') is not True:
            raise ValueError('Cannot verify an incomplete factorial run')
        manifest = read(folder / 'manifest.json')
        for name, digest in manifest['files_sha256'].items():
            if sha(folder / name) != digest:
                raise ValueError('Factorial artifact changed: ' + name)
            refs[str((folder / name).relative_to(ROOT))] = digest
        refs.update(read(folder / 'identity.json'))
        refs[str((folder / 'manifest.json').relative_to(ROOT))] = sha(folder / 'manifest.json')
        reports.append(report)
    if read(left / 'identity.json') != read(right / 'identity.json'):
        raise ValueError('Source identities differ')
    if set(reports[0]['cases']) != set(reports[1]['cases']) or len(reports[0]['cases']) != 12:
        raise ValueError('Both runs must contain twelve cases')
    for name in reports[0]['cases']:
        if read(left / 'cases' / (name + '.json')) != read(right / 'cases' / (name + '.json')):
            raise ValueError('Full account did not reproduce: ' + name)
    if reports[0]['analysis'] != reports[1]['analysis']:
        raise ValueError('Attribution changed between offline runs')
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Source changed before offline verification')
    proof = dict(schema='execution_factorial_offline_v1', passed=True, compared_cases=12,
        all_completed=all(r['all_completed'] for r in reports), source_sha256=refs, network_calls=0)
    write(output, proof)
    output.with_suffix('.sha256').write_text(sha(output) + '\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path, nargs=2)
    args = parser.parse_args()
    with file_lock(OUTPUT / '.run.lock', timeout=0):
        value = verify(*args.compare, args.output) if args.compare else run(args.output)
    print('passed', value.get('passed', value.get('all_completed')), flush=True)
