#!/usr/bin/env python3
"""Cross-compare frozen chip hypotheses, real account paths and event diagnostics."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import argparse
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from app.config import load_config
from app.file_lock import file_lock
from scripts import research_technical as parent
from scripts.research_exit_scenarios import read, write, sha, encoded, audit, summarize, TrackedCorporateActions
from skills.chip_research import ChipSignals, ChipReplay, FILTERS
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction

INPUT = ROOT / '.cache/chip-inputs'
OUTPUT = ROOT / '.cache/chip-research'
SPEC = ROOT / 'docs/prereg_chip_20260911.md'
ADDITIONS = ROOT / 'docs/chip_corporate_additions_20260911.json'
CODE = ('scripts/prepare_chip_inputs.py', 'scripts/research_chip.py', 'skills/chip_research.py')
MODES = ('control', *FILTERS, 'sell_full', 'sell_half')
PATH_MODES = ('control', 'trust_price', 'holder_margin', 'sell_half')


def block_comparison(frame, mode):
    """3-month moving blocks, preserving entry clustering and overlapping horizons."""
    known = frame.loc[frame[mode].notna() & frame.excess63.notna()].copy()
    yes = known[mode].astype(bool)
    def info(rows):
        return dict(n=len(rows), mean_excess63=None if rows.empty else float(rows.excess63.mean()),
                    win_rate=None if rows.empty else float(rows.excess63.gt(0).mean()))
    result = dict(passed=info(known[yes]), rejected=info(known[~yes]),
                  by_year={str(y): dict(passed=info(g[g[mode].astype(bool)]),
                    rejected=info(g[~g[mode].astype(bool)])) for y, g in known.groupby(known.entry_date.str[:4])})
    if not yes.any() or yes.all():
        return dict(**result, pass_minus_reject=None, ci95=None)
    months = pd.period_range('2022-01', '2026-09', freq='M').astype(str)
    groups = []
    for month in months:
        g = known.loc[known.entry_date.str[:7] == month]
        a, b = g[g[mode].astype(bool)], g[~g[mode].astype(bool)]
        groups.append([a.excess63.sum(), len(a), b.excess63.sum(), len(b)])
    values = np.array(groups)
    rng = np.random.default_rng(20260911)
    starts = rng.integers(0, len(months)-2, (2000, int(np.ceil(len(months)/3))))
    indices = (starts[:, :, None]+np.arange(3)).reshape(2000, -1)[:, :len(months)]
    sums = values[indices].sum(axis=1)
    valid = (sums[:, 1] > 0) & (sums[:, 3] > 0)
    differences = sums[valid, 0]/sums[valid, 1]-sums[valid, 2]/sums[valid, 3]
    return dict(**result, pass_minus_reject=float(known[yes].excess63.mean()-known[~yes].excess63.mean()),
        ci95=np.quantile(differences, [.025, .975]).tolist() if len(differences) else None,
        valid_bootstrap_samples=len(differences), multiplicity_adjusted=False)


def event_study(data, signals):
    rows = []
    close = data.features.adjusted_close
    cost_factor = lambda tax: (1-.001425-.0045-tax)/(1+.001425+.0045)
    for entry in data.entries:
        sid, day = entry['members'][0], pd.Timestamp(entry['entry_date'])
        i = data.days.get_loc(day)
        context = signals.context(i, sid, entry['event_id'])
        excess, stock_ret, etf_ret = None, None, None
        due = i+63
        if due < len(data.days) and data.days[due] <= pd.Timestamp(data.end):
            prices = [close.at[data.days[k], s] for k, s in ((i, sid), (due, sid), (i, '0050'), (due, '0050'))]
            if all(np.isfinite(v) and v > 0 for v in prices):
                stock_ret = prices[1]/prices[0]*cost_factor(.003)-1
                etf_ret = prices[3]/prices[2]*cost_factor(.001)-1
                excess = stock_ret-etf_ret
        rows.append(dict(event_id=entry['event_id'], stock_id=sid, entry_date=entry['entry_date'],
            **context, stock_return63=stock_ret, etf_return63=etf_ret, excess63=excess))
    frame = pd.DataFrame(rows)
    write(OUTPUT/'candidate_signals.json', frame.astype(object).where(pd.notna(frame), None).to_dict('records'))
    frame.to_csv(OUTPUT/'candidate_signals.csv', index=False)
    return dict(candidate_count=len(frame), completed_63_count=int(frame.excess63.notna().sum()),
        coverage={m: dict(known=int(frame[m].notna().sum()), passed=int(frame[m].eq(True).sum()),
                         unknown=int(frame[m].isna().sum())) for m in FILTERS},
        comparisons={m: block_comparison(frame, m) for m in FILTERS},
        is_executable_account=False)


def providers(data, offline):
    token = None if offline else load_config().finmind_token
    overrides = read(parent.OVERRIDES)['overrides'] | read(ADDITIONS)['overrides']
    return (ReplayMarketFeeds(INPUT/'execution-feeds', offline=offline, token=token),
        TrackedCorporateActions(data.events, INPUT/'dividends', token, offline=offline,
                                overrides=overrides))


def case(data, signals, mode, *, offline):
    feeds, corp = providers(data, offline)
    engine = ChipReplay(data.quotes, data.companies, data.days, data.entries, feeds, corp,
        technical_signals=data.features, chip_signals=signals, chip_mode=mode, start=data.start, end=data.end)
    try:
        account = engine.run()
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return dict(mode=mode, completed=False, reason=str(exc), live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(mode=mode, completed=False, reason=str(exc), live_qualified=False)
    checks = audit(account)
    result_summary = summarize(account)
    result_summary['corporate_assumptions'] = [r['action_id'] for r in account['corporate_actions']
        if r.get('stock_id') == '6691' and r.get('date') == '2023-07-17' and r.get('kind') == 'stock_dividend']
    return dict(mode=mode, completed=True, account=account, summary=result_summary, audit=checks,
        chip_decisions=engine.chip_decisions, half_states=engine.half_states,
        live_qualified=False)


def verify():
    meta = read(OUTPUT/'manifest.json')
    if meta.get('offline_identical') is not True or meta.get('live_qualified') is not False:
        raise ValueError('Unsealed chip research')
    for path, digest in meta['files_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Chip evidence changed: '+path)
    return meta


def research(*, offline=False, replay=False):
    began = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if replay:
        meta = verify()
    else:
        parent.verify_report()
        meta = read(INPUT/'manifest.json')
        if meta['spec_sha256'] != sha(SPEC):
            raise ValueError('Prerequisite changed')
        for path, digest in meta['files_sha256'].items():
            if sha(ROOT/path) != digest:
                raise ValueError('Raw chip source changed: '+path)
        for name in ('execution-feeds', 'dividends'):
            if not (INPUT/name).exists():
                shutil.copytree(parent.INPUT/name, INPUT/name)
        identity = dict(inputs_sha256=sha(INPUT/'manifest.json'), spec_sha256=sha(SPEC), corporate_additions_sha256=sha(ADDITIONS),
            code_sha256={name:sha(ROOT/name) for name in CODE})
        checkpoint = OUTPUT/'run_identity.json'
        if checkpoint.exists() and read(checkpoint) != identity:
            raise ValueError('Code or data changed since prior cases; archive old run explicitly')
        write(checkpoint, identity)
    data = parent.load_inputs()
    signals = ChipSignals(data, INPUT)
    print('chip features prepared', round(time.monotonic()-began, 2), flush=True)
    if not replay:
        event = event_study(data, signals)
        write(OUTPUT/'event_study.json', event)
    specs = [(m, None) for m in (*MODES, *('available_'+m for m in FILTERS))]
    specs += [(m, seed) for seed in range(20260911, 20260931) for m in PATH_MODES]
    results = {}
    for mode, seed in specs:
        name = mode if seed is None else str(seed)+'_'+mode
        path = OUTPUT/'cases'/(name+'.json')
        run_data = data
        if seed is not None:
            entries = deepcopy(data.entries)
            rng = np.random.default_rng(seed)
            for e in entries:
                e['priority'] = float(rng.random())
            run_data = replace(data, entries=entries)
        if path.exists() and not replay:
            result = read(path)
        else:
            tick = time.monotonic()
            result = case(run_data, signals, mode, offline=offline or replay)
            if mode == 'control' and seed is None:
                if not result['completed'] or encoded(result['account']) != encoded(data.parent['strategy']):
                    raise ValueError('Control must exactly reproduce original account')
            if replay:
                if encoded(result) != encoded(read(path)):
                    raise ValueError('Offline reproduction differs: '+name)
            else:
                write(path, result)
            print(name, 'complete' if result['completed'] else result['reason'],
                  round(time.monotonic()-tick, 2), flush=True)
        results[name] = {k: result[k] for k in ('mode', 'completed', 'summary', 'reason') if k in result}
    if replay:
        print('all chip cases identically reproduced offline', flush=True)
        return
    paired = {}
    for mode in PATH_MODES[1:]:
        pairs = [(results[str(seed)+'_'+mode], results[str(seed)+'_control']) for seed in range(20260911, 20260931)]
        differences = [a['summary']['final_nav']-b['summary']['final_nav'] for a, b in pairs if a['completed'] and b['completed']]
        paired[mode] = dict(complete_pairs=len(differences), wins=sum(v > 0 for v in differences),
            nav_difference_quantiles=np.quantile(differences, [.05, .5, .95]).tolist() if differences else None)
    write(OUTPUT/'summary.json', dict(cases=results, paired_priority=paired,
        benchmark=summarize(data.parent['benchmark']), event_study=read(OUTPUT/'event_study.json'),
        elapsed_seconds=time.monotonic()-began, live_qualified=False, unseen_validation=False))
    # Record the complete independently frozen source closure before offline replay.
    files = dict(read(parent.OUTPUT/'manifest.json')['verification_files_sha256'])
    for base in (INPUT, OUTPUT):
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.parquet', '.csv') and path != OUTPUT/'manifest.json':
                files[str(path.relative_to(ROOT))] = sha(path)
    for path in (ROOT/'.cache/chip-corporate-sources').glob('*.html'):
        files[str(path.relative_to(ROOT))] = sha(path)
    for path in (*[ROOT/name for name in CODE], SPEC, ADDITIONS, ROOT/'docs/chip_corporate_sources_20260911.md', ROOT/'docs/chip_data_correction_20260911.md'):
        files[str(path.relative_to(ROOT))] = sha(path)
    write(OUTPUT/'manifest.json', dict(files_sha256=files, offline_identical=False,
        live_qualified=False, unseen_validation=False, python=platform.python_version()))
    # Run all completed and blocked cases once more with no network or DB.
    for mode, seed in specs:
        name = mode if seed is None else str(seed)+'_'+mode
        entries = deepcopy(data.entries)
        if seed is not None:
            rng = np.random.default_rng(seed)
            for e in entries:
                e['priority'] = float(rng.random())
        actual = case(replace(data, entries=entries), signals, mode, offline=True)
        if encoded(actual) != encoded(read(OUTPUT/'cases'/(name+'.json'))):
            raise ValueError('Offline reproduction differs: '+name)
        print('offline identical', name, flush=True)
    sealed = read(OUTPUT/'manifest.json')
    for path, digest in sealed['files_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Evidence changed while reproducing: '+path)
    sealed['offline_identical'] = True
    write(OUTPUT/'manifest.json', sealed)
    print('chip research sealed', round(time.monotonic()-began, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--offline-replay', action='store_true')
    args = parser.parse_args()
    with file_lock(OUTPUT/'run.lock', timeout=1):
        if args.verify:
            verify()
            print('chip research hashes verified')
        else:
            research(offline=args.offline, replay=args.offline_replay)
