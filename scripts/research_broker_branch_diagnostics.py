#!/usr/bin/env python3
"""Offline fixed-cohort branch diagnostics, independent of portfolio slots."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.research_exit_scenarios import read, write, sha
from skills.broker_branch_diagnostics import NAMED_BRANCHES, branch_snapshot
from skills.independent_signals import SignalPath, observe
from skills.trial_registry import append_trial_registry

SPEC = ROOT/'docs/prereg_broker_branch_20261004.md'
INPUT = ROOT/'.cache/partial-risk-2019-20260929/inputs-final'
OLD = ROOT/'.cache/broker-persistence-20260924'
SIGNALS = ROOT/'.cache/five-axis-20260913/rebuild/signals.json'
POC = ROOT/'.cache/poc-latest-20261003/explorer-v2/payload.json'


def describe(rows):
    closed = [r for r in rows if r['outcome']['status'] == 'closed']
    net = [r['outcome']['net_return'] for r in closed]
    return dict(signals=len(rows), stocks=len({r['stock_id'] for r in rows}),
        closed=len(closed), statuses=dict(Counter(r['outcome']['status'] for r in rows)),
        win_rate=float(np.mean(np.asarray(net) > 0)) if net else None,
        mean_net_return=float(np.mean(net)) if net else None,
        median_net_return=float(np.median(net)) if net else None,
        stop_rate=float(np.mean([r['outcome']['reason'] == 'loss12' for r in closed])) if net else None,
        peak20_rate=float(np.mean([r['outcome']['peak_close_return'] >= .20 for r in closed])) if net else None)


def comparisons(rows):
    definitions = {
        'named_positive': lambda r: r['branch']['named_positive'],
        'concentrated_directional': lambda r: r['branch']['concentrated_directional'],
        'persistent5': lambda r: r['persistence5']['passed'],
        'combined': lambda r: r['branch']['concentrated_directional'] and r['persistence5']['passed'],
    }
    definitions.update({f'named_{sid}': lambda r, sid=sid: r['branch']['named'][sid]['positive_observed']
                        for sid in NAMED_BRANCHES})
    output = {}
    for name, predicate in definitions.items():
        known = [r for r in rows if r['branch']['known']
                 and (name not in ('persistent5', 'combined') or r['persistence5']['known'])]
        passed = [r for r in known if predicate(r)]
        other = [r for r in known if not predicate(r)]
        output[name] = dict(known_signals=len(known), unknown_signals=len(rows)-len(known),
            other_definition=('no_positive_named_flow_observed_not_verified_zero_trading'
                              if name.startswith('named_') else 'condition_not_met_on_same_data_coverage'),
            passed=describe(passed), other=describe(other),
            annual={year: dict(passed=describe([r for r in passed if r['signal_date'][:4] == year]),
                              other=describe([r for r in other if r['signal_date'][:4] == year]))
                    for year in sorted({r['signal_date'][:4] for r in rows})})
    return output


def run(output):
    if output.exists():
        raise ValueError('Choose a new output; prior research is immutable')
    refs = {}
    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Frozen input changed: '+str(path))
        refs[str(path.relative_to(ROOT))] = digest
        return digest
    for path in (SPEC, Path(__file__), ROOT/'skills/broker_branch_diagnostics.py',
                 ROOT/'skills/independent_signals.py', ROOT/'skills/exit_policy.py',
                 ROOT/'skills/million_replay.py', ROOT/'skills/trial_registry.py',
                 ROOT/'tests/test_broker_branch_diagnostics.py',
                 ROOT/'scripts/research_exit_scenarios.py'):
        bind(path)
    old_manifest = read(OLD/'manifest.json')['files_sha256']
    bind(OLD/'manifest.json')
    for path in (SIGNALS, OLD/'signals.json'):
        bind(path, old_manifest[str(path.relative_to(ROOT))])
    entries = sorted(read(SIGNALS)['entries'], key=lambda r: (r['signal_date'],r['members'][0]))
    if len(entries) != 458 or len({e['event_id'] for e in entries}) != 458:
        raise ValueError('Frozen cohort changed')
    persistence = read(OLD/'signals.json')
    price_manifest = read(INPUT/'manifest.json')['files_sha256']
    bind(INPUT/'manifest.json')
    for name in ('close-official.parquet','close-quality.parquet','eligibility.parquet',
                 'quotes-unmasked.parquet','companies.parquet'):
        bind(INPUT/name, price_manifest[name])
    ids = sorted({e['members'][0] for e in entries})
    frames = [pd.read_parquet(INPUT/name, columns=['date',*ids]).set_index('date')
              for name in ('close-official.parquet','close-quality.parquet','eligibility.parquet')]
    close, other, eligible = frames
    if eligible.isna().any().any() or any(t != np.dtype(bool) for t in eligible.dtypes):
        raise ValueError('Eligibility requires explicit booleans')
    days = pd.DatetimeIndex(close.index)
    if str(days[-1].date()) != '2026-09-09' or any(not f.index.equals(days) for f in frames):
        raise ValueError('Frozen calendar changed')
    raw = pd.read_parquet(INPUT/'quotes-unmasked.parquet')
    raw = raw[raw.stock_id.isin(ids)]
    prices = {k: raw.pivot(index='date',columns='stock_id',values=k).reindex(index=days,columns=ids)
              for k in ('close','high','low','volume')}
    paths = {sid: SignalPath(days, close[sid].to_numpy(float), other[sid].to_numpy(float),
        eligible[sid].to_numpy(bool), *[prices[k][sid].to_numpy(float) for k in prices]) for sid in ids}
    names = pd.read_parquet(INPUT/'companies.parquet').set_index('stock_id')['name'].to_dict()
    chip_manifest_path = ROOT/'.cache/chip-inputs/manifest.json'
    chip_manifest = read(chip_manifest_path)['files_sha256']
    bind(chip_manifest_path)
    rows, aliases, ambiguities, seen = [], {}, {}, set()
    for e in entries:
        sid, date = e['members'][0], e['signal_date']
        raw_path = ROOT/f'.cache/chip-inputs/raw/broker/{sid}_{date}_{date}.parquet'
        bind(raw_path, chip_manifest[str(raw_path.relative_to(ROOT))])
        frame = pd.read_parquet(raw_path)
        branch = branch_snapshot(frame, sid, date)
        if not frame.empty:
            for label, code in frame[['securities_trader','securities_trader_id']].drop_duplicates().itertuples(index=False,name=None):
                if code in NAMED_BRANCHES:
                    aliases.setdefault(code, set()).add(label)
                if any(word in label for word in ('竹科','嘉義','台中')):
                    ambiguities.setdefault(code,set()).add(label)
        index = int(days.get_loc(pd.Timestamp(date)))+1
        if index < len(days):
            result = observe(paths[sid], index)
            entry_date = str(days[index].date())
        else:
            result = dict(status='pending_entry', issue='no_next_session')
            entry_date = None
        row = dict(event_id=e['event_id'], stock_id=sid, name=names.get(sid), signal_date=date,
            entry_date=entry_date, first_for_stock=sid not in seen,
            branch=branch, persistence5=persistence[e['event_id']]['5'], outcome=result)
        seen.add(sid); rows.append(row)
    receipt_path = POC.with_name('receipt.json')
    bind(receipt_path)
    receipt = read(receipt_path)
    expected = receipt['output_sha256'].get(str(POC.relative_to(ROOT))) or receipt['output_sha256'].get('payload.json')
    if not expected:
        raise ValueError('Missing frozen POC payload digest')
    bind(POC, expected)
    current = read(POC)['signals']
    old_keys = {(r['stock_id'],r['signal_date']) for r in rows}
    overlap = [s for s in current if (s['stock_id'],s['signal_date']) in old_keys]
    coverage = dict(current_2026_candidates=len(current), old_cohort_overlap=len(overlap),
        current_2026_not_covered=len(current)-len(overlap))
    report = dict(schema='broker_branch_diagnostic_v1', network_requests=0,
        live_qualified=False, unseen_validation=False, cash_account=False, actual_fill_verified=False,
        period=dict(signal_start=rows[0]['signal_date'], signal_end=rows[-1]['signal_date'],
                    prices_through=str(days[-1].date())),
        outcome_model='T+1_HL2_loss12_time63_proportional_costs_not_cash_account',
        all_signals=describe(rows), coverage=coverage,
        branch_coverage=dict(Counter('known' if r['branch']['known'] else r['branch']['reason'] for r in rows)),
        comparisons=comparisons(rows), first_per_stock=comparisons([r for r in rows if r['first_for_stock']]),
        named_observation_coverage={sid: dict(
            observed=sum(r['branch']['known'] and r['branch']['named'][sid]['observed'] for r in rows),
            absent_from_known_snapshot=sum(r['branch']['known'] and not r['branch']['named'][sid]['observed'] for r in rows))
            for sid in NAMED_BRANCHES},
        named_aliases={k:sorted(v) for k,v in aliases.items()},
        ambiguous_locations={k:sorted(v) for k,v in ambiguities.items()}, source_sha256=refs)
    for path, digest in refs.items():
        if sha(ROOT/path) != digest:
            raise ValueError('Inputs changed during research: '+path)
    output.mkdir(parents=True)
    write(output/'rows.json', rows)
    report['rows_sha256'] = sha(output/'rows.json')
    write(output/'report.json', report)
    print(dict(output=str(output), signals=len(rows), coverage=coverage,
               branch_coverage=report['branch_coverage'], outcomes=report['all_signals']), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(ROOT)
    trial = dict(source='broker_branch_diagnostics_20261004',
                 timestamp=datetime.now(timezone.utc).isoformat(),
                 output=str(output.relative_to(ROOT)), command=' '.join(sys.argv),
                 cash_account=False, live_qualified=False)
    try:
        report = run(output)
    except Exception as exc:
        append_trial_registry(dict(trial, completed=False, error=type(exc).__name__+': '+str(exc)))
        raise
    # Log every compared condition, including failures and repeated diagnostics;
    # do not register only the attractive condition as a single successful trial.
    for condition in report['comparisons']:
        append_trial_registry(dict(trial, completed=True, condition=condition,
                                   report_sha256=sha(output/'report.json')))
