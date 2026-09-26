#!/usr/bin/env python3
"""Reproduce existing studies with two primary-source settlement repairs only."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from app.residual_slots_ui import load
from scripts import research_exit_mechanisms as exits
from scripts import research_volatility_budget as volatility
from scripts.research_exit_scenarios import read, write, sha, encoded
from skills.backtest_case_cache import file_identities
from skills.exit_corporate_completion import load_exit_completion, DOCUMENT
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

BASE = ROOT / '.cache/completed-corporate-studies-20260927'
FAMILIES = {'exits': (exits, ROOT / '.cache/exit-mechanisms-20260927/final-v1'),
            'volatility': (volatility, ROOT / '.cache/volatility-budget-20260927/final-v1')}


def check_manifest(folder):
    manifest = read(folder / 'manifest.json')
    for name, digest in manifest['files_sha256'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or sha(path) != digest:
            raise ValueError('Study artifact changed: ' + name)
    return manifest


def run(family, output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(BASE) or output == BASE:
        raise ValueError('Choose a new immutable completion directory')
    tick = time.monotonic()
    module, previous = FAMILIES[family]
    old_manifest = check_manifest(previous)
    old = read(previous / 'report.json')
    refs = read(previous / 'identity.json')
    for name, digest in old_manifest['files_sha256'].items():
        refs[str((previous / name).relative_to(ROOT))] = digest
    supplement, primary = load_exit_completion(ROOT)
    refs.update(primary)
    refs.update(file_identities([previous / 'manifest.json', Path(__file__), ROOT / DOCUMENT,
        ROOT / 'skills/exit_corporate_completion.py',
        ROOT / 'docs/exit_corporate_completion_20260927.md'], ROOT))
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Prior study source changed')
    write(output / 'identity.json', refs)
    original, selector = load(), exits.parent.load_selector()
    rows, trials = {}, []
    with offline_only():
        data, inputs, identity = exits.parent.parent.load_data(selector)
        additions = (exits.parent.parent.parent.load_corporate_completion(ROOT)
                     | exits.load_capital_terms(ROOT)[0] | supplement)
        for arm in module.ARMS:
            for mask in range(8):
                name = f'{arm}_{mask}'
                status, error = 'failed', None
                print(f'running {family} {name}', flush=True)
                target = arm if family == 'exits' else module.ARMS[arm]
                try:
                    value = module.case(data, inputs, identity, additions, mask, target)
                    write(output / 'cases' / (name + '.json'), value)
                    old_case = read(ROOT / old['cases'][name]['result']['path'])
                    if old_case['completed'] and (not value['completed'] or
                            encoded(old_case['account']) != encoded(value['account'])):
                        raise ValueError('Previously completed account changed: ' + name)
                    status = 'completed' if value['completed'] else 'blocked'
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    write(output / 'errors' / (name + '.json'), dict(error=error))
                    raise
                finally:
                    record = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                        source='corporate_completion_20260927', family=family, case=name,
                        command=' '.join(sys.argv), factor_mask=mask, status=status, error=error,
                        result_path=str((output / 'cases' / (name + '.json')).relative_to(ROOT)))
                    append_trial_registry(record)
                    trials.append(record)
                    write(output / 'trials.json', trials)
                row = dict(completed=value['completed'], config=value['config'],
                    summary=value.get('summary'), reason=value.get('reason'),
                    prior_completed=old_case['completed'],
                    prior_account_unchanged=True if old_case['completed'] else None,
                    result=dict(path=str((output / 'cases' / (name + '.json')).relative_to(ROOT)),
                                sha256=sha(output / 'cases' / (name + '.json'))))
                if value['completed']:
                    benchmark = read(ROOT / original['cases'][
                        'benchmark_combined' if mask & 1 else 'benchmark_control']['result']['path'])
                    metric_input = dict(value, volatility_decisions=[]) if family == 'exits' else value
                    row['metrics'] = volatility.metrics(metric_input, benchmark)
                    if family == 'exits':
                        for key in ('reduced_budget_decisions', 'missing_volatility_decisions'):
                            row['metrics'].pop(key)
                        row['exit_reasons'] = value['audit']['trigger_reasons']
                rows[name] = row
                print(name, value.get('summary', {}).get('total_return', value.get('reason')), flush=True)
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Completion source changed during replay')
    report = dict(schema=old['schema'], start=data.start, end=data.end, initial_cash=1_000_000,
        candidate_count=len(data.entries), cases=rows,
        all_completed=all(r['completed'] for r in rows.values()),
        prior_completed_unchanged=sum(r['prior_account_unchanged'] is True for r in rows.values()),
        repaired_cases=sum(not r['prior_completed'] and r['completed'] for r in rows.values()),
        elapsed_seconds=round(time.monotonic()-tick, 3), trial_count=len(trials),
        network_calls=0, database_writes=0, live_qualified=False, unseen_validation=False,
        settlement_supplement=DOCUMENT)
    if family == 'exits':
        report['screen'] = exits.screen(rows)
    else:
        report['validation'] = volatility.assess_family(rows)
    write(output / 'report.json', report)
    write(output / 'manifest.json', dict(files_sha256={str(p.relative_to(output)): sha(p)
        for p in output.rglob('*.json') if p.name != 'manifest.json'}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path, nargs=2)
    args = parser.parse_args()
    with file_lock(BASE / '.run.lock', timeout=0):
        result = (FAMILIES[args.family][0].verify(*args.compare, args.output) if args.compare
                  else run(args.family, args.output))
    print('complete', result['all_completed'], flush=True)
