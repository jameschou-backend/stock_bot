#!/usr/bin/env python3
"""Compare two independent offline executions, including blocked cases and matrices."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from scripts.audit_current_causality_20260925 import deterministic
from skills.backtest_case_cache import file_identities


def verify(left, right, output):
    left, right, output = (Path(p).resolve() for p in (left, right, output))
    if left == right or output.exists():
        raise ValueError('Compare separate runs and keep prior verification evidence')
    reports, refs = [], {}
    for directory in (left, right):
        manifest = read(directory / 'manifest.json')
        for name, expected in manifest['files_sha256'].items():
            path = directory / name
            if sha(path) != expected:
                raise ValueError('Replay artifact changed: ' + name)
            refs[str(path.relative_to(ROOT))] = expected
        refs.update(read(directory / 'identity.json'))
        refs[str((directory / 'manifest.json').relative_to(ROOT))] = sha(directory / 'manifest.json')
        reports.append(read(directory / 'report.json'))
    if read(left / 'identity.json') != read(right / 'identity.json'):
        raise ValueError('Offline executions used different input/code identities')
    if set(reports[0]['cases']) != set(reports[1]['cases']) or len(reports[0]['cases']) != 10:
        raise ValueError('Both executions must contain all ten cases')
    for name in reports[0]['cases']:
        if read(left / 'cases' / (name + '.json')) != read(right / 'cases' / (name + '.json')):
            raise ValueError('Offline account mismatch: ' + name)
    for arm in ('original', 'identity', 'omitted', 'combined'):
        if deterministic(read(left / arm / 'signals.json')) != deterministic(read(right / arm / 'signals.json')):
            raise ValueError('Offline selector mismatch: ' + arm)
        for path in (left / arm).glob('*.parquet'):
            if sha(path) != sha(right / arm / path.name):
                raise ValueError('Offline prepared matrix mismatch: ' + arm + '/' + path.name)
    refs[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Verified run source/code has changed')
    result = dict(schema='historical_selector_offline_v1', passed=True,
        compared_cases=10, compared_selectors=4, network_calls=0,
        all_completed=reports[0]['all_completed'], source_sha256=refs, live_qualified=False)
    write(output, result)
    output.with_suffix('.sha256').write_text(sha(output) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.left, args.right, args.output)
    print('passed', result['passed'], 'compared_cases', result['compared_cases'])
