#!/usr/bin/env python3
"""Run eight frozen recipes with separately verified corporate additions, offline."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from scripts import research_board_only_supplement as sealed
from scripts.research_exit_scenarios import read, write, sha
from skills.backtest_case_cache import file_identities
from skills.backtest_corporate_completion import DOCUMENT, load_corporate_completion
from skills.verified_backtest_tool import source_context, offline_only


def run(output):
    output = Path(output).resolve()
    allowed = (ROOT / '.cache/backtest-corporate-completion-20260925').resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError('Choose a new output directory under the corporate completion cache')
    identity, _ = source_context()
    additions = load_corporate_completion(ROOT)
    document = read(ROOT / DOCUMENT)
    extra = [ROOT / DOCUMENT, Path(__file__), ROOT / 'skills/backtest_corporate_completion.py',
             ROOT / 'docs/backtest_corporate_completion_20260925.md']
    refs = identity['source_sha256'] | document['evidence_sha256'] | file_identities(extra, ROOT)
    output.mkdir(parents=True)
    write(output / 'identity.json', refs)
    shutil.copytree(sealed.SOURCES / 'inputs', output / 'inputs')
    write(output / 'overrides.json', document)
    started = time.monotonic()
    cases = {}
    with offline_only():
        data, _ = sealed.parent.source.inputs()
        for name, config in sealed.parent.configurations():
            print('running', name, flush=True)
            result = sealed.run_case(data, config, output / 'inputs', additions)
            write(output / 'cases' / (name + '.json'), result)
            cases[name] = dict(completed=result['completed'], reason=result.get('reason'),
                              summary=result.get('summary'), config=config,
                              path=str((output / 'cases' / (name + '.json')).relative_to(ROOT)),
                              sha256=sha(output / 'cases' / (name + '.json')))
            print(name, result.get('summary', {}).get('total_return', result.get('reason')), flush=True)
    if file_identities([ROOT / name for name in refs], ROOT) != refs:
        raise ValueError('Source/code changed during corporate completion probe; do not publish')
    report = dict(all_completed=all(c['completed'] for c in cases.values()), cases=cases,
                  network_calls=0, elapsed_seconds=round(time.monotonic()-started, 3),
                  live_qualified=False, unseen_validation=False,
                  fractional_cash_is_verified_spendable=False)
    write(output / 'report.json', report)
    files = [p for p in output.rglob('*') if p.is_file()]
    write(output / 'manifest.json', dict(files_sha256={str(p.relative_to(output)): sha(p) for p in files}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with file_lock(ROOT / '.cache/backtest-corporate-completion-20260925.lock', timeout=0):
        result = run(args.output)
    print(json.dumps(dict(all_completed=result['all_completed'], elapsed_seconds=result['elapsed_seconds'])))
