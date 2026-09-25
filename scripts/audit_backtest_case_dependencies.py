#!/usr/bin/env python3
"""Reproduce the input dependency review without modifying sealed backtests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.backtest_data_evidence import checked, digest, verify_report
from skills.backtest_case_dependencies import configurations, dependencies

BASE = ROOT/'artifacts/forward_simulation/backtest_data_completion_20260925.json'
OUTPUT = ROOT/'.cache/backtest-case-dependencies-20260925/report-verified.json'
SOURCE_CLOSURES = (
    ('.cache/backtest-corporate-completion-20260925/probe-v2', 'identity.json', None),
    ('.cache/sector-accounts-20260925', 'report.json', 'source_sha256'),
)
CODE = (
    'scripts/audit_backtest_case_dependencies.py', 'skills/backtest_case_dependencies.py',
    'scripts/prepare_million_signals.py', 'skills/diffusion_signals.py', 'skills/regime_state.py',
    'skills/sector_account_replay.py', 'skills/surge_sector.py', 'skills/surge_anatomy.py',
    'skills/scenario_exit_replay.py', 'skills/replay_corporate_actions.py',
    'skills/pending_share_entitlements.py', 'skills/million_replay.py',
)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'


def build():
    base = verify_report(BASE, ROOT)
    if set(base['case_sources']) != set(configurations()) or set(base['cases']) != set(configurations()):
        raise ValueError('The reviewed twenty-case universe differs')
    refs = {str(BASE.relative_to(ROOT)):digest(BASE)}
    reviewed_code = {}
    for directory, name, key in SOURCE_CLOSURES:
        manifest_path = ROOT/directory/'manifest.json'
        manifest = json.loads(checked(manifest_path, digest(manifest_path), refs, ROOT).read_text())
        path = checked(ROOT/directory/name, manifest['files_sha256'][name], refs, ROOT)
        identity = json.loads(path.read_text())
        sources = identity[key] if key else identity
        for source, expected in sources.items():
            if source.endswith('.py'):
                checked(ROOT/source, expected, reviewed_code, ROOT)
    cases = {}
    for name, descriptor in sorted(base['case_sources'].items()):
        path = checked(ROOT/descriptor['path'], descriptor['sha256'], refs, ROOT)
        case = json.loads(path.read_text())
        cases[name] = dependencies(name, case['config'], base['cases'][name])
    counts = Counter(row['code'] for case in cases.values() for row in case['dependencies'] if row['required'])
    return dict(schema='backtest_case_dependencies_v1', cases=cases,
        summary=dict(case_count=len(cases), required_case_counts=dict(counts)),
        input_sha256=refs, code_sha256=reviewed_code | {name:digest(ROOT/name) for name in CODE},
        review_method='source-reviewed dependency contract bound to exact code and account bytes',
        performance_recomputed=False, strict_data_ready=False, live_qualified=False,
        limitations=[
            'This review is not dynamic taint analysis or proof that all required sources were available at decision time.',
            'Price-only selection still needs correct dated universe, price inputs and corporate-action accounting.',
            'No missing corporate announcement or delivery evidence is waived by excluding unrelated financial/news factors.',
            'Current sector membership remains non-PIT for both sector arms.',
            'Only known account paths are inventoried; changed fills or signals can create additional data requirements.',
        ])


def verify(path=OUTPUT):
    path = Path(path)
    if digest(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Dependency report hash mismatch')
    value = json.loads(path.read_text())
    if value != build():
        raise ValueError('Reviewed source/configuration dependencies changed')
    return value


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.verify:
        value=verify(args.output)
    else:
        value=build()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(encoded(value))
        args.output.with_suffix('.sha256').write_text(digest(args.output)+'\n')
    print(json.dumps(value['summary'],ensure_ascii=False))
