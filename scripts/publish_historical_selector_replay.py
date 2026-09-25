#!/usr/bin/env python3
"""Publish only accounts with bound offline and causality evidence."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from skills.backtest_case_cache import file_identities
from app.historical_selector_ui import REPORT, validate


def publish(source, offline, causal, output=REPORT):
    source, offline, causal = (Path(p).resolve() for p in (source, offline, causal))
    if output.exists():
        raise ValueError('Publication already exists; retain earlier evidence')
    def descriptor(path):
        return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
    refs = read(source / 'identity.json')
    for path in (offline, causal):
        proof = read(path)
        if sha(path) != path.with_suffix('.sha256').read_text().strip() or proof.get('passed') is not True:
            raise ValueError('Verification is incomplete or changed')
        for name, digest in proof['source_sha256'].items():
            if name in refs and refs[name] != digest:
                raise ValueError('Verification source versions disagree')
            refs[name] = digest
        refs.update(file_identities([path, path.with_suffix('.sha256')], ROOT))
    report = read(source / 'report.json')
    cases = {}
    for name, row in report['cases'].items():
        path = ROOT / row['path']
        if sha(path) != row['sha256']:
            raise ValueError('Case result changed')
        cases[name] = dict(completed=row['completed'], summary=row['summary'], reason=row['reason'],
                           result=descriptor(path), rolling252=row.get('rolling252'))
    extra = [Path(__file__), ROOT / 'app/historical_selector_ui.py',
             ROOT / 'app/dashboard_v2/pages/10_新版回測.py',
             ROOT / 'app/backtest_full_pass_ui.py', ROOT / 'app/backtest_tool_ui.py',
             source / 'report.json', source / 'manifest.json']
    execution_dir = (ROOT / read(source / 'execution-source.json')['path']).parent
    prep = [ROOT / '.cache/historical-selector-quality-20260925/manifest.json',
            execution_dir / 'manifest.json',
            ROOT / '.cache/historical-listing-prefix-v2-20260925/manifest.json']
    refs.update(file_identities(extra + prep, ROOT))
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Source/code changed before publication')
    value = dict(schema='historical_selector_publication_v1', published_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=refs, cases=cases, signals=report['signals'], start='2022-01-03', end='2026-09-09',
        initial_cash=1_000_000, execution_policy='board_only', live_qualified=False,
        strict_data_ready=False, unseen_validation=False, preparation_calls=sum(read(p)['calls_reserved'] for p in prep),
        causality_checks=read(causal)['expected_checks'], offline_verification=descriptor(offline),
        causality_verification=descriptor(causal), run_manifest=descriptor(source / 'manifest.json'),
        elapsed_seconds=report['elapsed_seconds'])
    validate(value, ROOT)
    write(output, value)
    output.with_suffix('.sha256').write_text(sha(output) + '\n')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--offline', type=Path, required=True)
    parser.add_argument('--causal', type=Path, required=True)
    args = parser.parse_args()
    value = publish(args.source, args.offline, args.causal)
    print('published_cases', len(value['cases']), 'causality_checks', value['causality_checks'])
