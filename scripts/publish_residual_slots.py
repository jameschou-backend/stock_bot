#!/usr/bin/env python3
"""Publish complete remnant-slot results only after an independent replay."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from skills.backtest_case_cache import file_identities
from app.residual_slots_ui import validate, REPORT, ORIGINAL


def publish(source, proof_path):
    source, proof_path = Path(source).resolve(), Path(proof_path).resolve()
    if REPORT.exists():
        raise ValueError('Publication already exists; retain original evidence')
    proof = read(proof_path)
    if sha(proof_path) != proof_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Offline proof changed')
    refs = dict(proof['source_sha256'])
    refs.update(file_identities([Path(__file__), ROOT / 'app/residual_slots_ui.py',
        ROOT / 'app/dashboard_v2/pages/12_餘股與名額.py', proof_path, proof_path.with_suffix('.sha256')], ROOT))
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Source or code changed before publication')
    report = read(source / 'report.json')
    report.update(schema='residual_slots_publication_v1', source_sha256=refs,
        published_at=datetime.now(timezone.utc).isoformat(),
        original_publication=dict(path=str(ORIGINAL.relative_to(ROOT)), sha256=sha(ORIGINAL)),
        offline_verification=dict(path=str(proof_path.relative_to(ROOT)), sha256=sha(proof_path)),
        run_manifest=dict(path=str((source / 'manifest.json').relative_to(ROOT)), sha256=sha(source / 'manifest.json')))
    validate(report, ROOT)
    write(REPORT, report)
    REPORT.with_suffix('.sha256').write_text(sha(REPORT) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    args = parser.parse_args()
    print('published', len(publish(args.source, args.proof)['cases']))
