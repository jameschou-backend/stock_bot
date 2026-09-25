#!/usr/bin/env python3
"""Bounded offline workbench entry point for the fixed twelve sector accounts."""
from pathlib import Path
import argparse
import hashlib
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import write
from skills.backtest_case_cache import file_identities
from app.backtest_completion_ui import validate_sector_report

CACHE = ROOT / '.cache/sector-account-sources-r2-20260925/inputs'
ADDITIONS = [ROOT / 'docs/backtest_corporate_completion_20260925.json',
             ROOT / 'docs/sector_account_corporate_20260925.json']


def run(output, *, preflight_only=False, strict_pit=False):
    output = Path(output).absolute()
    folder = ROOT / '.cache/workbench/jobs'
    if (output.parent != folder or not re.fullmatch(r'[0-9a-f]{32}\.result\.json', output.name)
            or any(p.is_symlink() for p in (output, *output.parents)) or output.exists()):
        raise ValueError('Use a new workbench jobs/<32hex>.result.json output')
    job_id = output.name.split('.')[0]
    run_dir = ROOT / '.cache/sector-account-jobs' / job_id
    if any(p.is_symlink() for p in (run_dir, *run_dir.parents)) or run_dir.exists():
        raise ValueError('Sector job output already exists or uses a symlink; start a new job')
    from scripts import research_sector_accounts as driver
    print('[TIMER] backtest start', flush=True)
    report = driver.run(run_dir, CACHE, ADDITIONS,
                        preflight_only=preflight_only, strict_pit=strict_pit)
    validate_sector_report(report, ROOT)
    if report['preflight_only'] != preflight_only or report['strict_pit'] != strict_pit:
        raise ValueError('Sector account result does not match requested mode')
    refs = report['source_sha256']
    if file_identities([ROOT / name for name in refs], ROOT) != refs:
        raise ValueError('Sector sources changed before job publication')
    report_path = run_dir / 'report.json'
    raw = report_path.read_bytes()
    # Verify the exact report bytes that the UI will consume.
    import json
    if json.loads(raw) != report:
        raise ValueError('Sector report differs from the returned account result')
    value = dict(format='sector_account_job_v1', job_id=job_id,
        status='blocked' if strict_pit else 'preflight_complete' if preflight_only else report['status'],
        request=dict(preflight_only=preflight_only, strict_pit=strict_pit),
        report=dict(path=str(report_path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest()),
        requirements=dict(path=str((run_dir / 'requirements.json').relative_to(ROOT)),
                          sha256=hashlib.sha256((run_dir / 'requirements.json').read_bytes()).hexdigest()),
        live_qualified=False, unseen_validation=False, network_calls=0,
        wrapper_sha256=file_identities([Path(__file__)], ROOT))
    if output.exists():
        raise ValueError('Job result was created during execution; refusing to overwrite it')
    write(output, value)
    print('[TIMER] backtest done', flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--strict-pit', action='store_true')
    args = parser.parse_args()
    try:
        result = run(args.output, preflight_only=args.preflight_only, strict_pit=args.strict_pit)
    except (OSError, ValueError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(result['status'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
