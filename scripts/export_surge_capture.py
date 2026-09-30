#!/usr/bin/env python3
"""Publish identical independent diagnostics, retaining all outcomes and missing labels."""
from pathlib import Path
import argparse
import gzip
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_surge_capture import read, write, sha


def export(left, right, output):
    left, right, output = (Path(p).resolve() for p in (left, right, output))
    output.relative_to(ROOT)
    if left == right or output.exists():
        raise ValueError('Two distinct runs and a new publication directory are required')
    reports = [read(p/'report.json') for p in (left, right)]
    if reports[0] != reports[1] or reports[0]['schema'] != 'surge_capture_20260930':
        raise ValueError('Independent diagnostic reports differ')
    report = reports[0]
    if not report['signal_reconstruction_exact'] or report['live_qualified'] or report['new_backtest']:
        raise ValueError('Unexpected scope or unverified signals')
    for name, digest in report['exports_sha256'].items():
        for folder in (left, right):
            if sha(folder/name) != digest:
                raise ValueError('Export changed: '+name)
    for name, digest in report['source_roots_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Source root changed: '+name)
    output.mkdir(parents=True)
    for name in report['exports_sha256']:
        data = (left/name).read_bytes()
        if name == 'signals.csv':
            (output/(name+'.gz')).write_bytes(gzip.compress(data, mtime=0))
        else:
            (output/name).write_bytes(data)
    write(output/'report.json', dict(report,
        original_exports_sha256=report['exports_sha256'],
        exports_sha256={p.name: sha(p) for p in sorted(output.iterdir())},
        offline_identical=True,
        run_reports=[dict(path=str((p/'report.json').relative_to(ROOT)), sha256=sha(p/'report.json'))
                     for p in (left, right)],
        export_code_sha256=sha(Path(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left', required=True, type=Path)
    p.add_argument('--right', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    export(a.left, a.right, a.output)
    print('Published', a.output)


if __name__ == '__main__':
    main()
