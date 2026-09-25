#!/usr/bin/env python3
"""Publish one verified full-pass index; never promote an account or buy data."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.backtest_full_pass_ui import load,KINDS,REPORT,CODE
from skills.publication_versions import digest,encoded


def descriptor(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Evidence must remain inside the repository')
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path.read_bytes()))


def publish(output,reports):
    output=Path(output).resolve()
    if (set(reports)!=KINDS or not output.is_relative_to(ROOT/'artifacts')
            or output.suffix!='.json' or output.exists() or output.with_suffix('.sha256').exists()):
        raise ValueError('Use a new artifact with the exact full-pass scope')
    value=dict(schema='backtest_full_pass_v1',strict_data_ready=False,live_qualified=False,
        performance_recomputed=False,
        parent_followup=descriptor(ROOT/'artifacts/forward_simulation/backtest_data_followup_20260925.json'),
        reports={key:descriptor(path) for key,path in sorted(reports.items())},
        code_sha256={name:digest((ROOT/name).read_bytes()) for name in CODE})
    raw=encoded(value)
    cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='full-pass-publish-',dir=cache) as directory:
        path=Path(directory)/'index.json'
        path.write_bytes(raw);path.with_suffix('.sha256').write_text(digest(raw)+'\n')
        load(path,ROOT)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(raw)
    with output.with_suffix('.sha256').open('x') as stream:stream.write(digest(raw)+'\n')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in sorted(KINDS):parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--output',type=Path,default=REPORT)
    args=parser.parse_args()
    publish(args.output,{key:getattr(args,key) for key in KINDS})
    print(str(args.output))
