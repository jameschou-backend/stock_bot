#!/usr/bin/env python3
"""Publish the exact verified incremental evidence, without replacing sealed results."""
import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from skills.publication_versions import digest,encoded
from app.backtest_data_followup_ui import load


def descriptor(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Evidence must remain inside this repository')
    return dict(path=str(path.relative_to(ROOT)),sha256=digest(path.read_bytes()))


def publish(output,board,identity,odd,publication):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'artifacts') or output.exists():
        raise ValueError('Use a new immutable project artifact')
    value=dict(schema='backtest_data_followup_v1',live_qualified=False,performance_recomputed=False,
        base_data_report=descriptor(ROOT/'artifacts/forward_simulation/backtest_data_completion_20260925.json'),
        reports={k:descriptor(p) for k,p in dict(ordinary=board,identity=identity,odd_lot=odd,publication=publication).items()})
    # Validate a staging file first so readers never see an unverified publication.
    raw=encoded(value)
    cache=ROOT/'.cache'
    cache.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='data-followup-publish-',dir=cache) as directory:
        staging=Path(directory)/'report.json'
        staging.write_bytes(raw)
        staging.with_suffix('.sha256').write_text(digest(raw)+'\n')
        load(staging)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw)
    with output.with_suffix('.sha256').open('x') as stream:
        stream.write(digest(raw)+'\n')
    print(str(output.relative_to(ROOT)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--odd',type=Path,required=True)
    p.add_argument('--publication',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/forward_simulation/backtest_data_followup_20260925.json')
    args=p.parse_args()
    publish(args.output,args.board,args.identity,args.odd,args.publication)
