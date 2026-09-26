#!/usr/bin/env python3
"""Publish all fixed ETF cases only after independent two-run reconstruction."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.file_lock import file_lock
from scripts import research_index_exposure as study
from scripts.research_exit_scenarios import read,write,sha

def descriptor(path):return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=sha(path))

def publish(left,right,proof_path,target):
    left,right,proof_path,target=(Path(p).resolve() for p in (left,right,proof_path,target))
    if any(p.exists() for p in (target,target.with_suffix('.sha256'),proof_path,proof_path.with_suffix('.sha256'))):
        raise ValueError('Do not overwrite a published research result')
    proof=study.verify(left,right,proof_path)
    report=read(left/'report.json')
    report.update(schema='index_exposure_publication_v1',adopted=False,
        original_publication=descriptor(study.BENCHMARK),source_identity=descriptor(left/'identity.json'),
        run_manifest=descriptor(left/'manifest.json'),offline_verification=descriptor(proof_path),
        publication_code=descriptor(Path(__file__)),reproduction_trial_count=proof['newly_executed_cases'],
        reproduction_cache_reuses=proof['benchmark_cache_reuses'],
        limitations=['00631L uses actual prices with daily 2x exposure; excess is not pure selection skill',
            'Provider price limits are zero; reconstructed 20% bounds are not observed historical official limits',
            'Daily volume capacity does not establish executable closing auction volume or queue priority',
            'Previously studied dates are not unseen data; neighboring parameters and annual gates still apply',
            'Historical gains include unrealised holdings; no real order or live qualification'],
        verified_evidence=dict(full_account_reproduction=True,causal_account_audit=True,
            official_historical_limits=False,live_qualified=False))
    write(target,report);target.with_suffix('.sha256').write_text(sha(target)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs=2,required=True)
    parser.add_argument('--proof',type=Path,required=True);parser.add_argument('--publication',type=Path,required=True)
    args=parser.parse_args()
    with file_lock(study.BASE/'.run.lock',timeout=0):
        report=publish(*args.runs,args.proof,args.publication)
    print('published',len(report['cases']),flush=True)
