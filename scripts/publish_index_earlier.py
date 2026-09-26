#!/usr/bin/env python3
"""Publish reproduced cross-period outcomes without promoting the winning control."""
from pathlib import Path
import argparse
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.file_lock import file_lock
from scripts import research_index_earlier as study
from scripts.research_exit_scenarios import read,write,sha
from skills.account_statistics import describe_account
from skills.trial_registry import TRIAL_REGISTRY_PATH

def descriptor(path):return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=sha(path))

def publish(left,right,proof_path,target):
    left,right,proof_path,target=(Path(p).resolve() for p in (left,right,proof_path,target))
    stats_path=proof_path.with_name('statistics-v1.json')
    if any(p.exists() for p in (proof_path,target,stats_path,target.with_suffix('.sha256'))):
        raise ValueError('Preserve existing earlier publications')
    proof=study.verify(left,right,proof_path);report=read(left/'report.json');stats={}
    benchmarks={k:read(ROOT/r['result']['path']) for k,r in report['benchmarks'].items()}
    calendar=[r['date'] for r in benchmarks['control']['account']['daily']]
    for name,row in report['cases'].items():
        value=read(ROOT/row['result']['path']);mask=value['config']['factor_mask']
        stats[name]=describe_account(value['account'],benchmarks['combined' if mask&1 else 'control']['account'],calendar)
    snapshot=proof_path.with_name('trial-registry-snapshot.jsonl');snapshot.write_bytes(TRIAL_REGISTRY_PATH.read_bytes())
    records=[json.loads(line) for line in snapshot.read_text().splitlines() if line.strip()]
    write(stats_path,dict(schema='index_earlier_statistics_v1',cases=stats,study_manifest=descriptor(left/'manifest.json'),
        registry_snapshot=descriptor(snapshot),recorded_executions=len(records),live_qualified=False,
        selection_adjusted_statistics_verified=False,unseen_validation=False,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'skills/account_statistics.py',ROOT/'skills/statistics.py')}))
    value=dict(report,schema='index_earlier_publication_v1',adopted=False,
        source_identity=descriptor(left/'identity.json'),run_manifest=descriptor(left/'manifest.json'),
        offline_verification=descriptor(proof_path),statistics=descriptor(stats_path),
        current_period=descriptor(ROOT/'artifacts/forward_simulation/index_exposure_20260927.json'),
        publication_code=descriptor(Path(__file__)),reproduction_trial_count=proof['newly_executed_cases'],
        limitations=['The fixed 200-day primary strategy remains the primary; a better control is not promoted after seeing results',
            'Another historical period is retrospective replication, not previously unseen prospective evidence',
            '00631L limits are reconstructed; daily volume does not prove real execution',
            'Dividend ex-dates/amounts match issuer; payment dates remain FinMind-only and announcements lack timestamps'])
    write(target,value);target.with_suffix('.sha256').write_text(sha(target)+'\n');return value

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs=2,required=True);parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--publication',type=Path,required=True);args=parser.parse_args()
    with file_lock(study.BASE/'.run.lock',timeout=0):report=publish(*args.runs,args.proof,args.publication)
    print('published',len(report['cases']),flush=True)
