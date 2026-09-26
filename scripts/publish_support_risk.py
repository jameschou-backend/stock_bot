#!/usr/bin/env python3
"""Publish only reproduced accounts whose entry metadata matches frozen signals."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.file_lock import file_lock
from app.residual_slots_ui import REPORT
from scripts import research_support_risk as study
from scripts.research_exit_scenarios import read,write,sha
from skills.backtest_case_cache import file_identities
from skills.support_risk_source_audit import audit_candidate_binding
from skills.verified_backtest_tool import offline_only


def descriptor(path):
    return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=sha(path))


def publish(left,right,proof_path,publication_path):
    left,right,proof_path,publication_path=(Path(p).resolve() for p in (left,right,proof_path,publication_path))
    if any(p.exists() for p in (proof_path,publication_path,proof_path.with_suffix('.sha256'),
                                publication_path.with_suffix('.sha256'))):
        raise ValueError('Choose new immutable proof and publication paths')
    basic=proof_path.with_name(proof_path.stem+'-reproduction.json')
    refs=file_identities([Path(__file__),ROOT/'skills/support_risk_source_audit.py'],ROOT)
    proof=study.verify(left,right,basic)
    bindings={}
    with offline_only():
        data,_,_=study.parent.parent.load_data(study.parent.load_selector())
        for folder in (left,right):
            bindings[str(folder.relative_to(ROOT))]={name:audit_candidate_binding(
                read(ROOT/row['result']['path']),data.entries,data.days)
                for name,row in read(folder/'report.json')['cases'].items()}
    if file_identities([ROOT/p for p in refs],ROOT)!=refs:
        raise ValueError('Source-binding audit code changed')
    proof.update(source_binding=bindings,source_binding_code_sha256=refs,
                 reproduction_proof=descriptor(basic))
    write(proof_path,proof);proof_path.with_suffix('.sha256').write_text(sha(proof_path)+'\n')
    report=read(left/'report.json')
    value=dict(report,schema='support_risk_publication_v1',adopted=False,strict_data_ready=False,
        completed_cases=sum(c['completed'] for c in report['cases'].values()),
        original_publication=descriptor(REPORT),source_identity=descriptor(left/'identity.json'),
        run_manifest=descriptor(left/'manifest.json'),offline_verification=descriptor(proof_path),
        reproduction_trial_count=len(report['cases'])*2,
        decision='Historical screen only; no strategy promoted to live trading',
        limitations=['Previously researched history, not unseen validation',
            'Planned risk is not a guaranteed maximum realised loss',
            'Daily-bar fill estimates do not reconstruct intraday order queues',
            'Rights certificate valuation and unpaid fractional cash retain the sealed account restrictions',
            'Complete historical source coverage and broker cash reconciliation remain unqualified'])
    write(publication_path,value)
    publication_path.with_suffix('.sha256').write_text(sha(publication_path)+'\n')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs=2,required=True)
    parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--publication',type=Path,required=True)
    args=parser.parse_args()
    with file_lock(study.OUTPUT/'.run.lock',timeout=0):
        value=publish(*args.runs,args.proof,args.publication)
    print('published',value['completed_cases'],'of',len(value['cases']),flush=True)
