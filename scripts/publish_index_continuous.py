#!/usr/bin/env python3
"""Publish continuous capital accounts and explicit ETF limit reconciliation."""
from pathlib import Path
import argparse
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.file_lock import file_lock
from scripts import research_index_continuous as study
from scripts.research_exit_scenarios import read,write,sha
from skills.account_statistics import describe_account
from skills.trial_registry import TRIAL_REGISTRY_PATH

def descriptor(path):return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=sha(path))

def reference_impact(report,refs):
    def sealed(path):
        path=Path(path);key=str(path.relative_to(ROOT))
        if refs.get(key)!=sha(path):raise ValueError('Comparison source was not sealed: '+key)
        return read(path)
    early=sealed(ROOT/'artifacts/forward_simulation/index_earlier_20260927.json')
    current=sealed(ROOT/'artifacts/forward_simulation/index_exposure_20260927.json')
    original=sealed(ROOT/current['original_publication']['path'])
    impact={}
    for name,row in report['reference_controls'].items():
        mode='combined' if name.endswith('combined') else 'control'
        old_ref=(early['benchmarks'][mode] if '_early_' in name else original['cases']['benchmark_'+mode])['result']
        old=sealed(ROOT/old_ref['path'])['account'];new=read(ROOT/row['result']['path'])['account']
        changed=[k for k in new if new[k]!=old[k]];order_diffs=[]
        for i in range(max(len(new['orders']),len(old['orders']))):
            before=old['orders'][i] if i<len(old['orders']) else None
            after=new['orders'][i] if i<len(new['orders']) else None
            if before!=after:order_diffs.append(dict(index=i,before=before,after=after))
        impact[name]=dict(changed_account_fields=changed,order_differences=order_diffs,
            daily_assets_identical=new['daily']==old['daily'],trades_identical=new['trades']==old['trades'],
            cash_ledger_identical=new['cash_ledger']==old['cash_ledger'],holdings_identical=new['holdings']==old['holdings'],
            return_difference=row['return_difference'],old_result=old_ref,new_result=row['result'])
    reset={name:dict(hypothetical_reset_product=(1+early['cases'][name]['summary']['total_return'])*(1+current['cases'][name]['summary']['total_return'])-1,
        actual_continuous_return=row['summary']['total_return']) for name,row in report['cases'].items()}
    return dict(schema='continuous_reference_impact_v1',controls=impact,reset_product_diagnostics=reset,
        reset_product_is_not_an_executable_account=True,live_qualified=False)

def publish(left,right,proof_path,target):
    left,right,proof_path,target=(Path(p).resolve() for p in (left,right,proof_path,target))
    stats_path=proof_path.with_name('statistics-v1.json');impact_path=proof_path.with_name('reference-impact-v1.json')
    if any(p.exists() for p in (proof_path,target,stats_path,impact_path,target.with_suffix('.sha256'))):
        raise ValueError('Preserve existing continuous publications')
    report=read(left/'report.json');impact=reference_impact(report,read(left/'identity.json'))
    proof=study.verify(left,right,proof_path);stats={};write(impact_path,impact)
    benchmarks={k:read(ROOT/r['result']['path']) for k,r in report['benchmarks'].items()}
    calendar=[r['date'] for r in benchmarks['control']['account']['daily']]
    for name,row in report['cases'].items():
        value=read(ROOT/row['result']['path']);mask=value['config']['factor_mask']
        stats[name]=describe_account(value['account'],benchmarks['combined' if mask&1 else 'control']['account'],calendar)
    snapshot=proof_path.with_name('trial-registry-snapshot.jsonl');snapshot.write_bytes(TRIAL_REGISTRY_PATH.read_bytes())
    records=[json.loads(line) for line in snapshot.read_text().splitlines() if line.strip()]
    write(stats_path,dict(schema='index_continuous_statistics_v1',cases=stats,study_manifest=descriptor(left/'manifest.json'),
        registry_snapshot=descriptor(snapshot),recorded_executions=len(records),live_qualified=False,
        selection_adjusted_statistics_verified=False,unseen_validation=False,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'skills/account_statistics.py',ROOT/'skills/statistics.py')}))
    value=dict(report,schema='index_continuous_publication_v1',adopted=False,
        source_identity=descriptor(left/'identity.json'),run_manifest=descriptor(left/'manifest.json'),
        offline_verification=descriptor(proof_path),statistics=descriptor(stats_path),
        earlier_period=descriptor(ROOT/'artifacts/forward_simulation/index_earlier_20260927.json'),
        current_period=descriptor(ROOT/'artifacts/forward_simulation/index_exposure_20260927.json'),
        limit_reconciliation=descriptor(left/'limit-audit.json'),
        reference_impact=descriptor(impact_path),
        publication_code=descriptor(Path(__file__)),reproduction_trial_count=proof['newly_executed_cases'],
        limitations=['The fixed 200-day primary strategy remains the primary; a better control is not promoted after seeing results',
            'Another historical period is retrospective replication, not previously unseen prospective evidence',
            '0050 and 00631L limits are reconstructed; daily volume does not prove real execution',
            'No intermediate cash reset; prior failed subperiods remain failures; payments and historical limit versions are not independently verified'])
    write(target,value);target.with_suffix('.sha256').write_text(sha(target)+'\n');return value

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs=2,required=True);parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--publication',type=Path,required=True);args=parser.parse_args()
    with file_lock(study.BASE/'.run.lock',timeout=0):report=publish(*args.runs,args.proof,args.publication)
    print('published',len(report['cases']),flush=True)
