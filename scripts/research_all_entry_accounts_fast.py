#!/usr/bin/env python3
"""Run all sealed candidates with an equivalent indexed prices and verified parsed odd-table reuse."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_all_entry_accounts import configured_runner,FROZEN,FROZEN_SHA
from skills.all_entry_account_data import AllEntryData
from skills.all_entry_accounts import BASE,registry
from skills.all_entry_corporate import load_terms
from skills.all_entry_reference_index import install_private_index
from skills.all_entry_prior_lookup import install_prior_lookup
from skills.all_entry_odd_cache import install_parsed_cache


def run(output,arms,*,candidate_bundle=BASE/'inputs-v3',document=None,fetch=False):
    candidate_bundle=Path(candidate_bundle).resolve()
    runner,refs=configured_runner(candidate_bundle,arms)
    install_private_index(runner)
    install_prior_lookup(runner)
    provider=AllEntryData(online=fetch)
    install_parsed_cache(provider)
    if document:
        additions,extra=load_terms(ROOT,document)
        for key,row in additions.items():
            if key in provider.corporate_overrides and provider.corporate_overrides[key]!=row:
                raise ValueError('Supplement changes existing corporate action: '+key)
        provider.corporate_overrides=dict(provider.corporate_overrides,**additions)
        refs=dict(refs,**extra)
    for name,sha in refs.items():provider._bind(ROOT/name,sha)
    provider._bind(FROZEN,FROZEN_SHA)
    for path in (Path(__file__),ROOT/'skills/all_entry_reference_index.py',ROOT/'skills/all_entry_corporate.py',
                 ROOT/'skills/all_entry_prior_lookup.py',ROOT/'skills/all_entry_odd_cache.py'):
        provider._bind(path)
    writer=runner.write
    def write(path,value):
        if Path(path).name=='report.json':
            value['reference_lookup']='private_index_exact_frozen_scalar_semantics'
            value['prior_lookup']='previous_close_required_by_frozen_tick_planner'
            value['odd_table_reuse']='parsed_table_cache_with_per_use_source_verification'
        return writer(path,value)
    runner.write=write
    return runner.run(output,tuple(arms),provider,candidate_bundle=candidate_bundle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arms',default=','.join([x['id'] for x in registry()]+['benchmark']))
    parser.add_argument('--candidate-bundle',type=Path,default=BASE/'inputs-v3')
    parser.add_argument('--terms',type=Path)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args()
    report=run(args.output,args.arms.split(','),candidate_bundle=args.candidate_bundle,document=args.terms,fetch=args.fetch)
    print(json.dumps(dict(all_completed=report['all_completed'],elapsed_seconds=report['elapsed_seconds'])))
