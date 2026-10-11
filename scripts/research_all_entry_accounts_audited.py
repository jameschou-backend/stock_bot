#!/usr/bin/env python3
"""Run all sealed candidates with indexed prices, independently audited quote rows and verified odd-table reuse."""
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
from skills.all_entry_audit_index import install_audit_index, SOURCE as AUDIT_SOURCE, SOURCE_SHA as AUDIT_SHA
from skills.all_entry_settlement_completion import install_completion
from skills.all_entry_zero_reentry import install_zero_reentry, POLICY as ZERO_REENTRY_POLICY


def run(output,arms,*,candidate_bundle=BASE/'inputs-v3',document=None,completion=None,fetch=False):
    candidate_bundle=Path(candidate_bundle).resolve()
    runner,refs=configured_runner(candidate_bundle,arms)
    install_private_index(runner)
    install_prior_lookup(runner)
    install_audit_index(runner)
    install_zero_reentry(runner)
    provider=AllEntryData(online=fetch)
    install_parsed_cache(provider)
    if document:
        additions,extra=load_terms(ROOT,document)
        for key,row in additions.items():
            if key in provider.corporate_overrides and provider.corporate_overrides[key]!=row:
                raise ValueError('Supplement changes existing corporate action: '+key)
        provider.corporate_overrides=dict(provider.corporate_overrides,**additions)
        refs=dict(refs,**extra)
    if completion:
        refs=dict(refs,**install_completion(runner,provider,completion,ROOT))
    for name,sha in refs.items():provider._bind(ROOT/name,sha)
    provider._bind(FROZEN,FROZEN_SHA)
    provider._bind(AUDIT_SOURCE,AUDIT_SHA)
    for path in (Path(__file__),ROOT/'skills/all_entry_reference_index.py',ROOT/'skills/all_entry_corporate.py',
                 ROOT/'skills/all_entry_prior_lookup.py',ROOT/'skills/all_entry_odd_cache.py',
                 ROOT/'skills/all_entry_audit_index.py',ROOT/'skills/all_entry_settlement_completion.py',
                 ROOT/'skills/all_entry_zero_reentry.py'):
        provider._bind(path)
    writer=runner.write
    def write(path,value):
        if Path(path).name=='report.json':
            value['reference_lookup']='private_index_exact_frozen_scalar_semantics'
            value['prior_lookup']='previous_close_required_by_frozen_tick_planner'
            value['independent_audit_quote_lookup']='private_index_exact_frozen_row_selection'
            value['zero_buy_reentry_policy']=ZERO_REENTRY_POLICY
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
    parser.add_argument('--settlement-completion',type=Path)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args()
    report=run(args.output,args.arms.split(','),candidate_bundle=args.candidate_bundle,document=args.terms,
        completion=args.settlement_completion,fetch=args.fetch)
    print(json.dumps(dict(all_completed=report['all_completed'],elapsed_seconds=report['elapsed_seconds'])))
