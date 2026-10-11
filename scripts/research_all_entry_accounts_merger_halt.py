#!/usr/bin/env python3
"""Replay sealed candidates with the verified capital, resource audits, and verified ordinary-share distributions and official execution aggregates."""
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
from skills.all_entry_capital_completion import install_capital_completion
from skills.all_entry_resource_rejections import install_resource_rejection_audit, POLICY as RESOURCE_AUDIT_POLICY
from skills.all_entry_capital_halt import install_capital_halt
from skills.all_entry_official_execution_v2 import install_official_execution
from skills.all_entry_distribution_completion import install_distributions
from skills.all_entry_albatron_completion import install_albatron
from skills.all_entry_certificate_completion import install_certificates
from skills.all_entry_late_certificate_completion import install_completion as install_late_certificate
from skills.all_entry_bank_distributions import install_distributions as install_bank_distributions
from skills.all_entry_issuer_distributions import install_distributions as install_issuer_distributions
from skills.all_entry_northstar_completion import install_completion as install_northstar
from skills.all_entry_context_distributions import install_distributions as install_context_distributions
from skills.all_entry_rescheduled_distributions import install_distributions as install_rescheduled_distributions
from skills.all_entry_poya_completion import install_completion as install_poya
from skills.all_entry_merger_halt import install_halt
from skills.all_entry_zero_reentry import install_zero_reentry, POLICY as ZERO_REENTRY_POLICY


def run(output,arms,*,candidate_bundle=BASE/'inputs-v3',document=None,completion=None,capital=None,official=None,distributions=None,albatron=None,certificates=None,late_certificate=None,bank_distributions=None,issuer_distributions=None,northstar=None,context_distributions=None,poya=None,merger_halt=None,fetch=False):
    candidate_bundle=Path(candidate_bundle).resolve()
    runner,refs=configured_runner(candidate_bundle,arms)
    install_private_index(runner)
    install_prior_lookup(runner)
    install_audit_index(runner)
    install_resource_rejection_audit(runner)
    provider=AllEntryData(online=fetch)
    if poya:
        refs=dict(refs,**install_poya(runner,provider,poya,ROOT))
    if context_distributions:
        refs=dict(refs,**install_context_distributions(runner,provider,context_distributions,ROOT))
    if official:
        refs=dict(refs,**install_official_execution(runner,provider,official,ROOT))
    install_zero_reentry(runner)
    install_parsed_cache(provider)
    if document:
        additions,extra=load_terms(ROOT,document)
        for key,row in additions.items():
            if key in provider.corporate_overrides and provider.corporate_overrides[key]!=row:
                raise ValueError('Supplement changes existing corporate action: '+key)
        provider.corporate_overrides=dict(provider.corporate_overrides,**additions)
        refs=dict(refs,**extra)
    if distributions:
        refs=dict(refs,**install_distributions(provider,distributions,ROOT))
    if bank_distributions:
        refs=dict(refs,**install_bank_distributions(provider,bank_distributions,ROOT))
    if issuer_distributions:
        refs=dict(refs,**install_issuer_distributions(runner,provider,issuer_distributions,ROOT))
    if northstar:
        refs=dict(refs,**install_northstar(runner,provider,northstar,ROOT))
    if albatron:
        refs=dict(refs,**install_albatron(provider,albatron,ROOT))
    if completion:
        refs=dict(refs,**install_completion(runner,provider,completion,ROOT))
    if capital:
        refs=dict(refs,**install_capital_completion(runner,provider,capital,ROOT))
        refs=dict(refs,**install_capital_halt(runner,capital,ROOT))
    if certificates:
        refs=dict(refs,**install_certificates(runner,provider,certificates,ROOT))
    if late_certificate:
        refs=dict(refs,**install_late_certificate(runner,provider,late_certificate,ROOT))
    refs=dict(refs,**install_rescheduled_distributions(provider,ROOT))
    if merger_halt:
        refs=dict(refs,**install_halt(runner,merger_halt,ROOT))
    for name,sha in refs.items():provider._bind(ROOT/name,sha)
    provider._bind(FROZEN,FROZEN_SHA)
    provider._bind(AUDIT_SOURCE,AUDIT_SHA)
    for path in (Path(__file__),ROOT/'skills/all_entry_reference_index.py',ROOT/'skills/all_entry_corporate.py',
                 ROOT/'skills/all_entry_prior_lookup.py',ROOT/'skills/all_entry_odd_cache.py',
                 ROOT/'skills/all_entry_audit_index.py',ROOT/'skills/all_entry_settlement_completion.py',
                 ROOT/'skills/all_entry_zero_reentry.py',ROOT/'skills/all_entry_capital_completion.py',
                 ROOT/'skills/prepared_corporate_settlement.py',ROOT/'skills/all_entry_resource_rejections.py',
                 ROOT/'skills/all_entry_capital_halt.py',ROOT/'skills/all_entry_official_execution_v2.py',ROOT/'skills/all_entry_distribution_completion.py',ROOT/'skills/all_entry_albatron_completion.py',ROOT/'skills/all_entry_certificate_completion.py',ROOT/'skills/all_entry_late_certificate_completion.py',ROOT/'skills/all_entry_bank_distributions.py',ROOT/'skills/all_entry_issuer_distributions.py',ROOT/'skills/all_entry_northstar_completion.py',ROOT/'skills/all_entry_context_distributions.py',ROOT/'skills/all_entry_rescheduled_distributions.py',ROOT/'skills/all_entry_poya_completion.py',ROOT/'skills/all_entry_merger_halt.py'):
        provider._bind(path)
    writer=runner.write
    def write(path,value):
        if Path(path).name=='report.json':
            value['reference_lookup']='private_index_exact_frozen_scalar_semantics'
            value['prior_lookup']='previous_close_required_by_frozen_tick_planner'
            value['independent_audit_quote_lookup']='private_index_exact_frozen_row_selection'
            value['zero_buy_reentry_policy']=ZERO_REENTRY_POLICY
            value['resource_slot_rejection_audit']=RESOURCE_AUDIT_POLICY
            value['official_daily_execution_completion']='sealed_evidenced_coordinates_complete_baskets_no_synthetic_ticks' if official else None
            value['capital_halt_completion']='2352_2025_09_25_through_2025_10_03' if capital else None
            value['capital_settlement_completion']='2352_2025_10_07_evidenced_historical_close' if capital else None
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
    parser.add_argument('--capital-completion',type=Path,required=True)
    parser.add_argument('--official-execution',type=Path,required=True)
    parser.add_argument('--distribution-completion',type=Path,required=True)
    parser.add_argument('--albatron-completion',type=Path,required=True)
    parser.add_argument('--certificate-completion',type=Path,required=True)
    parser.add_argument('--late-certificate-completion',type=Path,required=True)
    parser.add_argument('--bank-distributions',type=Path,required=True)
    parser.add_argument('--issuer-distributions',type=Path,required=True)
    parser.add_argument('--northstar-completion',type=Path,required=True)
    parser.add_argument('--context-distributions',type=Path,required=True)
    parser.add_argument('--poya-completion',type=Path,required=True)
    parser.add_argument('--merger-halt',type=Path,required=True)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args()
    report=run(args.output,args.arms.split(','),candidate_bundle=args.candidate_bundle,document=args.terms,
        completion=args.settlement_completion,capital=args.capital_completion,official=args.official_execution,distributions=args.distribution_completion,albatron=args.albatron_completion,certificates=args.certificate_completion,late_certificate=args.late_certificate_completion,bank_distributions=args.bank_distributions,issuer_distributions=args.issuer_distributions,northstar=args.northstar_completion,context_distributions=args.context_distributions,poya=args.poya_completion,merger_halt=args.merger_halt,fetch=args.fetch)
    print(json.dumps(dict(all_completed=report['all_completed'],elapsed_seconds=report['elapsed_seconds'])))
