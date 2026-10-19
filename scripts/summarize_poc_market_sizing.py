#!/usr/bin/env python3
"""Publish four fixed market_sizing cases beside eight unchanged original controls."""
import argparse
from datetime import datetime, timezone
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_poc_market_sizing import SPEC, BASE
from scripts.summarize_poc_risk_controls import PublicationSources, execution_coverage
from scripts.summarize_poc_broker_account import account_metrics
from skills.all_entry_accounts import digest, write
from skills.poc_market_sizing import audit_market_sizing, market_state, VARIANT
from skills.top3_validation import MODES, verify_costs
from app.account_analytics import realized_metrics

IDENTIFIER = 'poc_market_sizing_20261019'


def configuration(account):
    """Separate derived exclusion counts from unchanged trading parameters."""
    rows=account['data_gap_exclusions']
    children=[child for row in rows for child in row['excluded_children']]
    observed=dict(excluded_event_count=len(rows),
        excluded_board_children=sum(c['channel']=='board' for c in children),
        excluded_odd_children=sum(c['channel']=='odd' for c in children))
    if any(account['settings'].get(k)!=v for k,v in observed.items()):
        raise ValueError('Exclusion counters disagree with actual excluded orders')
    return {k:v for k,v in account['settings'].items() if k not in observed}


def compare(candidate, original):
    return dict(return_change=candidate['total_return']-original['total_return'],
        drawdown_improvement=candidate['max_drawdown']-original['max_drawdown'],
        profit_change=candidate['final_nav']-original['final_nav'],
        net_profit_retained=candidate['profit']/original['profit'] if original['profit'] else None)


def build(paths):
    sources, cases, accounts, families, refs = PublicationSources(ROOT), {}, {}, {}, []
    spec_ref=dict(path=str(SPEC.relative_to(ROOT)),sha256=digest(SPEC))
    spec=json.loads(sources.bind(spec_ref['path'],spec_ref['sha256']).read_text())
    p=spec['reuse']['publication'];sources.bind(p['path'],p['sha256'])
    def load(ref):
        return json.loads(sources.bind(ref['path'],ref['sha256']).read_text())
    def add(key, ref, reused):
        case=load(ref)
        if not case.get('completed'):
            raise ValueError('Incomplete market_sizing account cannot be published as performance')
        arm,variant,mode=key.split(':');account=case['account']
        verify_costs(account,mode)
        metrics=account_metrics(case)
        if (metrics['start'],metrics['end'],metrics['initial_cash'],metrics['trading_days']) != ('2024-01-02','2026-10-02',1000000,666):
            raise ValueError('Market sizing comparison scope differs')
        cases[key]=dict(id=arm,variant=variant,mode=mode,completed=True,reused=reused,
            path=ref['path'],sha256=ref['sha256'],metrics=metrics,summary=case['summary'],
            analytics=realized_metrics(account),execution_coverage=execution_coverage(account),
            data_gap_exclusions=len(case.get('data_gap_exclusions',[])),
            profile_data_exclusions=len(case.get('profile_data_exclusions',[])))
        accounts[key]=account
        families[key]=case.get('family_rules')
        return case
    for key,ref in spec['reuse']['cases'].items():
        add(key,ref,True)
    for supplied in paths:
        path=Path(supplied).resolve();path.relative_to(BASE)
        ref=dict(path=str(path.relative_to(ROOT)),sha256=path.with_suffix('.sha256').read_text().strip())
        report=load(ref);sources.closure(report,ref['path']);meta=report['risk_controls'];mode=meta['mode']
        if (meta['arm']!='poc_up_red' or mode not in MODES or meta['variant']!=VARIANT
                or meta['preregistration']!=spec_ref or not report.get('all_completed')
                or set(report['cases'])!={'poc_up_red'}):
            raise ValueError('Unregistered market_sizing attempt')
        key=f'poc_up_red:{VARIANT}:{mode}'
        if key in cases:raise ValueError('Duplicate completed configuration; no best-of retries')
        case=add(key,report['cases']['poc_up_red'],False)
        if case['risk_controls']!=meta:raise ValueError('Case metadata differs')
        adjusted_name='.cache/poc-latest-20261003/inputs-v1/close-official.parquet'
        sources.bind(adjusted_name,report['source_sha256'][adjusted_name])
        adjusted=pd.read_parquet(ROOT/adjusted_name,columns=['date','0050']).set_index('date')
        audit=audit_market_sizing(case['account'],adjusted)
        if audit!=case['account']['market_sizing_audit']:raise ValueError('Independent market_sizing audit differs')
        cases[key]['market_sizing_audit']=audit
        original=accounts[f'poc_up_red:original:{mode}'];candidate=case['account']
        if families[key]!=families[f'poc_up_red:original:{mode}']:
            raise ValueError('Original entry, ranking or exit family changed')
        # Same sizing/costs/exits and same account path before the first reduced new-buy budget.
        candidate_settings=configuration(candidate)
        for name,value in configuration(original).items():
            if candidate_settings.get(name)!=value:
                raise ValueError('Unrelated original setting changed: '+name)
        first=audit['first_trigger'] or '9999-12-31'
        for section,fields in [('daily',('date','nav','cash','market_value','receivable','drawdown')),
                ('trades',('date','stock_id','side','qty','reference_price','cash_change'))]:
            left=[{k:r[k] for k in fields} for r in original[section] if r['date']<first]
            right=[{k:r[k] for k in fields} for r in candidate[section] if r['date']<first]
            if left!=right:raise ValueError('Original account diverged before first market sizing intervention: '+section)
        cases[key]['unchanged_before_first_trigger']=True
        trial=path.parent/'trials.jsonl';sources.bind(str(trial.relative_to(ROOT)),digest(trial))
        rows=[json.loads(s) for s in trial.read_text().splitlines()]
        if len(rows)!=1 or rows[0]['risk_controls']!=meta:raise ValueError('Market sizing trial accounting differs')
        refs.append(ref)
    expected={f'{arm}:{variant}:{m}' for m in MODES for arm,variant in
        [('poc_up_red','original'),('benchmark','original'),('poc_up_red',VARIANT)]}
    if set(cases)!=expected:raise ValueError('All four cost comparisons must be present')
    comparisons={m:{'original':compare(cases[f'poc_up_red:{VARIANT}:{m}']['metrics'],cases[f'poc_up_red:original:{m}']['metrics']),
        'benchmark':compare(cases[f'poc_up_red:{VARIANT}:{m}']['metrics'],cases[f'benchmark:original:{m}']['metrics'])} for m in MODES}
    # Failed data attempts are part of the experiment history, never hidden
    # when a completed replacement is selected for the same fixed rules.
    selected={r['path'] for r in refs};attempts=[]
    for path in sorted(BASE.glob('run-v*/*/account/report.json')):
        ref=dict(path=str(path.relative_to(ROOT)),sha256=path.with_suffix('.sha256').read_text().strip())
        old=load(ref);meta=old.get('risk_controls',{})
        if (meta.get('variant')!=VARIANT or meta.get('arm')!='poc_up_red'
                or meta.get('mode') not in MODES or meta.get('preregistration')!=spec_ref):
            raise ValueError('Unrecognized attempt in market_sizing experiment')
        completed=old['all_completed'];chosen=ref['path'] in selected
        if completed and not chosen:raise ValueError('An already completed configuration was rerun')
        descriptor=old['cases']['poc_up_red'];load(descriptor)
        created=getattr((path.parent.parent/'run.log').stat(),'st_birthtime',None)
        attempts.append(dict(**ref,mode=meta['mode'],completed=completed,selected=chosen,
            reason=descriptor.get('reason'),last_date=descriptor.get('last_date'),
            completed_sessions=descriptor.get('completed_sessions'),elapsed_seconds=old['elapsed_seconds'],
            wall_seconds=round(path.stat().st_mtime-created,3) if created is not None else None))
    if len([r for r in attempts if r['completed']])!=4:raise ValueError('Missing complete attempt accounting')
    candidate_name='.cache/all-entry-accounts-20261011/inputs-v3/poc_up_red.json'
    candidate_ref=dict(path=candidate_name,sha256=report['source_sha256'][candidate_name])
    entries=load(candidate_ref)['entries']
    # Explain a zero-intervention result from the sealed opportunity set,
    # without changing its selection or rerunning any account.
    states={d:market_state(adjusted,d)['state'] for d in {e['signal_date'] for e in entries}}
    counts=Counter(states[e['signal_date']] for e in entries)
    overlap=dict(candidate_source=candidate_ref,price_source=dict(path=adjusted_name,
        sha256=report['source_sha256'][adjusted_name]),candidate_count=len(entries),
        candidate_states={k:counts[k] for k in ('normal','weak','unknown')},
        scope='original_red_candidates_before_POC_hard_filter',
        note='The original candidate ledger already requires 0050 above MA120; this fixed new-buy rule has no opportunity when all candidates are normal.')
    sources.finish()
    return dict(schema='poc_market_sizing_publication_v1',created_at=datetime.now(timezone.utc).isoformat(),
        start=spec['start'],end=spec['end'],initial_cash=1000000,position_count=3,
        variant=VARIANT,rule=spec['rule'],preregistration=spec_ref,
        modes={k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()},
        cases=cases,comparisons=comparisons,reports=refs,attempts=attempts,
        candidate_market_overlap=overlap,
        completed_new_accounts=4,reused_accounts=8,
        checked_source_count=len(sources.used),historical_period_already_researched=True,
        actual_fill_verified=False,unseen_validation=False,paper_enabled=False,live_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reports',nargs='+',type=Path);a=p.parse_args()
    report=build(a.reports)
    from app.poc_market_sizing_ui import validate
    validate(report)
    output=ROOT/'artifacts/forward_simulation'/f'{IDENTIFIER}.json'
    sha=write(output,report);output.with_suffix('.sha256').write_text(sha+'\n')
    print(json.dumps(dict(path=str(output),sha256=sha,comparisons=report['comparisons']),ensure_ascii=False))
