#!/usr/bin/env python3
"""Run preregistered protection, market-state and re-entry comparisons."""
from pathlib import Path
import sys, argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_midpoint as study
from scripts.research_midpoint_since_2025 import restart_inputs
from scripts.export_midpoint_2025_report import verify_cash
from skills.midpoint_risk_research import RiskResearchReplay, ARMS, protective_exit
from skills.trial_registry import append_trial_registry
from datetime import datetime
import pandas as pd

OUTPUT=ROOT/'.cache/midpoint-risk-20260928'


def run(output, prepare=False, arms=ARMS):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Use a new run directory')
    pub=study.old.load_selector()
    data,inputs,identity,repairs,refs=study.strict.repaired_data(pub)
    data=restart_inputs(data)
    extra,extra_refs=study.load_exit_completion(ROOT)
    additions=study.old.parent.parent.parent.load_corporate_completion(ROOT)|study.old.load_capital_terms(ROOT)[0]|extra
    overrides=(study.old.read(study.old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        study.old.read(study.old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        study.old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    sources=dict(pub['source_sha256'])|refs|extra_refs|study.old.file_identities([
        Path(__file__),ROOT/'skills/midpoint_risk_research.py',ROOT/'docs/prereg_midpoint_risk_20260928.md',
        ROOT/'scripts/research_midpoint_since_2025.py',*study.CODE],ROOT)
    provider=None
    if prepare:
        from scripts.prepare_mixed_odd_authorized import AuthorizedBudget,AuthorizedOdds,CACHE
        from skills.replay_market_feeds import ReplayMarketFeeds
        budget=AuthorizedBudget(CACHE/'budget.json',maximum={'finmind':0,'official':119})
        provider=ReplayMarketFeeds(OUTPUT/'prepared-feeds',offline=False,http_get=budget.official,official_min_interval=5)
    results={}

    def replay():
        for arm in arms:
            print('start',arm,flush=True)
            odds=AuthorizedOdds(ROOT,inputs/'execution-feeds',provider) if prepare else study.OddDailyCache(ROOT,inputs/'execution-feeds')
            ticks=study.strict.AdditionalTicks()
            feeds=study.old.ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
            corp=study.old.TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
            engine=RiskResearchReplay(data.quotes,data.companies,data.days,data.entries,feeds,corp,
                start=data.start,end=data.end,ticks=ticks,participation=.01,liquidity_identity=identity,
                odd_feeds=odds,ordering='original',position_count=3,factor_mask=0,residual_policy='release',
                identity_report=identity,exit_signals=data.features,
                action_dates=list(zip(data.events.stock_id,data.events.event_date)),risk_arm=arm)
            try:
                account=engine.run()
                study.old.validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
                audit=study.audit_high_return_resources(account,engine.resource_plans,engine.slot_decisions,
                    engine.board_decisions,engine.residual_days,data.quotes)
                markets=study.market_routes([*ticks.queries,*odds.queries])
                audit.update(study.audit_midpoint(account,ticks,odds,markets,data.quotes,data.days,corp,feeds))
                if arm=='control':
                    expected=study.old.read(ROOT/'.cache/midpoint-since-2025-20260928/final-a/original.json')['account']
                    if account!=expected:raise ValueError('Control changed')
                    audit['control_identical']=True
                cohorts={c['event_id']:c for c in account['cohorts']}
                for row in engine.risk_log:
                    i=data.days.get_loc(pd.Timestamp(row['date']))
                    c=cohorts[row['event_id']];sid=c['stock_id']
                    entry=data.days.get_loc(pd.Timestamp(c['entry_date']))
                    peak=data.features.adjusted_close[sid].iloc[entry:i].max()
                    state=dict(entry_index=entry,entry_price=data.features.price(entry,sid),peak_price=peak)
                    context=data.features.context(i,sid,state)
                    if context!=row['context'] or row['signal_date']!=str(data.days[i-1].date()):
                        raise ValueError('Risk decision differs from causal close prefix')
                    expected_reason=('market_defensive' if arm in ('market','combined','reentry') and row['market_slots']==0
                                     else protective_exit(context,{'trail_only':'trail','weak_only':'weak'}.get(arm,'both'))
                                     if arm in ('protect','combined','reentry','trail_only','weak_only','protect_reentry') else None)
                    if row['reason']!=expected_reason:raise ValueError('Risk decision reason changed')
                gates={row['date']:row for row in engine.entry_gate_log}
                for day,row in gates.items():
                    i=data.days.get_loc(pd.Timestamp(day))
                    signal=str(data.days[i-1].date())
                    if row['signal_date']!=signal or row['slots']!=engine.market_schedule.loc[signal,'slots']:
                        raise ValueError('Market gate used a different signal date')
                    if len(row['accepted'])>max(0,row['slots']-row['occupied']):
                        raise ValueError('Market entry slots exceeded')
                for trade in account['trades']:
                    if gates and trade['side']=='buy' and trade['event_id'] not in gates[trade['date']]['accepted']:
                        raise ValueError('Buy was not allowed by market gate')
                for row in engine.reentry_log:
                    i=data.days.get_loc(pd.Timestamp(row['date']));sid=row['stock_id']
                    close=data.features.adjusted_close[sid]
                    volume=engine.fields['volume'][sid]
                    old_i=data.days.get_loc(pd.Timestamp(row['exit_date']))
                    if not (row['signal_date']==str(data.days[i-1].date()) and 5<=i-1-old_i<=63
                            and close.iloc[i-1]>close.iloc[i-11:i-1].max()
                            and close.iloc[i-1]>data.features.ma20[sid].iloc[i-1]
                            and data.features.relative20[sid].iloc[i-1]>0
                            and volume.iloc[i-21:i-1].notna().sum()==20
                            and volume.iloc[i-1]>=1.5*volume.iloc[i-21:i-1].mean()):
                        raise ValueError('Re-entry conditions do not reconstruct')
                audit.update(risk_decisions_rebuilt=True,market_entry_gate_checked=True,
                             reentry_conditions_checked=True)
                result=dict(completed=True,summary=study.old.summarize(account),account=account,audit=audit,
                    risk_log=engine.risk_log,entry_gate_log=engine.entry_gate_log,reentry_log=engine.reentry_log)
                verify_cash(result)
            except (study.old.ReplayDataUnavailable,study.old.UnresolvedAction) as exc:
                result=dict(completed=False,summary=None,reason=str(exc),completed_sessions=len(engine.daily))
            if ticks.calls:raise ValueError('Unexpected tick request')
            sources.update(ticks.files|odds.files)
            path=output/(arm+'.json');study.old.write(path,result)
            results[arm]=dict(completed=result['completed'],summary=result['summary'],reason=result.get('reason'),
                             result_path=str(path.relative_to(ROOT)),sha256=study.old.sha(path))
            append_trial_registry(dict(timestamp=datetime.now().isoformat(timespec='seconds'),
                source='midpoint_risk_20260928',params=dict(arm=arm),result_path=str(path.relative_to(ROOT)),
                completed=result['completed'],preparation=prepare,live_qualified=False))
            print(arm,{k:result['summary'][k] for k in ('total_return','max_drawdown','final_nav')} if result['completed'] else result,flush=True)
    if prepare:replay()
    else:
        with study.old.offline_only():replay()
    if not prepare and study.old.file_identities([ROOT/p for p in sources],ROOT)!=sources:
        raise ValueError('Sources changed during replay')
    report=dict(start=data.start,end=data.end,cases=results,preparation=prepare,
        all_completed=all(r['completed'] for r in results.values()),source_sha256=sources,
        actual_fill_verified=False,live_qualified=False,unseen_validation=False)
    study.old.write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--prepare',action='store_true')
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    args=p.parse_args();run(args.output,args.prepare,tuple(args.arms))
