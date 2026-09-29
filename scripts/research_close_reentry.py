#!/usr/bin/env python3
"""Compare preregistered reclaim entries after close-confirmed stops."""
from pathlib import Path
import argparse
from datetime import datetime
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_midpoint as study
from scripts.research_midpoint_since_2025 import restart_inputs
from scripts.export_midpoint_2025_report import verify_cash
from skills.midpoint_exit_audit import audit_midpoint_exit
from skills.close_reentry import CloseReentry, ARMS
from skills.close_reentry_audit import audit_reentries
from skills.close_confirmed_audit import audit_close_stops
from skills.trial_registry import append_trial_registry

SPEC = ROOT/'docs/prereg_close_reentry_20260929.md'
CODE = [Path(__file__), ROOT/'skills/close_reentry.py', ROOT/'skills/close_reentry_audit.py', ROOT/'skills/close_confirmed_exit.py', ROOT/'skills/close_confirmed_audit.py', ROOT/'skills/midpoint_exit_replay.py', ROOT/'skills/midpoint_exit_audit.py', SPEC]


def run(output, prepare=False):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Use a new run directory')
    pub = study.old.load_selector()
    data, inputs, identity, repairs, refs = study.strict.repaired_data(pub)
    data = restart_inputs(data)
    extra, extra_refs = study.load_exit_completion(ROOT)
    additions = study.old.parent.parent.parent.load_corporate_completion(ROOT)|study.old.load_capital_terms(ROOT)[0]|extra
    overrides = (study.old.read(study.old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        study.old.read(study.old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        study.old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    sources = dict(pub['source_sha256'])|refs|extra_refs|study.old.file_identities([
        *CODE, ROOT/'scripts/research_midpoint_since_2025.py',*study.CODE], ROOT)
    provider = None
    if prepare:
        from scripts.prepare_mixed_odd_authorized import AuthorizedBudget,AuthorizedOdds,CACHE
        from skills.replay_market_feeds import ReplayMarketFeeds
        local = ROOT/'.cache/close-reentry-20260929'
        proof = local/'official-recovery.json'
        if not proof.exists():
            study.old.write(proof,study.old.read(CACHE/'official-recovery.json'))
        budget = AuthorizedBudget(local/'budget.json',maximum={'finmind':0,'official':40})
        provider = ReplayMarketFeeds(ROOT/'.cache/close-reentry-20260929/prepared-feeds',offline=False,
            http_get=budget.official,official_min_interval=5)
    cases = {}
    def replay():
        for arm in (ARMS[1:] if prepare else ARMS):
            print('start',arm,flush=True)
            odds = AuthorizedOdds(ROOT,inputs/'execution-feeds',provider) if prepare else study.OddDailyCache(ROOT,inputs/'execution-feeds')
            ticks = study.strict.AdditionalTicks()
            feeds = study.old.ReplayMarketFeeds(inputs/'execution-feeds',offline=True)
            corp = study.old.TrackedCorporateActions(data.events,inputs/'dividends',None,offline=True,overrides=overrides)
            engine = CloseReentry(data.quotes,data.companies,data.days,data.entries,feeds,corp,start=data.start,end=data.end,
                ticks=ticks,participation=.01,liquidity_identity=identity,odd_feeds=odds,ordering='original',
                position_count=3,factor_mask=0,residual_policy='release',identity_report=identity,
                exit_signals=data.features,action_dates=list(zip(data.events.stock_id,data.events.event_date)),
                stop_events=data.events,reentry_arm=arm)
            try:
                account = engine.run()
                study.old.validate_completed_account(account,[str(d.date()) for d in data.days],data.start,data.end)
                audit = study.audit_high_return_resources(account,engine.resource_plans,engine.slot_decisions,
                    engine.board_decisions,engine.residual_days,data.quotes)
                audit.update(audit_midpoint_exit(account,ticks,odds,study.market_routes([*ticks.queries,*odds.queries]),data.quotes,data.days,corp,feeds))
                if arm == 'control':
                    expected_path = ROOT/'.cache/close-confirmed-20260929/final-a/close15.json'
                    if account != study.old.read(expected_path)['account']: raise ValueError('Control changed')
                    sources.update(study.old.file_identities([expected_path],ROOT))
                audit['independent_close_stops'] = audit_close_stops(account,engine.exit_states,data,engine)
                if arm!='control':
                    audit['reentries'] = audit_reentries(account,data)
                result = dict(completed=True,summary=study.old.summarize(account),account=account,audit=audit,
                              intraday_stop=False,network_calls=ticks.calls)
                verify_cash(result)
            except (study.old.ReplayDataUnavailable,study.old.UnresolvedAction) as exc:
                result = dict(completed=False,summary=None,reason=str(exc),completed_sessions=len(engine.daily))
            if ticks.calls: raise ValueError('Unexpected tick request')
            if prepare:
                result = dict(completed=result['completed'],reason=result.get('reason'),summary=None)
            sources.update(ticks.files|odds.files)
            path = output/(arm+'.json'); study.old.write(path,result)
            cases[arm] = dict(completed=result['completed'],summary=result['summary'],reason=result.get('reason'),
                path=str(path.relative_to(ROOT)),sha256=study.old.sha(path))
            append_trial_registry(dict(timestamp=datetime.now().isoformat(timespec='seconds'),source='close_reentry_20260929',
                params=dict(arm=arm,sell_price='HL2',close_drawdown=.15,reentry_window=20),result_path=str(path.relative_to(ROOT)),
                completed=result['completed'],preparation=prepare,live_qualified=False))
            print(arm,{k:result['summary'][k] for k in ('total_return','max_drawdown','final_nav')} if result.get('summary') else result,flush=True)
    if prepare: replay()
    else:
        with study.old.offline_only(): replay()
    if not prepare and study.old.file_identities([ROOT/p for p in sources],ROOT)!=sources:
        raise ValueError('Sources changed')
    report = dict(start=data.start,end=data.end,cases=cases,source_sha256=sources,preparation=prepare,
        all_completed=all(c['completed'] for c in cases.values()),
        intraday_stop=False,actual_fill_verified=False,unseen_validation=False,live_qualified=False)
    study.old.write(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();run(a.output,a.prepare)
