#!/usr/bin/env python3
"""Fixed original/cap40/0050 accounts over an explicitly verified period."""
from datetime import datetime
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import research_midpoint as study
from scripts.research_midpoint_since_2025 import restart_inputs
from scripts.export_midpoint_2025_report import verify_cash
from skills.partial_risk import PartialRisk
from skills.partial_risk_audit import audit_partial_risk, audit_core_resources
from skills.midpoint_exit_replay import MidpointExitReplay
from skills.midpoint_exit_audit import audit_midpoint_exit
from skills.partial_risk_period import validate_period, reconcile_runs
from skills.trial_registry import append_trial_registry

CODE = [Path(__file__), ROOT/'skills/partial_risk_period.py',
        ROOT/'skills/partial_risk.py', ROOT/'skills/partial_risk_audit.py',
        ROOT/'scripts/research_midpoint_since_2025.py',
        ROOT/'skills/midpoint_exit_replay.py', ROOT/'skills/midpoint_exit_audit.py',
        ROOT/'docs/prereg_partial_risk_period_20260929.md', *study.CODE]


def run(output, start, prepare=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new run directory')
    pub = study.old.load_selector()
    data, inputs, identity, repairs, refs = study.strict.repaired_data(pub)
    # The quoted warmup calendar starts in 2021; the verified selector does not.
    validate_period(start, data.end, selector_start=data.start, selector_end=data.end,
        identity_start=identity['coverage_start'], identity_end=identity['coverage_end'],
        sessions=[str(d.date()) for d in data.days])
    data = restart_inputs(data, start=start)
    extra, extra_refs = study.load_exit_completion(ROOT)
    additions = study.old.parent.parent.parent.load_corporate_completion(ROOT) | study.old.load_capital_terms(ROOT)[0] | extra
    overrides = (study.old.read(study.old.parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        study.old.read(study.old.parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        study.old.read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    sources = dict(pub['source_sha256']) | refs | extra_refs | study.old.file_identities(CODE, ROOT)
    provider = None
    if prepare:
        from scripts.prepare_mixed_odd_authorized import AuthorizedBudget, AuthorizedOdds, CACHE
        from skills.replay_market_feeds import ReplayMarketFeeds
        local = ROOT/'.cache/partial-risk-period-20260929'
        proof = local/'official-recovery.json'
        if not proof.exists():
            study.old.write(proof, study.old.read(CACHE/'official-recovery.json'))
        budget = AuthorizedBudget(local/'budget.json', maximum={'finmind': 0, 'official': 80})
        provider = ReplayMarketFeeds(local/'feeds', offline=False, http_get=budget.official, official_min_interval=5)
    cases = {}

    def replay():
        for arm in ('original', 'cap40', 'benchmark'):
            print('start', start, arm, flush=True)
            odds = AuthorizedOdds(ROOT, inputs/'execution-feeds', provider) if prepare else study.OddDailyCache(ROOT, inputs/'execution-feeds')
            ticks = study.strict.AdditionalTicks()
            if arm == 'benchmark':
                result = study.case(data, inputs, identity, additions, benchmark=True, odds=odds, ticks=ticks)
                if result['completed']:
                    verify_cash(result)
            else:
                feeds = study.old.ReplayMarketFeeds(inputs/'execution-feeds', offline=True)
                corp = study.old.TrackedCorporateActions(data.events, inputs/'dividends', None, offline=True, overrides=overrides)
                cls = MidpointExitReplay if arm == 'original' else PartialRisk
                options = {} if arm == 'original' else dict(stop_events=data.events, risk_arm='cap40')
                engine = cls(data.quotes, data.companies, data.days, data.entries, feeds, corp,
                    start=data.start, end=data.end, ticks=ticks, participation=.01, liquidity_identity=identity,
                    odd_feeds=odds, ordering='original', position_count=3, factor_mask=0,
                    residual_policy='release', identity_report=identity, exit_signals=data.features,
                    action_dates=list(zip(data.events.stock_id, data.events.event_date)), **options)
                try:
                    account = engine.run()
                    study.old.validate_completed_account(account, [str(d.date()) for d in data.days], data.start, data.end)
                    audit = audit_core_resources(account, engine, data.quotes) if arm == 'cap40' else study.audit_high_return_resources(
                        account, engine.resource_plans, engine.slot_decisions, engine.board_decisions, engine.residual_days, data.quotes)
                    audit.update(audit_midpoint_exit(account, ticks, odds, study.market_routes([*ticks.queries, *odds.queries]),
                        data.quotes, data.days, corp, feeds))
                    if arm == 'cap40':
                        audit['partial_risk'] = audit_partial_risk(account, data, engine)
                    result = dict(completed=True, summary=study.old.summarize(account), account=account, audit=audit)
                    verify_cash(result)
                except (study.old.ReplayDataUnavailable, study.old.UnresolvedAction, ValueError) as exc:
                    result = dict(completed=False, summary=None, reason=str(exc), completed_sessions=len(engine.daily))
            if ticks.calls:
                raise ValueError('Unexpected tick network request')
            sources.update(ticks.files | odds.files)
            if prepare:
                result = dict(completed=result['completed'], reason=result.get('reason'), summary=None)
            path = output/(arm+'.json')
            study.old.write(path, result)
            cases[arm] = dict(completed=result['completed'], summary=result['summary'], reason=result.get('reason'),
                path=str(path.relative_to(ROOT)), sha256=study.old.sha(path))
            append_trial_registry(dict(timestamp=datetime.now().isoformat(timespec='seconds'), source='partial_risk_period_20260929',
                params=dict(start=start, end=data.end, arm=arm), preparation=prepare, completed=result['completed'],
                result_path=str(path.relative_to(ROOT)), live_qualified=False))
            print(arm, {k: result['summary'][k] for k in ('total_return', 'max_drawdown', 'final_nav')} if result.get('summary') else result, flush=True)

    if prepare:
        replay()
    else:
        with study.old.offline_only():
            replay()
    if study.old.file_identities([ROOT/p for p in sources], ROOT) != sources:
        raise ValueError('Sources changed during period comparison')
    report = dict(start=data.start, end=data.end, initial_cash=1_000_000, preparation=prepare, cases=cases,
        all_completed=all(r['completed'] for r in cases.values()), source_sha256=sources,
        actual_fill_verified=False, unseen_validation=False, live_qualified=False)
    study.old.write(output/'report.json', report)
    return report


def verify(left, right, output):
    paths = [Path(left).resolve(), Path(right).resolve()]
    if paths[0] == paths[1] or Path(output).exists():
        raise ValueError('Use independent runs and a new publication')
    reports = [study.old.read(p/'report.json') for p in paths]
    reconcile_runs(*reports)
    refs = dict(reports[0]['source_sha256'])
    for arm in reports[0]['cases']:
        cases = [study.old.read(p/(arm+'.json')) for p in paths]
        if cases[0] != cases[1]:
            raise ValueError('Independent accounts differ: '+arm)
        for folder, report, case in zip(paths, reports, cases):
            path = folder/(arm+'.json')
            if study.old.sha(path) != report['cases'][arm]['sha256']:
                raise ValueError('Run account changed')
            refs.update(study.old.file_identities([path, folder/'report.json'], ROOT))
            verify_cash(case)
    if study.old.file_identities([ROOT/p for p in refs], ROOT) != refs:
        raise ValueError('Publication sources changed')
    report = dict(reports[0], source_sha256=refs, offline_identical=True)
    study.old.write(output, report)
    Path(output).with_suffix('.sha256').write_text(study.old.sha(output)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--start', default='2022-01-03')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--compare', nargs=2, type=Path)
    a = p.parse_args()
    if a.prepare and a.compare:
        p.error('Preparation cannot publish performance')
    if a.compare:
        verify(*a.compare, a.output)
    else:
        run(a.output, a.start, a.prepare)
