"""Separate reproducible historical performance from verified live readiness."""
import math


def assess_family(cases, *, central='vol40', neighbors=('vol30','vol50'), evidence=None):
    evidence = evidence or {}
    required = {f'{arm}_{m}' for arm in ('equal', central, *neighbors) for m in range(8)}
    if set(cases) != required:
        raise ValueError('Validation requires every predeclared arm and stress')
    by_arm = {}
    for arm in ('equal', central, *neighbors):
        rows = [cases[f'{arm}_{m}'] for m in range(8)]
        complete = all(r.get('completed') is True for r in rows)
        if not complete:
            by_arm[arm] = dict(completed=False, historical_stress_pass=False)
            continue
        for r in rows:
            values=[r['summary']['total_return'], r['summary']['max_drawdown'],
                    r['metrics']['excess_return'], r['metrics']['rolling252_win_rate']]
            if not all(type(v) in (int,float) and math.isfinite(v) for v in values):
                raise ValueError('Nonfinite validation input')
        wins=sum(r['metrics']['excess_return']>0 for r in rows)
        worst=min(r['summary']['max_drawdown'] for r in rows)
        rolling=all(rows[m]['metrics']['rolling252_win_rate']>=.60 for m in (0,7))
        years=[sum(a['excess']>0 for a in rows[m]['metrics']['annual_excess'] if a['year'] in ('2022','2023','2024','2025'))
               for m in (0,7)]
        by_arm[arm]=dict(completed=True, benchmark_winning_stresses=wins,
            worst_drawdown=worst, all_stresses_beat_benchmark=wins==8,
            drawdown_within_user_limit=worst>=-.50, endpoint_rolling252_pass=rolling,
            endpoint_positive_full_years=years, endpoint_annual_pass=all(n>=3 for n in years),
            historical_stress_pass=wins==8 and worst>=-.50)
    middle=by_arm[central]
    neighbors_pass=all(by_arm[a]['historical_stress_pass'] for a in neighbors)
    historical=bool(middle['historical_stress_pass'] and middle.get('endpoint_rolling252_pass')
                    and middle.get('endpoint_annual_pass') and neighbors_pass)
    checks={key:evidence.get(key) is True for key in (
        'full_account_reproduction', 'causal_data_audit', 'historical_source_audit',
        'complete_trial_registry', 'selection_adjusted_statistics', 'unseen_validation',
        'broker_execution_reconciliation')}
    return dict(schema='strategy_validation_gate_v1', central_arm=central, arms=by_arm,
        neighbor_stability_pass=neighbors_pass, historical_robust_candidate=historical,
        evidence_checks=checks, missing_evidence=[k for k,v in checks.items() if not v],
        live_qualified=historical and all(checks.values()),
        note='Historical gates are necessary screens, not a probability or guarantee of future profit')
