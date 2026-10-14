from copy import deepcopy
import pytest

from app.poc_risk_controls_ui import validate, case_detail
from skills.top3_validation import MODES
from app.poc_risk_controls_ui import VARIANTS


@pytest.fixture
def report():
    cases = {}
    for arm in ('poc_up_red', 'poc_priority', 'benchmark'):
        for variant in (('original',) if arm == 'benchmark' else ('original', *VARIANTS)):
            for mode in MODES:
                ret = .8 if arm == 'benchmark' else 1. if variant == 'original' else .6
                metrics = dict(total_return=ret, max_drawdown=-.3, final_nav=1000000*(1+ret), trading_days=666)
                cases[f'{arm}:{variant}:{mode}'] = dict(id=arm, variant=variant, mode=mode, completed=True,
                    summary=deepcopy(metrics), metrics=metrics, excess_return=ret-.8,
                    same_funded_path_as_original=False,
                    control_audit=dict(passed=True, variant=variant),
                    analytics=dict(wins=1, losses=1, breakeven=0, settled=2, win_rate=.5))
    return dict(schema='poc_risk_controls_publication_v1', start='2024-01-02', end='2026-10-02',
        initial_cash=1000000, position_count=3, buy_fraction=.5, sell_fraction=.3,
        live_qualified=False, paper_enabled=False, actual_fill_verified=False, unseen_validation=False,
        modes={k:dict(commission=v[0], slippage=v[1]) for k,v in MODES.items()},
        expected_new_accounts=16, completed_new_accounts=16, cases=cases,
        variants=list(VARIANTS), execution_audit=dict(completed=True,passed=False,whole_account_replayed=False,blockers=["odd chronology missing"]),
        decisions={f"{arm}:{variant}":dict(improvement_pass=False, all_cost_improvement=False, historical_stress_pass=False,
            unseen_validation=False, causal_fill_verified=False, paper_enabled=False, live_qualified=False)
            for arm in ('poc_up_red','poc_priority') for variant in VARIANTS})


def test_fixed_matrix_and_incomplete_account_have_no_false_promotion(report):
    assert validate(report) is report
    case = report['cases']['poc_up_red:chase3:combined']
    case.update(completed=False, summary=None, metrics=None, analytics=None)
    report['completed_new_accounts'] = 15
    assert validate(report) is report


@pytest.mark.parametrize('mutation', ['missing_case', 'wrong_cost', 'wrong_benchmark', 'win_denominator',
    'nan', 'promote', 'gate', 'false_complete', 'partial_return'])
def test_rejects_inconsistent_or_unsafe_publications(report, mutation):
    row = report['cases']['poc_up_red:chase3:combined']
    if mutation == 'missing_case': del report['cases']['poc_priority:chase3:fee2']
    if mutation == 'wrong_cost': report['modes']['combined']['slippage'] = .0045
    if mutation == 'wrong_benchmark': row['excess_return'] += .1
    if mutation == 'win_denominator': row['analytics']['win_rate'] = .75
    if mutation == 'nan': row['metrics']['total_return'] = float('nan')
    if mutation == 'promote': report['paper_enabled'] = True
    if mutation == 'gate': report['decisions']['poc_up_red:chase3']['improvement_pass'] = True
    if mutation == 'false_complete': report['completed_new_accounts'] = 9
    if mutation == 'partial_return': row['completed'] = False
    with pytest.raises(ValueError): validate(report)


def test_case_file_cannot_escape_repo_or_bypass_digest(report, tmp_path):
    row = report['cases']['poc_up_red:chase3:baseline']
    row.update(path='../other.json', sha256='0'*64)
    with pytest.raises(ValueError, match='路徑'):
        case_detail(report, 'poc_up_red__chase3__baseline', tmp_path)
    row['path'] = 'case.json'
    (tmp_path/'case.json').write_text('{}')
    with pytest.raises(ValueError, match='指紋'):
        case_detail(report, 'poc_up_red__chase3__baseline', tmp_path)


@pytest.mark.parametrize('key', ['passed','whole_account_replayed'])
def test_frozen_order_diagnostic_cannot_be_promoted(report, key):
    report['execution_audit'][key]=True
    with pytest.raises(ValueError): validate(report)


def test_missing_independent_decision_audit_fails_closed(report):
    del report['cases']['poc_up_red:vol20:baseline']['control_audit']
    with pytest.raises(ValueError, match='獨立決策'):
        validate(report)


def test_changed_account_cannot_claim_unchanged_path(report):
    report['cases']['poc_up_red:vol20:baseline']['same_funded_path_as_original'] = True
    with pytest.raises(ValueError, match='相同帳戶路徑'):
        validate(report)


def test_baseline_improvement_is_separate_from_fee_stress(report):
    key = 'poc_up_red:chase3:baseline'
    row = report['cases'][key]
    for metrics in (row['summary'], row['metrics']):
        metrics.update(total_return=1.1, final_nav=2100000.)
    row['excess_return'] = 1.1-.8
    decision = report['decisions']['poc_up_red:chase3']
    decision['improvement_pass'] = True
    assert validate(report) is report
    assert decision['all_cost_improvement'] is False
    assert report['paper_enabled'] is False
