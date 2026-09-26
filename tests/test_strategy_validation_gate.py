from copy import deepcopy
import pytest
from skills.strategy_validation_gate import assess_family


def cases():
    return {f'{a}_{m}':dict(completed=True,summary=dict(total_return=1.,max_drawdown=-.2),
        metrics=dict(excess_return=.2,rolling252_win_rate=.7,
            annual_excess=[dict(year=str(y),excess=.1) for y in range(2022,2026)]))
        for a in ('equal','vol30','vol40','vol50') for m in range(8)}


def test_perfect_history_without_independent_evidence_cannot_promote_live():
    result=assess_family(cases())
    assert result['historical_robust_candidate'] and not result['live_qualified']
    assert 'unseen_validation' in result['missing_evidence']


def test_missing_stress_or_failed_neighbor_cannot_be_hidden():
    rows=cases();del rows['vol50_7']
    with pytest.raises(ValueError):assess_family(rows)
    rows=cases();rows['vol30_7']['metrics']['excess_return']=-.01
    assert not assess_family(rows)['historical_robust_candidate']
    rows=cases();rows['vol50_1']['completed']=False
    assert not assess_family(rows)['historical_robust_candidate']


def test_all_conditions_are_required_and_nonfinite_results_rejected():
    rows=cases();result=assess_family(rows)
    evidence={k:True for k in result['evidence_checks']}
    assert assess_family(rows,evidence=evidence)['live_qualified']
    for path,value in [(('summary','max_drawdown'),-.51),(('metrics','rolling252_win_rate'),.59)]:
        bad=deepcopy(rows);bad['vol40_7'][path[0]][path[1]]=value
        assert not assess_family(bad,evidence=evidence)['live_qualified']
    rows['vol40_7']['summary']['total_return']=float('nan')
    with pytest.raises(ValueError):assess_family(rows)
