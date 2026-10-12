from copy import deepcopy
import pytest

from app.top3_validation_ui import validate
from skills.top3_validation import ARMS, MODES, qualification


def publication():
    cases = {}
    for arm in ARMS:
        for mode in MODES:
            metrics = dict(trading_days=666, total_return=.2 if arm=='benchmark' else .4,
                max_drawdown=-.2 if arm=='benchmark' else -.3, final_nav=1_400_000)
            cases[f'{arm}:{mode}'] = dict(id=arm,mode=mode,completed=True,metrics=metrics,
                total_return=metrics['total_return'], max_drawdown=metrics['max_drawdown'],
                excess_return=0 if arm=='benchmark' else .2, benchmark_max_drawdown=-.2,
                benchmark_comparable=True)
    return dict(schema='top3_validation_publication_v1',start='2024-01-02',end='2026-10-02',
        buy_fraction=.5,sell_fraction=.3,live_qualified=False,unseen_validation=False,actual_fill_verified=False,
        cases=cases,all_completed=True,completed_cases=16,
        modes={k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()},
        qualification={arm:qualification({mode:cases[f'{arm}:{mode}'] for mode in MODES}) for arm in ARMS[:-1]},
        paper=dict(enabled_strategies=[],simulated_fills_written=0),out_of_sample=dict(sessions=0))


def test_historical_success_remains_unseen_and_paper_blocked():
    report=publication()
    assert validate(report) == report
    assert all(not g['paper_enabled'] for g in report['qualification'].values())


@pytest.mark.parametrize('change',['table_metric','relative_metric','cost','paper','unseen','partial'])
def test_inconsistent_or_promoted_results_are_rejected(change):
    report=deepcopy(publication())
    if change=='table_metric':report['cases']['poc_up_red:baseline']['total_return']=9
    elif change=='relative_metric':report['cases']['poc_up_red:baseline']['excess_return']=9
    elif change=='cost':report['modes']['baseline']['slippage']=0
    elif change=='paper':report['paper']['enabled_strategies']=['poc_up_red']
    elif change=='unseen':report['out_of_sample']['sessions']=126
    else:report['cases']['poc_up_red:baseline']['completed']=False
    with pytest.raises(ValueError):validate(report)
