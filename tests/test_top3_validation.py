from copy import deepcopy
import pytest

from skills.top3_validation import MODES, cost_function, qualification, verify_costs


def test_stress_costs_keep_tax_and_minimum_and_do_not_mutate_original():
    from skills.million_replay import costs
    original = costs(100, 1000, 'sell', '2330')
    assert cost_function('baseline')(100, 1000, 'sell', '2330') == original
    combined = cost_function('combined')(100, 1000, 'sell', '2330')
    assert combined['commission'] == 285
    assert combined['slippage'] == 900
    assert combined['tax'] == original['tax'] == 300
    assert costs(100, 1000, 'sell', '2330') == original
    assert cost_function('fee2')(10, 1, 'buy', '0050')['commission'] == 20
    assert cost_function('combined')(100, 1000, 'sell', '0050')['tax'] == 100


def test_independent_cost_oracle_detects_wrong_stress():
    trade = dict(reference_price=100, qty=1000, side='sell', stock_id='2330',
        **cost_function('combined')(100, 1000, 'sell', '2330'))
    verify_costs({'trades':[trade]}, 'combined')
    with pytest.raises(ValueError, match='cost mismatch'):
        verify_costs({'trades':[trade]}, 'baseline')


def good_history():
    return {mode: dict(completed=True, total_return=2, excess_return=.2,
        max_drawdown=-.2, benchmark_max_drawdown=-.3) for mode in MODES}


def test_historical_winners_cannot_enable_paper_without_unseen_or_causal_fills():
    result = qualification(good_history())
    assert result['checks']['historical_stress_pass']
    assert not result['paper_enabled']
    forward = dict(version_locked_before_signals=True, source_verified=True, sessions=126,
        settled_cohorts=30, net_return=.2, excess_return=.1,
        drawdown_not_worse_than_benchmark=True, causal_fill_verified=False)
    assert not qualification(good_history(), forward)['paper_enabled']
    forward['causal_fill_verified'] = True
    assert qualification(good_history(), forward)['paper_enabled']
    assert not qualification(good_history(), forward)['live_qualified']
    for field in forward:
        missing = deepcopy(forward)
        del missing[field]
        assert not qualification(good_history(), missing)['paper_enabled']


def test_missing_stress_and_relative_drawdown_failure_are_not_passes():
    rows = good_history()
    rows['combined']['max_drawdown'] = -.4
    assert not qualification(rows)['checks']['historical_stress_pass']
    del rows['combined']
    assert not qualification(rows)['checks']['complete_stress_matrix']


def test_rolling_windows_include_starting_cash_and_match_dates():
    from scripts.summarize_top3_validation import windows
    rows = [dict(date=f'2024-01-0{i+1}', opening_nav=100, nav=n) for i,n in enumerate([100,110,121])]
    base = [dict(r,nav=100) for r in rows]
    result = windows(rows, base, horizon=2)
    assert result['windows'] == 2
    assert result['worst_excess'] == pytest.approx(.1)
    assert result['best_excess'] == pytest.approx(.21)
    assert result['overlapping'] and not result['unseen']
    with pytest.raises(ValueError, match='calendar'):
        windows(rows, base[:-1])


def test_private_audit_and_engine_cost_adapters_leave_shared_modules_untouched():
    from types import SimpleNamespace
    from skills.all_entry_official_execution_v2 import official_auditor
    from skills.top3_validation import install
    from skills import poc_range_audit, poc_range_execution, million_replay
    class Base:
        def __init__(self, *, buy_fraction, sell_fraction):
            poc_range_execution.validate_fractions(buy_fraction, sell_fraction)
            self.buy_fraction, self.sell_fraction = buy_fraction, sell_fraction
    runner = SimpleNamespace(engine_types=lambda: (Base,Base), _all_entry_indexed_audit=official_auditor())
    install(runner, 'combined')
    cls,_ = runner.engine_types()
    engine = cls(buy_fraction=.5, sell_fraction=.3)
    assert (engine.buy_fraction,engine.sell_fraction) == (.5,.3)
    assert engine._costs(100,1000,'buy','2330')['slippage'] == 900
    assert million_replay.costs(100,1000,'buy','2330')['slippage'] == 450
    with pytest.raises(ValueError):
        poc_range_audit._fractions(.5,.3)
    with pytest.raises(ValueError):
        poc_range_execution.validate_fractions(.5,.3)
