from copy import deepcopy

import pytest

from skills.backtest_case_dependencies import configurations, dependencies


def evaluate(name, config=None, odd=None):
    config = configurations()[name] if config is None else config
    evidence = dict(name=name,ordinary={'required_sessions':12},
        odd_lot={'required_sessions':int(not config['board_only']) if odd is None else odd})
    return dependencies(name,config,evidence)


def test_both_sector_arms_depend_on_membership_even_without_turnover_filter():
    for arm in ('relative_strength','strength_with_turnover'):
        row=evaluate('sector:'+arm+'_control_mixed')
        assert row['historical_industry_required'] is True
        assert row['universe_required'] is True


def test_diffusion_correlation_groups_do_not_require_finmind_industry_archive():
    row=evaluate('corporate:capacity_control_mixed')
    assert row['historical_industry_required'] is False
    assert row['universe_required'] is True
    assert row['corporate_event_evidence_required'] is True


def test_explicit_benchmark_keeps_own_identity_actions_and_execution_requirements():
    row=evaluate('corporate:benchmark_control_mixed')
    required={d['code'] for d in row['dependencies'] if d['required']}
    assert not row['historical_industry_required'] and not row['universe_required']
    assert {'case_dated_market_identity','corporate_event_terms_and_delivery',
        'ordinary_complete_authenticated_sessions','odd_lot_complete_authenticated_sessions'} <= required
    assert row['strict_data_ready'] is False and row['live_qualified'] is False


def test_news_factor_exclusion_never_waives_corporate_action_timing():
    for name in configurations():
        row=evaluate(name)
        assert row['financial_news_archive_required'] is False
        assert row['corporate_event_evidence_required'] is True


def test_unknown_arm_or_changed_input_config_needs_new_review():
    name='sector:relative_strength_control_mixed'
    config=deepcopy(configurations()[name]); config['revenue_surprise']=True
    with pytest.raises(ValueError,match='Unreviewed'):
        evaluate(name,config)
    with pytest.raises(ValueError,match='Unreviewed'):
        evaluate('new:news_strategy',config)


def test_board_only_policy_cannot_hide_actual_executable_odd_order():
    with pytest.raises(ValueError,match='odd-lot'):
        evaluate('corporate:capacity_control_board_only',odd=1)
