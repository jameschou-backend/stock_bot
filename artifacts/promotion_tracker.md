# Promotion Tracker

- updated_at: `2026-02-28T16:56:49.370994Z`
- passed/total: `1/4`
- all_passed: `False`

| check | pass |
|---|---|
| A_stability | False |
| B_regime_consistency | False |
| C_operational_stability | True |
| D_tradability | False |

## Missing Conditions
- A_stability
- B_regime_consistency
- D_tradability

## D Tradability Sub-Checks
- D1_turnover: `True`
- D2_overlap: `False`
- D3_liquidity: `True`
- D3_liquidity_status: `pass`
- D3_observability_ready: `True`
- D4_switching_risk: `False`

## D Trend Readability
| criteria | current_status | previous_status | trend | blocker | notes |
|---|---|---|---|---|---|

| D1_turnover | pass | fail | improving |  | gap_3m=0.1333, gap_6m=0.0500 |
| D2_overlap | fail | fail | flat_or_worse | model 與 multi-agent picks 重疊不足 | overlap_3m=0.0, overlap_6m=0.0 |
| D3_liquidity | pass | pass | observable |  | avg_low_liq_ratio_6m=0.025000000000000005 |
| D4_switching_risk | fail | fail | flat_or_worse | 平均持股替換比例過高 | replace_ratio_3m=1.0, replace_ratio_6m=1.0 |

## Transition Policy (If Promoted)
- default_policy_when_all_pass: `C_gradual_adoption`

## Check Details
- A_stability: `{'checked_windows': [{'window': '1m', 'rebalance_dates': 2, 'qualified_for_A': False, 'ret_check': False, 'sharpe_check': False, 'mdd_check': True, 'window_pass': False}, {'window': '3m', 'rebalance_dates': 4, 'qualified_for_A': False, 'ret_check': True, 'sharpe_check': True, 'mdd_check': True, 'window_pass': True}, {'window': '6m', 'rebalance_dates': 7, 'qualified_for_A': True, 'ret_check': True, 'sharpe_check': True, 'mdd_check': True, 'window_pass': True}], 'qualified_windows': 1, 'passed_windows': 1}`
- B_regime_consistency: `{'non_negative_regimes': 1, 'no_major_bad_underperform': True, 'rows': [{'regime': '趨勢盤', 'periods': 5, 'model_avg_return': 0.010066625796653562, 'multi_agent_avg_return': 0.0225265072441816, 'multi_agent_minus_model': 0.012459881447528038, 'model_win_rate': 0.2, 'multi_agent_win_rate': 0.6, 'source': 'trend_regime', 'period_ratio': 0.45454545454545453}, {'regime': '震盪盤', 'periods': 6, 'model_avg_return': 0.1047024910752072, 'multi_agent_avg_return': 0.07797163676308728, 'multi_agent_minus_model': -0.02673085431211991, 'model_win_rate': 0.8333333333333334, 'multi_agent_win_rate': 0.8333333333333334, 'source': 'trend_regime', 'period_ratio': 0.5454545454545454}, {'regime': '低波動', 'periods': 5, 'model_avg_return': 0.056963997264449985, 'multi_agent_avg_return': 0.04783960589332091, 'multi_agent_minus_model': -0.009124391371129074, 'model_win_rate': 0.6, 'multi_agent_win_rate': 0.6, 'source': 'vol_regime', 'period_ratio': 0.45454545454545453}, {'regime': '高波動', 'periods': 6, 'model_avg_return': 0.06562134818537683, 'multi_agent_avg_return': 0.05687738788880453, 'multi_agent_minus_model': -0.008743960296572303, 'model_win_rate': 0.5, 'multi_agent_win_rate': 0.8333333333333334, 'source': 'vol_regime', 'period_ratio': 0.5454545454545454}]}`
- C_operational_stability: `{'model_invalid_result_ok': True, 'multi_agent_invalid_result_ok': True, 'degraded_ratio_ok': True, 'fund_yoy_coverage_ok': True, 'fund_mom_coverage_ok': True}`
- D_tradability: `{'D1_turnover': True, 'D2_overlap': False, 'D3_liquidity': True, 'D3_liquidity_status': 'pass', 'D3_observability_ready': True, 'D4_switching_risk': False, 'values': {'model_turnover_6m': 0.7666666666666667, 'multi_agent_turnover_6m': 0.8166666666666665, 'avg_overlap_6m': 0.0, 'avg_low_liquidity_ratio_6m': 0.025000000000000005, 'avg_replace_ratio_6m': 1.0}, 'improvement_trend': {'turnover_gap_1m': 0.19999999999999996, 'turnover_gap_3m': 0.1333333333333333, 'turnover_gap_6m': 0.04999999999999982, 'overlap_1m': 0.0, 'overlap_3m': 0.0, 'overlap_6m': 0.0}, 'readability_rows': [{'criteria': 'D1_turnover', 'current_status': 'pass', 'previous_status': 'fail', 'trend': 'improving', 'blocker': '', 'notes': 'gap_3m=0.1333, gap_6m=0.0500'}, {'criteria': 'D2_overlap', 'current_status': 'fail', 'previous_status': 'fail', 'trend': 'flat_or_worse', 'blocker': 'model 與 multi-agent picks 重疊不足', 'notes': 'overlap_3m=0.0, overlap_6m=0.0'}, {'criteria': 'D3_liquidity', 'current_status': 'pass', 'previous_status': 'pass', 'trend': 'observable', 'blocker': '', 'notes': 'avg_low_liq_ratio_6m=0.025000000000000005'}, {'criteria': 'D4_switching_risk', 'current_status': 'fail', 'previous_status': 'fail', 'trend': 'flat_or_worse', 'blocker': '平均持股替換比例過高', 'notes': 'replace_ratio_3m=1.0, replace_ratio_6m=1.0'}]}`