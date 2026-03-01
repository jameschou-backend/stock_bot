# OOS Summary

## Split
- split_rule: `rebalance_dates_count_70_30`
- development: `2025-03-01 ~ 2025-11-02`
- validation: `2025-11-03 ~ 2026-02-28`
- rebalance_count: `12`

## Development Results
| name | total_return | sharpe | max_drawdown | turnover | picks_stability |
|---|---:|---:|---:|---:|---:|
| baseline_round2 | -0.045968852917201986 | -0.06273581979275912 | -0.10250794856520995 | 0.9500000000000001 | 0.05 |
| slightly_tech_up | -0.02380920479960369 | 0.04092981799364026 | -0.12118672647107509 | 0.9357142857142858 | 0.0642857142857143 |
| slightly_flow_up | 0.08713419130442968 | 0.6168739306454133 | -0.08216795516637443 | 0.9071428571428573 | 0.09285714285714286 |
| slightly_defensive | 0.050828491323258396 | 0.4126730833352126 | -0.10088864824139465 | 0.942857142857143 | 0.05714285714285715 |

- development best by return: `slightly_flow_up` (total_return=0.08713419130442968)
- development best by risk-adjusted: `slightly_flow_up` (sharpe=0.6168739306454133)

## Validation Comparison
- model_research total_return: `0.2684100737110251`
- model_research sharpe: `2.2249074842850503`
- chosen_multi_agent (`slightly_flow_up`) total_return: `0.16098139343252038`
- chosen_multi_agent (`slightly_flow_up`) sharpe: `2.200795879775914`

## Overfit Assessment
- suspected_overfit: `True`
- rule: `suspected_when_chosen_multi_agent_underperforms_model_on_return_and_sharpe`