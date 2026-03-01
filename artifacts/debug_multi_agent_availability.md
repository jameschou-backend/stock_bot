# Debug Multi Agent Availability

## Agent Required Columns
- tech: `['ret_20', 'breakout_20', 'rsi_14', 'macd_hist', 'drawdown_60', 'vol_ratio_20']`
- flow: `['foreign_net_20', 'trust_net_20', 'dealer_net_20', 'chip_flow_intensity_20']`
- margin: `['margin_balance_chg_20', 'short_balance_chg_20', 'margin_short_ratio']`
- fund: `['fund_revenue_yoy', 'fund_revenue_mom', 'fund_revenue_trend_3m']`
- theme: `['theme_hot_score', 'theme_return_20', 'theme_turnover_ratio']`

## Experiment Diagnostics
### ma_strict_baseline_20260228
- feature_columns_observed_count(raw): `26`
- feature_columns_input_selector_count: `39`
- degenerate_multi_agent: `False`
- weights_used_avg: `{'tech': 0.3499999999999999, 'flow': 0.29999999999999993, 'margin': 0.10000000000000003, 'fund': 0.14999999999999997, 'theme': 0.10000000000000003}`

| agent | required_cols | present_in_feature_df |
|---|---:|---:|
| tech | 6 | 6 |
| flow | 4 | 4 |
| margin | 3 | 3 |
| fund | 3 | 3 |
| theme | 3 | 3 |

Top unavailable reasons:

### ma_research_baseline_20260228
- feature_columns_observed_count(raw): `26`
- feature_columns_input_selector_count: `39`
- degenerate_multi_agent: `False`
- weights_used_avg: `{'tech': 0.38088235294117645, 'flow': 0.32647058823529407, 'margin': 0.10882352941176475, 'theme': 0.10882352941176475, 'fund': 0.14999999999999997}`

| agent | required_cols | present_in_feature_df |
|---|---:|---:|
| tech | 6 | 6 |
| flow | 4 | 4 |
| margin | 3 | 3 |
| fund | 3 | 3 |
| theme | 3 | 3 |

Top unavailable reasons:
- fund: raw_fundamentals degraded (120)

### ma_research_tech_heavy_20260228
- feature_columns_observed_count(raw): `26`
- feature_columns_input_selector_count: `39`
- degenerate_multi_agent: `False`
- weights_used_avg: `{'tech': 0.5277777777777778, 'flow': 0.2111111111111112, 'margin': 0.1055555555555556, 'theme': 0.1055555555555556, 'fund': 0.10000000000000003}`

| agent | required_cols | present_in_feature_df |
|---|---:|---:|
| tech | 6 | 6 |
| flow | 4 | 4 |
| margin | 3 | 3 |
| fund | 3 | 3 |
| theme | 3 | 3 |

Top unavailable reasons:
- fund: raw_fundamentals degraded (120)

### ma_research_flow_heavy_20260228
- feature_columns_observed_count(raw): `26`
- feature_columns_input_selector_count: `39`
- degenerate_multi_agent: `False`
- weights_used_avg: `{'tech': 0.2638888888888889, 'flow': 0.47500000000000003, 'margin': 0.1055555555555556, 'theme': 0.1055555555555556, 'fund': 0.10000000000000003}`

| agent | required_cols | present_in_feature_df |
|---|---:|---:|
| tech | 6 | 6 |
| flow | 4 | 4 |
| margin | 3 | 3 |
| fund | 3 | 3 |
| theme | 3 | 3 |

Top unavailable reasons:
- fund: raw_fundamentals degraded (120)

### ma_research_defensive_20260228
- feature_columns_observed_count(raw): `26`
- feature_columns_input_selector_count: `39`
- degenerate_multi_agent: `False`
- weights_used_avg: `{'tech': 0.24285714285714288, 'flow': 0.24285714285714288, 'margin': 0.12142857142857144, 'theme': 0.24285714285714288, 'fund': 0.29999999999999993}`

| agent | required_cols | present_in_feature_df |
|---|---:|---:|
| tech | 6 | 6 |
| flow | 4 | 4 |
| margin | 3 | 3 |
| fund | 3 | 3 |
| theme | 3 | 3 |

Top unavailable reasons:
- fund: raw_fundamentals degraded (120)

## Root Cause for `degraded_datasets=['raw_fundamentals']` but others unavailable
- 主要原因是 `feature_df` 缺少 multi-agent 所需欄位，造成 `tech/flow/theme` 被判定為 `missing columns`，與 degraded dataset 無關。
- 修正後改為在評估階段強制補齊 `FEATURE_COLUMNS + AGENT_REQUIRED_COLUMNS`，缺值以 imputation 補齊，避免「欄位不存在」誤判 unavailable。