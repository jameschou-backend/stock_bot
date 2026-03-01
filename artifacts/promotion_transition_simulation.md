# Promotion Transition Simulation

- generated_at: `2026-02-28T16:48:05.004009Z`

## Baseline
- model_total_return_6m: 0.5520
- turnover(model/ma): 0.7667/0.8167
- overlap_6m: 0.0000
- replace_ratio_6m: 1.0000

## Scheme Comparison
### A_hard_switch
- total_return: 0.7423
- sharpe: 4.555757543268707
- max_drawdown: -0.0078
- turnover_impact: 0.8167
- overlap_proxy: 0.0000
- replacement_rate: 1.0000
- switching_risk_proxy: 2.0000
- delta_vs_model_total_return: 0.1903

### B_blended_70_30
- total_return: 0.6156
- sharpe: 3.236881415949673
- max_drawdown: -0.0087
- turnover_impact: 0.7817
- overlap_proxy: 0.7000
- replacement_rate: 0.3000
- switching_risk_proxy: 0.6000
- delta_vs_model_total_return: 0.0636

### B_blended_50_50
- total_return: 0.6551
- sharpe: 3.8749694586626866
- max_drawdown: -0.0085
- turnover_impact: 0.7917
- overlap_proxy: 0.5000
- replacement_rate: 0.5000
- switching_risk_proxy: 1.0000
- delta_vs_model_total_return: 0.1031

### C_gradual_adoption
- total_return: 0.6292
- sharpe: 4.100642751632745
- max_drawdown: -0.0085
- turnover_impact: 0.7892
- overlap_proxy: 0.5500
- replacement_rate: 0.4500
- switching_risk_proxy: 0.8500
- delta_vs_model_total_return: 0.0772

## Policy Recommendation
- recommended_now: `B_blended_70_30`
- default_policy_when_all_pass: `C_gradual_adoption`
- reason: 在滿足基本績效門檻下，切換風險代理值最低