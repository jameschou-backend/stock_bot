# Monthly Shadow Review

- generated_at: `2026-02-28T17:00:51.229687Z`
- month_window: `{'start_date': '2026-01-02', 'end_date': '2026-02-02', 'rebalance_dates': ['2026-01-02', '2026-02-02']}`

## 本月 model vs multi-agent 關鍵指標比較
| metric | model | multi_agent |
|---|---:|---:|
| total_return | 0.29209684483421205 | 0.1452587058674326 |
| max_drawdown | 0.0 | 0.0 |
| turnover | 0.65 | 0.85 |
| picks_stability | 0.35 | 0.15 |
| sharpe | N/A: zero std or <2 periods | N/A: zero std or <2 periods |

## Promotion criteria 變化摘要
- passed_checks: `1/4`
| criteria | current_status | previous_status | change | blocker |
|---|---|---|---|---|
| D1_turnover | pass | fail | improved |  |
| D2_overlap | fail | fail | no_change | model 與 multi-agent picks 重疊不足 |
| D3_liquidity | pass | pass | no_change |  |
| D4_switching_risk | fail | fail | no_change | 平均持股替換比例過高 |

## Month-over-Month Delta
- previous_generated_at: `2026-02-28T16:56:49.503897Z`
| metric | previous_month | current_month | delta | interpretation |
|---|---:|---:|---:|---|
| return | 0.145259 | 0.145259 | 0.000000 | flat |
| sharpe | N/A | N/A | N/A | N/A |
| mdd | 0.000000 | 0.000000 | 0.000000 | flat |
| turnover | 0.850000 | 0.850000 | 0.000000 | flat |
| picks_stability | 0.150000 | 0.150000 | 0.000000 | flat |
| overlap | 0.000000 | 0.000000 | 0.000000 | flat |
| promotion_pass_ratio | 0.250000 | 0.250000 | 0.000000 | flat |

## Regime 表現摘要
- strongest_relative_regime: `{'regime': '趨勢盤', 'periods': 5, 'model_avg_return': 0.010066625796653562, 'multi_agent_avg_return': 0.0225265072441816, 'multi_agent_minus_model': 0.012459881447528038, 'model_win_rate': 0.2, 'multi_agent_win_rate': 0.6}`
- non_negative_regime_count: `1.0`

## Tradability / Transition 狀態摘要
- D2_overlap: `False`
- D3_liquidity_status: `pass`
- D4_switching_risk: `False`
- avg_overlap_6m: `0.0`
- avg_replace_ratio_6m: `1.0`
- recommended_now: `B_blended_70_30`
- default_policy_if_promoted: `C_gradual_adoption`

## 建議 Action
- action: `continue shadow`
- reason: tradability 關鍵缺口仍在 overlap / switching risk
- action_candidates: `continue shadow` / `prepare blended pilot` / `trigger promotion review` / `hold`