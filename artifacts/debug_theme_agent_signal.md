# Debug Theme Agent Signal

- baseline_experiment_id: `full_baseline_20260228`

## Theme Features Distribution
| feature | count | mean | std | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| theme_hot_score_feat | 0 | None | None | None | None | None |
| theme_return_20_feat | 0 | None | None | None | None | None |
| theme_turnover_ratio_feat | 0 | None | None | None | None | None |

## Correlation
- theme_hot_score_feat: corr(final_score)=None, corr(next_period_return)=None
- theme_return_20_feat: corr(final_score)=None, corr(next_period_return)=None
- theme_turnover_ratio_feat: corr(final_score)=None, corr(next_period_return)=None

## 支持入選 vs 不支持入選（theme_signal）
- support(signal>=1): `{'count': 0, 'avg_next_period_return': None, 'win_rate_next_period': None}`
- non_support(signal<=0): `{'count': 240, 'avg_next_period_return': 0.03514167996827072, 'win_rate_next_period': 0.3958333333333333}`

## 建議
- 暫時停用