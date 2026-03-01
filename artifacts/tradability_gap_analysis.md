# Tradability Gap Analysis

- generated_at: `2026-02-28T16:47:56.543543Z`

## 1) Turnover Breakdown
- 1m: model=0.65, multi_agent=0.85
- 3m: model=0.7000000000000001, multi_agent=0.8333333333333334
- 6m: model=0.7666666666666667, multi_agent=0.8166666666666665

## 2) Overlap Breakdown
- 1m: avg_overlap=0.0000, low_overlap_dates=['2026-01-02']
- 3m: avg_overlap=0.0000, low_overlap_dates=['2025-11-03', '2025-12-01', '2026-01-02']
- 6m: avg_overlap=0.0000, low_overlap_dates=['2025-08-01', '2025-09-01', '2025-10-01', '2025-11-03', '2025-12-01', '2026-01-02']

## 3) Liquidity Breakdown
- 1m: avg_amt_20=3860241436.297625, avg_low_liquidity_ratio=0.1
- 3m: avg_amt_20=4081746505.3914666, avg_low_liquidity_ratio=0.03333333333333333
- 6m: avg_amt_20=4340384108.581004, avg_low_liquidity_ratio=0.025000000000000005

## 4) Switching Risk
- 1m: avg_replace_count=20.0, avg_replace_ratio=1.0000
- 3m: avg_replace_count=20.0, avg_replace_ratio=1.0000
- 6m: avg_replace_count=20.0, avg_replace_ratio=1.0000

## 5) 結論（主因排序）
1. overlap過低（策略分歧） (score=1.0000)
2. switching風險偏高（持股替換率） (score=1.0000)
3. turnover偏高 (score=0.0500)

- 策略問題：`turnover偏高`、`流動性偏弱`
- 治理問題：`overlap過低`、`switching風險偏高`