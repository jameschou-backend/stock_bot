# Debug Fund Agent Availability

- baseline_experiment_id: `full_baseline_20260228`

## Raw Fundamentals Coverage（按日期）
| date | exact_coverage | same_month_coverage |
|---|---:|---:|
| 2025-03-03 | 0.00% | 0.00% |
| 2025-04-01 | 94.57% | 94.57% |
| 2025-05-02 | 0.00% | 94.39% |
| 2025-06-02 | 0.00% | 95.20% |
| 2025-07-01 | 96.22% | 96.22% |
| 2025-08-01 | 96.91% | 96.91% |
| 2025-09-01 | 94.86% | 94.86% |
| 2025-10-01 | 95.50% | 95.50% |
| 2025-11-03 | 0.00% | 95.47% |
| 2025-12-01 | 97.01% | 97.01% |
| 2026-01-02 | 0.00% | 96.44% |
| 2026-02-02 | 0.00% | 92.54% |

## Feature 缺失率（fund）
| date | feature | missing_ratio |
|---|---|---:|
| 2025-03-03 | fund_revenue_yoy | 100.00% |
| 2025-03-03 | fund_revenue_mom | 100.00% |
| 2025-03-03 | fund_revenue_trend_3m | 100.00% |
| 2025-04-01 | fund_revenue_yoy | 100.00% |
| 2025-04-01 | fund_revenue_mom | 100.00% |
| 2025-04-01 | fund_revenue_trend_3m | 100.00% |
| 2025-05-02 | fund_revenue_yoy | 100.00% |
| 2025-05-02 | fund_revenue_mom | 100.00% |
| 2025-05-02 | fund_revenue_trend_3m | 100.00% |
| 2025-06-02 | fund_revenue_yoy | 100.00% |
| 2025-06-02 | fund_revenue_mom | 100.00% |
| 2025-06-02 | fund_revenue_trend_3m | 100.00% |
| 2025-07-01 | fund_revenue_yoy | 100.00% |
| 2025-07-01 | fund_revenue_mom | 100.00% |
| 2025-07-01 | fund_revenue_trend_3m | 100.00% |
| 2025-08-01 | fund_revenue_yoy | 100.00% |
| 2025-08-01 | fund_revenue_mom | 100.00% |
| 2025-08-01 | fund_revenue_trend_3m | 100.00% |
| 2025-09-01 | fund_revenue_yoy | 100.00% |
| 2025-09-01 | fund_revenue_mom | 100.00% |
| 2025-09-01 | fund_revenue_trend_3m | 100.00% |
| 2025-10-01 | fund_revenue_yoy | 100.00% |
| 2025-10-01 | fund_revenue_mom | 100.00% |
| 2025-10-01 | fund_revenue_trend_3m | 100.00% |
| 2025-11-03 | fund_revenue_yoy | 100.00% |
| 2025-11-03 | fund_revenue_mom | 100.00% |
| 2025-11-03 | fund_revenue_trend_3m | 100.00% |
| 2025-12-01 | fund_revenue_yoy | 100.00% |
| 2025-12-01 | fund_revenue_mom | 100.00% |
| 2025-12-01 | fund_revenue_trend_3m | 100.00% |
| 2026-01-02 | fund_revenue_yoy | 100.00% |
| 2026-01-02 | fund_revenue_mom | 100.00% |
| 2026-01-02 | fund_revenue_trend_3m | 100.00% |
| 2026-02-02 | fund_revenue_yoy | 100.00% |
| 2026-02-02 | fund_revenue_mom | 100.00% |
| 2026-02-02 | fund_revenue_trend_3m | 100.00% |

## 最常 unavailable 的日期 / 股票
- dates: `[('2025-03-03', 20), ('2025-05-02', 20), ('2025-06-02', 20), ('2025-11-03', 20), ('2026-01-02', 20), ('2026-02-02', 20)]`
- stocks: `[('2883', 4), ('2408', 3), ('2337', 3), ('2002', 2), ('2882', 2), ('2891', 2), ('2892', 2), ('1815', 2), ('2344', 2), ('2881', 2), ('2303', 2), ('6770', 2), ('2027', 1), ('1536', 1), ('2606', 1), ('4510', 1), ('8234', 1), ('2014', 1), ('4916', 1), ('2609', 1)]`

## unavailable 原因分類
- source missing: `21`
- feature engineering missing: `99`

## 建議修法（優先級）
1. 修正 fundamentals 對齊邏輯：以「月資料最近可得值」映射至 rebalance date，避免只用 trading_date 精確相等造成 coverage=0。
2. 在 feature engineering 增加 fund 欄位可追蹤旗標（source_found / aligned / imputed），把 unavailable 根因落地到 manifest。
3. 對長期無 fundamentals 的股票建立白名單排除或降權機制，避免 fund agent 對整體權重造成不必要擾動。