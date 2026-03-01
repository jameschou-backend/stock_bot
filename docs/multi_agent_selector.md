# Multi Agent Selector

## Agent 分工與使用欄位

- `tech`
  - `ret_20`, `breakout_20`, `rsi_14`, `macd_hist`, `drawdown_60`, `vol_ratio_20`
- `flow`
  - `foreign_net_20`, `trust_net_20`, `dealer_net_20`, `chip_flow_intensity_20`
- `margin`
  - `margin_balance_chg_20`, `short_balance_chg_20`, `margin_short_ratio`
- `fund`
  - `fund_revenue_yoy`, `fund_revenue_mom`, `fund_revenue_trend_3m`
- `theme`
  - `theme_hot_score`, `theme_return_20`, `theme_turnover_ratio`

## Deterministic Rules（簡述）

- 每個 agent 輸出：`signal(-2..2)` + `confidence(0..1)` + `reasons` + `risk_flags`
- Aggregator 先把 signal 映射成 `[-1,-0.5,0,0.5,1]`，再乘 confidence
- 最終分數：`sum(weight_agent * agent_score)`
- 若 `risk_flags` 含 `high_drawdown` / `liquidity_risk`，final_score 會扣分（乘上 0.9）

## Degraded 行為

- `raw_institutional` 缺失：`flow` agent 直接 `unavailable`
- `raw_margin_short` 缺失：`margin` agent 直接 `unavailable`
- `raw_fundamentals` 缺失：`fund` agent 直接 `unavailable`
- `raw_theme_flow` 缺失：`theme` agent 直接 `unavailable`
- unavailable agent 的權重會重分配（剩餘權重正規化為 1）

## 如何啟用與比較

- 啟用模式：
  - `SELECTION_MODE=model`
  - `SELECTION_MODE=multi_agent`
- 每次 `daily_pick` 成功會在 `artifacts/` 產生：
  - `run_manifest_daily_pick_{job_id}.json`
- 比較兩次 run：
  - `python scripts/compare_runs.py --a <job_id_a> --b <job_id_b>`
  - 或 `python scripts/compare_runs.py --path-a <pathA> --path-b <pathB>`
- 輸出：
  - stdout JSON
  - `artifacts/compare_{a}_{b}.md`
