# Multi-Agent Promotion Criteria（Shadow -> 候補主線）

## 目的

定義 4-agent multi-agent（`tech/flow/margin/fund`, `theme=0`）何時能從 shadow strategy 升級為候補主線，並對照目前 round5 狀態。

## Promotion 門檻（建議）

### A. 連續觀測穩定性（必要）

- 連續至少 `3` 個 shadow observation 視窗（每視窗 >= 6 個 rebalance dates）。
- 其中至少 `2/3` 視窗滿足：
  - multi-agent `total_return >= model - 5%`（相對差距）
  - multi-agent `Sharpe >= model - 0.20`
  - multi-agent `MDD <= model + 0.03`（允許略大但不可顯著惡化）

### B. Regime 一致性（必要）

- 在主要 regime（趨勢/震盪、高/低波動）中，至少有 `2` 個 regime 的 `multi_agent_minus_model >= 0`。
- 不可出現單一重要 regime（占比 >= 30%）下持續顯著落後（`ma_minus_model < -0.03`）。

### C. 可運行穩定性（必要）

- `invalid_result = false`、`degenerate_multi_agent = false`。
- `degraded_period_ratio` 在可接受範圍（建議 <= 20%）且不影響 fund/核心 agent 可用性。
- fund 對齊 debug 連續觀測中，`yoy/mom` 非空率維持 >= 80%。

### D. 交易可行性（必要）

- turnover 不可長期顯著高於 model（建議 <= model + 0.12）。
- picks overlap 若長期接近 0，需額外人工審核（策略分歧風險與 explainability）。

## 目前 round5 對照

依據：
- `artifacts/shadow_observation_summary.md`
- `artifacts/market_regime_analysis.md`

### 已達成

- fund 對齊可用性：`debug_fund_alignment_round4` 顯示 fund 核心欄位非空率明顯提升（round4 已驗證）。
- 近 6 個 rebalance 的 shadow 視窗：
  - multi-agent 績效優於 model（total_return / Sharpe / MDD）。

### 尚未達成

- 缺少「連續多視窗」證據：目前只有一個 shadow observation 視窗，不足以做 promotion。
- regime 一致性仍不夠：
  - `趨勢盤` multi-agent 相對較強
  - `震盪盤` multi-agent 相對較弱（ma_minus_model < 0）
- picks overlap 長期接近 0，需額外風險評估與人工審核。

## 目前建議

- 仍維持：`shadow only`（高優先 shadow）。
- 不升級到候補主線的主因：
  1. regime 表現尚未全面一致
  2. 觀測期數不足（缺連續多視窗）

## 下一步（最小增量）

1. 固定同一組 4-agent 權重，連續再跑 2 個 shadow observation 視窗。
2. 在每個視窗重算 regime report，確認「震盪盤」是否收斂，若無收斂再微調 `tech/flow` 配比（小幅）。
