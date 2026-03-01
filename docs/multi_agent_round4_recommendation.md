# Multi-Agent Round4 Recommendation

## 結論摘要

- fundamentals 對齊修正後（as-of snapshot, no lookahead），`fund` agent 已可實際參與。
- 在 round4 OOS 驗證，`4-agent (tech+flow+margin+fund, theme=0)` 相較 `3-agent (tech+flow+margin)` 出現明顯增益。
- 與 model 比較：multi-agent 已接近但整體仍略弱，建議維持 **shadow only / 次策略**，尚不建議取代 model 主流程。

## 1) fund 修正後是否有實質增益

有，且在 validation 期增益明確：

- `4-agent_tfmf vs 3-agent_tfm`（validation）
  - `total_return`: `+0.1054`
  - `sharpe`: `+1.7032`
  - `max_drawdown`: `0.0`（兩者同為 0）
- `fund_effective_gain=true`（見 `artifacts/round4_summary.json`）

同時，`debug_fund_alignment_round4.md` 顯示：
- fund feature 非空率由 `0%` 提升至約 `84%~88%`（yoy/mom/trend）
- 低覆蓋日期數（yoy/mom < 50%）為 `0`

## 2) 3-agent 是否比目前 5-agent 名義版更合理

是。現階段更合理的是：
- 以 `3-agent` 或 `4-agent(with fund)` 為主
- `theme` 維持停用（`theme weight=0`）

理由：
- round3/round4 診斷都顯示 theme 訊號貢獻弱、有效樣本不足。
- round4 加回 fund 後可觀察到增益，但 theme 仍未證明有效。

## 3) multi-agent 是否接近 model，或仍應 shadow only

目前判斷：**接近，但仍建議 shadow only**。

- round4 validation：
  - model baseline：`total_return=0.2684`, `sharpe=2.2249`, `max_dd=-0.0093`
  - best multi-agent（`ma_4agent_flow_up`）：`total_return=0.2570`, `sharpe=4.3340`, `max_dd=0.0`
- multi-agent 風險指標很漂亮，但 validation 期間短、picks_stability/turnover 與 model 仍有差異，尚不足以直接升主模式。

## 建議下一步（限 1~2 件）

1. 固定 `theme=0`，在 4-agent（tech/flow/margin/fund）做小範圍穩健性檢驗（多個 rolling split，不擴功能）。
2. 把 fund 對齊邏輯下沉到正式 feature pipeline 重算流程（非僅評估時覆寫），確保訓練/回測/線上一致口徑。
