# Multi-Agent Go-Forward Recommendation

## 明確結論

目前**不建議**直接把 multi-agent 設為主選股模式；建議維持 `model` 為主，multi-agent 作為平行研究線。

## 為什麼不是現在切主模式

- OOS 驗證中，最佳 round2 multi-agent（`slightly_flow_up`）在 validation 期：
  - total_return = `0.1610`，低於 model = `0.2684`
  - sharpe = `2.2008`，略低於 model = `2.2249`
- round1 全體平均也顯示 model 在報酬/回撤/穩定度上仍領先。
- `fund` agent 在 research 條件仍有較高 unavailable 比例，造成權重實際可用性不穩定。

## 若要持續使用 multi-agent（研究/次要策略）

建議暫定權重：
- `tech=0.30, flow=0.37, margin=0.10, fund=0.13, theme=0.10`（`slightly_flow_up`）

此組在 development period 同時拿到：
- best by return
- best by risk-adjusted（Sharpe）

## 下一輪最值得優化（只選 1~2 個）

1. `fund` agent  
   - 先提升 fundamentals coverage 與可用率，降低 degraded 時段 unavailable 對整體配置的扭曲。

2. `theme` agent  
   - 目前與 final_score 相關性接近 0，需先確認特徵定義/訊噪比，再決定是否保留 10% 權重。
