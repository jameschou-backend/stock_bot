# Multi-Agent Transition Policy

## 建議切換方式

若未來 multi-agent 達標，建議優先採用 `B_blended_70_30`（model 70% / multi-agent 30%）作為預設起始切換策略，而非直接 hard switch。  
原因是目前模擬中，`B_blended_70_30` 在維持正向報酬增益的前提下，`switching_risk_proxy` 最低，且 turnover 衝擊可控。

## 為什麼不是 hard switch

`A_hard_switch` 雖然 6m 報酬最高，但在目前結構下：

- model 與 multi-agent picks overlap 幾乎為 0
- 平均替換比例接近 100%
- 初始切換成本與操作風險最高（switching risk proxy 顯著高於 blended / gradual）

因此在 overlap 與 switching risk 未明顯改善前，不建議作為預設策略。

## 什麼條件下不應 hard switch

下列任一條件成立時，不應 hard switch：

- `avg_overlap_6m < 0.10`
- `avg_replace_ratio_6m > 0.80`
- `D_tradability` 尚未通過（特別是 `D2_overlap` 或 `D4_switching_risk` 未過）
- 近 3 次 rebalance 沒有連續穩定優於 model（風險調整後）

## 目前 transition readiness 判斷

目前不具備 hard switch readiness。  
雖然 D3_liquidity 已可觀測且可判定，但 D2/D4 仍未過，表示主要障礙仍在「組合分歧過大」與「切換成本過高」。

## 實務執行建議

1. Promotion 初期使用 `B_blended_70_30`，連續觀察 2~3 個 rebalance。  
2. 若 overlap 與 replacement 指標改善，才考慮移動到 `B_blended_50_50` 或 `C_gradual_adoption`。  
3. 只有在 D_tradability 四項皆穩定通過後，才評估 hard switch。
