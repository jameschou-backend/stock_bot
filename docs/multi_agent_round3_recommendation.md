# Multi-Agent Round3 Recommendation

## 最佳 agent 組合（ablation）
- 建議組合：`full_baseline`
- 權重：`{'tech': 0.3, 'flow': 0.37, 'margin': 0.1, 'fund': 0.13, 'theme': 0.1}`
- Sharpe: `1.3308490742910388`，total_return: `0.3841004626677331`

## Fund 是否值得優先補資料
- 結論：`是`（建議優先補）
- 原因：fund unavailable 多數與資料來源/對齊問題相關，會直接影響 agent 可用性與權重實際分配。

## Theme 是否應降權或關閉
- 建議：`暫時停用`
- 依據：theme 特徵與 final_score / 後續報酬相關性偏弱，支持訊號對入選後績效增益有限。

## 下一輪只做 1~2 件事
1. 修 fundamentals 月資料對齊到 rebalance date（最近可得值映射 + 覆蓋率監控）。
2. 對 theme 做特徵重建前先暫時降權/關閉，避免噪音拉低策略穩定度。