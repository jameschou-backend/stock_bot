# Multi-Agent Shadow Operating Mode

## 目前運行模式（Production vs Shadow）

- **Production 主線**：`model` 選股流程（現行主策略）。
- **Shadow 高優先並行**：`4-agent multi-agent`（`tech/flow/margin/fund`，`theme=0`，固定 round4/round5 權重）。
- 兩者在每次 rebalance 後都要輸出可比較報表，不互相覆蓋交易決策。

## 每次 rebalance 後必看報表

1. `artifacts/shadow_monitor_latest.md`
   - 看 1/3/6 個月 window：return / sharpe / mdd / turnover / overlap
   - 看 attribution trend（tech/flow/margin/fund）是否穩定
   - 看 per-period regime tags（趨勢/震盪、強弱、高低波動）

2. `artifacts/market_regime_analysis.md`
   - 看不同 regime 下 model vs multi-agent 誰更強
   - 看哪些 regime 仍顯著落後
   - 看各 agent 在 regime 的影響權重變化

3. `artifacts/promotion_tracker.md`
   - 看 promotion criteria 各項 pass/fail
   - 看已達成數量與缺口
   - 看是否仍需維持 shadow only

4. `artifacts/tradability_gap_analysis.md`
   - 當 `D_tradability` 未過時，優先看 D1~D4 子項
   - 看 turnover / overlap / liquidity / switching risk 哪一項拖累
   - 看 1m/3m/6m 是否有改善趨勢

5. `artifacts/monthly_shadow_review_latest.md`
   - 每月決策摘要：本月 KPI、promotion criteria 變化、regime 摘要、tradability/transition 狀態
   - `Month-over-Month Delta` 可快速看上月 vs 本月的改善/惡化方向
   - 快速判斷建議 action：`continue shadow` / `prepare blended pilot` / `trigger promotion review` / `hold`
   - 避免逐份報表手動拼湊，降低判讀落差

## 何時啟動升級審查（Promotion Review）

當下列條件同時成立才啟動審查：

- `promotion_tracker` 全部 checks pass（或至少達成你預先定義的最低門檻組合）。
- 連續多個 shadow 視窗（非單次）結果穩定，不是單一期間偶發勝出。
- regime 一致性可接受（不能只在單一 regime 有優勢，其他 regime 明顯落後）。
- 可運行穩定性持續達標（invalid/degraded/fund coverage）。
- `D_tradability` 的子項（D1~D4）至少連續 2 次更新全部通過，或僅剩 1 個輕微邊界未通過且有明顯改善趨勢。

## 現階段操作建議

- 維持 `model` 為 production。
- `multi-agent` 持續 shadow 自動監控。
- 每次 rebalance 先更新 `shadow_monitor`、`tradability_gap_analysis`、`promotion_tracker`，再做人工判讀是否需要進入升級審查會議。

## 每月 Review 建議流程

每月固定做一次 `monthly shadow review`，建議依序檢查：

1. `artifacts/monthly_shadow_review_latest.md`（先看 action 建議與 blocker）
2. `artifacts/promotion_tracker.md`（核對 criteria 表格與 trend）
3. `artifacts/market_regime_analysis.md`（確認 regime 一致性）
4. `artifacts/promotion_transition_simulation.md`（確認切換政策可行性）

`monthly_shadow_review` 的用途是把跨報表資訊濃縮成可決策摘要，不取代原始分析；若月報顯示 `prepare blended pilot` 或 `trigger promotion review`，仍需回看原始 artifacts 做最終核准。

### Delta 區塊判讀重點

- 主要看 `return / sharpe / mdd / turnover / picks_stability / overlap / promotion_pass_ratio` 的 MoM 方向。
- 接近 promotion review 的常見訊號：
  - `promotion_pass_ratio` 連續上升；
  - `overlap` 上升、`turnover` 下降；
  - `mdd`（越接近 0 越好）改善，且 `sharpe` 不惡化；
  - `D2_overlap`、`D4_switching_risk` 對應 blocker 在月報中由 `worsened/no_change` 轉為 `improved`。

## D_tradability 未過時的優先檢查順序

1. **D2_overlap**：若 overlap 長期過低，先判斷是否屬於治理風險（切換成本/組合差異）而非純績效問題。  
2. **D4_switching_risk**：看單次 rebalance 替換比例是否過高（切換衝擊）。  
3. **D1_turnover**：看 multi-agent 是否持續高於 model 太多。  
4. **D3_liquidity**：看是否常落入低 `amt_20` 股票群。

接近可 promotion 的訊號：
- D1~D4 至少 3 項連續 2 次更新通過；
- 未通過項目有明顯改善（例如 overlap 逐步上升、replace_ratio 逐步下降）；
- 且 A/B/C 同時維持通過或接近通過。
