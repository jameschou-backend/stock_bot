# Multi-Agent Round2 分析（基於最新修正 artifacts）

本分析使用下列輸入：
- `artifacts/experiment_summary_latest.md`
- `artifacts/experiment_matrix_summary.json`
- `artifacts/evaluation_*.json`
- `artifacts/agent_attribution_*.json`

## 1) model vs multi_agent 比較（round1）

以下為同一批 round1 實驗的平均指標（model: 2 組；multi-agent: 5 組）：

| group | total_return | max_drawdown | sharpe | turnover | picks_stability |
|---|---:|---:|---:|---:|---:|
| model (avg) | 0.6192 | -0.0289 | 2.1851 | 0.8173 | 0.1827 |
| multi_agent (avg) | 0.3563 | -0.0941 | 1.4789 | 0.9093 | 0.0907 |

觀察：
- 目前整體仍是 `model` 勝出（報酬、回撤、Sharpe、穩定度皆較好）。
- `multi_agent` 的 turnover 偏高，代表換手成本風險更大。
- 但在 strict 情境下，`ma_strict_baseline` 的 Sharpe（2.81）高於 `model_strict`（2.47），顯示 multi-agent 在某些資料品質條件下仍有潛力。

## 2) strict vs research 比較（round1）

| dq_mode | total_return(avg) | max_drawdown(avg) | sharpe(avg) | turnover(avg) | picks_stability(avg) |
|---|---:|---:|---:|---:|---:|
| strict | 0.4362 | -0.0342 | 2.6396 | 0.8950 | 0.1050 |
| research | 0.4295 | -0.0920 | 1.2971 | 0.8782 | 0.1218 |

觀察：
- strict 風險調整後績效較佳（Sharpe 明顯高），但 coverage 較少（會跳過 degraded period）。
- research 可以全期跑完，但在 fund degraded 期間效能被拉低。
- 研究/調參可持續用 research，實盤前仍需 strict gate 驗證。

## 3) baseline / tech-heavy / flow-heavy / defensive 比較（round1）

比較範圍：`ma_research_baseline`、`ma_research_tech_heavy`、`ma_research_flow_heavy`、`ma_research_defensive`。

| exp | total_return | max_drawdown | sharpe | turnover | picks_stability |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2720 | -0.1025 | 0.9114 | 0.9318 | 0.0682 |
| tech-heavy | 0.2989 | -0.1149 | 1.0045 | 0.9318 | 0.0682 |
| flow-heavy | 0.3787 | -0.0912 | 1.4845 | 0.8045 | 0.1955 |
| defensive | 0.3777 | -0.1029 | 1.1830 | 0.9182 | 0.0818 |

結論：
- `flow-heavy` 在報酬、Sharpe、turnover、stability 的綜合表現最佳。
- `defensive` 的報酬接近 `flow-heavy`，但穩定度與 Sharpe 較弱。
- 單純拉高 tech（tech-heavy）改善有限，且回撤更深。

## 4) agent attribution 結論

以 multi-agent 實驗聚合觀察：
- 最大貢獻 agent：`tech` 與 `flow` 交替主導，`flow-heavy` 版本由 `flow` 明顯主導（top contributor 168/240）。
- unavailable ratio 最高：`fund`（平均約 0.4；在 research degraded 期常見 `raw_fundamentals degraded`）。
- 與 final_score 相關性最高：`flow`（平均相關約 0.216），其次 `margin`（約 0.182）。
- 疑似噪音偏高：`theme`（平均相關約 0、平均 signal/confidence 近 0，且經常對最終排序貢獻有限）。

## 5) round2 權重建議

建議採用（研究模式）：
- `tech=0.30, flow=0.37, margin=0.10, fund=0.13, theme=0.10`

理由：
- 仍保留 tech 主軸，但把部分權重轉向在 attribution 與 round1 皆較有效的 flow。
- fund 因 unavailable 風險下修，但保留基本權重，避免日後 fundamental coverage 恢復後完全失去訊號。
- theme 先維持低配，等待下一輪先做特徵品質強化再決定是否放大。
