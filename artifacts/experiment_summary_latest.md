# Experiment Summary Latest

## Performance Table

| experiment_id | mode | dq | total_return | cagr | max_dd | sharpe | stability | degraded_ratio | overlap_vs_baseline | invalid_result | degenerate_multi_agent |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| model_strict_20260228 | model | strict | 0.41808788427864796 | 0.4197899744298663 | -0.009284771896615251 | 2.46802579851969 | 0.16999999999999998 | 0.5 | 1.0 | False | False |
| model_research_20260228 | model | research | 0.8202348936483455 | 0.823982764670731 | -0.0484971747152696 | 1.9021197133106282 | 0.19545454545454544 | 0.5 | 1.0 | False | False |
| ma_strict_baseline_20260228 | multi_agent | strict | 0.4543946701388655 | 0.45626675783288206 | -0.05920966035194786 | 2.8112645152513562 | 0.04 | 0.5 | 0.0 | False | False |
| ma_research_baseline_20260228 | multi_agent | research | 0.2719787198527215 | 0.2730299942644663 | -0.10250794856520995 | 0.9113508801900632 | 0.06818181818181818 | 0.5 | 0.0 | False | False |
| ma_research_tech_heavy_20260228 | multi_agent | research | 0.2989257185333496 | 0.3000928560868419 | -0.11490770219523816 | 1.004477260571825 | 0.06818181818181819 | 0.5 | 0.0 | False | False |
| ma_research_flow_heavy_20260228 | multi_agent | research | 0.37868845153538166 | 0.38020969498073964 | -0.09115631493212073 | 1.4845029141698027 | 0.19545454545454544 | 0.5 | 0.0 | False | False |
| ma_research_defensive_20260228 | multi_agent | research | 0.3776658209485433 | 0.3791824216772415 | -0.10294286726149782 | 1.1830133127717009 | 0.08181818181818182 | 0.5 | 0.0 | False | False |

## Model vs Multi-Agent
- model count: `2`
- multi_agent count: `5`

## Strict vs Research
- strict count: `2`
- research count: `5`

## Weights Comparison
- compare `weights_requested` / `weights_used` in `experiment_matrix_summary.json`

## Agent Attribution Summary
- model_strict_20260228: top_contributor_distribution={}
- model_research_20260228: top_contributor_distribution={}
- ma_strict_baseline_20260228: top_contributor_distribution={'flow': 21, 'tech': 99}
- ma_research_baseline_20260228: top_contributor_distribution={'tech': 200, 'flow': 39, 'margin': 1}
- ma_research_tech_heavy_20260228: top_contributor_distribution={'tech': 240}
- ma_research_flow_heavy_20260228: top_contributor_distribution={'flow': 168, 'tech': 67, 'margin': 5}
- ma_research_defensive_20260228: top_contributor_distribution={'tech': 165, 'flow': 43, 'margin': 32}

## Conclusion
- 最佳績效組：`model_research_20260228`（total_return=0.8202348936483455）
- 最穩定組：`model_research_20260228`（picks_stability=0.19545454545454544）
- degraded 下最韌性組：`model_research_20260228`（total_return=0.8202348936483455）