# Market Regime Analysis

- period: `2025-03-01 ~ 2026-02-28`
- multi_agent_weights(theme=0): `{'tech': 0.29, 'flow': 0.46, 'margin': 0.12, 'fund': 0.13, 'theme': 0.0}`

## Trend Regime
| regime | periods | model_avg_return | multi_agent_avg_return | ma_minus_model |
|---|---:|---:|---:|---:|
| 趨勢盤 | 5 | 0.010066625796653562 | 0.0225265072441816 | 0.012459881447528038 |
| 震盪盤 | 6 | 0.1047024910752072 | 0.07797163676308728 | -0.02673085431211991 |

## Strength Regime
| regime | periods | model_avg_return | multi_agent_avg_return | ma_minus_model |
|---|---:|---:|---:|---:|
| 弱勢盤 | 11 | 0.06168618867586464 | 0.0527693051635847 | -0.008916883512279941 |

## Volatility Regime
| regime | periods | model_avg_return | multi_agent_avg_return | ma_minus_model |
|---|---:|---:|---:|---:|
| 低波動 | 5 | 0.056963997264449985 | 0.04783960589332091 | -0.009124391371129074 |
| 高波動 | 6 | 0.06562134818537683 | 0.05687738788880453 | -0.008743960296572303 |

## Multi-Agent 相對較強 Regime
- 趨勢盤: ma_minus_model=0.0125
- 高波動: ma_minus_model=-0.0087
- 弱勢盤: ma_minus_model=-0.0089

## Agent Impact by Regime（avg_abs_contrib）
- trend_regime: `[{'regime': '趨勢盤', 'agent': 'fund', 'avg_abs_contrib': 0.22639963494148183, 'avg_signed_contrib': 0.22558419486525855}, {'regime': '趨勢盤', 'agent': 'tech', 'avg_abs_contrib': 0.19082374855777845, 'avg_signed_contrib': 0.18775839594213284}, {'regime': '趨勢盤', 'agent': 'flow', 'avg_abs_contrib': 0.18740283291998797, 'avg_signed_contrib': 0.18288295349933517}, {'regime': '趨勢盤', 'agent': 'margin', 'avg_abs_contrib': 0.09174460110442446, 'avg_signed_contrib': 0.08271278921169936}, {'regime': '震盪盤', 'agent': 'fund', 'avg_abs_contrib': 0.2290509732242631, 'avg_signed_contrib': 0.2271560511526289}, {'regime': '震盪盤', 'agent': 'tech', 'avg_abs_contrib': 0.20397659877486543, 'avg_signed_contrib': 0.20310165353851042}, {'regime': '震盪盤', 'agent': 'flow', 'avg_abs_contrib': 0.18617668063173193, 'avg_signed_contrib': 0.18458627824751}, {'regime': '震盪盤', 'agent': 'margin', 'avg_abs_contrib': 0.10008267237970499, 'avg_signed_contrib': 0.08599751601509587}]`
- strength_regime: `[{'regime': '弱勢盤', 'agent': 'fund', 'avg_abs_contrib': 0.22772530408287248, 'avg_signed_contrib': 0.2263701230089437}, {'regime': '弱勢盤', 'agent': 'tech', 'avg_abs_contrib': 0.19740017366632193, 'avg_signed_contrib': 0.19543002474032167}, {'regime': '弱勢盤', 'agent': 'flow', 'avg_abs_contrib': 0.18678975677585993, 'avg_signed_contrib': 0.18373461587342263}, {'regime': '弱勢盤', 'agent': 'margin', 'avg_abs_contrib': 0.09591363674206473, 'avg_signed_contrib': 0.08435515261339761}]`
- vol_regime: `[{'regime': '低波動', 'agent': 'fund', 'avg_abs_contrib': 0.22699310981238827, 'avg_signed_contrib': 0.2256340060759034}, {'regime': '低波動', 'agent': 'flow', 'avg_abs_contrib': 0.22015781112105692, 'avg_signed_contrib': 0.21807105798615833}, {'regime': '低波動', 'agent': 'tech', 'avg_abs_contrib': 0.19209377203585085, 'avg_signed_contrib': 0.19163739530609672}, {'regime': '低波動', 'agent': 'margin', 'avg_abs_contrib': 0.10685096647884737, 'avg_signed_contrib': 0.0910529434283849}, {'regime': '高波動', 'agent': 'fund', 'avg_abs_contrib': 0.22845749835335666, 'avg_signed_contrib': 0.22710623994198403}, {'regime': '高波動', 'agent': 'tech', 'avg_abs_contrib': 0.202706575296793, 'avg_signed_contrib': 0.19922265417454657}, {'regime': '高波動', 'agent': 'flow', 'avg_abs_contrib': 0.15342170243066294, 'avg_signed_contrib': 0.14939817376068684}, {'regime': '高波動', 'agent': 'margin', 'avg_abs_contrib': 0.08497630700528207, 'avg_signed_contrib': 0.07765736179841029}]`