# Round4 Summary

## Development
| name | mode | total_return | sharpe | max_drawdown | turnover | picks_stability |
|---|---|---:|---:|---:|---:|---:|
| model_baseline | model | 0.426420665864073 | 1.9401716863080487 | -0.0484971747152696 | 0.85 | 0.15 |
| ma_3agent_tfm | multi_agent | 0.08713419130442968 | 0.6168739306454133 | -0.08216795516637443 | 0.9071428571428573 | 0.09285714285714286 |
| ma_4agent_tfmf | multi_agent | 0.2701496981550662 | 1.6523620638053425 | -0.07465671800096585 | 0.9285714285714286 | 0.07142857142857142 |
| ma_4agent_flow_up | multi_agent | 0.17993839373415943 | 1.1831130381749142 | -0.0702470297068798 | 0.9 | 0.09999999999999999 |

## Validation
| name | mode | total_return | sharpe | max_drawdown | turnover | picks_stability |
|---|---|---:|---:|---:|---:|---:|
| model_baseline | model | 0.2684100737110251 | 2.2249074842850503 | -0.00928477189661514 | 0.7000000000000001 | 0.3 |
| ma_3agent_tfm | multi_agent | 0.16098139343252038 | 2.200795879775914 | 0.0 | 0.85 | 0.15 |
| ma_4agent_tfmf | multi_agent | 0.26634030525005103 | 3.9039691835908314 | 0.0 | 0.8666666666666667 | 0.13333333333333333 |
| ma_4agent_flow_up | multi_agent | 0.2569890012199354 | 4.333951544941467 | 0.0 | 0.8333333333333334 | 0.16666666666666666 |

## 4-agent vs 3-agent（fund 增益）
- development delta: return=0.1830155068506365, sharpe=1.0354881331599293, max_dd=0.00751123716540858
- validation delta: return=0.10535891181753065, sharpe=1.7031733038149173, max_dd=0.0
- fund_effective_gain: `True`

## Fund Alignment
- debug report: `/Users/james.chou/JamesProject/stock_bot/artifacts/debug_fund_alignment_round4.md`